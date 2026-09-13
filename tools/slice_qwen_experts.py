# slice_qwen_experts.py: slice Qwen3.5-35B-A3B GGUF expert tensors into a
# .vmex v2 pack (one shard per (layer, expert): ffn_down + ffn_gate + ffn_up).
#
# Usage: python slice_qwen_experts.py <model.gguf> <out.vmex> [maxLayers]
#
# Pack layout v2 (see include/vulkan_vm/storage/pack_format.hpp):
#   [PackHeader 32B][ShardEntry x N (24B each, LE)][pad to 4K][blob_0]...
# ShardEntry: id u32, offset u64, size u32, hash u64 (FNV-1a64).

import struct
import sys
import time

MAGIC = 0x32454D56          # 'VME2' little-endian
VERSION = 2
BLOB_ALIGN = 4096
FLAGS = (3) | (2 << 2) | (2 << 4) | (3 << 6)  # off=u64 size=u32 id=u32 hash=u64

Q8_BLOCK = 32
Q8_OUT = 34                  # Q8_0: 32 elems -> 34 bytes

def fnv1a64(data, h=1469598103934665603):
    for b in data:
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h

def align_up(v, a):
    return (v + a - 1) & ~(a - 1)

def parse_gguf_header(f):
    f.seek(0)
    magic = f.read(4)
    assert magic == b'GGUF', "not a GGUF"
    ver = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_kv = struct.unpack('<Q', f.read(8))[0]

    SCALAR = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    def read_str():
        n = struct.unpack('<Q', f.read(8))[0]
        return f.read(n).decode('utf-8', 'replace')
    def read_val(t):
        if t == 8:
            return read_str()
        if t == 9:
            et = struct.unpack('<I', f.read(4))[0]
            cnt = struct.unpack('<Q', f.read(8))[0]
            if et in SCALAR:
                f.seek(cnt * SCALAR[et], 1)
            else:
                for _ in range(cnt):
                    read_val(et)
            return
        f.seek(SCALAR[t], 1)

    kv = {}
    for _ in range(n_kv):
        k = read_str()
        t = struct.unpack('<I', f.read(4))[0]
        if t == 8:
            kv[k] = read_val(t)
        else:
            read_val(t)  # consume (position matters more than value here)
    return n_tensors, kv

def tensor_table(f, n_tensors):
    SCALAR = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    def read_str():
        n = struct.unpack('<Q', f.read(8))[0]
        return f.read(n).decode('utf-8', 'replace')
    def read_val(t):
        if t == 8:
            return read_str()
        if t == 9:
            et = struct.unpack('<I', f.read(4))[0]
            cnt = struct.unpack('<Q', f.read(8))[0]
            if et in SCALAR:
                f.seek(cnt * SCALAR[et], 1)
            else:
                for _ in range(cnt):
                    read_val(et)
            return
        f.seek(SCALAR[t], 1)

    # NOTE: caller must have already consumed KV pairs.
    tensors = {}
    for _ in range(n_tensors):
        name = read_str()
        nd = struct.unpack('<I', f.read(4))[0]
        dims = struct.unpack('<' + 'Q' * nd, f.read(8 * nd))
        ttype = struct.unpack('<I', f.read(4))[0]
        off = struct.unpack('<Q', f.read(8))[0]
        tensors[name] = (off, dims, ttype)
    return tensors

def main():
    if len(sys.argv) < 3:
        print("usage: slice_qwen_experts.py <model.gguf> <out.vmex> [maxLayers]")
        return 1
    model_path, out_path = sys.argv[1], sys.argv[2]
    max_layers = int(sys.argv[3]) if len(sys.argv) > 3 else 16

    t0 = time.time()
    with open(model_path, 'rb') as f:
        n_tensors, kv = parse_gguf_header(f)
        n_layers = kv.get('qwen35moe.block_count', 40)
        n_experts = kv.get('qwen35moe.expert_count', 256)
        ffn_len = kv.get('qwen35moe.expert_feed_forward_length', 512)
        tensors = tensor_table(f, n_tensors)
        pos = f.tell()
        data_start = align_up(pos, 32)

    use_layers = min(max_layers, n_layers)
    print(f"model: {n_layers} layers x {n_experts} experts x ffn={ffn_len}")
    print(f"slicing layers 0..{use_layers-1} ({use_layers * n_experts} shards)")

    # Q8_0 bytes per expert per tensor = ne00*ne01 / 32 * 34
    def expert_slice_bytes(dims):
        nelems = dims[0] * dims[1]
        assert nelems % Q8_BLOCK == 0, "Q8_0 expects 32-elem blocks"
        return nelems // Q8_BLOCK * Q8_OUT

    # per-tensor expert slice size (all three are 512x2048 or 2048x512 = same)
    sizes = {}
    for kind in ('ffn_down_exps', 'ffn_gate_exps', 'ffn_up_exps'):
        name = f'blk.0.{kind}.weight'
        off, dims, ttype = tensors[name]
        assert ttype == 8, f"{name} not Q8_0 (ttype={tt})"
        sizes[kind] = expert_slice_bytes(dims)
    shard_bytes = sizes['ffn_down_exps'] + sizes['ffn_gate_exps'] + sizes['ffn_up_exps']
    print(f"per-expert: down={sizes['ffn_down_exps']:,} gate={sizes['ffn_gate_exps']:,} "
          f"up={sizes['ffn_up_exps']:,} shard={shard_bytes:,} bytes")

    n_shards = use_layers * n_experts
    table_bytes = 24 * n_shards
    data_offset = align_up(32 + table_bytes, BLOB_ALIGN)

    entries = []
    cursor = data_offset
    with open(out_path, 'wb') as out:
        # placeholder header + table (patched at the end)
        out.write(b'\0' * (data_offset - 32 + 32))  # header+table+pad region

        with open(model_path, 'rb') as f:
            for layer in range(use_layers):
                base = {}
                for kind in ('ffn_down_exps', 'ffn_gate_exps', 'ffn_up_exps'):
                    name = f'blk.{layer}.{kind}.weight'
                    off, dims, ttype = tensors[name]
                    base[kind] = (data_start + off, dims)
                d_bytes = sizes['ffn_down_exps']
                g_bytes = sizes['ffn_gate_exps']
                for expert in range(n_experts):
                    shard_id = layer * n_experts + expert
                    blob = bytearray()
                    for kind in ('ffn_down_exps', 'ffn_gate_exps', 'ffn_up_exps'):
                        foff, dims = base[kind]
                        esz = expert_slice_bytes(dims)
                        f.seek(foff + expert * esz)
                        blob += f.read(esz)
                    h = fnv1a64(blob)
                    entries.append((shard_id, cursor, len(blob), h))
                    out.write(blob)
                    pad = align_up(len(blob), BLOB_ALIGN) - len(blob)
                    if pad:
                        out.write(b'\0' * pad)
                    cursor = align_up(cursor + len(blob), BLOB_ALIGN)
                if (layer + 1) % 4 == 0:
                    el = time.time() - t0
                    print(f"  layer {layer+1}/{use_layers} done ({el:.0f}s)")

        # patch header + table
        out.seek(0)
        out.write(struct.pack('<IIIIQQ', MAGIC, VERSION, FLAGS, 20, n_shards, data_offset))
        for shard_id, off, size, h in entries:
            out.write(struct.pack('<IQIQ', shard_id, off, size, h))

    total = cursor
    el = time.time() - t0
    print(f"wrote {out_path}: {n_shards} shards, {total:,} bytes ({total/1e9:.2f} GB) in {el:.0f}s")
    return 0

if __name__ == '__main__':
    sys.exit(main())