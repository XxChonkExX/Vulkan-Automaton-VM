# slice_qwen38_experts.py: slice Qwen3.8-Flash-Next multi-part GGUF expert
# tensors into a .vmex v2 pack (one shard per (layer, expert): ffn_down +
# ffn_gate + ffn_up). Handles the multi-part split (tensor tables in parts
# 2+3, each with its own GGUF header) and Q4_K/Q3_K block-quant byte math.
#
# Usage: python slice_qwen38_experts.py <part2.gguf> <part3.gguf> <out.vmex> [maxLayers]
#
# Per-expert byte math (256-elem blocks): Q4_K=144B, Q3_K=110B.
# Per-tensor size is derived from offset diffs within each part (safest —
# avoids padding assumptions), with the last tensor in each part from quant math.

import struct
import sys
import time

QBLOCK = 256
QBLOCK_BYTES = {20: 144, 18: 110}  # Q4_K, Q3_K (plus more if needed)
ALIGN = 4096

def fnv1a64(data, h=1469598103934665603):
    for b in data:
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h

def align_up(v, a):
    return (v + a - 1) & ~(a - 1)

FMT = {0:('B',1),1:('b',1),2:('H',2),3:('h',2),4:('I',4),5:('i',4),6:('f',4),
       7:('?',1),10:('Q',8),11:('q',8),12:('d',8)}

def read_str(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', 'replace')

def read_val(f, t):
    if t == 8:
        return read_str(f)
    if t == 9:
        et = struct.unpack('<I', f.read(4))[0]
        cnt = struct.unpack('<Q', f.read(8))[0]
        if et in FMT:
            c, sz = FMT[et]
            if cnt * sz > 4096:
                f.seek(sz * cnt, 1)
            else:
                f.read(sz * cnt)
            return
        for _ in range(cnt):
            read_val(f, et)
        return
    c, sz = FMT[t]
    return f.read(sz)

def parse_part(path):
    """Returns (tensor_table{name: (off, dims, ttype)}, data_start, file_len)."""
    f = open(path, 'rb')
    assert f.read(4) == b'GGUF'
    ver = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_kv = struct.unpack('<Q', f.read(8))[0]
    for _ in range(n_kv):
        read_str(f)
        t = struct.unpack('<I', f.read(4))[0]
        read_val(f, t)
    table = {}
    for _ in range(n_tensors):
        name = read_str(f)
        nd = struct.unpack('<I', f.read(4))[0]
        dims = struct.unpack('<' + 'Q' * nd, f.read(8 * nd))
        ttype = struct.unpack('<I', f.read(4))[0]
        off = struct.unpack('<Q', f.read(8))[0]
        table[name] = (off, dims, ttype)
    pos = f.tell()
    data_start = align_up(pos, 32)
    f.seek(0, 2)
    file_len = f.tell()
    f.close()
    return table, data_start, file_len

def main():
    import sys
    if len(sys.argv) < 4:
        print("usage: slice_qwen38_experts.py <part2> <part3> <out.vmex> [maxLayers]")
        return 1
    parts = sys.argv[1:-2] if False else [sys.argv[1], sys.argv[2]]
    out_path = sys.argv[3]
    max_layers = int(sys.argv[4]) if len(sys.argv) > 4 else 16

    t0 = time.time()
    # Parse all parts; merge tensor tables (data offsets are per-part).
    tables = []
    for p in parts:
        table, data_start, file_len = parse_part(p)
        tables.append((p, table, data_start, file_len))
        print(f"parsed {p}: {len(table)} tensors")

    # Merge: find expert tensors, keep (part_idx, off, dims, ttype).
    exp = {}
    for pi, (p, table, ds, fl) in enumerate(tables):
        for name, (off, dims, ttype) in table.items():
            if 'ffn' in name and 'exps' in name:
                exp[name] = (pi, off, dims, ttype)

    # Derive per-tensor byte sizes from offset diffs over the FULL tensor
    # list per part (data layout follows table order; consecutive-offset diff
    # gives the exact stride, alignment included). Per-expert = stride / 512.
    sizes = {}
    for pi, (p, table, ds, fl) in enumerate(tables):
        by_off = sorted(table.items(), key=lambda kv: kv[1][0])
        for i, (name, (off, dims, ttype)) in enumerate(by_off):
            if not ('ffn' in name and 'exps' in name):
                continue
            if i + 1 < len(by_off):
                size = by_off[i + 1][1][0] - off
            else:
                size = fl - ds - off  # last tensor in part: to end of file
            per_expert = size // dims[2]
            assert size % dims[2] == 0, f"{name}: size {size} not divisible by {dims[2]}"
            sizes[name] = (per_expert, size)

    # layer -> expert -> 3 slices
    layers = {}
    for name, (pi, off, dims, ttype) in sorted(exp.items()):
        seg = name.split('.')
        layer = int(seg[1])  # 'blk.0.ffn_down_exps.weight' -> seg[1] == '0'
        kind = '.'.join(seg[2:])
        layers.setdefault(layer, {})[kind] = (pi, off, dims, ttype, sizes[name][0])

    use_layers = min(max_layers, len(layers))
    n_experts = 512
    shard_bytes = sum(s[4] for s in layers[0].values())
    n_shards = use_layers * n_experts
    print(f"model: {len(layers)} layers x {n_experts} experts; "
          f"shard={shard_bytes:,} B ({shard_bytes/1048576:.2f} MiB); {n_shards} shards")

    table_bytes = 24 * n_shards
    data_offset = align_up(32 + table_bytes, ALIGN)

    entries = []
    cursor = data_offset
    with open(out_path, 'wb') as out:
        out.write(b'\0' * (data_offset - 32 + 32))

        open_files = [open(p, 'rb') for (p, t, ds, fl) in tables]
        for layer in range(use_layers):
            lt = layers[layer]
            for expert in range(n_experts):
                shard_id = layer * n_experts + expert
                blob = bytearray()
                for kind in ('ffn_down_exps.weight', 'ffn_gate_exps.weight', 'ffn_up_exps.weight'):
                    pi, off, dims, ttype, per = lt[kind]
                    open_files[pi].seek(off + expert * per)
                    blob += open_files[pi].read(per)
                h = fnv1a64(blob)
                entries.append((shard_id, cursor, len(blob), h))
                out.write(blob)
                pad = align_up(len(blob), ALIGN) - len(blob)
                if pad:
                    out.write(b'\0' * pad)
                cursor = align_up(cursor + len(blob), ALIGN)
            for fh in open_files:
                fh.close()
            open_files = [open(p, 'rb') for (p, t, ds, fl) in tables]
            el = time.time() - t0
            if (layer + 1) % 2 == 0:
                print(f"  layer {layer+1}/{use_layers} ({el:.0f}s)", flush=True)

        out.seek(0)
        out.write(struct.pack('<IIIIQQ', 0x32454D56, 2, (3) | (2 << 2) | (2 << 4) | (3 << 6),
                              20, n_shards, data_offset))
        for shard_id, off, size, h in entries:
            out.write(struct.pack('<IQIQ', shard_id, off, size, h))

    total = cursor
    el = time.time() - t0
    print(f"wrote {out_path}: {n_shards} shards, {total:,} bytes ({total/1e9:.2f} GB) in {el:.0f}s")
    return 0

if __name__ == '__main__':
    sys.exit(main())