import struct

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
    f.close()
    return table

t2 = parse_part(r'D:\AI_Bundle\qwen38next\UD-Q3_K_XL\Qwen3.8-Flash-Next-UD-Q3_K_XL-00002-of-00003.gguf')
t3 = parse_part(r'D:\AI_Bundle\qwen38next\UD-Q3_K_XL\Qwen3.8-Flash-Next-UD-Q3_K_XL-00003-of-00003.gguf')
print('part2 tensors:', len(t2), ' part3 tensors:', len(t3))
exp = {n for n in list(t2) + list(t3) if 'ffn' in n and 'exps' in n}
layers = sorted(set(int(n.split('.')[1][3:]) for n in exp))
print('expert tensors:', len(exp), ' layers:', len(layers), f'({layers[0]}..{layers[-1]})')
kinds = set(n.split('.')[2] for n in exp)
print('kinds:', kinds)
# spot: ttype distribution
from collections import Counter
tt = Counter()
for n in exp:
    v = t2.get(n) or t3[n]
    tt[v[2]] += 1
print('ttypes:', dict(tt))
