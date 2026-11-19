"""GGUF 文件的分段账本和合法性检查，只用标准库，不读权重本身。

    python3 gguf_check.py model.gguf            # 分段大小、KV 大户、张量类型分布、检查结果
    python3 gguf_check.py model.gguf --json     # 同样内容输出成一行 JSON
    python3 gguf_check.py model.gguf --strict   # 有 ERROR 时退出码 1，可以放进 CI

检查项逐条对照 ggml/src/gguf.cpp 的 gguf_init_from_file_impl（主干 0ec4027）：
  magic、版本（v1 不再支持，超大版本号多半是字节序反了）、n_tensors 和 n_kv 的范围、
  字符串长度、重复键、general.alignment 的类型和 2 的幂、张量名长度（< GGML_MAX_NAME=64）、
  重复张量名、维数（<= GGML_MAX_DIMS=4）、ne[0] 是块大小的整数倍、
  偏移必须等于前面所有张量按对齐补齐后的累计大小、文件长度覆盖整个数据段。
另外两条是 llama.cpp 加载层的约束，记为 WARN：
  mmap 时数据段起点要按 CPU 缓冲对齐（llama-model-loader.cpp 695 行）；
  分卷文件要有 split.no、split.count、split.tensors.count（llama-arch.cpp）。
"""
import json
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from gguf_lite import GGML_TYPES, GGUF_MAGIC, DEFAULT_ALIGNMENT, _SCALAR, T_STRING, T_ARRAY  # noqa: E402

GGML_MAX_NAME = 64
GGML_MAX_DIMS = 4
GGUF_VERSION = 3
MAX_STRING = 1 << 30          # gguf.cpp 的 GGUF_MAX_STRING_LENGTH 也是 1 GiB 量级，这里只防坏文件
TYPE_NAMES = {0: 'u8', 1: 'i8', 2: 'u16', 3: 'i16', 4: 'u32', 5: 'i32', 6: 'f32', 7: 'bool',
              8: 'str', 9: 'arr', 10: 'u64', 11: 'i64', 12: 'f64'}


class Bad(Exception):
    pass


class Reader:
    """带边界检查的读取器：每次读都确认文件里还有这么多字节。"""

    def __init__(self, f, size):
        self.f, self.size = f, size

    def pos(self):
        return self.f.tell()

    def take(self, fmt):
        n = struct.calcsize(fmt)
        b = self.f.read(n)
        if len(b) != n:
            raise Bad(f'文件在偏移 {self.pos()} 处提前结束')
        return struct.unpack(fmt, b)[0]

    def string(self):
        n = self.take('<Q')
        if n > MAX_STRING or n > self.size - self.pos():
            raise Bad(f'字符串长度 {n} 超过剩余文件大小')
        return self.f.read(n).decode('utf-8', errors='replace')

    def skip_value(self, t):
        """跳过一个值，返回（类型描述，元素数）。大数组不解析，只算字节。"""
        if t in _SCALAR:
            self.f.seek(struct.calcsize(_SCALAR[t]), 1)
            return TYPE_NAMES[t], 1
        if t == T_STRING:
            self.string()
            return 'str', 1
        if t == T_ARRAY:
            et, n = self.take('<I'), self.take('<Q')
            if et in _SCALAR:
                nb = n * struct.calcsize(_SCALAR[et])
                if nb > self.size - self.pos():
                    raise Bad(f'数组 {n} 项超过剩余文件大小')
                self.f.seek(nb, 1)
            elif et == T_STRING:
                for _ in range(n):
                    self.string()
            else:
                raise Bad(f'数组元素类型 {et} 不合法')
            return f'arr[{TYPE_NAMES.get(et, et)}]', n
        raise Bad(f'值类型 {t} 不合法')

    def value(self, t):
        if t in _SCALAR:
            return self.take(_SCALAR[t])
        if t == T_STRING:
            return self.string()
        raise Bad('value() 只读标量和字符串')


def pad(x, a):
    return (x + a - 1) // a * a


def check(path, cpu_align=32):
    """返回 dict：sections（分段字节）、kv（每个键的字节和类型）、tensors、errors、warns。"""
    size = os.path.getsize(path)
    out = {'path': path, 'file_size': size, 'errors': [], 'warns': [], 'kv': [], 'tensors': []}
    err, warn = out['errors'].append, out['warns'].append
    with open(path, 'rb') as f:
        r = Reader(f, size)
        try:
            magic = r.take('<I')
            if magic != GGUF_MAGIC:
                raise Bad(f'magic 是 {struct.pack("<I", magic)!r}，应为 b"GGUF"')
            ver = r.take('<I')
            out['version'] = ver
            if ver & 0x0000FFFF == 0 and ver:
                raise Bad(f'版本号 {ver} 大得离谱，多半是大端文件拿到小端机器上读')
            if ver == 1:
                raise Bad('GGUF v1 不再支持')
            if ver > GGUF_VERSION:
                raise Bad(f'版本 {ver} 高于本工具支持的 {GGUF_VERSION}')
            n_t, n_kv = r.take('<q'), r.take('<q')
            if n_t < 0 or n_t > (1 << 62) // 512:
                raise Bad(f'张量数 {n_t} 越界')
            if n_kv < 0 or n_kv > (1 << 62) // 64:
                raise Bad(f'KV 数 {n_kv} 越界')
            hdr_end = r.pos()
            seen, align = set(), DEFAULT_ALIGNMENT
            for i in range(n_kv):
                s = r.pos()
                k = r.string()
                if not k:
                    raise Bad(f'第 {i} 个键是空串')
                if k in seen:
                    raise Bad(f'键 {k} 重复')
                seen.add(k)
                t = r.take('<I')
                if k == 'general.alignment':
                    if t != 4:
                        raise Bad(f'general.alignment 必须是 u32，实际是 {TYPE_NAMES.get(t, t)}')
                    align = r.value(t)
                    desc, n = 'u32', 1
                elif k.startswith('split.') and t in _SCALAR:
                    out.setdefault('split', {})[k] = r.value(t)
                    desc, n = TYPE_NAMES[t], 1
                else:
                    desc, n = r.skip_value(t)
                out['kv'].append({'key': k, 'type': desc, 'n': n, 'bytes': r.pos() - s})
            if align == 0 or align & (align - 1):
                raise Bad(f'对齐 {align} 不是 2 的幂')
            kv_end = r.pos()
            names = set()
            for i in range(n_t):
                name = r.string()
                if len(name.encode()) >= GGML_MAX_NAME:
                    raise Bad(f'张量名 {name} 有 {len(name.encode())} 字节，上限 {GGML_MAX_NAME - 1}')
                if name in names:
                    raise Bad(f'张量名 {name} 重复')
                names.add(name)
                nd = r.take('<I')
                if nd > GGML_MAX_DIMS:
                    raise Bad(f'张量 {name} 有 {nd} 维，上限 {GGML_MAX_DIMS}')
                ne = [r.take('<q') for _ in range(nd)]
                if any(x < 0 for x in ne):
                    raise Bad(f'张量 {name} 的形状 {ne} 有负数')
                tid, off = r.take('<I'), r.take('<Q')
                if tid not in GGML_TYPES:
                    raise Bad(f'张量 {name} 的类型 {tid} 不认识')
                tn, blk, nb = GGML_TYPES[tid]
                if ne and ne[0] % blk:
                    raise Bad(f'张量 {name} 的 ne[0]={ne[0]} 不是 {tn} 块大小 {blk} 的整数倍')
                n = 1
                for x in ne:
                    n *= x
                out['tensors'].append({'name': name, 'shape': ne, 'type': tn, 'offset': off,
                                       'nbytes': n // blk * nb})
            ti_end = r.pos()
        except Bad as e:
            err(str(e))
            return out
    data_off = pad(ti_end, align) if out['tensors'] else ti_end
    acc = 0
    for t in out['tensors']:
        if t['offset'] != acc:
            err(f'张量 {t["name"]} 的偏移是 {t["offset"]}，按前面的张量累计应为 {acc}')
            break
        acc += pad(t['nbytes'], align)
    data_need = acc
    if data_off + data_need > size:
        err(f'数据段需要 {data_need} B，文件只剩 {size - data_off} B（文件被截断，或是只有文件头的夹具）')
    if out['tensors'] and data_off % cpu_align:
        warn(f'数据段起点 {data_off} 不是 {cpu_align} 的倍数，llama.cpp 用 mmap 加载会拒绝')
    sp = out.get('split')
    if sp is not None:
        miss = {'split.no', 'split.count', 'split.tensors.count'} - set(sp)
        if miss:
            warn(f'分卷文件缺少 {sorted(miss)}')
    tens = sum(t['nbytes'] for t in out['tensors'])
    out.update(alignment=align, data_offset=data_off, n_kv=len(out['kv']), n_tensors=len(out['tensors']))
    out['sections'] = {
        'header': hdr_end,
        'kv': kv_end - hdr_end,
        'tensor_info': ti_end - kv_end,
        'pad_before_data': data_off - ti_end,
        'tensor_data': tens,
        'pad_in_data': data_need - tens,
        'tail': max(0, size - data_off - data_need),
    }
    return out


def group(k):
    if k.startswith('tokenizer.ggml.'):
        return k
    if k.startswith('tokenizer.'):
        return 'tokenizer.*'
    if k.startswith('general.'):
        return 'general.*'
    return k.split('.', 1)[0] + '.*'


def report(o, top=6):
    MiB = 2 ** 20
    if 'sections' not in o:
        print(f'{o["path"]}: ERROR {o["errors"][0]}')
        return
    s = o['sections']
    print(f'{os.path.basename(o["path"])}  GGUF V{o["version"]}  {o["n_kv"]} KV  {o["n_tensors"]} 张量  '
          f'对齐 {o["alignment"]}  文件 {o["file_size"]:,} B')
    for k, v in s.items():
        print(f'  {k:16s} {v:>14,} B  {v / max(o["file_size"], 1) * 100:6.2f}%')
    g = {}
    for kv in o['kv']:
        g[group(kv['key'])] = g.get(group(kv['key']), 0) + kv['bytes']
    print('  KV 大户：' + '，'.join(f'{k} {v / MiB:.2f} MiB' if v > MiB / 10 else f'{k} {v:,} B'
                                for k, v in sorted(g.items(), key=lambda x: -x[1])[:top]))
    tc = {}
    for t in o['tensors']:
        a = tc.setdefault(t['type'], [0, 0])
        a[0] += 1
        a[1] += t['nbytes']
    print('  张量类型：' + '，'.join(f'{k} {v[0]} 个 {v[1] / MiB:.1f} MiB' for k, v in sorted(tc.items(), key=lambda x: -x[1][1])))
    print(f'  数据段起点 {o["data_offset"]:,}，模 4096 余 {o["data_offset"] % 4096}')
    for e in o['errors']:
        print('  ERROR', e)
    for w in o['warns']:
        print('  WARN ', w)
    if not o['errors'] and not o['warns']:
        print('  检查通过')


def main(argv):
    paths = [a for a in argv if not a.startswith('--')]
    bad = 0
    for p in paths:
        o = check(p)
        bad += bool(o['errors'])
        if '--json' in argv:
            o.pop('tensors')
            print(json.dumps(o, ensure_ascii=False))
        else:
            report(o)
    return 1 if bad and '--strict' in argv else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
