"""最小的 GGUF 写入器，只用标准库。测试用它造合法文件和各种坏文件，也可以拿来理解格式。

    from gguf_write import write_gguf
    write_gguf('tiny.gguf', kv=[('general.architecture', 'str', 'llama')],
               tensors=[('w', [32, 2], 'F32', b'\\0' * 256)])

kv 的每一项是 (键, 类型, 值)，类型取 u8 i8 u16 i16 u32 i32 f32 bool str u64 i64 f64，
或者 ('arr', 元素类型) 表示数组。tensors 的每一项是 (名字, 形状, 类型名, 数据字节)。
写法对照 ggml/src/gguf.cpp 的 gguf_write_to_buf：头、KV、张量信息、按对齐补零、数据段，
每个张量的数据也补零到对齐。hook 参数允许测试在写之前篡改字段。
"""
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from gguf_lite import TYPE_ID, GGML_TYPES  # noqa: E402

CODES = {'u8': (0, '<B'), 'i8': (1, '<b'), 'u16': (2, '<H'), 'i16': (3, '<h'), 'u32': (4, '<I'),
         'i32': (5, '<i'), 'f32': (6, '<f'), 'bool': (7, '<?'), 'str': (8, None),
         'u64': (10, '<Q'), 'i64': (11, '<q'), 'f64': (12, '<d')}


def _s(x):
    b = x.encode() if isinstance(x, str) else x
    return struct.pack('<Q', len(b)) + b


def _val(t, v):
    code, fmt = CODES[t]
    return _s(v) if t == 'str' else struct.pack(fmt, v)


def pad(x, a):
    return (x + a - 1) // a * a


def write_gguf(path, kv=(), tensors=(), alignment=32, version=3, magic=b'GGUF', hook=None):
    h = {'magic': magic, 'version': version, 'kv': list(kv), 'tensors': list(tensors),
         'offsets': None, 'n_tensors': None, 'n_kv': None, 'data': True}
    if hook:
        hook(h)
    tens = h['tensors']
    sizes = []
    for name, shape, tn, data in tens:
        _, blk, nb = GGML_TYPES[TYPE_ID[tn]]
        n = 1
        for d in shape:
            n *= d
        sizes.append(n // blk * nb if n % blk == 0 else len(data))
    offs, acc = [], 0
    for s in sizes:
        offs.append(acc)
        acc += pad(s, alignment)
    offs = h['offsets'] or offs
    out = bytearray(h['magic'] + struct.pack('<I', h['version']))
    out += struct.pack('<qq', len(tens) if h['n_tensors'] is None else h['n_tensors'],
                       len(h['kv']) if h['n_kv'] is None else h['n_kv'])
    for k, t, v in h['kv']:
        out += _s(k)
        if isinstance(t, tuple):
            et = t[1]
            out += struct.pack('<IIQ', 9, CODES[et][0], len(v))
            for x in v:
                out += _val(et, x)
        else:
            out += struct.pack('<I', CODES[t][0]) + _val(t, v)
    for (name, shape, tn, data), off in zip(tens, offs):
        out += _s(name) + struct.pack('<I', len(shape))
        for d in shape:
            out += struct.pack('<q', d)
        out += struct.pack('<IQ', TYPE_ID[tn], off)
    if tens and h['data']:
        out += b'\0' * (pad(len(out), alignment) - len(out))
        for (name, shape, tn, data), s in zip(tens, sizes):
            data = (data + b'\0' * s)[:s]
            out += data + b'\0' * (pad(s, alignment) - s)
    with open(path, 'wb') as f:
        f.write(out)
    return len(out)


if __name__ == '__main__':
    n = write_gguf(sys.argv[1] if len(sys.argv) > 1 else 'tiny.gguf',
                   kv=[('general.architecture', 'str', 'llama'), ('general.alignment', 'u32', 32),
                       ('tokenizer.ggml.tokens', ('arr', 'str'), ['a', 'b', 'c'])],
                   tensors=[('token_embd.weight', [32, 3], 'F32', b''),
                            ('blk.0.ffn_up.weight', [256, 4], 'Q4_K', b'')])
    print('wrote', n, 'bytes')
