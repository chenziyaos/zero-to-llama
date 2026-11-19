"""只用标准库读 GGUF 文件头：KV 元数据和张量目录，不读权重本身。

对照：
  ggml/src/gguf.cpp          gguf_init_from_file_impl 的读取顺序
  gguf-py/gguf/constants.py  GGUF_MAGIC、GGUFValueType、GGMLQuantizationType、GGML_QUANT_SIZES
基线 llama.cpp 0ec4027（v0.8.0，ggml 0.28.0）。

文件布局（小端）：
  magic u32 'GGUF' | version u32 | n_tensors u64 | n_kv u64
  n_kv 个 KV：key(str) + type u32 + value
  n_tensors 个张量信息：name(str) + n_dims u32 + ne[n_dims] u64 + type u32 + offset u64
  按 general.alignment（默认 32）对齐后是数据段，offset 相对数据段起点
str = 长度 u64 + utf-8 字节，不带结尾 0。
"""
import struct
import sys

GGUF_MAGIC = 0x46554747
DEFAULT_ALIGNMENT = 32
QK_K = 256

# GGUFValueType 0..12
_SCALAR = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?',
           10: '<Q', 11: '<q', 12: '<d'}
T_STRING, T_ARRAY = 8, 9

# GGMLQuantizationType -> (名字, 每块元素数, 每块字节数)；数字照抄 constants.py
GGML_TYPES = {
    0: ('F32', 1, 4), 1: ('F16', 1, 2),
    2: ('Q4_0', 32, 18), 3: ('Q4_1', 32, 20), 6: ('Q5_0', 32, 22), 7: ('Q5_1', 32, 24),
    8: ('Q8_0', 32, 34), 9: ('Q8_1', 32, 36),
    10: ('Q2_K', 256, 84), 11: ('Q3_K', 256, 110), 12: ('Q4_K', 256, 144),
    13: ('Q5_K', 256, 176), 14: ('Q6_K', 256, 210), 15: ('Q8_K', 256, 292),
    16: ('IQ2_XXS', 256, 66), 17: ('IQ2_XS', 256, 74), 18: ('IQ3_XXS', 256, 98),
    19: ('IQ1_S', 256, 50), 20: ('IQ4_NL', 32, 18), 21: ('IQ3_S', 256, 110),
    22: ('IQ2_S', 256, 82), 23: ('IQ4_XS', 256, 136),
    24: ('I8', 1, 1), 25: ('I16', 1, 2), 26: ('I32', 1, 4), 27: ('I64', 1, 8), 28: ('F64', 1, 8),
    29: ('IQ1_M', 256, 56), 30: ('BF16', 1, 2), 34: ('TQ1_0', 256, 54), 35: ('TQ2_0', 256, 66),
    39: ('MXFP4', 32, 17), 40: ('NVFP4', 64, 36), 41: ('Q1_0', 128, 18), 42: ('Q2_0', 64, 18),
    43: ('F8_E4M3', 1, 1),
}
TYPE_ID = {v[0]: k for k, v in GGML_TYPES.items()}


def type_bytes(type_id, n_elems):
    name, blk, nb = GGML_TYPES[type_id]
    assert n_elems % blk == 0, f'{name}: {n_elems} 不是块大小 {blk} 的整数倍'
    return n_elems // blk * nb


def bpw(type_name):
    _, blk, nb = GGML_TYPES[TYPE_ID[type_name]]
    return nb * 8 / blk


class _R:
    def __init__(self, f):
        self.f = f

    def take(self, fmt):
        n = struct.calcsize(fmt)
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError('文件在元数据中途结束')
        return struct.unpack(fmt, b)[0]

    def string(self):
        n = self.take('<Q')
        return self.f.read(n).decode('utf-8', errors='replace')

    def value(self, t, keep_arrays):
        if t in _SCALAR:
            return self.take(_SCALAR[t])
        if t == T_STRING:
            return self.string()
        if t == T_ARRAY:
            et, n = self.take('<I'), self.take('<Q')
            if not keep_arrays and et in _SCALAR:
                # tokenizer 词表这类大数组只跳过，记长度
                self.f.seek(n * struct.calcsize(_SCALAR[et]), 1)
                return ('<array>', et, n)
            if not keep_arrays and et == T_STRING:
                for _ in range(n):
                    self.f.seek(self.take('<Q'), 1)
                return ('<array>', et, n)
            return [self.value(et, keep_arrays) for _ in range(n)]
        raise ValueError(f'未知的 GGUF 值类型 {t}')


class Tensor:
    __slots__ = ('name', 'shape', 'type_id', 'offset')

    def __init__(self, name, shape, type_id, offset):
        self.name, self.shape, self.type_id, self.offset = name, shape, type_id, offset

    @property
    def type_name(self):
        return GGML_TYPES[self.type_id][0]

    @property
    def n_elems(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def nbytes(self):
        return type_bytes(self.type_id, self.n_elems)


class GGUF:
    def __init__(self, path, keep_arrays=False):
        self.path = path
        with open(path, 'rb') as f:
            r = _R(f)
            if r.take('<I') != GGUF_MAGIC:
                raise ValueError(f'{path} 不是 GGUF 文件')
            self.version = r.take('<I')
            n_tensors, n_kv = r.take('<Q'), r.take('<Q')
            self.kv = {}
            for _ in range(n_kv):
                k = r.string()
                self.kv[k] = r.value(r.take('<I'), keep_arrays)
            self.tensors = []
            for _ in range(n_tensors):
                name = r.string()
                nd = r.take('<I')
                shape = [r.take('<Q') for _ in range(nd)]
                self.tensors.append(Tensor(name, shape, r.take('<I'), r.take('<Q')))
            align = self.kv.get('general.alignment', DEFAULT_ALIGNMENT)
            pos = f.tell()
            self.data_offset = (pos + align - 1) // align * align
            f.seek(0, 2)
            self.file_size = f.tell()

    @property
    def arch(self):
        # imatrix.gguf 这类非模型文件没有 general.architecture，general.type 是 imatrix
        return self.kv.get('general.architecture', self.kv.get('general.type', '?'))

    def hp(self, key, default=None):
        """取 <arch>.<key>，比如 hp('block_count')。"""
        return self.kv.get(f'{self.arch}.{key}', default)

    @property
    def n_params(self):
        return sum(t.n_elems for t in self.tensors)

    @property
    def tensor_bytes(self):
        return sum(t.nbytes for t in self.tensors)

    def type_counts(self):
        c = {}
        for t in self.tensors:
            c[t.type_name] = c.get(t.type_name, 0) + 1
        return c


def main():
    g = GGUF(sys.argv[1])
    MiB = 2 ** 20
    print(f'GGUF V{g.version}  arch={g.arch}  {len(g.kv)} KV  {len(g.tensors)} tensors')
    print(f'数据段起点 {g.data_offset} B，文件 {g.file_size / MiB:.2f} MiB，'
          f'张量 {g.tensor_bytes / MiB:.2f} MiB，{g.n_params:,} 参数，'
          f'{g.tensor_bytes * 8 / max(g.n_params, 1):.2f} BPW')
    print('类型分布', ' '.join(f'{k}:{v}' for k, v in sorted(g.type_counts().items())))
    for k, v in g.kv.items():
        if isinstance(v, tuple):
            v = f'[{v[2]} 项]'
        print(f'  {k} = {v}')


if __name__ == '__main__':
    main()
