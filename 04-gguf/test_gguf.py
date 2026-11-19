"""gguf_check 的测试：先造一个合法文件，再逐个字段改坏，确认每种坏法都被抓到。

    python3 test_gguf.py            # 只用合成文件，不需要模型
    MODELS=../../models python3 test_gguf.py   # 另外检查真实模型的文件头

每条坏法对照 ggml/src/gguf.cpp 里 gguf_init_from_file_impl 的一条报错。
"""
import os
import struct
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gguf_check import check  # noqa: E402
from gguf_write import write_gguf  # noqa: E402

KV = [('general.architecture', 'str', 'llama'),
      ('tokenizer.ggml.tokens', ('arr', 'str'), ['a', 'b', 'c'])]
TS = [('token_embd.weight', [32, 3], 'F32', b''),
      ('blk.0.ffn_up.weight', [256, 4], 'Q4_K', b'')]


class T(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def make(self, name='x.gguf', **kw):
        p = os.path.join(self.d, name)
        kw.setdefault('kv', KV)
        kw.setdefault('tensors', TS)
        write_gguf(p, **kw)
        return check(p)

    def bad(self, needle, **kw):
        o = self.make(**kw)
        self.assertTrue(o['errors'], f'应报错却没报：{needle}')
        self.assertIn(needle, ' '.join(o['errors']))

    # 合法文件
    def test_ok(self):
        o = self.make()
        self.assertEqual(o['errors'], [])
        self.assertEqual(o['warns'], [])
        self.assertEqual(o['n_tensors'], 2)
        s = o['sections']
        self.assertEqual(sum(s.values()), o['file_size'])
        self.assertEqual(o['data_offset'] % 32, 0)

    def test_alignment_kv(self):
        o = self.make(kv=KV + [('general.alignment', 'u32', 64)], alignment=64)
        self.assertEqual(o['errors'], [])
        self.assertEqual(o['alignment'], 64)
        self.assertEqual(o['data_offset'] % 64, 0)

    def test_no_tensors(self):
        o = self.make(tensors=[])
        self.assertEqual(o['errors'], [])
        self.assertEqual(o['sections']['pad_before_data'], 0)

    # 文件头
    def test_magic(self):
        self.bad('magic', magic=b'GGML')

    def test_v1(self):
        self.bad('v1', version=1)

    def test_big_endian(self):
        self.bad('大端', version=struct.unpack('<I', struct.pack('>I', 3))[0])

    def test_future_version(self):
        self.bad('高于', version=4)

    def test_n_tensors_negative(self):
        self.bad('张量数', hook=lambda h: h.update(n_tensors=-1))

    def test_n_kv_too_many(self):
        # 多出来的键会把张量信息当成键来读，报字符串越界或提前结束都算抓到
        self.bad('', hook=lambda h: h.update(n_kv=len(KV) + 50))

    # KV
    def test_dup_key(self):
        self.bad('重复', kv=KV + [KV[0]])

    def test_empty_key(self):
        self.bad('空串', kv=[('', 'u32', 1)] + KV)

    def test_alignment_type(self):
        self.bad('必须是 u32', kv=KV + [('general.alignment', 'u8', 32)])

    def test_alignment_pow2(self):
        self.bad('2 的幂', kv=KV + [('general.alignment', 'u32', 48)], alignment=48)

    def test_alignment_zero(self):
        self.bad('2 的幂', kv=KV + [('general.alignment', 'u32', 0)])

    # 张量信息
    def test_name_len(self):
        self.bad('上限 63', tensors=[('w' * 64, [32, 1], 'F32', b'')])

    def test_name_63_ok(self):
        o = self.make(tensors=[('w' * 63, [32, 1], 'F32', b'')])
        self.assertEqual(o['errors'], [])

    def test_dup_tensor(self):
        self.bad('重复', tensors=[TS[0], TS[0]])

    def test_dims(self):
        self.bad('5 维', tensors=[('w', [32, 1, 1, 1, 1], 'F32', b'')])

    def test_block(self):
        self.bad('块大小 256', tensors=[('w', [128, 2], 'Q4_K', b'\0' * 144)])

    def test_offset(self):
        self.bad('偏移', hook=lambda h: h.update(offsets=[0, 64]))

    # 数据段
    def test_truncated(self):
        self.bad('截断', hook=lambda h: h.update(data=False))

    def test_cpu_align_warn(self):
        p = os.path.join(self.d, 'a16.gguf')
        write_gguf(p, kv=KV + [('general.alignment', 'u32', 16)], tensors=TS, alignment=16)
        o = check(p, cpu_align=32)
        self.assertEqual(o['errors'], [])
        if o['data_offset'] % 32:
            self.assertTrue(any('mmap' in w for w in o['warns']))

    def test_split_missing(self):
        o = self.make(kv=KV + [('split.no', 'u16', 0)])
        self.assertTrue(any('split.count' in w for w in o['warns']))

    def test_split_ok(self):
        o = self.make(kv=KV + [('split.no', 'u16', 0), ('split.count', 'u16', 1),
                               ('split.tensors.count', 'i32', 2)])
        self.assertEqual(o['warns'], [])


@unittest.skipUnless(os.environ.get('MODELS'), '设 MODELS 指向模型目录才跑')
class Real(unittest.TestCase):
    def test_q4km(self):
        p = os.path.join(os.environ['MODELS'], 'q-Q4_K_M.gguf')
        o = check(p)
        self.assertEqual(o['errors'], [])
        self.assertEqual(o['n_tensors'], 311)
        self.assertEqual(o['data_offset'], 5951872)

    def test_header_only(self):
        """截掉数据段的文件：文件头全对，只报截断。可以拿来做小体积的夹具。"""
        p = os.path.join(os.environ['MODELS'], 'q-Q4_K_M.gguf')
        q = os.path.join(tempfile.mkdtemp(), 'head.gguf')
        with open(p, 'rb') as f, open(q, 'wb') as g:
            g.write(f.read(5951872))
        o = check(q)
        self.assertEqual(len(o['errors']), 1)
        self.assertIn('截断', o['errors'][0])


if __name__ == '__main__':
    unittest.main()
