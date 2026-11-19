#!/usr/bin/env python3
"""ledger.py 的回归用例：把 LCPP-20 正文里引用的推算值钉住，改了公式或文件大小时会报出来。

python3 test_ledger.py
"""
import os
import sys
import unittest

import ledger as L

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
import hybrid_ledger as hl  # noqa: E402

GiB = L.GiB
GB = L.GB


def tot(*a, **k):
    _, t, _ = L.stack(*a, **k)
    return round(t / GiB, 2)


class KV(unittest.TestCase):
    def test_per_token(self):
        self.assertEqual(L.kv_per_token(L.MODELS['32b'], 'f16'), 256 * 1024)
        self.assertEqual(L.kv_per_token(L.MODELS['32b'], 'q8_0'), 136 * 1024)
        self.assertEqual(L.kv_per_token(L.MODELS['32b'], 'q4_0'), 72 * 1024)
        self.assertEqual(L.kv_per_token(L.MODELS['27b'], 'f16'), 64 * 1024)
        self.assertEqual(L.kv_per_token(L.MODELS['27b'], 'q8_0'), 34 * 1024)

    def test_deltanet_state(self):
        # n_embd_s = 128 x 6144，n_embd_r = 3 x (6144 + 2 x 16 x 128)，F32，48 层
        self.assertAlmostEqual(L.rec_per_seq(L.MODELS['27b']) / L.MiB, 149.6, places=1)
        self.assertAlmostEqual(L.rec_total(L.MODELS['27b'], 4, 3) / GiB, 2.34, places=2)
        self.assertAlmostEqual(L.rec_total(L.MODELS['27b'], 1, 2) / GiB, 0.44, places=2)


class Budget(unittest.TestCase):
    def test_budget(self):
        self.assertAlmostEqual(L.budget(0.5) / GiB, 22.5)

    def test_32b(self):
        self.assertEqual(tot('32b', 'Q4_K_M', 16384, 'q8_0'), 21.03)
        self.assertEqual(tot('32b', 'IQ4_XS', 32768, 'q8_0'), 21.23)
        self.assertEqual(tot('32b', 'Q4_K_M', 40960, 'q4_0'), 21.72)
        self.assertEqual(L.max_ctx('32b', 'Q4_K_M', 'q8_0'), 27648)
        self.assertEqual(L.max_ctx('32b', 'IQ4_XS', 'q8_0'), 40960)

    def test_32b_yarn(self):
        L.set_yarn(2)
        try:
            self.assertEqual(tot('32b', 'IQ4_XS', 65536, 'q4_0'), 21.48)
            self.assertEqual(tot('32b', 'IQ4_XS', 65536, 'q8_0'), 25.48)
            self.assertEqual(tot('32b', 'Q4_K_M', 65536, 'q4_0'), 23.40)
        finally:
            L.set_yarn(0)

    def test_32b_draft(self):
        d = 'Qwen3-0.6B-Q8_0'
        self.assertEqual(L.max_ctx('32b', 'Q4_K_M', 'q8_0', draft=d), 12544)
        self.assertEqual(L.max_ctx('32b', 'Q4_K_M', 'q8_0', draft=d, kvd='q8_0'), 15872)
        self.assertEqual(L.max_ctx('32b', 'IQ4_XS', 'q8_0', draft=d), 20736)
        self.assertEqual(L.max_ctx('32b', 'IQ4_XS', 'q8_0', draft=d, kvd='q8_0'), 26368)
        self.assertEqual(tot('32b', 'IQ4_XS', 16384, 'q8_0', draft=d, kvd='q8_0'), 20.63)

    def test_27b(self):
        q = 'UD-Q4_K_XL'
        self.assertEqual(L.max_ctx('27b', q, 'q8_0'), 169472)
        self.assertEqual(L.max_ctx('27b', q, 'q8_0', np_=4), 155904)
        self.assertEqual(L.max_ctx('27b', q, 'q8_0', mtp=2), 113920)
        self.assertEqual(L.max_ctx('27b', q, 'q8_0', np_=4, mtp=2), 75776)
        self.assertEqual(tot('27b', q, 65536, 'q8_0'), 19.13)
        self.assertEqual(tot('27b', q, 65536, 'q8_0', mtp=2), 20.83)
        self.assertEqual(tot('27b', q, 65536, 'q8_0', mtp=3), 20.97)
        self.assertEqual(tot('27b', q, 65536, 'q8_0', np_=4, mtp=2), 22.14)
        self.assertEqual(tot('27b', q, 65536, 'q8_0', np_=4, mtp=3), 22.73)
        self.assertEqual(tot('27b', 'UD-Q5_K_XL', 32768, 'q8_0'), 21.15)

    def test_27b_mmproj(self):
        q = 'UD-Q4_K_XL'
        self.assertEqual(L.max_ctx('27b', q, 'q8_0', mtp=2, mmproj=True), 88832)
        self.assertEqual(L.max_ctx('27b', q, 'q8_0', mmproj=True), 142848)


class Cache(unittest.TestCase):
    def test_entry(self):
        _, t = L.cache_entry('27b', 65536, 'q8_0', mtp=2, n_ckpt=32)
        self.assertEqual(round(t / GiB, 2), 7.08)
        _, t = L.cache_entry('27b', 65536, 'f16', mtp=2, n_ckpt=32)
        self.assertEqual(round(t / GiB, 2), 9.07)
        self.assertGreater(t, 8 * GiB)          # 超过 -cram 缺省 8192 MiB，整条跳过

    def test_32b_has_no_checkpoint(self):
        rows, t = L.cache_entry('32b', 32768, 'q8_0')
        self.assertEqual(len(rows), 1)
        self.assertEqual(round(t / GiB, 2), 4.25)


class Speed(unittest.TestCase):
    def test_eff(self):
        e = L.eff()
        self.assertAlmostEqual(e['RTX 4090'], 0.717, places=3)
        self.assertAlmostEqual(e['RTX 3090'], 0.661, places=3)
        self.assertAlmostEqual(e['kingy'], 0.833, places=3)

    def test_bound(self):
        self.assertAlmostEqual(L.tg_bound(19.76, 'RTX 4090'), 51.0, places=1)
        self.assertAlmostEqual(L.tg_bound(19.76, 'RTX 4090', 16384, '32b', 'f16'), 41.9, places=1)
        self.assertAlmostEqual(L.tg_bound(19.76, 'RTX 4090', 32768, '32b', 'q8_0'), 41.4, places=1)

    def test_split(self):
        self.assertAlmostEqual(L.tg_split(23.21, 1.0, 'RTX 4090'), 43.4, places=1)
        self.assertAlmostEqual(L.tg_split(23.21, 0.95, 'RTX 4090'), 28.7, places=1)
        self.assertAlmostEqual(L.tg_split(23.21, 0.80, 'RTX 4090'), 14.2, places=1)

    def test_spec(self):
        c = L.EXTRA['Qwen3-0.6B-Q8_0'] / L.QUANTS['32b']['Q4_K_M']
        self.assertAlmostEqual(L.spec_bound(2, 3, c), 1.82, places=2)
        self.assertAlmostEqual(L.spec_bound(2, 3, 4 * c), 1.44, places=2)


class SharedLedger(unittest.TestCase):
    """27B 混合架构几何和循环/KV 公式必须与共享账本同源；32B 仍是本篇历史对照，不进共享 manifest。"""

    @classmethod
    def setUpClass(cls):
        cls.man = hl.load_manifest()
        cls.g = hl.geometry(cls.man, 'qwen3.8-27b')
        cls.m27 = L.MODELS['27b']

    def test_27b_geometry_from_manifest(self):
        # 本篇 MODELS['27b'] 不再手抄，逐项等于共享几何
        for field, shared in (('n_layer', 'n_layer'), ('kv_layers', 'kv_layers'),
                              ('rec_layers', 'rec_layers'), ('n_kv', 'n_kv'),
                              ('head_dim', 'head_dim'), ('n_ctx_train', 'n_ctx_train'),
                              ('d_state', 'd_state'), ('v_heads', 'v_heads'),
                              ('k_heads', 'k_heads'), ('d_conv', 'd_conv')):
            self.assertEqual(self.m27[field], self.g[shared])

    def test_27b_kv_and_recurrent_delegate(self):
        for kv in ('f16', 'q8_0', 'q4_0'):
            self.assertEqual(L.kv_per_token(self.m27, kv), hl.kv_per_token(self.g, kv))
        self.assertEqual(L.rec_per_seq(self.m27), hl.rec_per_seq(self.g))
        for n_seq, rs in ((1, 0), (4, 0), (1, 2), (4, 3)):
            self.assertEqual(L.rec_total(self.m27, n_seq, rs), hl.rec_total(self.g, n_seq, rs))

    def test_32b_not_in_shared_manifest(self):
        # 32B 是全层历史对照，共享双模型基线里没有它
        self.assertNotIn('qwen3-32b', self.man['models'])
        self.assertEqual(L.MODELS['32b']['rec_layers'], 0)
        self.assertEqual(L.rec_per_seq(L.MODELS['32b']), 0)

    def test_unsloth_standard_quants_match_manifest_exact_bytes(self):
        # manifest 登记的是 unsloth 标准档精确字节；本篇 QUANTS 用 UD-* 动态档名，同档名映射后，
        # 展示 GB 必须落在精确字节的两位小数换算附近（配方不同，只核对同量级，不冒充同一文件）。
        local_name = {'IQ4_XS': 'UD-IQ4_XS', 'Q4_K_M': 'UD-Q4_K_M', 'Q5_K_M': 'UD-Q5_K_M',
                      'Q6_K': 'UD-Q6_K', 'Q8_0': 'Q8_0'}
        for q, exact in L.UNSLOTH_EXACT_BYTES.items():
            self.assertAlmostEqual(L.QUANTS['27b'][local_name[q]], exact / GB, places=1)
        # MTP sidecar 精确字节和本篇 1.37 GB 展示值一致
        self.assertAlmostEqual(L.EXTRA['mtp-Q4_0'], L.UNSLOTH_MTP_Q4_0_BYTES / GB, places=2)


if __name__ == '__main__':
    unittest.main()
