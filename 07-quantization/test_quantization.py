#!/usr/bin/env python3
"""07-quantization 的回归测试：只用标准库和仓库里的日志夹具，不需要模型文件。

    python3 test_quantization.py
"""
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'common'))

import collect_results as cr  # noqa: E402
import dequant as dq  # noqa: E402
import type_mix as tm  # noqa: E402
from gguf_lite import GGML_TYPES, TYPE_ID, bpw  # noqa: E402


def pack_scale_min_k4(sm):
    """get_scale_min_k4 的逆过程，照 quantize_row_q4_K_ref 的打包写法。"""
    s = [0] * 12
    for j, (ls, lm) in enumerate(sm):
        if j < 4:
            s[j] = ls
            s[j + 4] = lm
        else:
            s[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4)
            s[j - 4] |= (ls >> 4) << 6
            s[j] |= (lm >> 4) << 6
    return bytes(s)


class BlockLayout(unittest.TestCase):
    def test_sizes(self):
        # 结构体大小照 ggml-common.h 的 static_assert 手算
        self.assertEqual(GGML_TYPES[TYPE_ID['Q4_0']][1:], (32, 2 + 16))
        self.assertEqual(GGML_TYPES[TYPE_ID['Q8_0']][1:], (32, 2 + 32))
        self.assertEqual(GGML_TYPES[TYPE_ID['Q4_K']][1:], (256, 2 + 2 + 12 + 128))
        self.assertEqual(GGML_TYPES[TYPE_ID['Q6_K']][1:], (256, 128 + 64 + 16 + 2))
        self.assertEqual(GGML_TYPES[TYPE_ID['Q2_K']][1:], (256, 16 + 64 + 2 + 2))
        self.assertEqual(GGML_TYPES[TYPE_ID['IQ4_XS']][1:], (256, 2 + 2 + 4 + 128))

    def test_bpw(self):
        self.assertAlmostEqual(bpw('Q4_0'), 4.5)
        self.assertAlmostEqual(bpw('Q4_K'), 4.5)
        self.assertAlmostEqual(bpw('Q6_K'), 6.5625)
        self.assertAlmostEqual(bpw('Q8_0'), 8.5)
        self.assertAlmostEqual(bpw('IQ4_XS'), 4.25)
        self.assertAlmostEqual(bpw('IQ2_XXS'), 2.0625)


class Dequant(unittest.TestCase):
    def test_q8_0(self):
        b = struct.pack('<e', 0.5) + struct.pack('<32b', *range(-16, 16))
        self.assertEqual(dq.deq_q8_0(b), [0.5 * q for q in range(-16, 16)])

    def test_q4_0_nibble_order(self):
        # 第 i 字节的低 4 位是第 i 个值，高 4 位是第 i+16 个值
        qs = bytes((i & 0xF) | ((15 - i) << 4) for i in range(16))
        y = dq.deq_q4_0(struct.pack('<e', 1.0) + qs)
        self.assertEqual(y[:16], [i - 8 for i in range(16)])
        self.assertEqual(y[16:], [15 - i - 8 for i in range(16)])

    def test_scale_min_roundtrip(self):
        sm = [(35, 26), (51, 63), (27, 24), (38, 38), (63, 63), (55, 50), (43, 52), (48, 37)]
        s = pack_scale_min_k4(sm)
        self.assertEqual([dq.scale_min_k4(j, s) for j in range(8)], sm)

    def test_q4_k_block(self):
        sm = [(j + 1, 2 * j) for j in range(8)]
        q = bytes((i % 16) | ((15 - i % 16) << 4) for i in range(128))
        b = struct.pack('<ee', 0.25, 0.125) + pack_scale_min_k4(sm) + q
        self.assertEqual(len(b), 144)
        d, dmin, got_sm, y = dq.q4_k_block(b, 0)
        self.assertEqual(got_sm, sm)
        self.assertEqual(len(y), 256)
        # 子块 0：低 4 位，scale 1，min 0
        self.assertAlmostEqual(y[3], 0.25 * 1 * 3 - 0.125 * 0)
        # 子块 1：同一组 32 字节的高 4 位，scale 2，min 2
        self.assertAlmostEqual(y[32 + 3], 0.25 * 2 * 12 - 0.125 * 2)
        # 子块 7：最后一组的高 4 位
        self.assertAlmostEqual(y[224], 0.25 * 8 * 15 - 0.125 * 14)

    def test_q6_k_zero(self):
        # ql=qh=0 时 q=-32，值为 d*scale*(-32)
        b = bytes(192) + struct.pack('<16b', *([1] * 16)) + struct.pack('<e', 0.5)
        self.assertEqual(dq.deq_q6_k(b), [-16.0] * 256)

    def test_errors(self):
        s = dq.error_stats([1.0, 2.0], [1.0, 1.0])
        self.assertAlmostEqual(s['rmse'], (0.5) ** 0.5)
        self.assertAlmostEqual(s['maxabs'], 1.0)
        # 权重全压在第 0 个通道上，第 1 个通道的误差就不算
        self.assertEqual(dq.weighted_error([1.0, 9.0], [1.0, 1.0], 2, [1.0, 0.0]), 0.0)


class Recipe(unittest.TestCase):
    def test_use_more_bits_28(self):
        layers = [i for i in range(28) if tm.use_more_bits(i, 28)]
        self.assertEqual(layers, [0, 1, 2, 5, 8, 11, 14, 17, 20, 23, 24, 25, 26, 27])

    def test_predict(self):
        self.assertEqual(tm.predict_q4_k_m('output.weight', 28, 1024), 'Q6_K')
        self.assertEqual(tm.predict_q4_k_m('output.weight', 28, 1000), 'Q8_0')
        self.assertEqual(tm.predict_q4_k_m('blk.5.ffn_down.weight', 28), 'Q6_K')
        self.assertEqual(tm.predict_q4_k_m('blk.6.ffn_down.weight', 28), 'Q4_K')
        self.assertEqual(tm.predict_q4_k_m('blk.6.attn_norm.weight', 28), 'F32')


class Collect(unittest.TestCase):
    def setUp(self):
        self.rows = {r['name']: r for r in cr.collect(os.path.join(HERE, 'logs'),
                                                    [os.path.join(HERE, 'logs', 'round2')])}

    def test_q4_k_m(self):
        r = self.rows['q-Q4_K_M']
        self.assertAlmostEqual(r['kld'], 0.113755)
        self.assertAlmostEqual(r['top1'], 82.922)
        self.assertAlmostEqual(r['ppl_base'], 23.077265)
        self.assertAlmostEqual(r['size_mib'], 456.1, places=1)
        self.assertEqual(len(r['tg_runs']), 3)

    def test_imatrix_helps(self):
        for t in ('Q4_K_M', 'IQ4_XS', 'Q3_K_M', 'Q2_K'):
            self.assertLess(self.rows['qi-' + t]['kld'], self.rows['q-' + t]['kld'], t)

    def test_ppl_only_rows(self):
        self.assertIn('q-Q4_K-pure', self.rows)
        self.assertNotIn('size', self.rows['q-Q4_K-pure'])

    def test_pareto(self):
        front = cr.pareto(list(self.rows.values()))
        # 同体积下 imatrix 版本支配普通版本；Q4_0 被更小、KLD 更低的 qi-IQ4_XS 支配
        self.assertIn('qi-Q4_K_M', front)
        self.assertNotIn('q-Q4_K_M', front)
        self.assertNotIn('q-Q4_0', front)
        self.assertNotIn('q-Q2_K', front)

    def test_median(self):
        self.assertEqual(cr.median([3, 1, 2]), 2)
        self.assertEqual(cr.median([4, 1, 2, 3]), 2.5)


if __name__ == '__main__':
    unittest.main(verbosity=1)
