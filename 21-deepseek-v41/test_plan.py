#!/usr/bin/env python3
"""plan.py 的回归用例：把 LCPP-21 正文里引用的推算值钉住，改了公式、分配表或实测数字时会报出来。

python3 test_plan.py
"""
import unittest

import plan as P

GB, GiB = P.GB, P.GiB


class Bytes(unittest.TestCase):
    def test_backbone_matches_card(self):
        # 算出来的骨干和模型卡差不到 0.02 GB，差的是 GGUF 头和元数据
        for b in P.BUILDS:
            s = P.summary(b)
            self.assertLess(abs(s['bytes'] / GB - s['published'][1]), 0.02, b)

    def test_bpw_matches_header(self):
        for b in P.BUILDS:
            s = P.summary(b)
            self.assertAlmostEqual(s['bpw'], s['measured_bpw'], places=4)

    def test_params(self):
        self.assertAlmostEqual(P.summary('IQ2_XXS')['params'] / 1e9, 551.9, places=1)

    def test_split(self):
        s = P.summary('IQ2_XXS')
        self.assertAlmostEqual(s['by_kind']['exps'] / GB, 132.07, places=2)
        self.assertAlmostEqual(s['by_kind']['dense'] / GB, 5.70, places=2)
        self.assertAlmostEqual(s['by_kind']['tok_embd'] / GB, 0.455, places=3)
        self.assertAlmostEqual(s['engram_bytes'] / GB, 110.59, places=2)

    def test_expert_layers(self):
        e = P.summary('IQ2_XXS')['exp_layer']
        self.assertAlmostEqual(min(e) / GB, 2.97, places=2)
        self.assertAlmostEqual(max(e) / GB, 3.93, places=2)
        self.assertEqual(e.index(max(e)), 39)


class KV(unittest.TestCase):
    def test_per_token(self):
        self.assertEqual(P.kv_per_token('f16'), 3200)
        self.assertEqual(P.kv_per_token('q8_0'), 1700)

    def test_totals(self):
        self.assertAlmostEqual(P.kv_total(65536, 'f16', 1, 2048) / GB, 0.304, places=3)
        self.assertAlmostEqual(P.kv_total(65536, 'f16', 4, 512) / GB, 0.965, places=3)
        self.assertAlmostEqual(P.kv_total(1048576, 'q8_0', 1, 512) / GB, 1.799, places=3)
        self.assertAlmostEqual(P.kv_total(1048576, 'f16', 4, 512) / GB, 13.548, places=3)


class Fit(unittest.TestCase):
    def test_other(self):
        o = P.other_from_peak()
        self.assertAlmostEqual(o[25] / GB, 6.60, places=2)
        self.assertAlmostEqual(o[28] / GB, 6.79, places=2)
        self.assertLess(P.other_from_peak(GB)[28], 0)  # 按 GB 读峰值说不通

    def test_sweep(self):
        f = P.fit_sweep()
        self.assertAlmostEqual(f['t0'] * 1e3, 12.56, places=2)
        self.assertAlmostEqual(P.cpu_rate(f) / GB, 49.4, places=1)
        self.assertAlmostEqual(f['eff'], 0.161, places=3)
        self.assertGreater(f['t0'], P.gpu_floor())
        for n, tg in P.SWEEP.items():
            self.assertLess(abs(f['pred'][n] - tg) / tg, 0.04, n)

    def test_cards(self):
        self.assertEqual(P.fit_card('IQ2_XXS', 96, 65536, 'f16', 2048), 27)
        self.assertEqual(P.fit_card('IQ2_XXS', 24, 16384, 'q8_0', 2048), 4)
        self.assertEqual(P.fit_card('IQ1_M', 96, 65536, 'f16', 2048), 31)
        self.assertEqual(P.fit_card('IQ2_XS', 96, 65536, 'f16', 2048), 24)

    def test_vram_points(self):
        v = lambda n: sum(b for _, b in P.vram('IQ2_XXS', n, 65536, 'f16', 2048)) / GiB
        self.assertAlmostEqual(v(25), 85.90, places=2)
        self.assertAlmostEqual(v(28), 95.92, places=2)

    def test_tg(self):
        r = lambda *a: tuple(round(x, 1) for x in P.predict_tg(*a))
        self.assertEqual(r('IQ2_XXS', 25), (34.3, 34.3))
        self.assertEqual(r('IQ2_XXS', 4, 89.6), (7.0, 19.8))
        self.assertEqual(r('IQ2_XXS', 4, 179.2), (12.9, 19.8))
        self.assertEqual(r('IQ2_XXS', 0, 89.6), (6.4, 18.4))


if __name__ == '__main__':
    unittest.main()
