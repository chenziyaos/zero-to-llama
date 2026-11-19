"""测 estimate_speed、estimate_budget 和 fit_ledger：python3 test_ledger.py

夹具全在 logs/：ledgers.json 是从 19 个 Qwen3-0.6B GGUF 导出的账本，其余是本机实测。
有 GGUF 时设 MODELS=../../models，会额外核对账本导出和预算计算。
"""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import estimate_speed as es  # noqa: E402
import fit_ledger as fl  # noqa: E402

LED = json.load(open(os.path.join(HERE, 'logs', 'ledgers.json')))
MB = 1e6


class Ledger(unittest.TestCase):
    def test_bytes_per_token(self):
        # token_embd 不计，output 整张计：bf16 1192 MB，Q4_K_M 391 MB
        self.assertAlmostEqual(LED['qwen3-0.6b-bf16']['w_bytes'] / MB, 1192.2, delta=0.1)
        self.assertAlmostEqual(LED['q-Q4_K_M']['w_bytes'] / MB, 390.8, delta=0.1)
        self.assertAlmostEqual(LED['q-Q2_K']['w_bytes'] / MB, 290.3, delta=0.1)

    def test_flops_same_for_all_quants(self):
        f = {v['tg_flops'] for v in LED.values()}
        self.assertEqual(len(f), 1)
        self.assertAlmostEqual(f.pop() / 1e9, 1.192, delta=0.001)

    def test_kv_per_token(self):
        # 28 层 x 8 KV 头 x 128 x (2 + 2) 字节 = 114688 B，即 112 KiB
        self.assertEqual(LED['q-Q4_K_M']['kv_per_token'], 28 * 8 * 128 * 4)

    def test_path(self):
        self.assertEqual(LED['qwen3-0.6b-bf16']['path'], 'bf16')
        self.assertEqual(LED['q-Q4_K_M']['path'], 'int8')
        self.assertGreater(LED['qi-IQ2_M']['codebook_frac'], 0.5)
        self.assertEqual(LED['q-Q4_K_M']['codebook_frac'], 0)

    def test_at_depth(self):
        d = es.at_depth(LED['q-Q4_K_M'], 4096)
        self.assertAlmostEqual(d['kv_bytes'] / MB, 469.8, delta=0.1)
        self.assertEqual(d['tg_flops'] - LED['q-Q4_K_M']['tg_flops'], 4 * 28 * 16 * 128 * 4096)


class Bounds(unittest.TestCase):
    def setUp(self):
        self.bw, self.peak = es.machine(4)

    def test_machine(self):
        self.assertAlmostEqual(self.bw, 35.5, delta=0.5)
        self.assertAlmostEqual(self.peak['int8'], 1662.4, delta=1)

    def test_roofline(self):
        r = es.bounds(LED['q-Q4_K_M'], 35.5, self.peak)
        self.assertAlmostEqual(r['tg'], 90.9, delta=0.2)
        self.assertLess(r['intensity'], 4)          # 单序列 decode 3 op/B，远在屋脊左边
        self.assertGreater(r['ridge'], 40)          # int8 屋脊 47 op/B

    def test_predict_matches_slow_phase(self):
        # sandwich 第 1、4 轮带宽约 35.5 GB/s，实测 bf16 27.5、Q8_0 47.0 至 48.9、Q4_K_M 72.0、Q2_K 85.2 至 87.9
        want = {'qwen3-0.6b-bf16': 27.7, 'q-Q8_0': 48.0, 'q-Q4_K_M': 72.0, 'q-Q2_K': 86.5}
        for m, tg in want.items():
            p = es.predict(LED[m], 35.5, self.peak)['tg']
            self.assertLess(abs(p / tg - 1), 0.07, m)

    def test_codebook_slower(self):
        p = es.predict(LED['qi-IQ2_M'], 35.5, self.peak)
        self.assertEqual(p['tg_limit'], 'compute')
        self.assertLess(abs(p['tg'] / 60.7 - 1), 0.1)

    def test_batched(self):
        # llama-batched-bench：B=1 62.3，B=32 199.1 t/s
        b = dict(es.predict(LED['q-Q4_K_M'], 35.5, self.peak, (1, 16, 32))['batch'])
        self.assertLess(abs(b[32] / 199.1 - 1), 0.1)
        self.assertLess(abs(b[16] / 182.3 - 1), 0.1)

    def test_depth(self):
        # depth.jsonl：FA off 4096 深度 24.9 t/s，FA on 17.1
        for fa, tg in (('off', 24.9), ('on', 17.1)):
            p = es.predict(es.at_depth(LED['q-Q4_K_M'], 4096), 35.9, self.peak, fa=fa)['tg']
            self.assertLess(abs(p / tg - 1), 0.15, fa)


class Fit(unittest.TestCase):
    def test_sandwich(self):
        _, peak = es.machine(4)
        (t0, eff, err), _ = fl.fit_sandwich(LED, peak, fl.jl('sandwich.jsonl'))
        self.assertAlmostEqual(t0, es.CAL['t0_ms'], delta=0.3)
        self.assertAlmostEqual(eff, es.CAL['eff_tg'], delta=0.01)
        self.assertLess(err, 0.07)

    def test_batched_table(self):
        rows = fl.read_batched(os.path.join(HERE, 'logs', 'batched-q4km.md'))
        self.assertEqual([r['B'] for r in rows], [1, 2, 4, 8, 16, 32])
        _, peak = es.machine(4)
        a, b, eff, _ = fl.fit_batched(LED, peak, rows)
        self.assertAlmostEqual(b, 4.62, delta=0.05)
        self.assertAlmostEqual(eff, es.CAL['eff_batch'], delta=0.01)

    def test_pp(self):
        _, peak = es.machine(4)
        pp = fl.fit_pp(LED, peak, fl.jl('interleave.jsonl'))
        self.assertAlmostEqual(pp['q-Q4_K_M'], es.CAL['eff_pp']['int8'], delta=0.02)
        self.assertAlmostEqual(pp['qwen3-0.6b-bf16'], es.CAL['eff_pp']['bf16'], delta=0.02)

    def test_depth(self):
        d = fl.fit_depth(LED, fl.jl('depth.jsonl'))
        self.assertAlmostEqual(d['off'][1], es.CAL['eff_attn']['off'], delta=0.03)
        self.assertAlmostEqual(d['on'][1], es.CAL['eff_attn']['on'], delta=0.03)


@unittest.skipUnless(os.environ.get('MODELS'), '设 MODELS=GGUF 目录才跑')
class WithModels(unittest.TestCase):
    def test_ledger_export(self):
        sys.path.insert(0, os.path.join(HERE, '..', 'common'))
        from gguf_lite import GGUF
        g = GGUF(os.path.join(os.environ['MODELS'], 'q-Q4_K_M.gguf'))
        self.assertEqual(es.ledger(g)['w_bytes'], LED['q-Q4_K_M']['w_bytes'])

    def test_budget(self):
        import estimate_budget as eb
        from gguf_lite import GGUF

        class A:
            ubatch, ctk, ctv, fa = 512, 'f16', 'f16', 'on'
        g = GGUF(os.path.join(os.environ['MODELS'], 'q-Q4_K_M.gguf'))
        self.assertEqual(eb.max_ctx(g, 2048 * 2 ** 20, A), 11520)


if __name__ == '__main__':
    unittest.main(verbosity=1)
