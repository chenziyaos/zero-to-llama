"""测 bench_stats 和 parse_ppl：python3 test_bench_ppl.py

夹具都在 logs/ 下，是基线 0ec4027 上 Qwen3-0.6B 在 4 核 EPYC 上的实机输出。
"""
import io
import math
import os
import sys
import unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LOGS = os.path.join(HERE, 'logs')
Q7 = os.path.join(HERE, '..', '07-quantization', 'logs')

import bench_stats as bs  # noqa: E402
import parse_ppl as pp  # noqa: E402


class TestStats(unittest.TestCase):
    def test_t95_table(self):
        self.assertAlmostEqual(bs.t95(1), 12.706)
        self.assertAlmostEqual(bs.t95(9), 2.262)
        self.assertAlmostEqual(bs.t95(1000), 1.96, places=2)

    def test_sd_matches_llama_bench(self):
        # llama-bench.cpp:110 的 stdev 是样本标准差（n-1），和这里一致
        xs = [138.002, 137.978, 137.921]
        self.assertAlmostEqual(bs.sd(xs), 0.041596, places=4)  # jsonl 里的 samples_ts 只保留 3 位小数

    def test_welch_identical(self):
        d, h, t, df = bs.welch([1, 2, 3, 4], [1, 2, 3, 4])
        self.assertEqual(d, 0)
        self.assertGreater(h, 0)

    def test_need_reps(self):
        # cv 5%、要分辨 5% → 2*(2.8)^2 ≈ 16
        self.assertEqual(bs.need_reps(0.05, 0.05), 16)
        self.assertGreater(bs.need_reps(0.06, 0.03), 50)

    def test_test_name(self):
        self.assertEqual(bs.test_name({'n_prompt': 512, 'n_gen': 0, 'n_depth': 0}), 'pp512')
        self.assertEqual(bs.test_name({'n_prompt': 0, 'n_gen': 64, 'n_depth': 4096}), 'tg64 @ d4096')

    def test_noise_fixture(self):
        files = [os.path.join(LOGS, f'noise-{i}.jsonl') for i in range(1, 6)]
        g = bs.group(files)
        tests = {k[1]: v for k, v in g.items()}
        self.assertEqual(set(tests), {'pp512', 'tg128'})
        s = bs.summarize(tests['tg128']['files'])
        self.assertEqual(s['n'], 50)
        self.assertEqual(s['n_files'], 5)
        # 进程间离散不小于 2%，这是这一章的核心结论
        self.assertGreater(s['cv_between'], 0.02)

    def test_aa_compare_flags_difference(self):
        # 同一配置、两个进程：Welch 检验仍可能判“不同”，说明单进程 -r 的置信区间不可信
        buf = io.StringIO()
        with redirect_stdout(buf):
            bs.main(['--compare', os.path.join(LOGS, 'noise-2.jsonl'), os.path.join(LOGS, 'noise-3.jsonl')])
        self.assertIn('tg128', buf.getvalue())
        self.assertIn('B faster', buf.getvalue())


class TestPPL(unittest.TestCase):
    def test_plain_ppl(self):
        d = pp.load(os.path.join(Q7, 'ppl-qwen3-0.6b-bf16.log'))
        self.assertEqual(pp.kind(d), 'ppl')
        self.assertAlmostEqual(d['final'][0], 23.3744)
        self.assertEqual(len(d['cum']), 20)
        self.assertEqual(d['n_ctx'], 512)

    def test_kld_log(self):
        d = pp.load(os.path.join(Q7, 'ppl-q-Q4_K_M.log'))
        self.assertEqual(pp.kind(d), 'kld')
        self.assertAlmostEqual(d['kld'][0], 0.113755)
        self.assertAlmostEqual(d['top1'][0], 82.922)
        self.assertEqual(len(d['kld_rows']), 20)

    def test_per_chunk_inverts_cumulative(self):
        d = pp.load(os.path.join(Q7, 'ppl-q-Q4_K_M.log'))
        pc = pp.per_chunk(d)
        # 逐块值的均值应该等于最后一行的累计值
        self.assertAlmostEqual(sum(pc['kld']) / len(pc['kld']), d['kld_rows'][-1][5], places=4)
        self.assertAlmostEqual(math.exp(sum(pc['ln_ppl']) / len(pc['ln_ppl'])), d['kld_rows'][-1][1], places=2)

    def test_chunk_se_wider_than_token_se_for_ppl(self):
        d = pp.load(os.path.join(Q7, 'ppl-q-Q4_K_M.log'))
        se = pp.chunk_se(pp.per_chunk(d)['ln_ppl'])
        tok = d['ppl_q'][1] / d['ppl_q'][0]
        self.assertGreater(se / tok, 1.5)

    def test_pair(self):
        a = pp.load(os.path.join(Q7, 'ppl-q-Q4_K_M.log'))
        b = pp.load(os.path.join(Q7, 'ppl-qi-Q4_K_M.log'))
        out = pp.pair(a, b)
        self.assertIn('显著', out)

    def test_hellaswag_line(self):
        text = 'task\tacc_norm\t95% confidence interval\n400\t40.25000000%\t[35.5590%, 45.1255%]\n'
        d = pp.parse(text)
        self.assertEqual(d['hellaswag'][0], 400)
        self.assertAlmostEqual(d['hellaswag'][1], 40.25)

    def test_hs_outcomes_roundtrip(self):
        # 由一串 0/1 生成累计准确率日志，再还原回来
        truth = [1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1]
        lines, s = [], 0
        for i, x in enumerate(truth, 1):
            s += x
            lines.append(f'{i}\t{100*s/i:.8f}%\t[0.0000%, 100.0000%]')
        self.assertEqual(pp.hs_outcomes('\n'.join(lines)), truth)

    def test_mcnemar(self):
        a = [1] * 20 + [0] * 20
        self.assertEqual(pp.mcnemar(a, a)[3], 1.0)
        b = [0] * 20 + [1] * 20          # 20 对 20，不显著
        self.assertGreater(pp.mcnemar(a, b)[3], 0.5)
        c = [1] * 20 + [1] * 15 + [0] * 5  # B 多对 15 题，A 多对 0 题
        n01, n10, chi2, p = pp.mcnemar(a, c)
        self.assertEqual((n01, n10), (15, 0))
        self.assertLess(p, 0.001)

    def test_hs_fixture_pair(self):
        a = open(os.path.join(LOGS, 'hs-qwen3-0.6b-bf16.log'), encoding='utf-8').read()
        b = open(os.path.join(LOGS, 'hs-q-Q4_K_M.log'), encoding='utf-8').read()
        oa, ob = pp.hs_outcomes(a), pp.hs_outcomes(b)
        self.assertEqual(len(oa), 400)
        self.assertEqual(sum(oa), 187)   # 46.75%
        self.assertGreater(pp.mcnemar(oa, ob)[3], 0.5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
