#!/usr/bin/env python3
"""13-speculative 的回归测试：期望收益模型用公式测，实验结论用 logs 核对。

    python3 test_spec.py
没有 logs 的用例会跳过。
"""
import glob
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
L = os.path.join(HERE, 'logs')
sys.path.insert(0, HERE)

import spec_model as sm  # noqa: E402


def read(name):
    p = os.path.join(L, name)
    if not os.path.exists(p):
        return ''
    with open(p, errors='replace') as f:
        return f.read()


def need(*names):
    miss = [n for n in names if not os.path.exists(os.path.join(L, n))]
    return unittest.skipIf(miss, 'no logs: ' + ','.join(miss))


def acc(name):
    m = re.search(r'accept\s+= ([\d.]+)%', read(name))
    return float(m.group(1)) if m else None


def tps(name):
    t = read(name)
    m = re.search(r'decoded\s+\d+ tokens in\s+[\d.]+ seconds, speed:\s+([\d.]+)', t)
    if m:
        return float(m.group(1))
    m = re.search(r'(?<!prompt )\beval time =\s+[\d.]+ ms /\s+\d+ runs.*?([\d.]+) tokens per second', t)
    return float(m.group(1)) if m else None


def stats(name):
    m = re.search(r'#gen drafts =\s+(\d+).*?#acc tokens =\s+(\d+)(?:, #mean acc len = ([\d.]+))?(?:, #acc rate/pos = \(([^)]*)\))?', read(name))
    if not m:
        return None
    return dict(gen_drafts=int(m.group(1)), acc_tok=int(m.group(2)),
                mean_len=float(m.group(3)) if m.group(3) else None,
                rate_pos=[float(x) for x in m.group(4).split(',')] if m.group(4) else [])


def clean(s):
    s = re.sub(r'<\|im_(start|end)\|>', '', s).replace('[end of text]', '')
    return ''.join(s.split())


class Model(unittest.TestCase):
    def test_expected_len(self):
        # Leviathan 式 (1)：α = 0.5、γ = 4 时 1 + 0.5 + 0.25 + 0.125 + 0.0625
        self.assertAlmostEqual(sm.expected_len(0.5, 4), 1.9375)
        self.assertEqual(sm.expected_len(1.0, 4), 5.0)
        self.assertAlmostEqual(sm.expected_len(0.9, 16), 8.332, places=3)

    def test_alpha_roundtrip(self):
        for a in (0.3, 0.6, 0.88, 0.95):
            for k in (1, 4, 16):
                self.assertAlmostEqual(sm.alpha_from_mean_len(sm.expected_len(a, k), k), a, places=6)

    def test_synth_p(self):
        p = sm.synth_p(3.0, 4)
        self.assertAlmostEqual(sum(p ** j for j in range(1, 5)), 2.0, places=5)

    def test_break_even(self):
        # α 0.88、k 4、v 2：c 低于 0.484 才回本
        self.assertGreater(sm.speedup(0.88, 4, 0.48, 2.0), 1.0)
        self.assertLess(sm.speedup(0.88, 4, 0.49, 2.0), 1.0)

    def test_temp_curve(self):
        tc = sm.temp_curve(trials=100)
        t0, g0, r0 = tc[0]
        self.assertLess(abs(g0 - r0), 0.03)          # 温度趋近 0 时两种验证一样
        for t, g, r in tc:
            if t >= 0.4:
                self.assertGreater(r, g)              # 拒绝采样接受得更多
        self.assertLess(tc[-1][1], tc[1][1])          # 逐位比对随温度下降


class Logs(unittest.TestCase):
    @need('base-q-Q8_0-code.out', 'draft-q-Q8_0-q-Q4_0-k4-code.out')
    def test_lossless_temp0(self):
        for a, b in (('base-q-Q8_0-code.out', 'draft-q-Q8_0-q-Q4_0-k4-code.out'),
                     ('base-qwen3-0.6b-bf16-code.out', 'draft-qwen3-0.6b-bf16-q-Q4_0-k4-code.out'),
                     ('base-q-Q8_0-rewrite.out', 'ngram-ngram-cache-rewrite.out')):
            x, y = clean(read(a)), clean(read(b))
            n = min(len(x), len(y))
            self.assertGreater(n, 200)
            self.assertEqual(x[:n], y[:n], (a, b))

    @need('cost-q-Q8_0.csv', 'cost-qwen3-0.6b-bf16.csv')
    def test_cost_not_flat(self):
        q = sm.read_cost(L, 'q-Q8_0')
        self.assertGreater(q[5] / q[1], 1.5)          # CPU 上喂 5 个比 1 个贵一半以上
        self.assertGreater(q[17] / q[1], 3.0)

    @need('draft-q-Q8_0-q-Q4_0-k16-prose.err')
    def test_rate_pos_nonincreasing(self):
        for f in glob.glob(os.path.join(L, 'draft-*.err')):
            s = stats(os.path.basename(f))
            if s and s['rate_pos']:
                rp = s['rate_pos']
                self.assertTrue(all(a >= b - 1e-9 for a, b in zip(rp, rp[1:])), f)

    @need('pmin-0.0-prose.err', 'pmin-0.9-prose.err')
    def test_pmin(self):
        for p in ('code', 'prose'):
            a = [acc(f'pmin-{v}-{p}.err') for v in ('0.0', '0.3', '0.6', '0.8', '0.9')]
            ml = [stats(f'pmin-{v}-{p}.err')['mean_len'] for v in ('0.0', '0.3', '0.6', '0.8', '0.9')]
            self.assertEqual(a, sorted(a))            # 接受率随 p_min 涨
            self.assertEqual(ml, sorted(ml, reverse=True))  # 接受长度随 p_min 缩
            self.assertGreater(tps(f'pmin-0.8-{p}.err'), tps(f'pmin-0.0-{p}.err'))

    @need('temp-greedy-s1-code.err', 'temp-probabilistic-s1-code.err')
    def test_simple_ignores_probabilistic(self):
        # speculative-simple 不给 result_q，--spec-draft-sampling probabilistic 不起作用
        for s in range(1, 6):
            for p in ('code', 'prose'):
                self.assertEqual(acc(f'temp-greedy-s{s}-{p}.err'), acc(f'temp-probabilistic-s{s}-{p}.err'))

    @need('stoch-s1-code.err', 'temp-greedy-s1-code.err')
    def test_rejection_accepts_more(self):
        for p in ('code', 'prose'):
            g = sum(acc(f'temp-greedy-s{s}-{p}.err') for s in range(1, 6)) / 5
            r = sum(acc(f'stoch-s{s}-{p}.err') for s in range(1, 6)) / 5
            self.assertGreater(r, g + 10)

    @need('ngram-ngram-mod-code.err')
    def test_ngram_mod_default_no_draft(self):
        for p in ('code', 'prose'):
            self.assertEqual(stats(f'ngram-ngram-mod-{p}.err')['gen_drafts'], 0)
            self.assertEqual(stats(f'ngram-ngram-simple-{p}.err')['gen_drafts'], 0)
        self.assertGreater(stats('ngram-ngram-mod-rewrite.err')['gen_drafts'], 0)

    @need('inter-k4-q-Q8_0-r1.err', 'inter-base-q-Q8_0-r1.err')
    def test_interleaved(self):
        for r in (1, 2, 3):
            self.assertLess(tps(f'inter-k4-q-Q8_0-r{r}.err') / tps(f'inter-base-q-Q8_0-r{r}.err'), 1.0)
            self.assertGreater(tps(f'inter-cache-rewrite-r{r}.err') / tps(f'inter-nd-rewrite-r{r}.err'), 1.3)

    @need('granite-k2.err', 'granite-base.err')
    def test_granite_slower(self):
        b = tps('granite-base.err')
        a = [acc(f'granite-k{k}.err') for k in (2, 4, 8)]
        self.assertEqual(a, sorted(a, reverse=True))
        for k in (2, 4, 8):
            self.assertLess(tps(f'granite-k{k}.err'), b)

    @need('lookahead-rewrite.err')
    def test_lookahead_consts(self):
        t = read('lookahead-rewrite.err')
        for k, v in (('W', 15), ('N', 5), ('G', 15)):
            self.assertRegex(t, rf'\b{k} = +{v}\b')
        self.assertLess(tps('lookahead-rewrite.err'), 10)

    @need('unit-rejection.txt')
    def test_unit_ran(self):
        t = read('unit-rejection.txt')
        self.assertIn('Empirical vs target distribution after 5 trials', t)

    def test_no_listening_programs(self):
        with open(os.path.join(HERE, 'exp13.sh')) as f:
            body = [ln for ln in f if not ln.lstrip().startswith('#')]
        s = ''.join(body)
        for bad in ('llama-server', 'rpc-server', 'llama-cli'):
            self.assertNotIn(bad, s)


if __name__ == '__main__':
    unittest.main(verbosity=1)
