#!/usr/bin/env python3
"""09-sampling-grammar 的回归测试：用 logs 里存下的 logits 和日志，不需要模型。

    python3 test_samp.py
需要 numpy。
"""
import os
import re
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
L = os.path.join(HERE, 'logs')
sys.path.insert(0, HERE)

import samp_sim as ss  # noqa: E402


def rd(fn):
    with open(os.path.join(L, fn)) as f:
        return f.read()


def logits(name):
    return np.fromfile(os.path.join(L, f'logits-{name}.f32'), dtype=np.float32).astype(np.float64)


def micro(name):
    out = {}
    for line in rd(f'micro-{name}.txt').splitlines():
        m = re.match(r'(\S+)\s+kept\s+(\d+)\s+us\s+([\d.]+)', line)
        if m:
            out[m.group(1)] = (int(m.group(2)), float(m.group(3)))
    return out


NAMES = ('fact', 'story', 'code')


class AgainstC(unittest.TestCase):
    """numpy 重算的候选数和 samp_bench（真正的 llama.cpp 采样器）对拍。"""

    def test_exact(self):
        for n in NAMES:
            lg, c = logits(n), micro(n)
            self.assertEqual(ss.top_k(lg, 40).sum(), c['top_k40'][0], n)
            self.assertEqual(ss.min_p(lg, 0.05).sum(), c['min_p0.05'][0], n)
            self.assertEqual(ss.top_n_sigma(lg, 1.0).sum(), c['top_nsigma1'][0], n)
            self.assertEqual(ss.chain(lg).sum(), c['chain_default'][0], n)

    def test_top_p_close(self):
        # C 里 softmax 和累加都是 float，长尾分布上边界会差几个 token
        for n in NAMES:
            a, b = ss.top_p(logits(n), 0.95).sum(), micro(n)['top_p0.95'][0]
            self.assertLessEqual(abs(a - b), max(2, 0.02 * b), n)


class Rules(unittest.TestCase):
    def test_min_p_is_prob_ratio(self):
        lg = logits('fact')
        pr = ss.softmax(lg)
        self.assertTrue(np.array_equal(ss.min_p(lg, 0.05), pr >= 0.05 * pr.max() * (1 - 1e-12)))

    def test_penalty_sign(self):
        # 负 logit 乘 1.1 变得更负，正 logit 除以 1.1 变小：两边都是压低
        lg = np.array([2.2, -2.0, 1.0])
        out = ss.penalties(lg, {0: 1, 1: 1}, repeat=1.1)
        self.assertAlmostEqual(out[0], 2.0)
        self.assertAlmostEqual(out[1], -2.2)
        self.assertEqual(out[2], 1.0)
        out = ss.penalties(lg, {0: 3}, freq=0.5, present=1.5)
        self.assertAlmostEqual(out[0], 2.2 - 1.5 - 1.5)

    def test_temp_last_keeps_count(self):
        # 缺省链里温度在最后，截断结果和温度无关
        for n in NAMES:
            lg = logits(n)
            ks = {int(ss.chain(lg, k=0, temp=t).sum()) for t in (0.3, 0.8, 1.5, 3.0)}
            self.assertEqual(len(ks), 1, n)

    def test_temp_first_top_p_explodes(self):
        lg = logits('fact')
        self.assertGreater(ss.top_p(lg / 2.0, 0.95).sum(), 10 * ss.top_p(lg, 0.95).sum())
        self.assertLess(ss.min_p(lg / 2.0, 0.05).sum(), 100)

    def test_nsigma_temp_invariant(self):
        for n in NAMES:
            lg = logits(n)
            self.assertEqual(ss.top_n_sigma(lg, 1.0).sum(), ss.top_n_sigma(lg / 2.0, 1.0).sum(), n)


class Logs(unittest.TestCase):
    def test_gguf_defaults(self):
        t = rd('params.txt')
        # 不写参数：top_k 20、temp 0.6 来自 GGUF；写了 --top-k 40 --temp 0.8 就用用户的
        self.assertIn('top_k = 20, top_p = 0.950, min_p = 0.050', t)
        self.assertIn('temp = 0.600', t)
        self.assertIn('top_k = 40', t)
        self.assertIn('-> greedy', t)

    def test_think_block(self):
        t = rd('tmpl-render.txt')
        self.assertIn('enable_thinking=false（81 字符）', t)
        self.assertIn('单轮，缺省（62 字符）', t)

    def test_reasoning_flag_ignored_by_completion(self):
        t = rd('tmpl-flags.txt')
        n = re.findall(r'number of tokens in prompt = (\d+)', t)
        self.assertEqual(n[:2], ['15', '15'])
        self.assertIn('invalid argument: --chat-template-kwargs', t)

    def test_repetition(self):
        rows = {}
        for line in rd('rep_summary.txt').splitlines()[1:]:
            f = line.split()
            rows[f[0]] = float(f[2].rstrip('%'))
        self.assertGreater(rows['none'], 50)
        for k in ('repeat', 'presence', 'dry'):
            self.assertLess(rows[k], 15, k)


if __name__ == '__main__':
    unittest.main(verbosity=1)
