"""回归用例：不需要模型文件，用小矩阵验证 compare.py、inject.py、hf_last.py 里的几处关键计算。

用法：python3 test_add_model.py
有 logs/compare.jsonl 时，再核对几条实验结论（RoPE 类错误在位置 0 和基线一样、单层注入在第 14 层第一次跳出来）。
"""
import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compare import logits_stats  # noqa: E402

# inject.py 在导入时要找 gguf-py；只用它的 permute_rows，找不到就跳过相关用例
try:
    from inject import permute_rows
except Exception:  # pragma: no cover
    permute_rows = None


def rope_neox(x, pos, base=10000.0):
    # HF 的 rotate_half：前一半和后一半配对
    d = x.shape[-1]
    inv = base ** (-np.arange(0, d, 2) / d)
    a = pos * inv
    c, s = np.cos(a), np.sin(a)
    x1, x2 = x[: d // 2], x[d // 2:]
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c])


def rope_norm(x, pos, base=10000.0):
    # GPT-J / 原版 LLaMA 的写法：相邻两维配对
    d = x.shape[-1]
    inv = base ** (-np.arange(0, d, 2) / d)
    a = pos * inv
    c, s = np.cos(a), np.sin(a)
    y = x.copy()
    y[0::2] = x[0::2] * c - x[1::2] * s
    y[1::2] = x[0::2] * s + x[1::2] * c
    return y


def grade(nmse):
    for th, s in [(1e-6, 'essentially identical'), (1e-4, 'excellent'), (1e-3, 'very good'),
                  (1e-2, 'good'), (0.1, 'acceptable'), (1.0, 'poor')]:
        if nmse < th:
            return s
    return 'very poor'


class TestPermute(unittest.TestCase):
    def setUp(self):
        if permute_rows is None:
            self.skipTest('gguf-py 不在 ../../../llama.cpp/gguf-py')
        self.rng = np.random.default_rng(0)
        self.d = 8

    def test_norm_after_permute_equals_neox(self):
        # LLaMA 转换器的 permute 把 NEOX 布局改成 NORM 布局：rope_norm(P q) == P rope_neox(q)
        q = self.rng.standard_normal(self.d)
        pq = permute_rows(q.reshape(self.d, 1), 1).ravel()
        for pos in (0, 1, 7, 100):
            np.testing.assert_allclose(rope_norm(pq, pos), permute_rows(rope_neox(q, pos).reshape(self.d, 1), 1).ravel(), atol=1e-12)

    def test_neox_after_permute_breaks_except_pos0(self):
        # 把 permute 过的权重交给 NEOX 架构：位置 0 的打分不变，其余位置变了
        q, k = self.rng.standard_normal(self.d), self.rng.standard_normal(self.d)
        P = lambda v: permute_rows(v.reshape(self.d, 1), 1).ravel()  # noqa: E731
        ref0 = rope_neox(q, 0) @ rope_neox(k, 0)
        self.assertAlmostEqual(rope_neox(P(q), 0) @ rope_neox(P(k), 0), ref0, places=12)
        ref = rope_neox(q, 5) @ rope_neox(k, 2)
        bad = rope_neox(P(q), 5) @ rope_neox(P(k), 2)
        self.assertGreater(abs(bad - ref), 1e-3)

    def test_permute_per_head(self):
        # 多头时每个头各自重排，头和头之间不串
        w = np.arange(2 * self.d).reshape(2 * self.d, 1)
        p = permute_rows(w, 2).ravel()
        self.assertEqual(sorted(p[: self.d]), list(range(self.d)))
        self.assertEqual(list(p[: 4]), [0, 4, 1, 5])


class TestStats(unittest.TestCase):
    def test_identical(self):
        r = np.random.default_rng(1).standard_normal((4, 50)).astype(np.float32)
        s = logits_stats(r, r.copy())
        self.assertEqual(s['max_abs'], 0.0)
        self.assertEqual(s['top1'], 1.0)
        self.assertEqual(s['first_flip'], -1)
        self.assertLess(abs(s['kld_mean']), 1e-12)

    def test_nmse_var_matches_check_nmse(self):
        # check-nmse.py：mean((t-r)^2) / var(r)
        rng = np.random.default_rng(2)
        r = rng.standard_normal((3, 100))
        t = r + 0.01 * rng.standard_normal((3, 100))
        s = logits_stats(r, t)
        self.assertAlmostEqual(s['nmse_var'], float(((t - r) ** 2).mean() / r.var()), places=12)
        self.assertEqual(grade(s['nmse_var']), 'excellent')

    def test_first_flip_and_pos0(self):
        r = np.zeros((3, 5))
        r[:, 0] = 1.0
        t = r.copy()
        t[2, 0], t[2, 3] = 0.0, 2.0
        s = logits_stats(r, t)
        self.assertEqual(s['first_flip'], 2)
        self.assertEqual(s['pos0'], 0.0)

    def test_grade_edges(self):
        self.assertEqual(grade(9.9e-7), 'essentially identical')
        self.assertEqual(grade(5e-3), 'good')
        self.assertEqual(grade(1.0), 'very poor')


class TestLogs(unittest.TestCase):
    def setUp(self):
        p = os.path.join(HERE, 'logs', 'compare.jsonl')
        if not os.path.exists(p):
            self.skipTest('没有 logs/compare.jsonl')
        self.r = {}
        with open(p) as f:
            for line in f:
                x = json.loads(line)
                self.r[x['tag']] = x

    def test_rope_bugs_keep_pos0(self):
        base = self.r['bf16']['logits']['pos0']
        for tag in ('kv-theta1e4', 'bug-qk-permute'):
            self.assertAlmostEqual(self.r[tag]['logits']['pos0'], base, places=6)
        self.assertGreater(self.r['bug-norm-plus1']['logits']['pos0'], 100 * base)

    def test_single_layer_bug_located(self):
        lay = self.r['bug-qk-permute-l14']['layers']
        first = next(x['layer'] for x in lay if x['cos_min'] < 0.99)
        self.assertEqual(first, 14)

    def test_bf16_beats_hf_bf16(self):
        self.assertLess(self.r['bf16']['logits']['kld_mean'], self.r['hf-bf16']['logits']['kld_mean'])


if __name__ == '__main__':
    unittest.main(verbosity=1)
