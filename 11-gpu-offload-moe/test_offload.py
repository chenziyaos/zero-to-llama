#!/usr/bin/env python3
"""11-gpu-offload-moe 的回归测试：缓存语义用手写的小路由测，显存账本用 GGUF 张量头测，实测结论用 logs 核对。

    python3 test_offload.py
没有模型文件或张量头的用例会跳过。
"""
import json
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
L = os.path.join(HERE, 'logs')
MD = os.environ.get('MODELS', os.path.join(HERE, '..', '..', '..', 'models'))
sys.path.insert(0, HERE)

import moe_cache_sim as sim  # noqa: E402
import vram_ledger as vl  # noqa: E402

M1 = os.path.join(MD, 'moe', 'granite-1b-a400m-Q4_K_M.gguf')
QH = os.path.join(MD, 'hdr', 'qwen3-30b-a3b-Q4_K_M.head')
MiB = 2 ** 20


def read(name):
    p = os.path.join(L, name)
    if not os.path.exists(p):
        return ''
    with open(p) as f:
        return f.read()


def table(name):
    """sim-*.txt 的结果表：{倍数: (layers, hot, oracle, lru, slru)}"""
    out = {}
    for line in read(name).splitlines():
        f = line.split()
        if len(f) == 8 and re.match(r'^\d+\.\d+$', f[0]):
            out[float(f[2])] = tuple(map(float, f[3:]))
    return out


def trace(n_layer, n_expert, n_used, rows):
    """rows[t][il] 是这一层选中的专家，拼成 moe_trace 的数据区"""
    data = bytes(e for tok in rows for sel in tok for e in sel)
    return n_layer, n_expert, n_used, len(rows), data


class CacheSemantics(unittest.TestCase):
    def test_repeat_hits(self):
        # 同一组专家反复出现，第一次全缺，之后全中
        nl, ne, nu, nt, data = trace(1, 8, 2, [[[0, 1]]] * 10)
        for seg in (False, True):
            self.assertAlmostEqual(sim.replay(sim.SLRU(4, seg), data, nl, ne, nu, nt, 1), 1.0)

    def test_cycle_larger_than_cache(self):
        # 6 个专家轮着用，4 个槽：LRU 每次换掉的正好是下一个要用的
        rows = [[[2 * (t % 3), 2 * (t % 3) + 1]] for t in range(30)]
        nl, ne, nu, nt, data = trace(1, 8, 2, rows)
        self.assertEqual(sim.replay(sim.SLRU(4, False), data, nl, ne, nu, nt, 3), 0.0)

    def test_protected_half(self):
        # protected 段最多 n_slots/2 个，超出的退回 probation
        c = sim.SLRU(4, True)
        for e in range(4):
            c.plan(0, [e], 8)
        for e in range(4):
            c.plan(0, [e], 8)
        self.assertEqual(len(c.seg[1]), 2)
        self.assertEqual(len(c.seg[0]), 2)

    def test_victim_skips_current_request(self):
        # 同一次请求里选中的专家不会被自己挤掉
        c = sim.SLRU(2, False)
        self.assertEqual(c.plan(0, [3, 5], 8), (0, 2))
        self.assertEqual(c.plan(0, [3, 5], 8), (2, 0))
        self.assertIsNone(c.plan(0, [1, 2, 3], 8))


class SimLogs(unittest.TestCase):
    def check(self, name):
        t = table(name)
        if not t:
            self.skipTest(name + ' 不在')
        # 槽位不到一个 token 的用量：两种都是 0
        self.assertEqual(t[0.5][3], 0.0)
        self.assertEqual(t[0.5][4], 0.0)
        # 正好一个 token：SLRU 远低于 LRU，保护段只有一半槽位
        self.assertLess(t[1.0][4], t[1.0][3] * 0.5)
        # 两倍以上：SLRU 追上并略超 LRU
        for m in (2.0, 2.5, 3.0):
            self.assertGreaterEqual(t[m][4], t[m][3])
        # oracle 是静态放法的上限，按层放永远最差
        for m, (lay, hot, orc, lru, slru) in t.items():
            self.assertGreaterEqual(orc, hot)
            self.assertLessEqual(lay, hot)

    def test_1b(self):
        self.check('sim-1b-wiki.txt')

    def test_3b(self):
        self.check('sim-3b-wiki.txt')

    def test_skew_and_locality(self):
        s = read('sim-1b-wiki.txt')
        if not s:
            self.skipTest('sim-1b-wiki.txt 不在')
        top = float(re.search(r'最热 1/4 专家占的选择：均匀时 [\d.]+，各层 [\d.]+ 到 [\d.]+，平均 ([\d.]+)', s).group(1))
        reuse = float(re.search(r'和上一个 token 重合的专家比例：随机时 ([\d.]+)，各层 [\d.]+ 到 [\d.]+，平均 ([\d.]+)', s).group(2))
        self.assertGreater(top, 0.40)
        self.assertGreater(reuse, 0.40)
        n32 = float(re.search(r'n=32 ([\d.]+)', s).group(1))
        self.assertGreater(n32, 0.9)

    def test_profile_shift(self):
        # 用 wiki 的频次钉生成代码时的专家，离 oracle 差一截
        t = table('sim-1b-code.txt')
        if not t:
            self.skipTest('sim-1b-code.txt 不在')
        self.assertGreater(t[2.0][2] - t[2.0][1], 0.08)


class Ledger(unittest.TestCase):
    def test_granite_1b(self):
        if not os.path.exists(M1):
            self.skipTest('没有 granite-1b')
        g = vl.GGUF(M1)
        start, dev, host = vl.ledger(g, -1, 0, 4096, 2)
        self.assertEqual(start, 0)
        self.assertEqual(round(dev['exps'] / MiB), 698)
        self.assertEqual(round(dev['kv'] / MiB), 192)
        _, dev2, host2 = vl.ledger(g, -1, 24, 4096, 2)
        self.assertEqual(dev2['exps'], 0)
        self.assertEqual(host2['exps'], dev['exps'])

    def test_ngl_counts_output(self):
        # -ngl N 从后往前数，输出层算第 n_layer 层：-ngl 1 只放输出层
        if not os.path.exists(M1):
            self.skipTest('没有 granite-1b')
        g = vl.GGUF(M1)
        start, dev, _ = vl.ledger(g, 1, 0, 4096, 2)
        self.assertEqual(start, 24)
        self.assertEqual(dev['exps'], 0)
        self.assertGreater(dev['output'], 0)
        start, _, _ = vl.ledger(g, 1, 0, 4096, 2, legacy=True)
        self.assertEqual(start, 23)

    def test_qwen_budget_and_cache(self):
        if not os.path.exists(QH):
            self.skipTest('没有 qwen3-30b-a3b 的张量头')
        g = vl.GGUF(QH)
        n, dev = vl.min_ncmoe(g, 16384, 1024, 4096, 2)
        self.assertEqual(n, 7)
        gr = vl.cache_groups(g, 48, 1700)
        self.assertEqual(len(gr), 2)
        for layers, types, slots, top, foot in gr:
            self.assertEqual(foot, 24 * 8)
            self.assertLess(slots, 2 * foot)
        gr = vl.cache_groups(g, 48, 3400)
        self.assertTrue(all(s >= 2 * f for _, _, s, _, f in gr))

    def test_cache_too_small(self):
        if not os.path.exists(M1):
            self.skipTest('没有 granite-1b')
        g = vl.GGUF(M1)
        self.assertTrue(all(s == 0 for _, _, s, _, _ in vl.cache_groups(g, 24, 2)))


class CpuLogs(unittest.TestCase):
    def test_fit_ignores_repack(self):
        s = read('fit.txt')
        if not s:
            self.skipTest('fit.txt 不在')
        host = int(re.search(r'- Host\s+\|\s+(\d+)', s).group(1))
        rep = int(re.search(r'- CPU_REPACK\s+\|\s+(\d+)', s).group(1))
        proj = int(re.search(r'projected to use (\d+) MiB of host', s).group(1))
        self.assertEqual(proj, host)
        self.assertGreater(rep, host)

    def test_override_keeps_repack(self):
        s = read('cpu-override.txt')
        if not s:
            self.skipTest('cpu-override.txt 不在')
        self.assertIn('q6_K) buffer type overridden to CPU\n', s)
        self.assertIn('q4_K) buffer type overridden to CPU_REPACK', s)

    def test_cpu_only_same_decode(self):
        rows = [json.loads(x) for x in read('cpu.jsonl').splitlines() if x.strip()]
        if not rows:
            self.skipTest('cpu.jsonl 不在')
        for m in ('1b', '3b'):
            v = {}
            for cfg in ('default', 'cmoe'):
                xs = [r['avg_ts'] for r in rows if r['model'] == m and r['cfg'] == cfg and r['n_gen']]
                v[cfg] = sum(xs) / len(xs)
            # 纯 CPU 构建里 -cmoe 只是换缓冲类型，decode 差别在一成以内
            self.assertLess(abs(v['cmoe'] - v['default']) / v['default'], 0.1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
