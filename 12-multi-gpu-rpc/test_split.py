#!/usr/bin/env python3
"""12-multi-gpu-rpc 的回归测试：切层规则和通信量用公式测，假多卡的结论用 logs 核对。

    python3 test_split.py
没有模型文件或 logs 的用例会跳过。
"""
import os
import re
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
L = os.path.join(HERE, 'logs')
MD = os.environ.get('MODELS', os.path.join(HERE, '..', '..', '..', 'models'))
sys.path.insert(0, HERE)

import comm_volume as cv  # noqa: E402
import split_plan as sp  # noqa: E402

Q = os.path.join(MD, 'q-Q4_0.gguf')
H14 = os.path.join(MD, 'hdr', 'Qwen3-14B-Q4_K_M.head')
MiB = 2 ** 20


def read(name):
    p = os.path.join(L, name)
    if not os.path.exists(p):
        return ''
    with open(p) as f:
        return f.read()


def fake_layers(text):
    return re.findall(r'layers\s+(\d+)-\s*(\d+) -> (\S+)', text)


def breakdown(text):
    out = {}
    for line in text.splitlines():
        f = line.split()
        if len(f) == 5 and re.match(r'^[\d.]+$', f[1]) and f[0] != 'buffer':
            out[f[0]] = tuple(map(float, f[1:]))
    return out


class SplitRule(unittest.TestCase):
    def test_points_equal(self):
        self.assertEqual(sp.split_points([8000, 8000]), [0.5, 1.0])

    def test_points_ts(self):
        self.assertEqual(sp.split_points([3, 1]), [0.75, 1.0])

    def test_assign_two(self):
        # 28 层 + 输出层，两张卡平分：0-14 一张，15-28 一张
        d = sp.assign(28, -1, [0.5, 1.0])
        self.assertEqual(d.count(0), 15)
        self.assertEqual(d.count(1), 14)
        self.assertEqual(d[-1], 1)

    def test_assign_partial_ngl(self):
        d = sp.assign(28, 20, [0.5, 1.0])
        self.assertEqual(d[:9], [-1] * 9)
        self.assertEqual(d[9], 0)


class Comm(unittest.TestCase):
    def test_pp_bytes(self):
        self.assertEqual(cv.pp(1024, 2, 512)['per_hop'], 2 * MiB)
        self.assertEqual(cv.pp(1024, 2, 1)['per_hop'], 4096)
        self.assertEqual(cv.pp(1024, 4, 1)['total'], 3 * 4096)

    def test_tp_two_per_layer(self):
        self.assertEqual(cv.tp(1024, 28, 2, 1)['n_ar'], 56)

    def test_nccl_threshold(self):
        self.assertFalse(cv.nccl_bf16(2, 32767))
        self.assertTrue(cv.nccl_bf16(2, 32768))
        self.assertFalse(cv.nccl_bf16(4, 262143))

    def test_butterfly(self):
        self.assertEqual([cv.butterfly_steps(n) for n in (2, 4, 8)], [1, 2, 3])
        self.assertEqual(cv.butterfly_rounds(3), 3)

    def test_rpc_graph(self):
        self.assertEqual(cv.rpc_graph(1000, 1500), (452029, 21))


@unittest.skipUnless(os.path.exists(Q), '缺 q-Q4_0.gguf')
class Plan(unittest.TestCase):
    def test_q_two(self):
        from gguf_lite import GGUF
        _, _, w, kv, layers = sp.plan(GGUF(Q), [8000, 8000])
        self.assertAlmostEqual((w[0] + kv[0]) / MiB, 366.7, delta=0.1)
        self.assertAlmostEqual((w[1] + kv[1]) / MiB, 439.5, delta=0.1)

    @unittest.skipUnless(os.path.exists(H14), '缺 14B 张量头')
    def test_rpc_first(self):
        from gguf_lite import GGUF
        _, _, w, kv, layers = sp.plan(GGUF(H14), [24000, 12000])
        self.assertEqual(layers[0][-1], 27)
        self.assertAlmostEqual((w[0] + kv[0]) / MiB, 5688.8, delta=0.1)


class FakeLogs(unittest.TestCase):
    def need(self, name):
        t = read(name)
        if not t:
            self.skipTest(f'缺 logs/{name}，先跑 bash exp12.sh fake')
        return t

    def test_layer_split_matches_plan(self):
        t = self.need('fake-n2.txt')
        self.assertEqual(fake_layers(t), [('0', '14', 'FAKE0'), ('15', '28', 'FAKE1')])
        b = breakdown(t)
        # llama.cpp 自己的账本和 split_plan.py 的权重、KV 对得上
        self.assertAlmostEqual(b['FAKE0'][0] + b['FAKE0'][1], 366.7, delta=0.1)
        self.assertAlmostEqual(b['FAKE1'][0] + b['FAKE1'][1], 439.5, delta=0.1)

    def test_rpc_goes_first(self):
        t = self.need('fake-rpc.txt')
        self.assertEqual(fake_layers(t)[0][2], 'RPC0')
        self.assertLess(t.index('using device RPC0'), t.index('using device FAKE0'))

    def test_ts_equals_free(self):
        a, b = self.need('fake-ts31.txt'), self.need('fake-free124.txt')
        self.assertEqual(fake_layers(a), fake_layers(b))

    def test_pipeline_needs_events(self):
        self.assertNotIn('pipeline parallelism enabled', self.need('fake-n2.txt'))
        t = self.need('fake-n2-events.txt')
        self.assertIn('pipeline parallelism enabled', t)
        self.assertIn('sched copies = 4', t)

    def test_tp_one_meta_device(self):
        t = self.need('fake-tp4.txt')
        self.assertIn('creating a Meta device for tensor parallelism from 4 devices', t)
        self.assertIn('model devices = 1', t)
        self.assertIn('splits = 2', t)

    def test_sched_boundary(self):
        t = self.need('fake-sched.txt')
        self.assertIn('[l_out-9 (   2M)]', t)
        self.assertIn('[l_out-9 (   4K)]', t)
        self.assertIn('splits = 4', t)


if __name__ == '__main__':
    unittest.main(verbosity=1)
