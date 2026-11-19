#!/usr/bin/env python3
"""10-cpu-tuning 的回归测试：cpu_probe 的解析逻辑用假的 /sys 目录树测，实测结论用 logs 里的数据核对，不需要模型。

    python3 test_cpu.py
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
L = os.path.join(HERE, 'logs')
sys.path.insert(0, HERE)

import cpu_probe as cp  # noqa: E402


def jl(name):
    out = []
    p = os.path.join(L, name)
    if not os.path.exists(p):
        return out
    for line in open(p):
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def avg(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def ts(rows, **kw):
    """按字段过滤，返回 avg_ts 的平均；test 写成 pp128 或 tg32"""
    t = kw.pop('test', None)
    r = []
    for d in rows:
        if any(str(d.get(k)) != str(v) for k, v in kw.items()):
            continue
        name = f"pp{d['n_prompt']}" if d['n_gen'] == 0 else f"tg{d['n_gen']}"
        if t is None or name == t:
            r.append(d['avg_ts'])
    return avg(r)


def fake_sys(root, siblings, nodes):
    """siblings: {cpu: '0-1'}，nodes: {'node0': '0-3'}"""
    base = os.path.join(root, 'sys/devices/system/cpu')
    os.makedirs(base)
    open(os.path.join(base, 'online'), 'w').write(f'0-{len(siblings) - 1}\n')
    for c, s in siblings.items():
        d = os.path.join(base, f'cpu{c}/topology')
        os.makedirs(d)
        open(os.path.join(d, 'thread_siblings_list'), 'w').write(s + '\n')
    for n, lst in nodes.items():
        d = os.path.join(root, 'sys/devices/system/node', n)
        os.makedirs(d)
        open(os.path.join(d, 'cpulist'), 'w').write(lst + '\n')
    os.makedirs(os.path.join(root, 'proc'))
    open(os.path.join(root, 'proc/cpuinfo'), 'w').write('flags\t\t: fpu sse2 avx2 fma f16c\n')


class Probe(unittest.TestCase):
    def test_parse_list(self):
        self.assertEqual(cp.parse_list('0-1,4'), [0, 1, 4])
        self.assertEqual(cp.parse_list('3'), [3])
        self.assertEqual(cp.parse_list(''), [])

    def test_mask(self):
        # 每个物理核取第一个超线程：CPU 0 和 2，掩码 0x5
        self.assertEqual(cp.mask_of([0, 2]), '0x5')
        self.assertEqual(cp.mask_of([0, 1, 2, 3]), '0xf')

    def test_smt_topology(self):
        with tempfile.TemporaryDirectory() as root:
            fake_sys(root, {0: '0-1', 1: '0-1', 2: '2-3', 3: '2-3'}, {'node0': '0-3'})
            online, cores, nodes = cp.topology(root)
            adv = cp.advise(online, cores, nodes)
            self.assertEqual(adv['threads'], 2)
            self.assertEqual(adv['mask'], '0x5')
            self.assertTrue(adv['smt'])
            self.assertIn('空操作', adv['numa'])

    def test_interleaved_siblings(self):
        # 有的机器把超线程编号成 0 和 4、1 和 5：掩码应该是 0xf
        with tempfile.TemporaryDirectory() as root:
            sib = {c: f'{c % 4},{c % 4 + 4}' for c in range(8)}
            fake_sys(root, sib, {'node0': '0-3', 'node1': '4-7'})
            online, cores, nodes = cp.topology(root)
            adv = cp.advise(online, cores, nodes)
            self.assertEqual(adv['threads'], 4)
            self.assertEqual(adv['mask'], '0xf')
            self.assertIn('distribute', adv['numa'])

    def test_repack_avx2_only(self):
        # 只有 AVX2 时 Q2_K 不走 repack，留给 tiled
        with tempfile.TemporaryDirectory() as root:
            fake_sys(root, {0: '0', 1: '1'}, {})
            out = subprocess.run([sys.executable, os.path.join(HERE, 'cpu_probe.py'), '--root', root],
                                 capture_output=True, text=True).stdout
            line = [x for x in out.splitlines() if x.startswith('走 repack')][0]
            self.assertIn('Q4_K', line)
            self.assertNotIn('Q2_K', line)
            self.assertNotIn('-C', out)  # 没有超线程就不建议掩码


class Logs(unittest.TestCase):
    TH = jl('threads.jsonl')

    def test_threads_scale_prefill(self):
        # prefill 随线程数涨，4 线程比 1 线程快两倍以上
        if not self.TH:
            self.skipTest('no logs')
        p1 = ts(self.TH, build='omp', cfg='none', n_threads=1, test='pp128')
        p4 = ts(self.TH, build='omp', cfg='none', n_threads=4, test='pp128')
        self.assertGreater(p4 / p1, 2.0)

    def test_decode_saturates(self):
        # decode 从 3 线程到 4 线程涨不到 15%：已经贴着带宽
        if not self.TH:
            self.skipTest('no logs')
        t3 = ts(self.TH, build='omp', cfg='none', n_threads=3, test='tg32')
        t4 = ts(self.TH, build='omp', cfg='none', n_threads=4, test='tg32')
        self.assertLess(t4 / t3, 1.15)

    def test_oversubscribed_strict(self):
        # 3 个线程绑到 2 个 CPU 上：比不绑慢一个数量级
        if not self.TH:
            self.skipTest('no logs')
        over = [d for d in self.TH if d.get('cfg') == '5-strict-over']
        if not over:
            self.skipTest('no oversub rows')
        base = ts(self.TH, build='omp', cfg='none', n_threads=1, test='pp128')
        self.assertLess(max(d['avg_ts'] for d in over), base)

    def test_sticky(self):
        # 同一进程里先跑超订绑核，后面不绑核的测试也被拖慢
        rows = jl('sticky.jsonl')
        if not rows:
            self.skipTest('no logs')
        fresh = [d['avg_ts'] for d in rows if d['order'] == 'fresh' and d['build'] == 'omp']
        after = [d['avg_ts'] for d in rows if d['order'] == 'after' and d['build'] == 'omp' and d['cpu_mask'] == '0x0']
        self.assertLess(after[0], fresh[0] / 3)

    def test_kmp_blocktime_noop(self):
        # GCC 构建下 KMP_BLOCKTIME=0 和缺省差不多，OMP_WAIT_POLICY=passive 才有影响
        rows = jl('omp.jsonl')
        if not rows:
            self.skipTest('no logs')
        base = ts(rows, env='X=1', test='tg32')
        kmp = ts(rows, env='KMP_BLOCKTIME=0', test='tg32')
        self.assertLess(abs(kmp / base - 1), 0.08)

    def test_tiled_beats_vecdot(self):
        # Q6_K 关掉 repack 后，tiled 在 128 行的 prefill 上比 vec_dot 快
        rows = jl('tiled.jsonl')
        if not rows:
            self.skipTest('no logs')
        t = ts(rows, variant='q-Q6_K', path='tiled', test='pp128')
        v = ts(rows, variant='q-Q6_K', path='vecdot', test='pp128')
        self.assertGreater(t, v)


if __name__ == '__main__':
    unittest.main(verbosity=2)
