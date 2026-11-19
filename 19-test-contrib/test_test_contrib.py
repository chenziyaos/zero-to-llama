"""回归用例：不需要模型和编好的 llama.cpp，验证 test-backend-ops 的误差口径、尾部漏算的预测、日志解析和 SQLite 改标签。

用法：python3 test_test_contrib.py
有 logs/ 时，再核对几条实验结论（单算子测试全过、融合测试失败的用例恰好是预测的那些、修好后全过）。
"""
import os
import sqlite3
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import parse_tbo as P  # noqa: E402
from load_sql import load  # noqa: E402

LOGS = os.path.join(HERE, 'logs')


def nmse(ref, out):
    # test-backend-ops.cpp 的 nmse：sum((a-b)^2) / sum(a^2)，a 是参考
    ref, out = np.asarray(ref, np.float64), np.asarray(out, np.float64)
    return float(((ref - out) ** 2).sum() / (ref ** 2).sum())


def rms_norm_mul(x, w, eps=1e-6, tail_bug=False):
    scale = 1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    y = x * scale * w
    if tail_bug:
        n = x.shape[-1] & ~15
        y[..., n:] = 0.0  # 输出缓冲区没写的那几列；真实情况是残留值，这里取 0
    return y


def read(name):
    p = os.path.join(LOGS, name)
    if not os.path.exists(p):
        raise unittest.SkipTest(f'没有 logs/{name}')
    return open(p, errors='replace').read()


class TestNmse(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(nmse([1, 2, 3], [1, 2, 3]), 0.0)

    def test_scale_free(self):
        # 同样的相对误差，放大 1000 倍 nmse 不变
        a, b = np.array([1.0, -2.0, 3.0]), np.array([1.01, -2.0, 3.0])
        self.assertAlmostEqual(nmse(a, b), nmse(a * 1000, b * 1000))

    def test_tail_error_shrinks_with_length(self):
        # 漏掉的列数是 ne0 % 16；都只漏 1 列时，行越长占比越小，nmse 跟着变小，但都远高于 1e-7
        rng = np.random.default_rng(0)
        errs = []
        for n in (129, 1025, 16385):
            x = rng.uniform(-10, 10, (6, n))
            w = rng.uniform(-10, 10, n)
            errs.append(nmse(rms_norm_mul(x, w), rms_norm_mul(x, w, tail_bug=True)))
        self.assertTrue(errs[0] > errs[1] > errs[2] > 1e-7)

    def test_multiple_of_16_is_exact(self):
        rng = np.random.default_rng(1)
        x, w = rng.uniform(-1, 1, (3, 512)), rng.uniform(-1, 1, 512)
        self.assertEqual(nmse(rms_norm_mul(x, w), rms_norm_mul(x, w, tail_bug=True)), 0.0)


class TestParse(unittest.TestCase):
    LOG = ('\x1b[1;31m[ADD] ERR = 0.477 > 0.0000001   \x1b[0mRMS_NORM_MUL_ADD(type=f32,ne=[1,3,2,1],eps=0.000001,'
           'broadcast=0,multi_add=0,post_mul=0,alias_rms_input=0,weight_broadcast=1): FAIL\n'
           '  RMS_NORM_MUL_ADD(type=f32,ne=[1025,5,4,3],eps=0.000001,broadcast=1,multi_add=0,post_mul=0,'
           'alias_rms_input=0,weight_broadcast=0): OK\n'
           '  ARGSORT(type=f32,ne=[8,1,1,1],order=0): not supported [CPU]\n'
           '  27/45 tests passed\n== rc=1 wall_s=2.4 maxrss_mib=293\n')

    def test_cases(self):
        cs = P.cases(self.LOG)
        self.assertEqual([c['result'] for c in cs], ['FAIL', 'OK', 'unsupported'])
        self.assertEqual(cs[0]['node'], 'ADD')
        self.assertAlmostEqual(cs[0]['err'], 0.477)
        self.assertEqual(cs[0]['thr'], 1e-7)
        self.assertEqual(cs[1]['ne'], [1025, 5, 4, 3])

    def test_summary(self):
        s = P.summary(self.LOG)
        self.assertEqual((s['passed'], s['total'], s['rc']), (27, 45, 1))

    def test_fusable(self):
        cs = P.cases(self.LOG)
        self.assertTrue(P.fusable(cs[0]))
        self.assertFalse(P.fusable(cs[1]))  # broadcast=1：权重宽度和输入不同，不融合

    def test_ctest(self):
        t = ('1/2 Test #28: test-a ....   Passed    0.60 sec\n'
             '2/2 Test #16: test-b ....***Failed    3.62 sec\n'
             '50% tests passed, 1 tests failed out of 2\nTotal Test time (real) =   4.30 sec\n')
        c = P.ctest(t)
        self.assertEqual(c['tests'][1], ('test-b', '***Failed', 3.62))
        self.assertEqual((c['failed'], c['n'], c['real_s']), (1, 2, 4.3))


class TestLoadSql(unittest.TestCase):
    SQL = ("CREATE TABLE IF NOT EXISTS t (build_commit TEXT, v REAL);\n"
           "INSERT INTO t (build_commit, v) VALUES ('0ec4027', 1.0);\n"
           "INSERT INTO t (build_commit, v) VALUES ('0ec4027', 2.0);\n")

    def test_relabel_only_new_rows(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, 'x.sqlite')
            self.assertEqual(load(db, self.SQL, 'fuse0'), {'t': 2})
            self.assertEqual(load(db, self.SQL, 'fuse1'), {'t': 2})
            rows = sqlite3.connect(db).execute('SELECT build_commit, COUNT(*) FROM t GROUP BY 1').fetchall()
            self.assertEqual(sorted(rows), [('fuse0', 2), ('fuse1', 2)])


class TestLogs(unittest.TestCase):
    def test_single_ops_blind(self):
        # 同一个带错的构建，单算子测试全过
        for op in ('RMS_NORM', 'MUL'):
            s = P.summary(read(f'bug-{op}.log'))
            self.assertEqual(s['passed'], s['total'])

    def test_prediction_matches(self):
        for op in ('RMS_NORM_MUL_ADD', 'RMS_NORM_MUL_ROPE'):
            for c in P.cases(read(f'bug-{op}.log')):
                with self.subTest(op=op, params=c['params']):
                    self.assertEqual(c['result'] == 'FAIL', P.tail_bug_predicts(c['ne'][0], fused=P.fusable(c)))

    def test_disable_fusion_and_fix(self):
        for name in ('bugnofuse-RMS_NORM_MUL_ADD', 'fix-RMS_NORM_MUL_ADD', 'fix-RMS_NORM_MUL_ROPE'):
            s = P.summary(read(f'{name}.log'))
            self.assertEqual((s['passed'], s['rc']), (s['total'], 0), name)

    def test_full_cpu(self):
        s = P.summary(read('tbo-cpu-full.log'))
        self.assertEqual(s['passed'], s['total'])


if __name__ == '__main__':
    unittest.main()
