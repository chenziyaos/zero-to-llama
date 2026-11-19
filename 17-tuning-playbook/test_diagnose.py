"""diagnose.py 的回归用例：用 logs/ 下保存的真实日志，加几段手写的最小日志。用法：python3 test_diagnose.py"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import diagnose as dg

LOGS = os.path.join(HERE, 'logs')
MODELS = os.environ.get('MODELS', os.path.join(HERE, '..', '..', '..', 'models'))


def log(name):
    with open(os.path.join(LOGS, name), errors='replace') as f:
        return f.read()


def has(name):
    return os.path.exists(os.path.join(LOGS, name))


SYNTH_SILENT = '''
I system_info: n_threads = 8 (n_threads_batch = 8) / 4 | CPU : AVX2 = 1 |
I generate: n_ctx = 4096, n_batch = 256, n_predict = 4, n_keep = 0
I llama_context: n_seq_max             = 4
I llama_context: n_ctx                 = 4096
I llama_context: n_ctx_seq             = 1024
I llama_context: n_ubatch              = 256
I llama_context: kv_unified            = true
'''


class ParseTest(unittest.TestCase):
    def test_synthetic_fields(self):
        p = dg.parse(SYNTH_SILENT)
        self.assertEqual((p['threads'], p['hw_threads']), (8, 4))
        self.assertEqual((p['n_ctx'], p['n_batch'], p['n_ubatch']), (4096, 256, 256))
        self.assertEqual((p['n_seq_max'], p['n_ctx_seq']), (4, 1024))

    def test_synthetic_silent(self):
        what = ' '.join(w for _, w, _ in dg.silent_checks(dg.parse(SYNTH_SILENT)))
        for key in ('-t 8', '截到 256', '每条序列只有 1024', 'KV 池'):
            self.assertIn(key, what)

    def test_rules_order(self):
        text = ('W n_ctx_seq (100096) > n_ctx_train (40960) -- possible training context overflow\n'
                'E ggml_backend_cpu_buffer_type_alloc_buffer: failed to allocate buffer of size 11479810048\n'
                'E llama_completion: error: unable to create context\n')
        hits = dg.match_rules(text)
        self.assertEqual([h['level'] for h in hits], ['warn', 'error', 'error'])
        self.assertEqual(hits[1]['groups'], ('11479810048',))

    def test_verdict(self):
        self.assertEqual(dg.speed_verdict(10, 60)[1], 'far')
        self.assertEqual(dg.speed_verdict(40, 60)[1], 'low')
        self.assertEqual(dg.speed_verdict(58, 60)[1], 'ok')


@unittest.skipUnless(has('default-v4.log'), '先跑 exp17.sh defaults')
class DefaultsTest(unittest.TestCase):
    def test_no_c_means_train_ctx(self):
        p = dg.parse(log('default-v4.log'))
        self.assertEqual(p['ctx'], p['n_ctx_train'])
        self.assertEqual(p['kv_cells'], 40960)
        self.assertTrue(any('训练长度' in w for _, w, _ in dg.silent_checks(p)))

    def test_c4096(self):
        p = dg.parse(log('c4096-v4.log'))
        self.assertEqual(p['ctx'], 4096)
        self.assertEqual(p['threads'], 2)
        self.assertEqual((p['n_batch'], p['n_ubatch']), (2048, 512))
        self.assertEqual(p['load_mode'], 'mmap')
        self.assertGreater(p['repack_mib'], 0)
        self.assertEqual(p['top_k'], 20)  # Qwen3 GGUF 里的 general.sampling.top_k


@unittest.skipUnless(has('mis-c100k.log'), '先跑 exp17.sh misconfig')
class MisconfigTest(unittest.TestCase):
    def levels(self, name):
        text = log(name)
        return [h['level'] for h in dg.match_rules(text)] + [a for a, _, _ in dg.silent_checks(dg.parse(text))]

    def test_errors(self):
        self.assertIn('error', self.levels('mis-c100k.log'))
        self.assertIn('error', self.levels('mis-ctv-fa-off.log'))
        self.assertIn('error', self.levels('mis-nofile.log'))

    def test_warns(self):
        for n in ('mis-ngl.log', 'mis-smrow.log', 'mis-mask.log'):
            self.assertIn('warn', self.levels(n), n)

    def test_silent(self):
        for n in ('mis-t8.log', 'mis-b256.log', 'mis-np4.log', 'mis-ub16.log'):
            self.assertIn('silent', self.levels(n), n)

    def test_base_clean(self):
        self.assertEqual(self.levels('mis-base.log'), [])


@unittest.skipUnless(os.path.exists(os.path.join(MODELS, 'q-Q4_K_M.gguf')), '没有模型文件')
class PlanTest(unittest.TestCase):
    def test_default_kv(self):
        r = dg.plan(os.path.join(MODELS, 'q-Q4_K_M.gguf'), 0)
        self.assertEqual(r['n_ctx'], 40960)
        self.assertAlmostEqual(r['kv_mib'], 4480, delta=1)

    def test_c4096_q8(self):
        r = dg.plan(os.path.join(MODELS, 'q-Q4_K_M.gguf'), 4096, ctk='q8_0', ctv='q8_0')
        self.assertAlmostEqual(r['kv_mib'], 238, delta=1)  # 和日志里的 llama_kv_cache size 一致

    def test_np_note(self):
        r = dg.plan(os.path.join(MODELS, 'q-Q4_K_M.gguf'), 4096, np_=4)
        self.assertTrue(any('1024' in n for n in r['notes']))


if __name__ == '__main__':
    import warnings
    warnings.simplefilter('ignore', ResourceWarning)  # 02-ledger 的 machine() 读文件没关句柄，和这里无关
    unittest.main(verbosity=2, warnings='ignore')
