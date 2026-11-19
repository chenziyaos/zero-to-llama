"""三个复刻（preset_tool、router_sim、metrics_parse）的回归用例。用法：python3 test_router.py"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preset_tool as pt
import router_sim as rs
import metrics_parse as mp


class PresetTest(unittest.TestCase):
    def test_canonical_tag(self):
        self.assertEqual(pt.canonical_tag('q8_0'), 'Q8_0')
        self.assertEqual(pt.canonical_tag('UD-q4_k_xl'), 'Q4_K_XL')
        self.assertEqual(pt.canonical_tag('Qwen3.q4_0'), 'Q4_0')

    def test_section_tag_normalized(self):
        ps, _, _ = pt.load_from_ini('[m:q4_k_m]\nctx-size = 1\n')
        self.assertIn('m:Q4_K_M', ps)

    def test_negated_bool_and_version(self):
        ps, _, _ = pt.load_from_ini('version = 1\n[m]\nno-jinja = true\n')
        self.assertEqual(ps['m'].get('LLAMA_ARG_JINJA'), 'false')
        self.assertIn('--no-jinja', ps['m'].to_args())

    def test_unknown_key(self):
        with self.assertRaises(ValueError):
            pt.load_from_ini('[m]\nfoo = 1\n')
        _, _, w = pt.load_from_ini('[m]\nfoo = 1\n', ignore_unknown=True)
        self.assertEqual(len(w), 1)

    def test_cascade_and_router_cli_wins(self):
        ini = '[*]\nctx-size = 2048\nparallel = 2\n[a]\nmodel = a.gguf\nctx-size = 8192\n[b]\nmodel = b.gguf\n'
        out = pt.build(ini)
        self.assertEqual(out['a'][0].get('LLAMA_ARG_CTX_SIZE'), '8192')
        self.assertEqual(out['b'][0].get('LLAMA_ARG_CTX_SIZE'), '2048')
        out = pt.build(ini, router_argv=['-c', '4096'])
        self.assertEqual(out['a'][0].get('LLAMA_ARG_CTX_SIZE'), '4096')

    def test_reserved_removed(self):
        out = pt.build('[a]\nmodel = a.gguf\n', router_argv=['--api-key', 'k', '--models-max', '2', '-m', 'x.gguf'])
        p = out['a'][0]
        self.assertIsNone(p.get('LLAMA_API_KEY'))
        self.assertIsNone(p.get('LLAMA_ARG_MODELS_MAX'))
        self.assertEqual(p.get('LLAMA_ARG_MODEL'), 'a.gguf')
        args = pt.child_args('a', p, 39001)
        self.assertEqual(args[args.index('--host') + 1], '127.0.0.1')
        self.assertEqual(args[args.index('--alias') + 1], 'a')

    def test_preset_only_keys_not_rendered(self):
        out = pt.build('[a]\nmodel = a.gguf\nload-on-startup = true\nstop-timeout = 3\n')
        p = out['a'][0]
        self.assertNotIn('load-on-startup', ' '.join(pt.child_args('a', p, 1)))
        self.assertEqual(pt.stop_timeout(p), 3)

    def test_stop_timeout_fallback(self):
        out = pt.build('[a]\nmodel = a.gguf\nstop-timeout = soon\n')
        self.assertEqual(pt.stop_timeout(out['a'][0]), 10)

    def test_models_dir(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, 'vl'))
            for f in ('vl/model.gguf', 'vl/mmproj-f16.gguf', 'vl/mtp-draft.gguf', 'solo.gguf', 'mmproj-x.gguf'):
                open(os.path.join(d, f), 'w').close()
            ps = pt.load_from_models_dir(d)
            self.assertEqual(sorted(ps), ['solo', 'vl'])
            self.assertTrue(ps['vl'].get('LLAMA_ARG_MMPROJ').endswith('mmproj-f16.gguf'))
            self.assertTrue(ps['vl'].get('LLAMA_ARG_SPEC_DRAFT_MODEL').endswith('mtp-draft.gguf'))


def models(dev='host', extra=None, fit=None):
    return rs.mk({}, {}, dev=dev, extra=extra, fit=fit)


class RouterTest(unittest.TestCase):
    def test_busy_not_evicted(self):
        r = rs.Router(models(), models_max=1).run([(0, 'A', 30.0), (5, 'B', 2.0)])
        a, s = r.lat[1]
        self.assertGreater(s - a, 25)

    def test_lru_victim(self):
        r = rs.Router(models(), models_max=2).run(rs.trace('ABAC'))
        self.assertEqual(r.resident(), ['A', 'C'])

    def test_models_max_zero_unlimited(self):
        r = rs.Router(models(), models_max=0).run(rs.trace('ABCABC'))
        self.assertEqual(rs.summary(r)['loads'], 3)

    def test_cpu_margin_ignored(self):
        r = rs.Router(models(), models_max=4, margin=10 ** 6).run(rs.trace('ABC'))
        self.assertEqual(r.resident(), ['A', 'B', 'C'])

    def test_gpu_margin_evicts(self):
        r = rs.Router(models('GPU0'), models_max=4, margin=2048, devices={'GPU0': 6144}, external={'GPU0': 512})
        r.dm['GPU0'] = [6144, 512]
        r.run(rs.trace('ABC', gap=20))
        self.assertEqual(r.resident(), ['C'])

    def test_loading_anyway(self):
        r = rs.Router(models('GPU0'), models_max=4, margin=1024, devices={'GPU0': 2048})
        r.dm['GPU0'] = [2048, 0]
        r.run([(0, 'C', 1.0)])
        self.assertTrue(any('照样加载' in x for x in r.log))
        self.assertEqual(r.resident(), ['C'])

    def test_unseen_mmproj_goes_negative(self):
        r = rs.Router(models('GPU0', extra={'q-Q8_0': 600}), models_max=4, margin=512,
                      devices={'GPU0': 3072})
        r.dm['GPU0'] = [3072, 0]
        r.run(rs.trace('AB', gap=20))
        self.assertLess(r.ledger('GPU0')['real_free'], 0)


class MetricsTest(unittest.TestCase):
    def test_roundtrip(self):
        txt = mp.render({'tokens_predicted_total': 12345678}, {'requests_processing': 2}, per_pos=[5, 3])
        v, t = mp.parse(txt)
        self.assertEqual(v['tokens_predicted_total'], 1.23457e7)   # 6 位有效数字
        self.assertEqual(t['tokens_predicted_total'], 'counter')
        self.assertEqual(v['spec_decode_num_accepted_tokens_per_pos_total']['{position="1"}'], 3)

    def test_first_token_free_and_reset(self):
        e = mp.Emu()
        e.request(0.0, 10, 0, 1.0, 11, 0.1)       # 10 步 1 秒
        sc = e.scrape_at([5.0, 10.0])
        a, _ = mp.parse(sc[0][1])
        b, _ = mp.parse(sc[1][1])
        self.assertAlmostEqual(a['predicted_tokens_seconds'], 10.0, places=3)
        self.assertEqual(b['predicted_tokens_seconds'], 0.0)
        self.assertEqual(b['tokens_predicted_total'], 11)

    def test_long_request_invisible_until_done(self):
        e = mp.Emu()
        e.request(0.0, 10, 0, 0.1, 1001, 0.02)    # 20 s 才结束
        sc = e.scrape_at([5.0, 25.0])
        a, _ = mp.parse(sc[0][1])
        self.assertEqual(a['tokens_predicted_total'], 0)

    def test_derive(self):
        a = {'tokens_predicted_total': 0, 'tokens_predicted_seconds_total': 0,
             'prompt_tokens_total': 0, 'prompt_tokens_cached_total': 0}
        b = {'tokens_predicted_total': 600, 'tokens_predicted_seconds_total': 30,
             'prompt_tokens_total': 100, 'prompt_tokens_cached_total': 300}
        dv = mp.derive(a, b, 10.0)
        self.assertEqual(dv['gen_tps_wall'], 60)
        self.assertEqual(dv['gen_tps_per_req'], 20)
        self.assertEqual(dv['gen_busy'], 3)
        self.assertEqual(dv['cache_hit'], 0.75)


if __name__ == '__main__':
    unittest.main(verbosity=2)
