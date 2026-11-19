"""测 parse_load_log 和 estimate_memory：python3 test_first_run.py

夹具 sample_load.log 是基线 0ec4027 上 Qwen3-0.6B Q4_K_M、-c 4096 的实机日志。
预测器的期望值来自同一台机器上 9 组配置的日志实测（README 的对账表）。
没有 GGUF 文件时，预测器用一个按 Qwen3-0.6B Q4_K_M 张量目录构造的假 GGUF 对象来测。
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'common'))

from parse_load_log import parse  # noqa: E402
from estimate_memory import estimate  # noqa: E402
from gguf_lite import Tensor, TYPE_ID  # noqa: E402

MiB = 2 ** 20


class FakeQwen3Q4KM:
    """Qwen3-0.6B Q4_K_M 的张量目录：每层 11 个张量，attn_v 和 ffn_down 是 Q6_K。"""
    arch = 'qwen3'

    def __init__(self):
        E, F, L, V, H, KV, D = 1024, 3072, 28, 151936, 16, 8, 128
        self.kv = {'qwen3.block_count': L, 'qwen3.attention.head_count': H,
                   'qwen3.attention.head_count_kv': KV, 'qwen3.attention.key_length': D,
                   'qwen3.attention.value_length': D, 'qwen3.embedding_length': E,
                   'qwen3.context_length': 40960, 'tokenizer.ggml.tokens': ('<array>', 8, V)}
        T = []
        add = lambda n, s, t: T.append(Tensor(n, s, TYPE_ID[t], 0))
        add('token_embd.weight', [E, V], 'Q4_K')
        add('output.weight', [E, V], 'Q6_K')
        add('output_norm.weight', [E], 'F32')
        # Q4_K_M 把部分层的 attn_v、ffn_down 升到 Q6_K，规则抄自 src/llama-quant.cpp:490 use_more_bits：
        # 前 1/8、后 1/8，中间每 3 层一层。28 层时是 0,1,2,5,8,...,23,24,25,26,27，共 14 层
        q6 = {i for i in range(L) if i < L // 8 or i >= 7 * L // 8 or (i - L // 8) % 3 == 2}
        q6_v = q6_down = q6
        for i in range(L):
            p = f'blk.{i}.'
            add(p + 'attn_norm.weight', [E], 'F32')
            add(p + 'ffn_norm.weight', [E], 'F32')
            add(p + 'attn_q_norm.weight', [D], 'F32')
            add(p + 'attn_k_norm.weight', [D], 'F32')
            add(p + 'attn_q.weight', [E, H * D], 'Q4_K')
            add(p + 'attn_k.weight', [E, KV * D], 'Q4_K')
            add(p + 'attn_v.weight', [E, KV * D], 'Q6_K' if i in q6_v else 'Q4_K')
            add(p + 'attn_output.weight', [H * D, E], 'Q4_K')
            add(p + 'ffn_gate.weight', [E, F], 'Q4_K')
            add(p + 'ffn_up.weight', [E, F], 'Q4_K')
            add(p + 'ffn_down.weight', [F, E], 'Q6_K' if i in q6_down else 'Q4_K')
        self.tensors = T

    def hp(self, key, default=None):
        return self.kv.get(f'qwen3.{key}', default)

    @property
    def tensor_bytes(self):
        return sum(t.nbytes for t in self.tensors)


def model():
    p = os.environ.get('GGUF')
    if p:
        from gguf_lite import GGUF
        return GGUF(p)
    return FakeQwen3Q4KM()


class TestParse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = parse(open(os.path.join(HERE, 'sample_load.log'), encoding='utf-8').read())

    def test_model_meta(self):
        r = self.r
        self.assertEqual(r['arch'], 'qwen3')
        self.assertEqual((r['n_layer'], r['n_head'], r['n_head_kv'], r['head_dim']), ('28', '16', '8', '128'))
        self.assertEqual(r['file_size'], ['456.11', '5.09'])
        self.assertEqual(r['n_kv'], ['40', '311'])

    def test_last_load_wins(self):
        # fit 的空载那次 load_mode = none、buffer 全 0，必须跳过
        self.assertEqual(self.r['load_mode'], 'mmap')
        self.assertEqual(self.r['weight_buffers'], {'CPU_Mapped': 251.1, 'CPU_REPACK': 204.75})

    def test_context(self):
        r = self.r
        self.assertEqual((r['n_ctx'], r['n_ubatch']), ('4096', '512'))
        self.assertEqual(r['fa_resolved'], 'enabled')
        self.assertEqual(r['kv'][:3], ['448.00', '4096', '28'])
        self.assertEqual(r['compute'], ['CPU', '312.77'])

    def test_breakdown(self):
        b = self.r['breakdown']
        self.assertEqual(b['Host'], dict(self=1012, model=251, context=448, compute=312))
        self.assertEqual(b['CPU_REPACK']['self'], 204)

    def test_fit(self):
        self.assertEqual(self.r['fit_used'][0], '1012')
        self.assertEqual(self.r['fit_leave'][1], '1024')


class TestEstimate(unittest.TestCase):
    # (ctx, ubatch, ctk, ctv, fa) -> 日志里 Host 一行的 self（MiB）
    CASES = [
        ((4096, 512, 'f16', 'f16', 'on'), 1012),
        ((4096, 128, 'f16', 'f16', 'on'), 777),
        ((16384, 128, 'f16', 'f16', 'on'), 2124),
        ((16384, 512, 'f16', 'f16', 'on'), 2366),
        ((4096, 1024, 'f16', 'f16', 'on'), 1324),
        ((16384, 1024, 'f16', 'f16', 'on'), 2688),
        ((4096, 512, 'q8_0', 'q8_0', 'on'), 804),
        ((4096, 512, 'q4_0', 'q4_0', 'on'), 692),
        ((4096, 512, 'f16', 'f16', 'off'), 1018),
    ]

    def test_host_total(self):
        g = model()
        for args, want in self.CASES:
            e = estimate(g, *args, 'avx512')
            with self.subTest(args=args):
                self.assertLessEqual(abs(e['host'] / MiB - want), 3)

    def test_parts(self):
        e = estimate(model(), 4096, 512, 'f16', 'f16', 'on', 'avx512')
        self.assertEqual(e['k'] / MiB, 224)
        self.assertAlmostEqual(e['compute'] / MiB, 312.77, places=2)
        self.assertAlmostEqual(e['repack'] / MiB, 204.75, places=2)

    def test_ctx_padding_and_zero(self):
        g = model()
        self.assertEqual(estimate(g, 4000, 512, 'f16', 'f16', 'on', 'avx512')['n_ctx'], 4096)
        e = estimate(g, 0, 512, 'f16', 'f16', 'on', 'avx512')
        self.assertEqual(e['n_ctx'], 40960)
        self.assertEqual((e['k'] + e['v']) / MiB, 4480)

    def test_no_repack_without_avx2(self):
        self.assertEqual(estimate(model(), 4096, 512, 'f16', 'f16', 'on', 'none')['repack'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=1)
