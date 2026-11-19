#!/usr/bin/env python3
"""08-kv-context 的回归测试：只用标准库，不需要模型文件。

    python3 test_kv.py
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import kv_ledger as kl  # noqa: E402

MiB = 2 ** 20
Q = kl.PRESETS['qwen3-0.6b']


class Ledger(unittest.TestCase):
    def test_qwen3_matches_log(self):
        # logs/kv_size.txt 里 llama_kv_cache: size = ... 的实测值
        for c, t, mib in ((4096, 'f16', 448), (4096, 'q8_0', 238), (4096, 'q4_0', 126),
                          (40960, 'f16', 4480), (40960, 'q8_0', 2380), (40960, 'q4_0', 1260)):
            self.assertEqual(kl.ledger(Q, c, t, t)['total'], mib * MiB, (c, t))

    def test_default_ctx_is_train(self):
        self.assertEqual(kl.ledger(Q)['n_ctx'], 40960)

    def test_pad_256(self):
        r = kl.ledger(Q, 1000)
        self.assertEqual(r['n_ctx'], 1024)
        self.assertEqual(r['total'], 1024 * 112 * 1024)

    def test_np_split(self):
        # 非 unified：每条序列 n_ctx/np 再取整到 256，总量不变
        r = kl.ledger(Q, 4096, n_seq=4)
        self.assertEqual((r['size_base'], r['n_stream']), (1024, 4))
        self.assertEqual(r['total'], 448 * MiB)
        r = kl.ledger(Q, 4096, n_seq=3)
        self.assertEqual(r['size_base'], 1536)  # 4096/3=1365 取整到 1536

    def test_mixed_types(self):
        r = kl.ledger(Q, 4096, 'q8_0', 'q4_0')
        self.assertEqual((r['k'], r['v']), (119 * MiB, 63 * MiB))

    def test_block_must_divide(self):
        m = dict(Q, head_k=80, head_v=80)  # 查的是单个头，80 不被 32 整除
        kl.ledger(m, 256, 'f16', 'f16')
        with self.assertRaises(ValueError):
            kl.ledger(m, 256, 'q8_0', 'q8_0')

    def test_swa_pattern(self):
        # gemma3 pattern 6：每 6 层第 6 层是全局
        flags = [kl.is_swa(i, 6) for i in range(12)]
        self.assertEqual(flags, [True] * 5 + [False] + [True] * 5 + [False])
        r = kl.ledger(kl.PRESETS['gemma3-27b'], 131072)
        self.assertEqual(r['n_swa_layers'], 52)
        self.assertEqual(r['size_swa'], 1536)  # 1024 + 512 取整到 256
        self.assertLess(r['total'], kl.ledger(kl.PRESETS['gemma3-27b'], 131072, swa_full=True)['total'] / 5)

    def test_mla_only_k(self):
        r = kl.ledger(kl.PRESETS['deepseek-v3'], 4096)
        self.assertEqual(r['v'], 0)
        self.assertEqual(r['per_token_full'], 61 * 576 * 2)


class Hybrid(unittest.TestCase):
    """双模型混合架构：只有 full attention 层存 KV，DeltaNet 存固定循环状态。"""

    def test_hybrid_presets_registered(self):
        self.assertEqual(set(kl.HYBRID_MODELS), {'qwen3.8-27b', 'qwen3.5-35b-a3b'})
        g27 = kl.hybrid_geometry('qwen3.8-27b')
        g35 = kl.hybrid_geometry('qwen3.5-35b-a3b')
        self.assertEqual((g27['kv_layers'], g27['rec_layers']), (16, 48))
        self.assertEqual((g35['kv_layers'], g35['rec_layers']), (10, 30))

    def test_27b_kv_and_recurrent(self):
        # f16 每 token 64 KiB；q8_0 34816 B；循环状态每序列 156893184 B
        self.assertEqual(kl.hl.kv_per_token(kl.hybrid_geometry('qwen3.8-27b'), 'f16'), 64 * 1024)
        self.assertEqual(kl.hl.kv_per_token(kl.hybrid_geometry('qwen3.8-27b'), 'q8_0'), 34816)
        r = kl.hybrid_ledger('qwen3.8-27b', 32768, 'q8_0')
        self.assertEqual(r['n_ctx'], 32768)
        self.assertEqual(r['kv_bytes'], 34816 * 32768)
        self.assertEqual(r['rec_per_seq'], 156893184)
        # 循环状态不随上下文变
        self.assertEqual(kl.hybrid_ledger('qwen3.8-27b', 8192)['rec_per_seq'],
                         kl.hybrid_ledger('qwen3.8-27b', 262144)['rec_per_seq'])

    def test_35b_kv_and_recurrent(self):
        # f16 每 token 20 KiB；q8_0 10880 B；循环状态每序列 65863680 B
        self.assertEqual(kl.hl.kv_per_token(kl.hybrid_geometry('qwen3.5-35b-a3b'), 'f16'), 20 * 1024)
        self.assertEqual(kl.hl.kv_per_token(kl.hybrid_geometry('qwen3.5-35b-a3b'), 'q8_0'), 10880)
        r = kl.hybrid_ledger('qwen3.5-35b-a3b', 32768, 'q8_0')
        self.assertEqual(r['kv_bytes'], 10880 * 32768)
        self.assertEqual(r['rec_per_seq'], 65863680)

    def test_hybrid_unified_pool_and_snapshot(self):
        g = kl.hybrid_geometry('qwen3.5-35b-a3b')
        # unified 下 4 槽 KV 共池，不翻倍
        one = kl.hybrid_ledger('qwen3.5-35b-a3b', 32768, 'q8_0', n_seq=1, unified=True)
        four = kl.hybrid_ledger('qwen3.5-35b-a3b', 32768, 'q8_0', n_seq=4, unified=True)
        self.assertEqual(one['kv_bytes'], four['kv_bytes'])
        # 循环态每槽一份；mtp 回滚快照按 (1+n_rs_seq) 倍增
        self.assertEqual(four['rec_bytes'], 4 * one['rec_per_seq'])
        snap = kl.hybrid_ledger('qwen3.5-35b-a3b', 32768, 'q8_0', n_seq=4, n_rs_seq=3)
        self.assertEqual(snap['rec_bytes'], kl.hl.rec_per_seq(g) * 16)

    def test_hybrid_non_unified_is_total_pool_not_double_pad(self):
        # 入参是整池总上下文。非 unified 平分 4 槽，整池字节不随槽数虚增，
        # 且只在共享账本里 pad 一次，不能先 pad 整池再平分（旧实现的双重 pad 会高估）。
        one = kl.hybrid_ledger('qwen3.5-35b-a3b', 32768, 'q8_0', n_seq=1, unified=False)
        four = kl.hybrid_ledger('qwen3.5-35b-a3b', 32768, 'q8_0', n_seq=4, unified=False)
        self.assertEqual((four['n_ctx'], four['ctx_per_seq']), (32768, 8192))
        self.assertEqual(four['kv_bytes'], one['kv_bytes'])
        # 不对齐整池：32769 先 pad 33024，平分再 pad 到每槽 8448、回写整池 33792
        odd = kl.hybrid_ledger('qwen3.5-35b-a3b', 32769, 'q8_0', n_seq=4, unified=False)
        self.assertEqual((odd['n_ctx'], odd['ctx_per_seq']), (33792, 8448))
        uni = kl.hybrid_ledger('qwen3.5-35b-a3b', 32769, 'q8_0', n_seq=4, unified=True)
        self.assertEqual(uni['n_ctx'], 33024)
        self.assertGreater(odd['kv_bytes'], uni['kv_bytes'])
        # 缺省 ctx=0 落到训练长度，且非 unified 下也能平分（旧实现 pad(0)=0 再平分会触发零槽报错）
        default = kl.hybrid_ledger('qwen3.5-35b-a3b', 0, 'f16', n_seq=4, unified=False)
        g = kl.hybrid_geometry('qwen3.5-35b-a3b')
        self.assertEqual(default['ctx_per_seq'], g['n_ctx_train'] // 4)


if __name__ == '__main__':
    unittest.main()
