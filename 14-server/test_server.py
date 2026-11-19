"""slot_sim 的回归用例：每条都对着 0ec4027 的源码行为写。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slot_sim as S  # noqa: E402


class T(unittest.TestCase):

    def test_pad_unified(self):
        assert S.ctx_layout(4096, 4, True) == (4096, 4096)
        assert S.ctx_layout(4000, 1, True) == (4096, 4096)

    def test_pad_split(self):
        # 10000 先补到 10240，除以 3 得 3413，再补到 3584，总量改写成 3 × 3584
        assert S.ctx_layout(10000, 3, False) == (10752, 3584)
        assert S.ctx_layout(16384, 8, False) == (16384, 2048)
        # 和 batched-bench -v 的日志对账：-c 4096 -npl 3 实际分到 4608 格，每条 1536
        assert S.ctx_layout(4096, 3, False) == (4608, 1536)
        assert S.ctx_layout(10000, 4, False) == (10240, 2560)

    def test_kv_bytes(self):
        assert S.KV_BYTES_PER_TOKEN == 112 * 1024
        assert S.kv_mib(4096) == 448.0

    def test_lcp(self):
        assert S.lcp((1, 2, 3), (1, 2, 4)) == 2
        assert S.lcp((), (1,)) == 0

    def test_cache_skip_contained(self):
        c = S.PromptCache(0, 0)
        assert c.alloc((1, 2, 3, 4))
        assert not c.alloc((1, 2))            # 已经整段被缓存包含

    def test_cache_drop_obsolete(self):
        c = S.PromptCache(0, 0)
        c.alloc((1, 2))
        c.alloc((1, 2, 3))                    # 旧条目是新条目的前缀，删掉
        assert c.states == [(1, 2, 3)]

    def test_cache_over_limit_skipped(self):
        c = S.PromptCache(1, 0, bpt=S.MIB)    # 1 MiB 上限，每 token 1 MiB
        assert not c.alloc((1, 2))

    def test_cache_evict_oldest(self):
        c = S.PromptCache(3, 0, bpt=S.MIB)
        c.alloc((1,))
        c.alloc((2,))
        c.alloc((3,))
        c.alloc((4,))
        assert c.states == [(2,), (3,), (4,)]
        assert c.evicted == 1

    def test_load_keep_threshold(self):
        c = S.PromptCache(0, 0)
        c.alloc(tuple(range(100)))
        # 新提示只和缓存条目共享 10 个 token，f_keep = 0.1 ＜ 0.25，不换
        assert c.load((), tuple(range(10)) + (999,)) == ()
        # 共享 60 个，f_keep 0.6，换进来，缓存里同时删掉
        got = c.load((), tuple(range(60)) + (999,))
        assert got == tuple(range(100))
        assert c.states == []

    def test_load_needs_both_better(self):
        c = S.PromptCache(0, 0)
        c.alloc((1, 2, 3, 4, 5, 6, 7, 8))
        cur = (1, 2, 3, 9)                    # 槽位里已有：f_keep 0.75，f_sim 0.5
        new = (1, 2, 3, 4, 0, 0)
        # 缓存条目 lcp 4：f_keep 0.5 不比 0.75 好，不换
        assert c.load(cur, new) == cur

    def test_slot_prefers_empty_then_lru(self):
        slots = [S.Slot(i) for i in range(3)]
        slots[0].tokens, slots[0].t_last = (5, 6), 2
        slots[1].tokens, slots[1].t_last = (7, 8), 1
        s, how = S.get_available_slot(slots, (1, 2), 0.1, None)
        assert how == 'lru' and s.id == 2     # 空槽优先
        slots[2].tokens, slots[2].t_last = (9,), 3
        s, how = S.get_available_slot(slots, (1, 2), 0.1, None)
        assert s.id == 1                      # 都不空时选最久没用的

    def test_slot_lcp_threshold(self):
        slots = [S.Slot(0), S.Slot(1)]
        slots[0].tokens = tuple(range(10))
        slots[1].tokens = tuple(range(3)) + (99,)
        task = tuple(range(10)) + (1000,)
        s, how = S.get_available_slot(slots, task, 0.1, None)
        assert how == 'lcp' and s.id == 0
        # 阈值抬到 0.95，10/11 不够，退回 LRU
        s, how = S.get_available_slot(slots, task, 0.95, None)
        assert how == 'lru'

    def test_slot_skips_busy(self):
        slots = [S.Slot(0), S.Slot(1)]
        slots[0].tokens, slots[0].busy = (1, 2, 3), True
        s, how = S.get_available_slot(slots, (1, 2, 3, 4), 0.1, None)
        assert s.id == 1 and how == 'lru'

    def test_low_keep_saves_to_cache(self):
        c = S.PromptCache(0, 0)
        slots = [S.Slot(0)]
        slots[0].tokens = tuple(range(100))
        task = tuple(range(20)) + (5000,)
        # f_sim = 20/21 选中，f_keep = 0.2 ＜ 0.5，旧提示先存进缓存
        s, how = S.get_available_slot(slots, task, 0.1, c)
        assert how == 'lcp'
        assert c.states == [tuple(range(100))]

    def test_sim_cache_helps(self):
        a = S.simulate(4, 16, 6, 300, 0.1, 0, 16384)
        b = S.simulate(4, 16, 6, 300, 0.1, 8192, 16384)
        assert b['recomputed'] < a['recomputed']
        assert a['prompt_tokens'] == b['prompt_tokens']

    def test_sim_few_users_no_gain(self):
        # 用户数不超过槽位数时，每个用户一直占着自己的槽，缓存用不上
        a = S.simulate(4, 4, 6, 300, 0.1, 0, 16384)
        b = S.simulate(4, 4, 6, 300, 0.1, 8192, 16384)
        assert a['recomputed'] == b['recomputed']


if __name__ == '__main__':
    unittest.main(verbosity=1)
