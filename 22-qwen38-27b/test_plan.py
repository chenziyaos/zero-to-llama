#!/usr/bin/env python3
"""test_plan.py：22-qwen38-27b 推算的回归测试，只用标准库。运行 python3 test_plan.py。"""
import plan

GB = 1e9
GiB = 2 ** 30
MiB = 2 ** 20


def close(a, b, rel=1e-9):
    return abs(a - b) <= rel * max(abs(a), abs(b), 1)


def test_geometry():
    # full-attn 只有 16 层：层号 3,7,...,63；四分之三（48 层）是循环层
    layers = [i for i in range(64) if (i + 1) % 4 == 0]
    assert len(layers) == 16 and layers[0] == 3 and layers[-1] == 63
    assert plan.M['rec_layers'] == 48
    # d_inner = 128 value head dim x 48 value heads
    assert plan.d_inner() == 6144


def test_kv_per_token():
    # f16：16 层 x 2 x 4 头 x 256 x 2 字节 = 65536 B = 64 KiB
    assert close(plan.kv_per_token('f16'), 16 * 2 * 4 * 256 * 2)
    assert plan.kv_per_token('f16') == 64 * KiB
    # q8_0：每块 32 int8 加 1 个 f16 scale（34 字节）。head_dim 256 = 8 块，
    # 每 K/V 头 256 x 34/32 字节，16 层 x 2(K,V) x 4 头
    assert close(plan.kv_per_token('q8_0'), 16 * 2 * 4 * 256 * 34 / 32)
    assert close(plan.kv_per_token('q8_0'), 34816)
    assert close(plan.kv_per_token('q8_0'), 34 * KiB, 1e-3)
    # 32k / 128k / 256k 的 f16 KV
    assert close(plan.kv_per_token('f16') * 32768, 2.0 * GiB, 1e-3)
    assert close(plan.kv_per_token('f16') * 131072, 8.0 * GiB, 1e-3)
    assert close(plan.kv_per_token('f16') * 262144, 16.0 * GiB, 1e-3)


def test_recurrent_state_constant():
    # 每层每序列 F32：(786432 s + 30720 r) x 4 = 3,268,608 B
    per_layer = (786432 + 30720) * 4
    assert close(plan.rec_per_seq(), 48 * per_layer)
    assert close(plan.rec_per_seq(), 149.63 * MiB, 2e-3)
    # 循环态不随上下文变：这里没有 ctx 参数，显式验证它是常数
    assert plan.rec_total(1, 0) == plan.rec_per_seq()
    # 4 槽不开回滚：x4；draft-mtp n=3 时每槽 (1+3) 倍
    assert plan.rec_total(4, 0) == 4 * plan.rec_per_seq()
    assert plan.rec_total(4, 3) == 16 * plan.rec_per_seq()


KiB = 1024


def test_weights_published():
    # 字节以共享 manifest 为唯一来源，本篇只做精确字节换算后的 GB 展示，不再手抄两位小数
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
    import hybrid_ledger as hl
    man = hl.load_manifest()
    gb = lambda b: b / GB  # noqa: E731
    assert plan.BARTOWSKI['Q4_K_M'] == gb(hl.weight_bytes(man, 'qwen3.8-27b', 'bartowski', 'Q4_K_M'))
    assert plan.BARTOWSKI['Q8_0'] == gb(hl.weight_bytes(man, 'qwen3.8-27b', 'bartowski', 'Q8_0'))
    assert plan.LMSTUDIO['Q4_K_M'] == gb(hl.weight_bytes(man, 'qwen3.8-27b', 'lmstudio-community', 'Q4_K_M'))
    assert plan.LMSTUDIO['Q8_0'] == gb(hl.weight_bytes(man, 'qwen3.8-27b', 'lmstudio-community', 'Q8_0'))
    # bartowski mmproj bf16 精确字节 931145952
    assert close(plan.MMPROJ_GB, 0.931145952)
    # kingy 实测钉的是 unsloth 旧修订常规 Q4_K_M 的精确字节，故意独立于 manifest 保留
    assert plan.KINGY['file_gb'] == 17.106775008


def test_plan_delegates_to_shared_ledger():
    # 本篇几何和账不再本地硬编码：与共享账本逐项一致，manifest 改了这里自动跟随
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
    import hybrid_ledger as hl
    man = hl.load_manifest()
    g = hl.geometry(man, 'qwen3.8-27b')
    assert plan.M['kv_layers'] == g['kv_layers'] == 16
    assert plan.M['rec_layers'] == g['rec_layers'] == 48
    assert plan.d_inner() == hl.d_inner(g)
    # 同参数下本篇 stack 与共享 stack 总额一致（bartowski 内嵌 MTP，不另加 sidecar）
    _, t_plan = plan.stack('Q4_K_M', 32768, 'q8_0', 1, 0, True)
    _, t_hl, _ = hl.stack(man, 'qwen3.8-27b', 'bartowski', 'Q4_K_M', 32768, 'q8_0',
                          n_seq=1, mtp=0, mmproj=True)
    assert close(t_plan, t_hl)
    assert plan.max_ctx('Q4_K_M', 'q8_0', 1, 0, True, card_gib=24) == \
        hl.max_ctx(man, 'qwen3.8-27b', 'bartowski', 'Q4_K_M', 'q8_0', 1, 0, True, card_gib=24)


def test_fit_24g():
    # bartowski Q4_K_M 17.44 GB 折算 GiB 后，24 GiB 挂 mmproj + 32k q8 KV 必须放得下
    rows, total = plan.stack('Q4_K_M', 32768, 'q8_0', 4, 0, True)
    assert total <= plan.budget(24)
    # 不开 mmproj 的最长上下文必须比开 mmproj 长
    c_mm = plan.max_ctx('Q4_K_M', 'q8_0', 4, 0, True, card_gib=24)
    c_no = plan.max_ctx('Q4_K_M', 'q8_0', 4, 0, False, card_gib=24)
    assert c_no > c_mm > 0
    # 开 draft-mtp（循环态 x4）后最长上下文明显变短
    c_mtp = plan.max_ctx('Q4_K_M', 'q8_0', 4, 3, True, card_gib=24)
    assert c_mtp < c_mm
    # 16 GiB 卡 bartowski Q4_K_M 放不下
    assert plan.max_ctx('Q4_K_M', 'q8_0', 4, 0, True, card_gib=16) == 0
    # 32 GiB 卡 Q6_K 也能给一段非零上下文
    assert plan.max_ctx('Q6_K', 'q8_0', 4, 0, True, card_gib=32) > 0
    # 最长上下文不超过训练长度
    assert plan.max_ctx('IQ2_M', 'q8_0', 4, 0, True, card_gib=48) <= plan.M['n_ctx_train']


def test_lmstudio_text_lane():
    # 精简三档里 Q4_K_M 不挂 mmproj 必须在 24 GiB 放下，Q8_0 放不下
    assert plan.max_ctx('Q4_K_M', 'q8_0', 1, 0, False, 'lmstudio', card_gib=24) > 0
    assert plan.max_ctx('Q8_0', 'q8_0', 1, 0, False, 'lmstudio', card_gib=24) == 0
    # bartowski 内嵌 MTP：开 mtp3 不增加权重，只增加循环态
    base = plan.stack('Q4_K_M', 0, 'q8_0', 1, 0, False)[1]
    mtp = plan.stack('Q4_K_M', 0, 'q8_0', 1, 3, False)[1]
    assert close(mtp - base, 3 * plan.rec_per_seq())


def test_tg_bound_monotone():
    b0 = plan.tg_bound(plan.BARTOWSKI['Q4_K_M'], 'RTX 4090', 0)
    b32 = plan.tg_bound(plan.BARTOWSKI['Q4_K_M'], 'RTX 4090', 32768)
    assert b0 > b32 > 0
    # kingy 实测 49.1 必须落在两个效率系数夹出的区间里
    lo = plan.tg_bound(plan.KINGY['file_gb'], 'RTX 4090') * plan.eff()['RTX 4090']
    hi = plan.tg_bound(plan.KINGY['file_gb'], 'RTX 4090') * plan.eff()['kingy']
    assert lo < plan.KINGY['tg'] <= hi


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for fn in fns:
        fn()
        print('ok', fn.__name__)
    print('%d passed' % len(fns))
