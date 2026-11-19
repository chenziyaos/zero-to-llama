#!/usr/bin/env python3
"""test_hybrid_ledger.py：双模型混合架构账本的回归测试，只用标准库。

固定四类已核实常量，防止改公式时静默漂移：
- 27B full-attn KV 每 token、DeltaNet 循环状态每序列；
- 35B-A3B full-attn KV 每 token、DeltaNet 循环状态每序列；
- manifest 里的官方 revision、架构名和制品字节；
- 账本堆叠、预算和最长上下文的基本性质。

运行：python3 test_hybrid_ledger.py
"""
import json
import os

import hybrid_ledger as hl

HERE = os.path.dirname(os.path.abspath(__file__))
MAN = hl.load_manifest(os.path.join(HERE, 'dual_models.json'))
G27 = hl.geometry(MAN, 'qwen3.8-27b')
G35 = hl.geometry(MAN, 'qwen3.5-35b-a3b')
GB = 1e9
GiB = 2 ** 30
MiB = 2 ** 20
KiB = 1024


def close(a, b, rel=1e-9):
    return abs(a - b) <= rel * max(abs(a), abs(b), 1)


def test_manifest_identity():
    assert MAN['source_baseline']['llama_cpp_commit'].startswith('0ec4027')
    d27, d35 = MAN['models']['qwen3.8-27b'], MAN['models']['qwen3.5-35b-a3b']
    assert d27['official_revision'] == '37561816e66b5b9153b1b64c10968c18c1e6f4a0'
    assert d35['official_revision'] == '59d61f3ce65a6d9863b86d2e96597125219dc754'
    assert d27['llama_cpp_arch'] == 'qwen35' and d27['ffn'] == 'dense'
    assert d35['llama_cpp_arch'] == 'qwen35moe' and d35['ffn'] == 'moe'
    # 两个模型都还没有真实 GPU 实测
    for d in (d27, d35):
        assert d['measurement_status'] == 'pending_gpu_measurement'


def test_geometry_layer_split():
    # 27B：64 层里 16 层 full attention（层号 3,7,...,63），48 层 DeltaNet
    assert G27['n_layer'] == 64 and G27['kv_layers'] == 16 and G27['rec_layers'] == 48
    assert G27['n_kv'] == 4 and G27['head_dim'] == 256
    # 35B：40 层里 10 层 full attention（层号 3,7,...,39），30 层 DeltaNet
    assert G35['n_layer'] == 40 and G35['kv_layers'] == 10 and G35['rec_layers'] == 30
    assert G35['n_kv'] == 2 and G35['head_dim'] == 256
    # MoE 几何带专家信息，稠密不带
    assert G35['moe']['n_routed_experts'] == 256 and G35['moe']['n_routed_per_tok'] == 8
    assert G27['moe'] is None


def test_kv_per_token_27b():
    # f16：16 层 x 2(K,V) x 4 头 x 256 x 2 字节 = 65536 B = 64 KiB
    assert hl.kv_per_token(G27, 'f16') == 64 * KiB == 65536
    # q8_0：每块 32 int8 + 1 个 f16 scale = 34 字节
    assert close(hl.kv_per_token(G27, 'q8_0'), 16 * 2 * 4 * 256 * 34 / 32)
    assert close(hl.kv_per_token(G27, 'q8_0'), 34816)
    assert close(hl.kv_per_token(G27, 'q4_0'), 18432)
    # 32k / 256k 满长 f16 KV
    assert close(hl.kv_per_token(G27, 'f16') * 32768, 2.0 * GiB, 1e-3)
    assert close(hl.kv_per_token(G27, 'f16') * 262144, 16.0 * GiB, 1e-3)


def test_kv_per_token_35b():
    # f16：10 层 x 2 x 2 头 x 256 x 2 字节 = 20480 B = 20 KiB
    assert hl.kv_per_token(G35, 'f16') == 20 * KiB == 20480
    # q8_0：10 x 2 x 2 x 256 x 34/32 = 10880；q4_0 = 5760
    assert close(hl.kv_per_token(G35, 'q8_0'), 10880)
    assert close(hl.kv_per_token(G35, 'q4_0'), 5760)
    # 32k q8_0 KV = 340 MiB
    assert close(hl.kv_per_token(G35, 'q8_0') * 32768, 340 * MiB, 1e-3)
    # 同上下文下 35B 的 KV 明显小于 27B（层数和 KV 头都更少）
    assert hl.kv_per_token(G35, 'q8_0') < hl.kv_per_token(G27, 'q8_0')


def test_d_inner_and_recurrent_27b():
    # d_inner = 128 value head dim x 48 value heads = 6144
    assert hl.d_inner(G27) == 6144
    # 每层 (s=786432 + r=30720) x 4，48 层
    per_layer = (786432 + 30720) * 4
    assert close(hl.rec_per_seq(G27), 48 * per_layer)
    assert close(hl.rec_per_seq(G27), 156893184)
    assert close(hl.rec_per_seq(G27) / MiB, 149.63, 2e-3)


def test_d_inner_and_recurrent_35b():
    # d_inner = 128 x 32 value heads = 4096
    assert hl.d_inner(G35) == 4096
    # s = 128 x 4096 = 524288；r = 3 x (4096 + 2 x 16 x 128) = 24576
    s = 128 * 4096
    r = 3 * (4096 + 2 * 16 * 128)
    assert s == 524288 and r == 24576
    assert close(hl.rec_per_seq(G35), 30 * (s + r) * 4)
    assert close(hl.rec_per_seq(G35), 65863680)
    assert close(hl.rec_per_seq(G35) / MiB, 62.8125, 1e-3)


def test_recurrent_is_constant_and_snapshot():
    # 循环状态不随上下文变：函数签名里没有 ctx，这里显式断言同值
    base27 = hl.rec_per_seq(G27)
    assert hl.rec_total(G27, 1, 0) == base27
    # 4 槽不开回滚 x4；draft-mtp n=3 时每槽 (1+3) 倍 -> 16 倍
    assert hl.rec_total(G27, 4, 0) == 4 * base27
    assert hl.rec_total(G27, 4, 3) == 16 * base27
    base35 = hl.rec_per_seq(G35)
    assert hl.rec_total(G35, 4, 0) == 4 * base35
    assert close(hl.rec_total(G35, 4, 0) / MiB, 251.25, 1e-3)


def test_unified_kv_pool():
    # 入参 ctx 语义统一为“整池总上下文”。unified 下 4 槽共用一个池，字节不随槽数翻倍
    single = hl.kv_bytes(G35, 32768, 'q8_0', n_seq=1, unified=True)
    four = hl.kv_bytes(G35, 32768, 'q8_0', n_seq=4, unified=True)
    assert single == four
    # 非 unified：总池 32768 平分到 4 个 stream，每 stream 8192，总和仍覆盖整池 32768，
    # 所以整池字节等于 unified 单池（而不是再 x4）。x4 只发生在“每槽都要整池长度”的旧错误语义里。
    separate = hl.kv_bytes(G35, 32768, 'q8_0', n_seq=4, unified=False)
    per_tok = hl.kv_per_token(G35, 'q8_0')
    assert separate == per_tok * 8192 * 4 == single
    # 同样覆盖 131072 的整池：非 unified 平分 4 槽（每槽 32768，无额外对齐浪费），总和等于 unified
    big = hl.kv_bytes(G35, 131072, 'q8_0', n_seq=4, unified=False)
    uni_big = hl.kv_bytes(G35, 131072, 'q8_0', n_seq=4, unified=True)
    assert big == uni_big
    # 非 unified 因每槽独立 pad，遇到不对齐的总池会有浪费：32769 被 pad 到 33024，
    # 平分后每槽再 pad 到 8448、回写整池 33792，比 unified 的 33024 多占
    nu = hl.kv_bytes(G35, 32769, 'q8_0', n_seq=4, unified=False)
    un = hl.kv_bytes(G35, 32769, 'q8_0', n_seq=4, unified=True)
    assert nu > un


def test_weight_and_sidecar_bytes():
    # 精确字节回读，和 HF tree API 登记值一致
    assert hl.weight_bytes(MAN, 'qwen3.8-27b', 'bartowski', 'Q4_K_M') == 17442399968
    assert hl.weight_bytes(MAN, 'qwen3.5-35b-a3b', 'unsloth', 'Q4_K_M') == 22016023168
    assert hl.sidecar_bytes(MAN, 'qwen3.8-27b', 'bartowski', 'mmproj_bf16') == 931145952
    assert hl.sidecar_bytes(MAN, 'qwen3.5-35b-a3b', 'unsloth', 'mmproj_bf16') == 902822592
    # 27B unsloth 有独立 MTP sidecar，35B unsloth 标准仓库没有
    assert hl.sidecar_bytes(MAN, 'qwen3.8-27b', 'unsloth', 'mtp_q4_0') == 1369590656
    pub35 = MAN['models']['qwen3.5-35b-a3b']['gguf']['unsloth']['sidecars']
    assert 'mtp_q4_0' not in pub35


def test_manifest_all_file_bytes_positive_and_unique_publishers():
    for key, d in MAN['models'].items():
        for pub, rec in d['gguf'].items():
            for quant, byt in rec['files'].items():
                assert isinstance(byt, int) and byt > 0, (key, pub, quant)
            for nm, byt in rec['sidecars'].items():
                assert isinstance(byt, int) and byt > 0, (key, pub, nm)


def test_stack_and_budget_24g_27b():
    # bartowski Q4_K_M + 32k q8 KV + 单槽 + mmproj 必须在 24 GiB 内
    rows, total, notes = hl.stack(MAN, 'qwen3.8-27b', 'bartowski', 'Q4_K_M', 32768,
                                  'q8_0', n_seq=1, mtp=0, mmproj=True)
    assert total <= hl.budget(24)
    # 单槽 unified：整池 = 每槽，32k 已 256 对齐
    assert notes['ctx_padded_total'] == 32768 and notes['ctx_per_seq'] == 32768
    # 开 mtp 回滚快照只增加循环态，权重/KV 不变 -> 总额变大
    _, total_mtp, _ = hl.stack(MAN, 'qwen3.8-27b', 'bartowski', 'Q4_K_M', 32768,
                               'q8_0', n_seq=1, mtp=3, mmproj=True)
    assert total_mtp > total
    # 挂 mmproj 比不挂可用上下文短
    c_mm = hl.max_ctx(MAN, 'qwen3.8-27b', 'bartowski', 'Q4_K_M', 'q8_0', 1, 0, True, card_gib=24)
    c_no = hl.max_ctx(MAN, 'qwen3.8-27b', 'bartowski', 'Q4_K_M', 'q8_0', 1, 0, False, card_gib=24)
    assert c_no > c_mm > 0
    # 上下文按 256 对齐且不超训练长度
    assert c_mm % 256 == 0 and c_no <= G27['n_ctx_train']


def test_ctx_padding_and_split_matches_llama_cpp():
    # 照 llama-context.cpp：先 pad 总 n_ctx，再决定每槽
    assert hl.pad_ctx(32768) == 32768
    assert hl.pad_ctx(32769) == 33024
    assert hl.pad_ctx(0) == 0
    # unified：n_ctx_seq 就是 pad 后的总长度
    total, per = hl.split_ctx(32769, n_seq=4, unified=True)
    assert total == 33024 and per == 33024
    # 非 unified：n_ctx_seq = pad(pad(总)/n_seq)，总长度回写为 per*n_seq
    total, per = hl.split_ctx(32768, n_seq=4, unified=False)
    assert per == 8192 and total == 32768
    total, per = hl.split_ctx(32769, n_seq=4, unified=False)
    # pad(32769)=33024，/4=8256，再 pad 到 256 = 8448，回写总 33792
    assert per == 8448 and total == 33792
    # 每槽为 0 直接报错，而不是静默给一份空账：总 ctx=0 时 pad 后仍为 0，平分后每槽 0
    try:
        hl.split_ctx(0, n_seq=4, unified=False)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_non_unified_kv_scales_with_streams():
    per_tok = hl.kv_per_token(G35, 'q8_0')
    # 入参 ctx 是整池总上下文。非 unified 下总池平分到 n_seq 个 stream，
    # 各 stream 只存自己那份（32768/4=8192），整池字节不随槽数虚增：
    single = hl.kv_bytes(G35, 32768, 'q8_0', n_seq=1, unified=False)
    four = hl.kv_bytes(G35, 32768, 'q8_0', n_seq=4, unified=False)
    assert single == per_tok * 32768
    assert four == per_tok * 8192 * 4 == single
    # 整池请求放大到 4 倍（131072），非 unified 平分 4 槽后每槽 32768，整池也放大 4 倍
    per_seq_full = hl.kv_bytes(G35, 131072, 'q8_0', n_seq=4, unified=False)
    assert per_seq_full == 4 * single
    # unified 共池不随槽数翻倍
    u1 = hl.kv_bytes(G35, 32768, 'q8_0', n_seq=1, unified=True)
    u4 = hl.kv_bytes(G35, 32768, 'q8_0', n_seq=4, unified=True)
    assert u1 == u4
    # 同整池请求下非 unified 因平分后再对齐，KV 不少于 unified
    assert four >= u4


def test_max_ctx_inverse_of_stack_non_unified():
    # max_ctx 给出的 ctx 放回 stack，不能越过预算（互逆性，非 unified 也要成立）
    for unified in (True, False):
        c = hl.max_ctx(MAN, 'qwen3.5-35b-a3b', 'bartowski', 'Q4_K_M', 'q8_0',
                       n_seq=4, mtp=0, mmproj=True, card_gib=24, unified=unified)
        if c == 0:
            continue
        _, total, _ = hl.stack(MAN, 'qwen3.5-35b-a3b', 'bartowski', 'Q4_K_M', c,
                               'q8_0', n_seq=4, mtp=0, mmproj=True, unified=unified)
        assert total <= hl.budget(24)
        # 再多一个对齐档就该超预算（确认不是过度保守）。
        # unified 的对齐档作用在整池（+256）；非 unified 的对齐档作用在每槽，整池步长 = 256*n_seq。
        step = 256 if unified else 256 * 4
        if c + step <= G35['n_ctx_train']:
            _, over, _ = hl.stack(MAN, 'qwen3.5-35b-a3b', 'bartowski', 'Q4_K_M', c + step,
                                  'q8_0', n_seq=4, mtp=0, mmproj=True, unified=unified)
            assert over > hl.budget(24)


def test_invalid_args_rejected():
    for bad in (-1, 0):
        try:
            hl.kv_bytes(G27, 32768, 'q8_0', n_seq=bad)
            ok = False
        except (ValueError, TypeError):
            ok = True
        assert ok
    try:
        hl.stack(MAN, 'qwen3.8-27b', 'bartowski', 'Q4_K_M', -1)
        ok = False
    except ValueError:
        ok = True
    assert ok
    try:
        hl.stack(MAN, 'qwen3.8-27b', 'bartowski', 'Q4_K_M', 32768, cuda_ctx=float('nan'))
        ok = False
    except ValueError:
        ok = True
    assert ok


def test_tg_bound_pure_architecture():
    # 稠密模型：带宽上限随上下文深度下降，且不依赖任何实测校准
    b0 = hl.tg_bound(17.44, 1008, ffn='dense')
    b32 = hl.tg_bound(17.44, 1008, G27, 32768, 'q8_0', ffn='dense')
    assert b0 > b32 > 0
    # 手算：1008 GB/s / 17.44 GB
    assert close(b0, 1008 / 17.44)
    # 缺省 ffn 就是 dense，旧调用方不传 ffn 也能跑
    assert close(hl.tg_bound(17.44, 1008), 1008 / 17.44)


def test_tg_bound_rejects_moe():
    # MoE 必须拒绝整文件扫描口径，不能把稠密公式静默套到 35B-A3B
    try:
        hl.tg_bound(22.0, 1008, G35, 32768, 'q8_0', ffn='moe')
        ok = False
    except ValueError:
        ok = True
    assert ok


def test_mtp_sidecar_explicit_only():
    # 默认 stack 不含独立 MTP sidecar
    _, base, notes = hl.stack(MAN, 'qwen3.8-27b', 'unsloth', 'Q4_K_M', 32768,
                              'q8_0', n_seq=1, mtp=0, mmproj=False)
    assert notes['mtp_sidecar_included'] is False
    # 显式传入 unsloth 独立 MTP sidecar 后总额恰好多出该字节
    mtp_b = hl.sidecar_bytes(MAN, 'qwen3.8-27b', 'unsloth', 'mtp_q4_0')
    _, with_mtp, notes2 = hl.stack(MAN, 'qwen3.8-27b', 'unsloth', 'Q4_K_M', 32768,
                                   'q8_0', n_seq=1, mtp=0, mmproj=False,
                                   mtp_sidecar_bytes=mtp_b)
    assert notes2['mtp_sidecar_included'] is True
    assert with_mtp - base == mtp_b


def test_35b_heavier_than_27b_on_24g():
    # 同发布者口径下 35B Q4_K_M 权重大于 27B，24 GiB 单卡留给上下文的空间更紧
    w27 = hl.weight_bytes(MAN, 'qwen3.8-27b', 'bartowski', 'Q4_K_M')
    w35 = hl.weight_bytes(MAN, 'qwen3.5-35b-a3b', 'bartowski', 'Q4_K_M')
    assert w35 > w27
    c35 = hl.max_ctx(MAN, 'qwen3.5-35b-a3b', 'bartowski', 'Q4_K_M', 'q8_0', 1, 0, True, card_gib=24)
    c27 = hl.max_ctx(MAN, 'qwen3.8-27b', 'bartowski', 'Q4_K_M', 'q8_0', 1, 0, True, card_gib=24)
    # 两者都给得出非负上下文；35B 更紧（可能为 0，表示 24G 挂 mmproj 放不下）
    assert c35 >= 0 and c35 <= c27


def test_json_is_valid_and_schema_version():
    with open(os.path.join(HERE, 'dual_models.json'), encoding='utf-8') as f:
        raw = json.load(f)
    assert raw['schema_version'] == 1
    assert set(raw['models']) == {'qwen3.8-27b', 'qwen3.5-35b-a3b'}


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for fn in fns:
        fn()
        print('ok', fn.__name__)
    print('%d passed' % len(fns))
