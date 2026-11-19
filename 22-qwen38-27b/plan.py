#!/usr/bin/env python3
"""plan.py：把 Qwen3.8-27B（llama.cpp 架构名 qwen35，混合 Gated DeltaNet 注意力）装进一台机器的账。

几何和字节不再在本篇硬编码，全部从 common/dual_models.json 读、走 common/hybrid_ledger.py 统一账本，
避免各章各抄一份常量、改一处漏一处。本篇只保留 Qwen3.8-27B 特有的校准（Discussion #15013、kingy 实测）、
多卡上下文表和命令行入口。

结构口径（官方 config，厂商口径，精确值以 manifest 为准）：
- 64 层，full_attention_interval=4：层号 3、7、…、63 是 full attention，共 16 层，存 softmax KV；
  其余 48 层（四分之三）是 Gated DeltaNet 线性注意力，只存固定大小的循环状态，不随上下文变长。
- full-attn：24 Q 头、4 KV 头、head_dim 256，partial rotary 64 维。
- DeltaNet：16 个 key 头、48 个 value 头，key/value head_dim 128，conv kernel 4，状态 F32。
- 带 1 个 MTP 头（mtp_num_hidden_layers=1）。原生上下文 262144，模型卡写明可扩到 1000000；这里仍封顶 262144。

公式照 llama.cpp 0ec4027（固定基线），实现全部在 common/hybrid_ledger.py：
- KV 每 token = 2 (K、V) x 16 层 x n_kv 4 x head_dim 256 x 每元素字节。f16 64 KiB，q8_0 约 34 KiB。
- DeltaNet 循环状态 F32 写死，是常数，不随上下文变长；draft-mtp 按 (1 + n_rs_seq) 倍增回滚快照。
- llama-server 缺省 -np auto = 4 槽。kv_unified 下 4 槽共用一个 KV 池，循环状态每槽一份。

文件口径（Hugging Face tree API，精确字节在 manifest，十进制 GB 只是展示换算）：
- bartowski/Qwen3.8-27B-GGUF：MTP 头编进主文件，imatrix 档 MTP 存 Q4_0（Q8_0 除外）。
- lmstudio-community/Qwen3.8-27B-GGUF：精简三档，模型卡没写 MTP 怎么处理，不为它算 mtp 情形。
- unsloth/Qwen3.8-27B-GGUF：MTP 单独文件；kingy 实测钉的是旧修订常规 Q4_K_M 17106775008 B，
  main 已换文件，字节不可照搬，所以 kingy 用单独的历史文件字节，不从 manifest 取。

没有 GPU 实测。CUDA 上下文 --cuda-ctx 缺省 0.5 GiB、compute buffer --compute 缺省 0.5 GiB 都是假设，
上机后用 llama-fit-params -fitp on 核对。权重大小是发布者文件的十进制 GB，显存预算按 GiB，两者分开。

用法：
  python3 plan.py                       # 打印各量化在 16/24/32/48 GB 上能给多长上下文、带宽上限
  python3 plan.py --quant Q4_K_M --ctx 32768 --kv q8_0 --card 24 --mmproj
  python3 plan.py --quant Q4_K_M --ctx 32768 --kv q8_0 --card 24 --mmproj --mtp 3
  python3 plan.py --fit-card 24 --quant Q4_K_M --kv q8_0 --mmproj
  python3 plan.py --emit gen22_data.py
"""
import argparse
import os
import sys

GiB = 2 ** 30
MiB = 2 ** 20
GB = 1e9

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))

import hybrid_ledger as hl  # noqa: E402

MODEL_KEY = 'qwen3.8-27b'
MAN = hl.load_manifest()
G = hl.geometry(MAN, MODEL_KEY)
# 向后兼容旧测试/旧调用方：M 仍是同一组几何，但值来自 manifest，不再本地硬编码
M = dict(name=G['name'], n_layer=G['n_layer'], kv_layers=G['kv_layers'], n_kv=G['n_kv'],
         head_dim=G['head_dim'], n_ctx_train=G['n_ctx_train'], rec_layers=G['rec_layers'],
         d_state=G['d_state'], v_heads=G['v_heads'], k_heads=G['k_heads'],
         v_head_dim=G['v_head_dim'], d_conv=G['d_conv'])

# 发布者 key（manifest）到本篇命令行 source 的映射
PUB = {'bartowski': 'bartowski', 'lmstudio': 'lmstudio-community'}

# 展示用十进制 GB，从 manifest 精确字节换算，不再手抄
BARTOWSKI = {q: b / GB for q, b in MAN['models'][MODEL_KEY]['gguf']['bartowski']['files'].items()}
LMSTUDIO = {q: b / GB for q, b in MAN['models'][MODEL_KEY]['gguf']['lmstudio-community']['files'].items()}
MMPROJ_GB = hl.sidecar_bytes(MAN, MODEL_KEY, 'bartowski', 'mmproj_bf16') / GB

GPU_BW = {'RTX 4090': 1008, 'RTX 3090': 936, 'RTX 4070 Ti Super': 672, 'RTX 5090': 1792}  # GB/s，TechPowerUp 规格
# kingy 社区实测（2026-08-17，b10453 commit 3cb7ffb，CUDA 11.8，f16 KV，5 次合成 256 token decode 取中位）：
# unsloth 常规 Q4_K_M 旧修订 f1bfb12 文件 17106775008 B，RTX 4090 decode 49.1 t/s。
# 这是历史修订的独立实测锚点，main 已换文件，字节故意不从 manifest 取。
KINGY = dict(file_gb=17106775008 / GB, tg=49.1, gpu='RTX 4090')
CAL_MODEL_GIB = 3.56
CAL = {'RTX 4090': 188.96, 'RTX 3090': 161.89}   # Discussion #15013，Llama2 7B Q4_0 tg128 FA
FIT_MARGIN_GIB = hl.DEFAULT_FIT_MARGIN_GIB


def _pub(source):
    return PUB[source]


def d_inner():
    return hl.d_inner(G)


def kv_per_token(kv='f16'):
    """16 个 full-attn 层每 token 的 KV 字节。"""
    return hl.kv_per_token(G, kv)


def rec_per_seq():
    """一条序列的 48 层 DeltaNet 循环状态（F32），常数，不随上下文变。"""
    return hl.rec_per_seq(G)


def rec_total(n_seq, n_rs_seq):
    return hl.rec_total(G, n_seq, n_rs_seq)


def weight_gb(quant, source='bartowski'):
    return hl.weight_bytes(MAN, MODEL_KEY, _pub(source), quant) / GB


def stack(quant, ctx, kv='q8_0', n_seq=4, mtp=0, mmproj=False, source='bartowski',
          cuda_ctx=0.5, compute=0.5):
    """显存账 [(项, 字节)] 与总数。直接转发共享账本，KV 按 kv_unified 共池，循环态每槽一份并按 mtp 倍增。

    bartowski 主文件已内嵌 MTP 头，开 draft-mtp 不增加权重；lmstudio 模型卡没写明 MTP 处理，
    本函数不为它计算 mtp 情形。
    """
    rows, total, _ = hl.stack(MAN, MODEL_KEY, _pub(source), quant, ctx, kv, n_seq, mtp,
                              mmproj, cuda_ctx, compute, unified=True)
    return rows, total


def budget(card_gib):
    """CUDA 上下文已作为一项算进 stack，这里只扣每张卡 1 GiB 的 fit 余量。"""
    return hl.budget(card_gib, FIT_MARGIN_GIB)


def max_ctx(quant, kv='q8_0', n_seq=4, mtp=0, mmproj=False, source='bartowski',
            cuda_ctx=0.5, compute=0.5, card_gib=24.0):
    """预算里能放下的最长整池上下文，按 256 向下取整，封顶训练长度 262144。"""
    return hl.max_ctx(MAN, MODEL_KEY, _pub(source), quant, kv, n_seq, mtp, mmproj,
                      cuda_ctx, compute, card_gib, unified=True, fit_margin=FIT_MARGIN_GIB)


def tg_bound(file_gb, gpu, ctx_depth=0, kv='q8_0'):
    """全层上卡 decode 的带宽上限：每 token 读一遍权重，再读当前深度的 KV。27B 是稠密模型，整文件口径成立。"""
    return hl.tg_bound(file_gb, GPU_BW[gpu], G, ctx_depth, kv, ffn='dense')


def eff():
    out = {g: CAL[g] / (bw * GB / (CAL_MODEL_GIB * GiB)) for g, bw in GPU_BW.items() if g in CAL}
    out['kingy'] = KINGY['tg'] / (GPU_BW[KINGY['gpu']] / KINGY['file_gb'])
    return out


def fmt(rows, total, card_gib, cuda_ctx=0.5):
    for n, b in rows:
        print('  %-42s %8.2f GiB' % (n, b / GiB))
    bud = budget(card_gib) / GiB
    print('  %-42s %8.2f GiB / 预算 %.2f GiB  %s' % (
        '合计', total / GiB, bud, 'OK' if total <= bud * GiB else '超出'))


def collect():
    e = eff()
    D = {'kv_tok': {kv: kv_per_token(kv) for kv in hl.KV_BYTES}, 'rec_seq': rec_per_seq(),
         'bartowski': BARTOWSKI, 'lmstudio': LMSTUDIO, 'mmproj_gb': MMPROJ_GB,
         'gpu_bw': GPU_BW, 'eff': e,
         'kingy': {'file_gb': KINGY['file_gb'], 'tg': KINGY['tg'],
                   'bound': tg_bound(KINGY['file_gb'], 'RTX 4090')}}
    D['ctx24'] = {(q, mm): max_ctx(q, 'q8_0', 4, 0, mm, card_gib=24)
                  for q in BARTOWSKI for mm in (False, True)}
    D['ctx24_mtp'] = {(q, mm): max_ctx(q, 'q8_0', 4, 3, mm, card_gib=24)
                      for q in BARTOWSKI for mm in (False, True)}
    # 单卡单用户主推 -np 1：循环态只一份。给两张主表：单槽挂 mmproj，单槽开 mtp3 挂 mmproj
    D['ctx24_np1'] = {q: max_ctx(q, 'q8_0', 1, 0, True, card_gib=24) for q in BARTOWSKI}
    D['ctx24_np1_mtp3'] = {q: max_ctx(q, 'q8_0', 1, 3, True, card_gib=24) for q in BARTOWSKI}
    D['ctx_cards'] = {card: {q: max_ctx(q, 'q8_0', 1, 0, True, card_gib=card)
                             for q in ('IQ3_M', 'Q3_K_M', 'Q4_K_M', 'Q5_K_M', 'Q6_K')}
                      for card in (16, 24, 32, 48)}
    # 24 GB 主推：bartowski Q4_K_M，q8 KV，单槽，挂 mmproj（run_24g.sh 口径）
    D['stack24_32k'] = stack('Q4_K_M', 32768, 'q8_0', 1, 0, True)[0]
    D['stack24_32k_mtp3'] = stack('Q4_K_M', 32768, 'q8_0', 1, 3, True)[0]
    D['stack24_32k_text'] = stack('Q4_K_M', 32768, 'q8_0', 1, 0, False, 'lmstudio')[0]
    # 4 槽服务口径，对照用
    D['stack24_32k_np4'] = stack('Q4_K_M', 32768, 'q8_0', 4, 0, True)[0]
    D['rec'] = {(ns, rs): rec_total(ns, rs) for ns in (1, 4) for rs in (0, 3)}
    D['kv_table'] = {ctx: {kv: kv_per_token(kv) * ctx for kv in ('f16', 'q8_0')}
                     for ctx in (8192, 32768, 131072, 262144)}
    D['tg'] = {(q, g): (b, b * e['RTX 4090'] if g == 'RTX 4090' else b * e['kingy'])
               for q, fg in BARTOWSKI.items() for g, b in
               ((g, tg_bound(fg, g)) for g in GPU_BW)}
    D['tg_depth'] = {d: tg_bound(BARTOWSKI['Q4_K_M'], 'RTX 4090', d, 'q8_0') for d in (0, 32768, 131072)}
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quant', default='Q4_K_M')
    ap.add_argument('--source', choices=tuple(PUB), default='bartowski')
    ap.add_argument('--ctx', type=int, default=32768)
    ap.add_argument('--kv', default='q8_0', choices=sorted(hl.KV_BYTES))
    ap.add_argument('--np', type=int, default=4)
    ap.add_argument('--mtp', type=int, default=0, help='draft-mtp 草稿最大 token 数，0 不开（bartowski 内嵌 MTP）')
    ap.add_argument('--mmproj', action='store_true', help='挂视觉投影器（看图才需要）')
    ap.add_argument('--card', type=float, default=24, help='显存，GiB')
    ap.add_argument('--cuda-ctx', type=float, default=0.5)
    ap.add_argument('--compute', type=float, default=0.5)
    ap.add_argument('--fit-card', type=float, metavar='GIB')
    ap.add_argument('--emit')
    a = ap.parse_args()
    if a.emit:
        with open(a.emit, 'w') as f:
            f.write('"""由 22-qwen38-27b/plan.py --emit 生成，不要手改。"""\n')
            for k, v in collect().items():
                f.write('%s = %r\n' % (k.upper(), v))
        print('wrote', a.emit)
        return
    if a.fit_card:
        c = max_ctx(a.quant, a.kv, a.np, a.mtp, a.mmproj, a.source, a.cuda_ctx, a.compute, a.fit_card)
        print('%s（%s），%g GiB 卡，%s KV，%d 槽，mtp=%d，mmproj=%s：最长上下文 %d' % (
            a.quant, a.source, a.fit_card, a.kv, a.np, a.mtp, a.mmproj, c))
        return
    src = a.source
    table = BARTOWSKI if src == 'bartowski' else LMSTUDIO
    if a.quant not in table:
        sys.exit('%s 不在 %s 的文件里' % (a.quant, src))
    rows, total = stack(a.quant, a.ctx, a.kv, a.np, a.mtp, a.mmproj, src, a.cuda_ctx, a.compute)
    fmt(rows, total, a.card, a.cuda_ctx)
    print('  最长上下文 %d' % max_ctx(a.quant, a.kv, a.np, a.mtp, a.mmproj, src,
                                      a.cuda_ctx, a.compute, a.card))
    e = eff()
    b = tg_bound(weight_gb(a.quant, src), 'RTX 4090', a.ctx, a.kv)
    print('  RTX 4090 全层上卡带宽上限 %.1f t/s，按校准效率预测 %.1f-%.1f t/s（无 KV 深度时）' % (
        b, b * e['RTX 4090'], b * e['kingy']))
    print('  kingy 实测锚点：unsloth 旧修订 Q4_K_M 17.1068 GB，4090 f16 KV decode 49.1 t/s（社区实测）')


if __name__ == '__main__':
    sys.exit(main())
