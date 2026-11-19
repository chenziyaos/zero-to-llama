#!/usr/bin/env python3
"""hybrid_ledger.py：Qwen3.5 混合架构（Gated DeltaNet 线性注意力 + 少量 full attention）的统一显存账本。

一个模块同时服务稠密 Qwen3.8-27B（llama.cpp 架构名 qwen35）和 MoE Qwen3.5-35B-A3B（qwen35moe）。
模型几何和制品字节不在本文件硬编码，全部从 common/dual_models.json 读，避免各章各写一份常量。

公式照 llama.cpp 0ec4027（v0.8.0，ggml 0.28.0）：
- full-attention KV 每 token = 2(K、V) x kv_layers x n_kv_heads x head_dim x 每元素字节。
  只有 full_attention 层存 softmax KV；DeltaNet 层不存随上下文增长的 KV。
- DeltaNet 循环状态（src/llama-hparams.cpp 的 n_embd_r / n_embd_s，llama-memory-recurrent.cpp，F32 写死）：
  d_inner = value_head_dim x value_heads
  n_embd_s = d_state x d_inner
  n_embd_r = (d_conv - 1) x (d_inner + 2 x key_heads x d_state)
  每循环层每序列 (s + r) x 4 字节；是常数，不随上下文变长。
- draft-mtp / eagle3 / dflash / dspark 需要循环态回滚快照（common/common.h need_n_rs_seq），
  循环状态按 n_seq x (1 + n_rs_seq) 倍增，n_rs_seq 等于草稿最大 token 数；不开投机解码就是 1 倍。
- llama-server 缺省 -np auto = 4 槽（tools/server/server.cpp）。kv_unified 下多槽共用一个 KV 池，
  循环状态每槽一份。

MoE 的激活参数（35B-A3B 的 3B）只决定每 token 计算量，不决定常驻显存；常驻权重按整份量化文件字节算。

没有 GPU 实测。CUDA 上下文和 compute buffer 沿用 ch20/ch22 的假设默认值，命令行可改，
上机后用 llama-fit-params -fitp on 核对。本模块只做可复算的字节账本，不产出任何实测速度。
"""
import argparse
import json
import math
import os

GiB = 2 ** 30
MiB = 2 ** 20
KiB = 1024
GB = 1e9
CTX_ALIGN = 256

# KV 元素类型 -> 每元素字节（按 ggml 块大小折算），照 ggml-common.h
KV_BYTES = {'f32': 4.0, 'f16': 2.0, 'bf16': 2.0,
            'q8_0': 34 / 32, 'q4_0': 18 / 32, 'q4_1': 20 / 32,
            'iq4_nl': 18 / 32, 'q5_0': 22 / 32, 'q5_1': 24 / 32}

DEFAULT_CUDA_CTX_GIB = 0.5
DEFAULT_COMPUTE_GIB = 0.5
DEFAULT_FIT_MARGIN_GIB = 1.0
HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(HERE, 'dual_models.json')


def load_manifest(path=MANIFEST_PATH):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def pad_ctx(ctx, align=CTX_ALIGN):
    """n_ctx 向上取整到 256，照 llama-context.cpp 的 GGML_PAD(n_ctx, 256)。"""
    if not isinstance(ctx, int) or isinstance(ctx, bool):
        raise TypeError('ctx 必须是整数 token 数，收到 %r' % type(ctx).__name__)
    if ctx < 0:
        raise ValueError('ctx 不能为负，收到 %d' % ctx)
    return ((ctx + align - 1) // align) * align


def _check_pos_int(value, name, minimum=1):
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError('%s 必须是整数，收到 %r' % (name, type(value).__name__))
    if value < minimum:
        raise ValueError('%s 必须 >= %d，收到 %d' % (name, minimum, value))
    return value


def _check_nonneg_int(value, name):
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError('%s 必须是整数，收到 %r' % (name, type(value).__name__))
    if value < 0:
        raise ValueError('%s 不能为负，收到 %d' % (name, value))
    return value


def split_ctx(ctx, n_seq, unified=True, align=CTX_ALIGN):
    """照 llama-context.cpp 算 (总 n_ctx, 每槽 n_ctx_seq)。

    unified：总池一个 stream，n_ctx_seq = pad(n_ctx)。
    非 unified：先 pad 总长度，再 n_ctx_seq = pad(n_ctx / n_seq)，
     llama.cpp 还会把总长度回写为 n_ctx_seq * n_seq，这里把回写后的总值也返回。
    """
    _check_pos_int(n_seq, 'n_seq')
    total = pad_ctx(ctx, align)
    if unified:
        return total, total
    per = pad_ctx(total // n_seq, align)
    if per == 0:
        raise ValueError('非 unified 下每槽上下文为 0：n_ctx=%d，n_seq=%d' % (ctx, n_seq))
    return per * n_seq, per


def geometry(manifest, model_key):
    """从 manifest 取一个模型，摊平成本账本要的几何 dict。"""
    m = manifest['models'][model_key]
    g = m['geometry']
    fa, dn = g['full_attention'], g['deltanet']
    out = dict(
        name=m['official_id'], key=model_key, ffn=m['ffn'],
        n_layer=g['n_layer'], kv_layers=g['kv_layers'], rec_layers=g['rec_layers'],
        n_kv=fa['n_kv_heads'], head_dim=fa['head_dim'],
        n_ctx_train=m['context_native'],
        d_state=dn['d_state'], k_heads=dn['k_heads'], v_heads=dn['v_heads'],
        v_head_dim=dn['v_head_dim'], d_conv=dn['d_conv'],
    )
    out['moe'] = m['moe']
    return out


def d_inner(g):
    """DeltaNet 内部宽度 = value head dim x value heads。"""
    return g['v_head_dim'] * g['v_heads']


def kv_per_token(g, kv='f16'):
    """所有 full-attention 层每 token 的 KV 字节（K、V 各一份）。"""
    if kv not in KV_BYTES:
        raise KeyError('不支持的 KV 类型 %s，可选 %s' % (kv, sorted(KV_BYTES)))
    return 2 * g['kv_layers'] * g['n_kv'] * g['head_dim'] * KV_BYTES[kv]


def rec_per_seq(g):
    """一条序列占用的全部 DeltaNet 循环状态字节（F32），常数，不随上下文变。"""
    if not g['rec_layers']:
        return 0
    di = d_inner(g)
    s = g['d_state'] * di
    r = (g['d_conv'] - 1) * (di + 2 * g['k_heads'] * g['d_state'])
    return g['rec_layers'] * (s + r) * 4


def rec_total(g, n_seq, n_rs_seq=0):
    """n_seq 个槽、每槽 (1 + n_rs_seq) 份循环状态（回滚快照）。

    照 llama-memory-hybrid.cpp：recurrent_kv_size = n_seq_max，
     llama-memory-recurrent.cpp 每层 R/S 张量行数 = n_seq_max x (1 + n_rs_seq)，F32。
    """
    _check_pos_int(n_seq, 'n_seq')
    _check_nonneg_int(n_rs_seq, 'n_rs_seq')
    return rec_per_seq(g) * n_seq * (1 + n_rs_seq)


def kv_bytes(g, ctx, kv='f16', n_seq=1, unified=True):
    """给定总上下文 ctx 的 full-attention KV 池字节。

    照 llama-context.cpp / llama-kv-cache.cpp：
    - unified（llama-server 缺省 kv_unified）：一个 stream，kv_size = n_ctx_seq = pad(ctx)，只存一份；
    - 非 unified：n_stream = n_seq，每 stream 的 kv_size = pad(pad(ctx) / n_seq)，
      总字节 = 每 token 字节 x 每槽长度 x 槽数（等价于按回写后的 n_ctx_seq * n_seq 算）。
    传入的 ctx 语义统一为“整池总上下文”，不再让调用方自己除以槽数。
    """
    _check_pos_int(n_seq, 'n_seq')
    total, per = split_ctx(ctx, n_seq, unified)
    per_tok = kv_per_token(g, kv)
    if unified:
        return per_tok * per
    return per_tok * per * n_seq


def weight_bytes(manifest, model_key, publisher, quant):
    """某发布者某量化主文件的精确字节。"""
    pub = manifest['models'][model_key]['gguf'][publisher]
    return pub['files'][quant]


def sidecar_bytes(manifest, model_key, publisher, name):
    """sidecar（mmproj/mtp）的精确字节，name 如 mmproj_bf16。"""
    return manifest['models'][model_key]['gguf'][publisher]['sidecars'][name]


def stack(manifest, model_key, publisher, quant, ctx, kv='q8_0', n_seq=1, mtp=0,
          mmproj=False, cuda_ctx=DEFAULT_CUDA_CTX_GIB, compute=DEFAULT_COMPUTE_GIB,
          unified=True, mtp_sidecar_bytes=0):
    """显存账 [(项, 字节)] 与合计。KV 按 unified 共池；循环态每槽一份并按 mtp 倍增。

    只算主权重 + full-attn KV + DeltaNet 循环态 + compute/CUDA 假设项 + 可选 mmproj。
    MTP 头权重和 MTP 自身 KV 因发布者处理方式不同（bartowski 内嵌主文件、unsloth 独立 sidecar、
    35B 另起整模型仓库），默认不自动计入。要把独立 MTP sidecar 算进来，显式传
    mtp_sidecar_bytes=sidecar_bytes(...)；内嵌口径的主文件字节已含 MTP，不要再加。
    mtp>0 只倍增主模型循环态回滚快照，不代表已覆盖独立 draft 模型的权重与状态。
    返回第三项是完整性标记 dict，说明哪些成本未计入，避免把部分账误读成整账。
    """
    _check_nonneg_int(ctx, 'ctx')
    _check_pos_int(n_seq, 'n_seq')
    _check_nonneg_int(mtp, 'mtp')
    for nm, val in (('cuda_ctx', cuda_ctx), ('compute', compute)):
        if not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(val) or val < 0:
            raise ValueError('%s 必须是非负有限数，收到 %r' % (nm, val))
    _check_nonneg_int(mtp_sidecar_bytes, 'mtp_sidecar_bytes')
    g = geometry(manifest, model_key)
    total_ctx, per_ctx = split_ctx(ctx, n_seq, unified)
    rows = [('权重 %s（%s）' % (quant, publisher), weight_bytes(manifest, model_key, publisher, quant)),
            ('full-attn KV %s，%d 层 x 总 %d / 每槽 %d（%d 槽 %s）' % (
                kv, g['kv_layers'], total_ctx, per_ctx, n_seq,
                'kv_unified 共池' if unified else '各槽独立'),
             kv_bytes(g, ctx, kv, n_seq, unified))]
    if g['rec_layers']:
        rows.append(('DeltaNet 循环状态 x %d 槽 x %d' % (n_seq, 1 + mtp),
                     rec_total(g, n_seq, mtp)))
    rows.append(('compute buffer（假设）', compute * GiB))
    rows.append(('CUDA 上下文（假设）', cuda_ctx * GiB))
    if mmproj:
        rows.append(('mmproj bf16', sidecar_bytes(manifest, model_key, publisher, 'mmproj_bf16')))
    if mtp_sidecar_bytes:
        rows.append(('独立 MTP sidecar（显式传入）', mtp_sidecar_bytes))
    notes = {
        'ctx_padded_total': total_ctx,
        'ctx_per_seq': per_ctx,
        'mtp_sidecar_included': bool(mtp_sidecar_bytes),
        'mtp_draft_kv_included': False,
        'assumption_items': ('compute buffer=%.2f GiB, CUDA ctx=%.2f GiB（均为假设，待上机核对）'
                             % (compute, cuda_ctx)),
    }
    return rows, sum(b for _, b in rows), notes


def budget(card_gib, fit_margin=DEFAULT_FIT_MARGIN_GIB):
    """单卡可用预算：扣每张卡 fit 余量。CUDA 上下文已在 stack 里单列，不在这里重复扣。"""
    return (card_gib - fit_margin) * GiB


def max_ctx(manifest, model_key, publisher, quant, kv='q8_0', n_seq=1, mtp=0, mmproj=False,
            cuda_ctx=DEFAULT_CUDA_CTX_GIB, compute=DEFAULT_COMPUTE_GIB, card_gib=24.0,
            unified=True, fit_margin=DEFAULT_FIT_MARGIN_GIB, mtp_sidecar_bytes=0):
    """预算内最长“整池总上下文”，按 256 向下取整（llama.cpp n_ctx 对齐），封顶原生训练长度。

    和 stack 互逆：用这里返回的 ctx 调 stack，KV 占用不会越过预算。
    非 unified 下整池要在 n_seq 个槽间平分，每 token 的整池成本 = 每 token 字节 x n_seq，
    因此求解时按槽数折算，避免旧版漏乘 n_seq 高估可放下的上下文。
    """
    g = geometry(manifest, model_key)
    # 固定项（权重 + 循环态 + compute/CUDA + 可选 sidecar）与 unified 无关，
    # ctx=0 时 unified 的 KV 恰为 0，用它取固定项，避开非 unified 平分零 ctx 的报错。
    _, fixed, _ = stack(manifest, model_key, publisher, quant, 0, kv, n_seq, mtp, mmproj,
                        cuda_ctx, compute, True, mtp_sidecar_bytes)
    room = budget(card_gib, fit_margin) - fixed
    per_tok = kv_per_token(g, kv)
    if room <= 0 or per_tok <= 0:
        return 0
    cap = g['n_ctx_train']
    if unified:
        # 单 stream：整池就是一个连续 KV，按 256 向下取整即可
        raw = int(room // per_tok)
        total = raw // CTX_ALIGN * CTX_ALIGN
        return min(total, cap)
    # 非 unified：n_seq 个 stream，每 stream 独立按 256 对齐。先求每槽最长，
    # 再回写整池 = per_seq * n_seq（和 split_ctx 的回写一致），不能只对整池做一次对齐，
    # 否则平分再 pad 会把实际 KV 顶破预算（round-trip 会失败）。
    per_seq_raw = int(room // (per_tok * n_seq))
    per_seq = per_seq_raw // CTX_ALIGN * CTX_ALIGN
    if per_seq == 0:
        return 0
    return min(per_seq * n_seq, cap)


def tg_bound(file_gb, gpu_bw_gbs, g=None, ctx_depth=0, kv='q8_0', ffn='dense'):
    """全层上卡 decode 的带宽上限（t/s）：每 token 读一遍权重，再读当前深度的 KV。

    gpu_bw_gbs 是显存带宽 GB/s（十进制）。file_gb 是权重文件的十进制 GB。
    纯架构账本，不含任何实测校准系数。

    适用范围要分清：
    - 稠密模型（ffn='dense'）：decode 每 token 确实要读完整份权重，整文件扫描是站得住的带宽上限。
    - MoE（ffn='moe'）：每 token 只激活部分专家，读不完整份权重，拿整文件当分母会得到一个
      被严重低估、没有“上限”含义的数。这里直接拒绝给 MoE 算这个 bound，逼调用方改用按激活
      专家字节建模的口径，而不是把稠密公式套到 35B-A3B 上。
    """
    if ffn == 'moe':
        raise ValueError(
            'tg_bound 的整文件扫描口径不适用于 MoE：每 token 只激活部分专家，'
            '请按激活专家字节单独建模，不要把稠密带宽上限套到 MoE。')
    byt = file_gb * GB
    if g is not None and ctx_depth:
        byt += kv_per_token(g, kv) * ctx_depth
    return gpu_bw_gbs * GB / byt


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--manifest', default=MANIFEST_PATH)
    ap.add_argument('--model', required=True, help='dual_models.json 的模型 key，如 qwen3.8-27b')
    ap.add_argument('--publisher', default=None, help='GGUF 发布者 key，如 bartowski/unsloth/lmstudio-community')
    ap.add_argument('--quant', default='Q4_K_M')
    ap.add_argument('--ctx', type=int, default=32768)
    ap.add_argument('--kv', default='q8_0', choices=sorted(KV_BYTES))
    ap.add_argument('--np', type=int, default=1)
    ap.add_argument('--mtp', type=int, default=0)
    ap.add_argument('--mmproj', action='store_true')
    ap.add_argument('--card', type=float, default=24)
    ap.add_argument('--cuda-ctx', type=float, default=DEFAULT_CUDA_CTX_GIB,
                    help='CUDA 上下文假设占用 GiB，上机后用 llama-fit-params 核对回填')
    ap.add_argument('--compute', type=float, default=DEFAULT_COMPUTE_GIB,
                    help='compute buffer 假设占用 GiB，上机后核对回填')
    ap.add_argument('--fit-margin', type=float, default=DEFAULT_FIT_MARGIN_GIB,
                    help='每张卡预留的 fit 余量 GiB')
    ap.add_argument('--no-unified', action='store_true',
                    help='按非 kv_unified（各槽独立 KV stream）算账，缺省是 unified 共池')
    args = ap.parse_args()

    man = load_manifest(args.manifest)
    if args.model not in man['models']:
        raise SystemExit('未知模型 %s，可选 %s' % (args.model, sorted(man['models'])))
    g = geometry(man, args.model)
    unified = not args.no_unified
    pub = args.publisher or sorted(man['models'][args.model]['gguf'])[0]
    print('%s（%s，ffn=%s）来源 %s' % (g['name'], g['key'], g['ffn'], pub))
    print('  full-attn %d 层 / DeltaNet %d 层，KV 头 %d，head_dim %d' % (
        g['kv_layers'], g['rec_layers'], g['n_kv'], g['head_dim']))
    print('  KV 每 token：f16 %.0f B（%.1f KiB），q8_0 %.0f B，q4_0 %.0f B' % (
        kv_per_token(g, 'f16'), kv_per_token(g, 'f16') / KiB,
        kv_per_token(g, 'q8_0'), kv_per_token(g, 'q4_0')))
    print('  DeltaNet 循环状态每序列 %.3f MiB；%d 槽 mtp=%d 合计 %.3f MiB' % (
        rec_per_seq(g) / MiB, args.np, args.mtp, rec_total(g, args.np, args.mtp) / MiB))
    rows, total, notes = stack(man, args.model, pub, args.quant, args.ctx, args.kv,
                               args.np, args.mtp, args.mmproj,
                               args.cuda_ctx, args.compute, unified)
    for name, b in rows:
        print('  %-46s %8.2f GiB' % (name, b / GiB))
    bud = budget(args.card, args.fit_margin) / GiB
    print('  %-46s %8.2f GiB / 预算 %.2f GiB  %s' % (
        '合计', total / GiB, bud, 'OK' if total <= bud * GiB else '超出'))
    print('  上下文对齐：请求 %d -> 整池 %d，每槽 %d（%s）' % (
        args.ctx, notes['ctx_padded_total'], notes['ctx_per_seq'],
        'unified 共池' if unified else '各槽独立'))
    print('  最长上下文 %d token（%g GiB 卡，%s KV，%d 槽，mmproj=%s，%s）' % (
        max_ctx(man, args.model, pub, args.quant, args.kv, args.np, args.mtp, args.mmproj,
                args.cuda_ctx, args.compute, args.card, unified, args.fit_margin),
        args.card, args.kv, args.np, args.mmproj,
        'unified' if unified else 'non-unified'))
    if args.mtp and not notes['mtp_sidecar_included']:
        print('  注意：mtp=%d 只倍增了主模型循环态快照，独立 draft 权重/KV 未计入（见 stack docstring）'
              % args.mtp)


if __name__ == '__main__':
    main()
