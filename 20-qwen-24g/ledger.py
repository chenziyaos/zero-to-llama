#!/usr/bin/env python3
"""ledger.py：24 GB 单卡跑 Qwen3-32B 和 Qwen3.8-27B 的显存账和速度上限，全部是推算。

公式照 llama.cpp 0ec4027：
- KV：每 token = 2 (K、V) x 存 KV 的层数 x n_head_kv x head_dim x 每元素字节。
  Qwen3-32B 64 层都存；Qwen3.8-27B 只有 16 层 Gated Attention 存，另外 48 层 Gated DeltaNet 存固定大小的状态。
- DeltaNet 状态（llama-hparams.cpp 的 n_embd_r / n_embd_s，类型 F32）：
  s = d_state x d_inner = 128 x (48 x 128)，r = (d_conv - 1) x (d_inner + 2 x n_group x d_state)。
  llama-memory-recurrent.cpp 按 n_rows = n_seq_max x (1 + n_rs_seq) 分配，n_rs_seq 在 draft-mtp、eagle3、
  dflash、dspark 时等于 --spec-draft-n-max（common.h need_n_rs_seq）。
- llama-server 的 -np 缺省是 auto：n_parallel = 4，kv_unified = true（server.cpp）。KV 是 4 个槽共用一个池，
  DeltaNet 状态却是每个槽一份。
- fit 缺省打开，每张卡留 1024 MiB（common.h fit_params_target），-c 0 时上下文最少压到 4096。

没有 GPU 实测。显存里算不准的两项用假设值，命令行可改：CUDA 上下文 --cuda-ctx（缺省 0.5 GiB），
compute buffer --compute（缺省 0.5 GiB，前提是 flash attention 打开）。上机以后用 llama-fit-params -fitp on 核对。

用法：
  python3 ledger.py                       # 打印两张表：各量化在 24 GB 上能给多长上下文、速度上限
  python3 ledger.py --model 27b --quant UD-Q4_K_XL --ctx 65536 --kv q8_0 --np 1 --mtp 2
  python3 ledger.py --model 32b --quant IQ4_XS --ctx 65536 --kv q4_0 --yarn 2   # 32B 开 YaRN 2 倍
  python3 ledger.py --model 32b --quant IQ4_XS --ctx 16384 --kv q8_0 --draft Qwen3-0.6B-Q8_0 --kvd q8_0
  python3 ledger.py --model 27b --ctx 65536 --kv q8_0 --mtp 2 --cache 32     # 一条提示缓存多大
  python3 ledger.py --emit gen20_data.py  # 把文中用到的数字写成一个 Python 模块
"""
import argparse
import os
import sys

GiB = 2 ** 30
MiB = 2 ** 20
GB = 1e9
KV_BYTES = {'f16': 2.0, 'q8_0': 34 / 32, 'q4_0': 18 / 32}

# 混合架构（Qwen3.8-27B）的几何和精确制品字节以共享 manifest 为唯一来源，本篇不再手抄，
# 避免各章各写一份常量、改一处漏一处。32B 是全 64 层传统架构、不在双模型基线里，仍是本篇历史对照，保留本地。
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
import hybrid_ledger as hl  # noqa: E402

MAN = hl.load_manifest()
MODEL_KEY = 'qwen3.8-27b'
G27 = hl.geometry(MAN, MODEL_KEY)      # 27B 混合架构几何，来自 common/dual_models.json

# 结构参数：32B 来自它自己的 config.json（历史对照，不进共享 manifest）；
# 27B 整块从共享几何 G27 派生，字段名保留本篇历史叫法，下游 stack/max_ctx/cache 不用改。
MODELS = {
    '32b': dict(name='Qwen3-32B', n_layer=64, kv_layers=64, n_kv=8, head_dim=128, n_vocab=151936,
                n_ctx_train=40960, rec_layers=0),
    '27b': dict(name='Qwen3.8-27B', n_layer=G27['n_layer'], kv_layers=G27['kv_layers'],
                n_kv=G27['n_kv'], head_dim=G27['head_dim'],
                n_ctx_train=G27['n_ctx_train'], rec_layers=G27['rec_layers'],
                d_state=G27['d_state'], v_heads=G27['v_heads'], k_heads=G27['k_heads'],
                d_conv=G27['d_conv']),
}

# 文件大小，单位 GB（十进制），抄自 Hugging Face 文件列表，2026-11-02。
# 注意：27b 这里全是 unsloth 的 UD-* 动态量化档，和共享 manifest 登记的标准档（IQ4_XS/Q4_K_M/...）
# 不是同一批文件，精确字节没有登记，保留两位小数的历史展示值，不拿 manifest 的标准档字节冒充。
# manifest 已登记的 unsloth 标准档精确字节另见 UNSLOTH_EXACT_BYTES，回归测试用它交叉核对数量级。
UNSLOTH_EXACT_BYTES = MAN['models'][MODEL_KEY]['gguf']['unsloth']['files']
UNSLOTH_MTP_Q4_0_BYTES = hl.sidecar_bytes(MAN, MODEL_KEY, 'unsloth', 'mtp_q4_0')
QUANTS = {
    '32b': {  # bartowski/Qwen_Qwen3-32B-GGUF
        'IQ2_M': 11.36, 'IQ3_XXS': 12.82, 'IQ3_M': 14.93, 'Q3_K_M': 15.97, 'IQ4_XS': 17.69,
        'Q4_K_S': 18.77, 'Q4_K_M': 19.76, 'Q4_K_L': 20.34, 'Q5_K_S': 22.64, 'Q5_K_M': 23.21,
        'Q6_K': 26.88, 'Q8_0': 34.82,
    },
    '27b': {  # unsloth/Qwen3.8-27B-GGUF
        'UD-IQ2_XXS': 7.27, 'UD-Q2_K_XL': 9.83, 'UD-IQ3_XXS': 10.93, 'UD-Q3_K_XL': 13.15,
        'UD-IQ4_XS': 14.25, 'UD-Q4_K_S': 15.36, 'UD-Q4_K_M': 16.46, 'UD-Q4_K_XL': 17.56,
        'UD-Q5_K_M': 19.77, 'UD-Q5_K_XL': 20.88, 'UD-Q6_K': 21.98, 'Q8_0': 29.05,
    },
}
EXTRA = {'mmproj-F16': 0.93, 'mtp-Q4_0': 1.37, 'Qwen3-0.6B-Q8_0': 0.64, 'Qwen3-1.7B-Q8_0': 1.83,
         'eagle3-32B-bf16': 1.455}
DRAFT_06B = dict(kv_layers=28, n_kv=8, head_dim=128)   # Qwen3-0.6B config.json

GPUS = {'RTX 4090': 1008, 'RTX 3090': 936}          # GB/s，TechPowerUp 规格
CARD_GIB = 24.0
FIT_MARGIN_GIB = 1.0

# 校准：Discussion #15013，Llama 2 7B Q4_0（3.56 GiB），tg128，FA 打开
CAL = {'RTX 4090': 188.96, 'RTX 3090': 161.89}
CAL_MODEL_GIB = 3.56
# 交叉核对：kingy.ai，RTX 4090，unsloth Q4_K_M 17.11 GB，llama-bench decode 49.1 t/s
KINGY = dict(gpu='RTX 4090', file_gb=17.11, tg=49.1)


def set_yarn(factor):
    """32B 开 YaRN 后 llama.cpp 把 n_ctx_train 换成 yarn_orig_ctx / rope_freq_scale，即 32768 x factor。"""
    MODELS['32b']['n_ctx_train'] = int(32768 * factor) if factor else 40960


def kv_per_token(m, kv='f16'):
    # 27b 混合几何来自共享账本，直接委托保证两边公式同源；32b 是本篇历史全层架构，走本地公式。
    if m.get('rec_layers'):
        return hl.kv_per_token(G27, kv)
    return 2 * m['kv_layers'] * m['n_kv'] * m['head_dim'] * KV_BYTES[kv]


def rec_per_seq(m):
    """一条序列的 DeltaNet 状态字节数（F32）。混合模型委托共享账本，全层模型没有循环态，返回 0。"""
    if not m['rec_layers']:
        return 0
    return hl.rec_per_seq(G27)


def rec_total(m, n_seq, n_rs_seq):
    if not m['rec_layers']:
        return 0
    return hl.rec_total(G27, n_seq, n_rs_seq)


def budget(cuda_ctx):
    return (CARD_GIB - cuda_ctx - FIT_MARGIN_GIB) * GiB


def stack(key, quant, ctx, kv='f16', np_=1, mtp=0, draft=None, mmproj=False, cuda_ctx=0.5, compute=0.5, kvd='f16'):
    """返回 [(项, 字节)]，以及总数和预算。"""
    m = MODELS[key]
    rows = [('权重 ' + quant, QUANTS[key][quant] * GB),
            ('KV %s x %d' % (kv, ctx), kv_per_token(m, kv) * ctx)]
    if m['rec_layers']:
        rows.append(('DeltaNet 状态 %d 槽 x %d' % (np_, 1 + mtp), rec_total(m, np_, mtp)))
    rows.append(('compute buffer（假设）', compute * GiB))
    if mmproj:
        rows.append(('mmproj-F16', EXTRA['mmproj-F16'] * GB))
    if mtp:
        # MTP 头是一层完整的 Gated Attention，有自己的 KV，长度等于目标上下文（speculative.cpp）
        rows.append(('MTP 头 Q4_0', EXTRA['mtp-Q4_0'] * GB))
        rows.append(('MTP 头 KV', 2 * 1 * m['n_kv'] * m['head_dim'] * KV_BYTES[kv] * ctx))
    if draft:
        rows.append((draft, EXTRA[draft] * GB))
        if draft.startswith('Qwen3-0.6B'):
            d = DRAFT_06B
            # -ctkd / -ctvd 缺省 f16，长度等于目标上下文
            rows.append(('草稿 KV %s x %d' % (kvd, ctx), 2 * d['kv_layers'] * d['n_kv'] * d['head_dim'] * KV_BYTES[kvd] * ctx))
    total = sum(b for _, b in rows)
    return rows, total, budget(cuda_ctx)


def max_ctx(key, quant, kv='f16', np_=1, mtp=0, draft=None, mmproj=False, cuda_ctx=0.5, compute=0.5, kvd='f16'):
    """预算里能放下的最长上下文，按 256 向下取整，封顶 n_ctx_train。"""
    m = MODELS[key]
    _, fixed, bud = stack(key, quant, 0, kv, np_, mtp, draft, mmproj, cuda_ctx, compute, kvd)
    per = kv_per_token(m, kv)
    if mtp:
        per += 2 * m['n_kv'] * m['head_dim'] * KV_BYTES[kv]
    if draft and draft.startswith('Qwen3-0.6B'):
        d = DRAFT_06B
        per += 2 * d['kv_layers'] * d['n_kv'] * d['head_dim'] * KV_BYTES[kvd]
    room = bud - fixed
    if room <= 0:
        return 0
    return min(int(room // per) // 256 * 256, m['n_ctx_train'])


def eff():
    """7B 校准给的效率，以及 kingy 的 27B 数字反推的效率。"""
    out = {g: CAL[g] / (bw * GB / (CAL_MODEL_GIB * GiB)) for g, bw in GPUS.items()}
    out['kingy'] = KINGY['tg'] / (GPUS[KINGY['gpu']] / KINGY['file_gb'])
    return out


def tg_bound(file_gb, gpu, ctx_depth=0, key=None, kv='f16'):
    """全部在卡上时 decode 的带宽上限：每 token 读一遍权重，再读一遍已有的 KV。"""
    byt = file_gb * GB
    if key and ctx_depth:
        byt += kv_per_token(MODELS[key], kv) * ctx_depth
    return GPUS[gpu] * GB / byt


def tg_split(file_gb, gpu_frac, gpu, cpu_bw=89.6):
    """部分层留在内存：一步时间 = 卡上那部分 / 显存带宽 + 内存那部分 / 内存带宽。"""
    byt = file_gb * GB
    t = byt * gpu_frac / (GPUS[gpu] * GB) + byt * (1 - gpu_frac) / (cpu_bw * GB)
    return 1 / t


def spec_bound(accept_len, n_draft, c):
    """投机解码的加速上限：每轮出 accept_len 个 token，花 1 次目标前向加 n_draft 次草稿前向。
    c 是草稿一次前向相对目标的耗时比，带宽受限时近似为两者每 token 字节之比。验证 n_draft+1 个 token
    在带宽受限时和 1 个 token 差不多，这里按相同算，所以是上限。"""
    return accept_len / (1 + n_draft * c)


def cache_entry(key, ctx, kv, np_=1, mtp=0, n_ckpt=32):
    """一个槽存进提示缓存（-cram）的大小：主模型 KV + DeltaNet 状态 + 全部检查点 + MTP 头的 KV。
    server-task.cpp 存之前比大小，单条超过 -cram 就整条跳过。检查点只在有循环状态时创建。"""
    m = MODELS[key]
    rows = [('KV %s x %d' % (kv, ctx), kv_per_token(m, kv) * ctx)]
    if m['rec_layers']:
        rows.append(('DeltaNet 状态', rec_per_seq(m)))
        rows.append(('检查点 %d 个' % n_ckpt, rec_per_seq(m) * n_ckpt))
    if mtp:
        rows.append(('MTP 头 KV', 2 * m['n_kv'] * m['head_dim'] * KV_BYTES[kv] * ctx))
    return rows, sum(b for _, b in rows)


def collect(cuda_ctx=0.5, compute=0.5):
    e = eff()
    D = {'eff': e, 'kv_tok': {k: {kv: kv_per_token(m, kv) for kv in KV_BYTES} for k, m in MODELS.items()},
         'rec_seq': rec_per_seq(MODELS['27b']), 'budget_gib': budget(cuda_ctx) / GiB,
         'quants': QUANTS, 'extra': EXTRA, 'gpus': GPUS}
    D['ctx'] = {}
    for key in MODELS:
        for q in QUANTS[key]:
            for kv in ('f16', 'q8_0'):
                D['ctx'][(key, q, kv)] = max_ctx(key, q, kv, cuda_ctx=cuda_ctx, compute=compute)
    # 27B 的几种组合：单槽、4 槽、加 MTP、加 mmproj
    q = 'UD-Q4_K_XL'
    D['ctx27'] = {
        'np1': max_ctx('27b', q, 'q8_0', 1),
        'np4': max_ctx('27b', q, 'q8_0', 4),
        'np1_mtp2': max_ctx('27b', q, 'q8_0', 1, mtp=2),
        'np4_mtp2': max_ctx('27b', q, 'q8_0', 4, mtp=2),
        'np1_mtp2_mm': max_ctx('27b', q, 'q8_0', 1, mtp=2, mmproj=True),
    }
    D['rec'] = {(n, r): rec_total(MODELS['27b'], n, r) for n in (1, 4) for r in (0, 2, 3)}
    D['ctx32'] = {
        'q4km_f16': max_ctx('32b', 'Q4_K_M', 'f16'),
        'q4km_q8': max_ctx('32b', 'Q4_K_M', 'q8_0'),
        'iq4xs_q8': max_ctx('32b', 'IQ4_XS', 'q8_0'),
        'iq4xs_q8_d06': max_ctx('32b', 'IQ4_XS', 'q8_0', draft='Qwen3-0.6B-Q8_0'),
        'q4km_q8_d06': max_ctx('32b', 'Q4_K_M', 'q8_0', draft='Qwen3-0.6B-Q8_0'),
        'q4km_q8_e3': max_ctx('32b', 'Q4_K_M', 'q8_0', draft='eagle3-32B-bf16'),
    }
    # 账本堆叠图用的三组
    D['stacks'] = {
        '32b': stack('32b', 'Q4_K_M', 16384, 'q8_0')[0],
        '27b': stack('27b', 'UD-Q4_K_XL', 65536, 'q8_0', np_=1, mtp=2)[0],
        '27b_np4': stack('27b', 'UD-Q4_K_XL', 65536, 'q8_0', np_=4, mtp=2)[0],
    }
    D['tg'] = {}
    for key, qs in (('32b', ('IQ4_XS', 'Q4_K_M')), ('27b', ('UD-Q4_K_XL', 'UD-Q5_K_M'))):
        for q in qs:
            for g in GPUS:
                b = tg_bound(QUANTS[key][q], g)
                D['tg'][(key, q, g)] = (b, b * e[g], b * e['kingy'])
    # 32B Q5_K_M 放不下时卸几层：卡上放 90%，DDR5 双通道 5600 理论 89.6 GB/s
    D['split'] = {f: tg_split(QUANTS['32b']['Q5_K_M'], f, 'RTX 4090') for f in (1.0, 0.95, 0.9, 0.8)}
    # 深上下文对 decode 的拖累：32B Q4_K_M f16 KV 在 0、16K、32K 深度
    D['depth'] = {d: tg_bound(QUANTS['32b']['Q4_K_M'], 'RTX 4090', d, '32b') for d in (0, 16384, 32768)}
    c06 = EXTRA['Qwen3-0.6B-Q8_0'] / QUANTS['32b']['Q4_K_M']
    ce3 = EXTRA['eagle3-32B-bf16'] / QUANTS['32b']['Q4_K_M']
    cmtp = EXTRA['mtp-Q4_0'] / QUANTS['27b']['UD-Q4_K_XL']
    D['spec'] = {'c06': c06, 'ce3': ce3, 'cmtp': cmtp,
                 'e3_191': spec_bound(1.91, 2, ce3), 'd06_l2': spec_bound(2.0, 3, c06),
                 'd06_l2_c4': spec_bound(2.0, 3, c06 * 4), 'mtp_l18': spec_bound(1.8, 2, cmtp)}
    return D


def fmt_stack(rows, total, bud):
    for n, b in rows:
        print('  %-28s %7.2f GiB' % (n, b / GiB))
    print('  %-28s %7.2f GiB / 预算 %.2f GiB  %s' % ('合计', total / GiB, bud / GiB, 'OK' if total <= bud else '超出'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=MODELS)
    ap.add_argument('--quant')
    ap.add_argument('--ctx', type=int, default=32768)
    ap.add_argument('--kv', default='f16', choices=KV_BYTES)
    ap.add_argument('--np', type=int, default=1)
    ap.add_argument('--mtp', type=int, default=0, help='--spec-draft-n-max，0 表示不用 MTP')
    ap.add_argument('--draft', choices=[k for k in EXTRA if k not in ('mmproj-F16', 'mtp-Q4_0')])
    ap.add_argument('--mmproj', action='store_true')
    ap.add_argument('--kvd', default='f16', choices=KV_BYTES, help='草稿模型的 KV 类型（-ctkd/-ctvd）')
    ap.add_argument('--cache', type=int, metavar='N_CKPT', help='打印一条提示缓存的大小，N_CKPT 是检查点数（-ctxcp）')
    ap.add_argument('--cuda-ctx', type=float, default=0.5)
    ap.add_argument('--compute', type=float, default=0.5)
    ap.add_argument('--yarn', type=float, default=0, help='32B 的 YaRN 倍数，0 表示不开')
    ap.add_argument('--emit')
    a = ap.parse_args()
    set_yarn(a.yarn)
    if a.emit:
        D = collect(a.cuda_ctx, a.compute)
        with open(a.emit, 'w') as f:
            f.write('"""由 20-qwen-24g/ledger.py --emit 生成，不要手改。"""\n')
            for k, v in D.items():
                f.write('%s = %r\n' % (k.upper(), v))
        print('wrote', a.emit)
        return
    if a.model:
        q = a.quant or list(QUANTS[a.model])[7]
        if a.cache is not None:
            rows, total = cache_entry(a.model, a.ctx, a.kv, a.np, a.mtp, a.cache)
            for n, b in rows:
                print('  %-28s %7.2f GiB' % (n, b / GiB))
            print('  %-28s %7.2f GiB / -cram 缺省 8.00 GiB  %s' % ('一条缓存', total / GiB, 'OK' if total <= 8 * GiB else '超出，跳过'))
            return
        rows, total, bud = stack(a.model, q, a.ctx, a.kv, a.np, a.mtp, a.draft, a.mmproj, a.cuda_ctx, a.compute, a.kvd)
        fmt_stack(rows, total, bud)
        print('  最长上下文 %d' % max_ctx(a.model, q, a.kv, a.np, a.mtp, a.draft, a.mmproj, a.cuda_ctx, a.compute, a.kvd))
        return
    e = eff()
    print('预算 %.2f GiB（24 GiB - CUDA 上下文 %.1f - fit 余量 %.1f）' % (budget(a.cuda_ctx) / GiB, a.cuda_ctx, FIT_MARGIN_GIB))
    print('效率：' + '，'.join('%s %.2f' % kv for kv in e.items()))
    for key, m in MODELS.items():
        print('\n%s  KV/token f16 %.0f KiB  q8_0 %.0f KiB' % (m['name'], kv_per_token(m) / 1024, kv_per_token(m, 'q8_0') / 1024))
        for q, gb in QUANTS[key].items():
            c16, c8 = max_ctx(key, q, 'f16'), max_ctx(key, q, 'q8_0')
            b = tg_bound(gb, 'RTX 4090')
            print('  %-11s %6.2f GB  ctx f16 %7d  q8_0 %7d  4090 上限 %5.1f t/s，预测 %5.1f-%5.1f' % (
                q, gb, c16, c8, b, b * e['RTX 4090'], b * e['kingy']))
    print('\nDeltaNet 状态每条序列 %.1f MiB' % (rec_per_seq(MODELS['27b']) / MiB))


if __name__ == '__main__':
    sys.exit(main())
