#!/usr/bin/env python3
"""plan.py：DeepSeek-V4.1-Flash 的 ISTA GSQ-RCO GGUF 装进一台机器的账，全部是推算。

字节账：
- 每个张量的形状照 llama.cpp 0ec4027 的 src/models/deepseek41.cpp load_arch_tensors，结构参数取自
  config.json 的 text_config；每个张量的量化类型读 alloc/<build>.txt（发布者随模型附带的 RCO 分配表）。
- 字节数 = 元素数 x 每元素位数 / 8，位数取自 ggml-common.h 的块大小（例如 IQ2_XXS 每 256 个元素 66 字节）。
- 核对：算出来的骨干（除 Engram 表以外的全部张量）位宽要等于分配表头里的 measured bpw，
  字节数要和模型卡的 120.72 / 138.24 / 155.23 GB 对得上（差的是 GGUF 头和元数据）。

放置：
- -ngl 99 时 token_embd 留在 CPU（输入层固定在 CPU），per_layer_token_embd（Engram 表）是 TENSOR_READ_LAZY，
  --lazy-mode auto 对超过 4 GiB 的张量按需从 mmap 读行，不占显存，也不常驻内存。
- -ot 'blk\\.(N|...)\\.ffn_(up|down|gate)_exps=CPU' 把一部分层的路由专家留在内存，其余张量全在卡上。

KV 账（llama-kv-cache-dsv4.cpp）：
- 原始 KV：40 层都是 SWA（窗口 128），只存 K（512 维），每条序列 GGML_PAD(min(ctx, 128 + n_ubatch), 256) 格。
- CSA 压缩 KV：只有 kv 源层 2、8、14 存，每层 GGML_PAD(ceil(ctx/2), 256) 格，每格 512 + 128 = 640 维。
- HCA 压缩 KV：只有 kv 源层 20 存，每层 GGML_PAD(ctx, 256) 格，每格 640 维。
- 压缩器状态：F32，CSA 源层每层 2 x 2 x 512，HCA 源层 2 x 1 x 512，乘 n_seq x (1 + n_rs_seq)。
- 这几块缓存不按 kv_unified 合并：-np 4 时每个槽各有一份完整长度，所以 llama-server 缺省的 -np auto
  会把压缩 KV 乘 4。-ctk 和 -ctv 必须相同（llama-context.cpp 对 DEEPSEEK41 检查）。

速度：
- decode 一步读的字节 = 卡上的稠密权重 + 每层 6/384 的路由专家。专家在卡上走显存带宽，在内存走内存带宽。
- 模型 t = t0 + 内存侧字节 / 内存侧吞吐。t0 和吞吐用 HumanjavaEnterprises/deepseek-v41-upstream 的
  RECIPE.md 扫描（社区实测，RTX PRO 6000 Max-Q + Xeon w5-3525 八通道 DDR5-4800，16 线程）做最小二乘拟合。
- 换机器时给一个区间：一端假设吞吐按内存带宽等比缩放（eff 不变），另一端假设吞吐不变、只受内存带宽封顶，
  小的当低端，大的当高端。两端都是假设，给的是量级，不是承诺。

没有 GPU 实测。显存峰值里算不准的 compute buffer 和 CUDA 上下文，用 RECIPE 的 85.9 峰值反推一个"其余"项，
再拿 96.1 那个点核对。RECIPE 写的是 GB，按 GB 算 28 层那个点的其余项为负，按 GiB 算两点一致，所以按 GiB 读。

用法：
  python3 plan.py                                   # 三档量化的字节核对、放置表、KV 表、速度拟合
  python3 plan.py --build IQ2_XXS --gpu-layers 25 --ctx 65536 --kv f16 --ub 2048 --card 96
  python3 plan.py --build IQ2_XXS --gpu-layers 0 --ctx 16384 --kv q8_0 --card 24 --ram-bw 89.6
  python3 plan.py --fit-card 24 --build IQ2_XXS --ctx 32768 --kv q8_0      # 给定显存最多放几层专家
  python3 plan.py --emit gen21_data.py
"""
import argparse
import functools
import os
import sys

GiB = 2 ** 30
MiB = 2 ** 20
GB = 1e9

HERE = os.path.dirname(os.path.abspath(__file__))
BUILDS = ('IQ1_M', 'IQ2_XXS', 'IQ2_XS')

# 每元素位数，按 ggml-common.h 的块大小换算
BPW = {'F32': 32.0, 'BF16': 16.0, 'F16': 16.0, 'Q8_0': 34 * 8 / 32, 'Q5_K': 176 * 8 / 256, 'IQ4_NL': 18 * 8 / 32,
       'IQ3_XXS': 98 * 8 / 256, 'IQ2_S': 82 * 8 / 256, 'IQ2_XS': 74 * 8 / 256, 'IQ2_XXS': 66 * 8 / 256,
       'IQ1_M': 56 * 8 / 256, 'IQ1_S': 50 * 8 / 256}
KV_BYTES = {'f16': 2.0, 'bf16': 2.0, 'q8_0': 34 / 32}

# config.json text_config
CFG = dict(n_embd=5120, n_layer=40, n_head=64, head_dim=512, n_vocab=129280, n_ff_exp=2304, n_expert=384,
           n_used=6, n_shared=1, q_lora=1280, o_lora=1024, o_groups=8, hc=4, idx_heads=32, idx_dim=128,
           ple_heads=(4 - 1) * 8, ple_dim=256, n_swa=128,
           ratios=[0, 0] + [2] * 18 + [1] * 20, kv_src=(2, 8, 14, 20), idx_src=(2, 8, 14, 20, 24, 28, 32, 36),
           engram=(1, 14))

# 模型卡（发布者口径）：bpw、骨干 GB、Engram 文件 GB；Engram 表 196.6B 参数，IQ4_NL
PUBLISHED = {'IQ1_M': (1.75, 120.72), 'IQ2_XXS': (2.00, 138.24), 'IQ2_XS': (2.25, 155.23)}
ENGRAM_GB = 110.60
ENGRAM_PARAMS = 196.6e9
MMPROJ_GB = 0.97

# 社区实测：HumanjavaEnterprises/deepseek-v41-upstream RECIPE.md，IQ2_XXS 重打包（权重逐字节不变），
# llama-bench tg128，16 线程。键是放到卡上的专家层数（0-9 即 10 层），值是 decode t/s；39-40 取 39.5
SWEEP = {0: 18.5, 10: 21.8, 16: 26.2, 22: 31.9, 25: 33.5, 27: 36.3, 28: 39.5}
# 同一来源的显存峰值，64K 上下文，-ub 2048，-np 1，KV 缺省 f16
PEAK = {25: 85.9, 28: 96.1}
PEAK_CTX = 65536
PEAK_UB = 2048

# 带宽，GB/s：Xeon w5-3525 八通道 DDR5-4800 理论值 8 x 4800 x 8 / 1000；RTX PRO 6000 Max-Q 取 NVIDIA 规格页
RAM_BW = {'8ch DDR5-4800': 8 * 4800 * 8 / 1000, '2ch DDR5-5600': 2 * 5600 * 8 / 1000,
          '4ch DDR5-5600': 4 * 5600 * 8 / 1000, '12ch DDR5-6400': 12 * 6400 * 8 / 1000}
GPU_BW = {'RTX PRO 6000 Max-Q': 1792, 'RTX 4090': 1008, 'RTX 3090': 936}
HJE_RAM = '8ch DDR5-4800'
HJE_GPU = 'RTX PRO 6000 Max-Q'


def pad(x, n=256):
    return (x + n - 1) // n * n


def shape_elems(base):
    """base 是去掉 blk.N. 前缀的张量名，返回元素数。"""
    c = CFG
    e, h, hd = c['n_embd'], c['n_head'], c['head_dim']
    hcd, hmix = c['hc'] * e, (2 + c['hc']) * c['hc']
    table = {
        'output.weight': e * c['n_vocab'], 'token_embd.weight': e * c['n_vocab'], 'output_norm.weight': e,
        'attn_norm.weight': e, 'attn_sinks.weight': h, 'attn_q_a.weight': e * c['q_lora'],
        'attn_q_a_norm.weight': c['q_lora'], 'attn_q_b.weight': c['q_lora'] * h * hd, 'attn_kv.weight': e * hd,
        'attn_kv_a_norm.weight': hd, 'attn_output_a.weight': h * hd // c['o_groups'] * c['o_lora'] * c['o_groups'],
        'attn_output_b.weight': c['o_groups'] * c['o_lora'] * e,
        'hc_attn_fn.weight': hcd * hmix, 'hc_attn_base.weight': hmix, 'hc_attn_scale.weight': 3,
        'hc_ffn_fn.weight': hcd * hmix, 'hc_ffn_base.weight': hmix, 'hc_ffn_scale.weight': 3,
        'attn_compressor_kv.weight': e * hd, 'attn_compressor_norm.weight': hd, 'attn_compressor_gate.weight': e * hd,
        'indexer.attn_k.weight': hd * c['idx_dim'], 'indexer.k_norm.weight': c['idx_dim'],
        'indexer.attn_q_b.weight': c['q_lora'] * c['idx_heads'] * c['idx_dim'],
        'indexer.proj.weight': e * c['idx_heads'],
        'engram_kv.weight': c['ple_heads'] * c['ple_dim'] * e * (c['hc'] + 1), 'engram_gate.weight': e * c['hc'],
        'ffn_gate_inp.weight': e * c['n_expert'], 'exp_probs_b.bias': c['n_expert'], 'exp_probs_b_vl.bias': c['n_expert'],
        'ffn_norm.weight': e,
        'ffn_gate_exps.weight': e * c['n_ff_exp'] * c['n_expert'], 'ffn_up_exps.weight': e * c['n_ff_exp'] * c['n_expert'],
        'ffn_down_exps.weight': c['n_ff_exp'] * e * c['n_expert'],
        'ffn_gate_shexp.weight': e * c['n_ff_exp'] * c['n_shared'], 'ffn_up_shexp.weight': e * c['n_ff_exp'] * c['n_shared'],
        'ffn_down_shexp.weight': c['n_ff_exp'] * c['n_shared'] * e,
        'per_layer_token_embd.weight': int(ENGRAM_PARAMS),
    }
    return table[base]


def load_alloc(build):
    out = {}
    with open(os.path.join(HERE, 'alloc', build + '.txt')) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            name, typ = line.split(': ')
            out[name] = typ
        return out


def header(build):
    """分配表头里的 measured bpw。"""
    with open(os.path.join(HERE, 'alloc', build + '.txt')) as f:
        for line in f:
            if 'measured:' in line:
                return float(line.split('measured:')[1].strip(' )\n'))
    return None


def tensors(build):
    """[(名字, 层号或 None, 类别, 类型, 元素数, 字节数)]。类别：exps、engram_table、tok_embd、dense。"""
    rows = []
    for name, typ in load_alloc(build).items():
        layer = None
        base = name
        if name.startswith('blk.'):
            _, n, base = name.split('.', 2)
            layer = int(n)
        n_el = shape_elems(base)
        if base.endswith('_exps.weight'):
            kind = 'exps'
        elif base == 'per_layer_token_embd.weight':
            kind = 'engram_table'
        elif base == 'token_embd.weight':
            kind = 'tok_embd'
        else:
            kind = 'dense'
        rows.append((name, layer, kind, typ, n_el, n_el * BPW[typ] / 8))
    return rows


@functools.lru_cache(maxsize=None)
def summary(build):
    """骨干元素数、字节、位宽，按类别分的字节，和每层专家字节。"""
    T = tensors(build)
    bb = [t for t in T if t[2] != 'engram_table']
    n_el = sum(t[4] for t in bb)
    byt = sum(t[5] for t in bb)
    by_kind = {}
    for t in T:
        by_kind[t[2]] = by_kind.get(t[2], 0) + t[5]
    exp_layer = [0.0] * CFG['n_layer']
    for t in T:
        if t[2] == 'exps':
            exp_layer[t[1]] += t[5]
    return dict(params=n_el, bytes=byt, bpw=byt * 8 / n_el, by_kind=by_kind, exp_layer=exp_layer,
                engram_bytes=by_kind['engram_table'], measured_bpw=header(build), published=PUBLISHED[build])


def placement(build, gpu_layers):
    """前 gpu_layers 层的专家放卡上（HJE 的 -ot 写法），返回卡上、内存、磁盘三处的权重字节。"""
    s = summary(build)
    k = s['by_kind']
    exp_gpu = sum(s['exp_layer'][:gpu_layers])
    exp_cpu = sum(s['exp_layer'][gpu_layers:])
    return dict(gpu=k['dense'] + exp_gpu, cpu=k['tok_embd'] + exp_cpu, disk=k['engram_table'],
                dense=k['dense'], exp_gpu=exp_gpu, exp_cpu=exp_cpu, tok_embd=k['tok_embd'])


def kv_ledger(ctx, kv='f16', n_seq=1, ub=512, n_rs_seq=0):
    """[(项, 字节)]。ctx 是每个槽的长度（-c 65536 -np 1 时就是 65536；-np 4 时 kv_unified 下仍是 65536）。"""
    c = CFG
    b = KV_BYTES[kv]
    comp = c['head_dim'] + c['idx_dim']
    csa_src = [i for i in c['kv_src'] if c['ratios'][i] == 2]
    hca_src = [i for i in c['kv_src'] if c['ratios'][i] == 1]
    raw_cells = pad(min(pad(ctx), c['n_swa'] + ub))
    planes = n_seq * (1 + n_rs_seq)
    return [
        ('原始 KV（SWA，40 层，%d 格）' % raw_cells, c['n_layer'] * raw_cells * c['head_dim'] * b * n_seq),
        ('CSA 压缩 KV（%d 层，%d 格）' % (len(csa_src), pad(-(-ctx // 2))), len(csa_src) * pad(-(-ctx // 2)) * comp * b * n_seq),
        ('HCA 压缩 KV（%d 层，%d 格）' % (len(hca_src), pad(ctx)), len(hca_src) * pad(ctx) * comp * b * n_seq),
        ('压缩器状态（F32）', (len(csa_src) * 2 * 2 * c['head_dim'] + len(hca_src) * 2 * 1 * c['head_dim']) * 4 * planes),
    ]


def kv_total(*a, **k):
    return sum(b for _, b in kv_ledger(*a, **k))


def kv_per_token(kv='f16'):
    """长上下文时每多一个 token 多出的字节：CSA 3 层各 1/2 格，HCA 1 层 1 格，每格 640 维。"""
    c = CFG
    comp = c['head_dim'] + c['idx_dim']
    return (3 / 2 + 1) * comp * KV_BYTES[kv]


@functools.lru_cache(maxsize=None)
def other_from_peak(unit=GiB):
    """RECIPE 峰值反推的"其余"：峰值 - 卡上权重 - KV。返回 {层数: 其余字节}，unit 是峰值数字的单位假设。
    按 GB 读，28 层那个点的其余项是负数，不可能；按 GiB 读（nvidia-smi 显示 MiB），两个点得到的其余项
    相差不到 0.2 GB，所以缺省按 GiB。"""
    out = {}
    for n, pk in PEAK.items():
        p = placement('IQ2_XXS', n)
        out[n] = pk * unit - p['gpu'] - kv_total(PEAK_CTX, 'f16', 1, PEAK_UB)
    return out


def active_bytes(build, gpu_layers):
    """decode 一个 token 读的字节：卡上 = 稠密 + 卡上专家 x 6/384；内存 = 内存专家 x 6/384 + 一行 embedding。"""
    p = placement(build, gpu_layers)
    f = CFG['n_used'] / CFG['n_expert']
    return dict(gpu=p['dense'] + p['exp_gpu'] * f, cpu=p['exp_cpu'] * f + p['tok_embd'] / CFG['n_vocab'])


@functools.lru_cache(maxsize=None)
def fit_sweep():
    """t = t0 + cpu_bytes / (BW x eff) 的最小二乘：返回 t0（秒）、eff、残差。"""
    xs, ys = [], []
    for n, tg in SWEEP.items():
        xs.append(active_bytes('IQ2_XXS', n)['cpu'])
        ys.append(1 / tg)
    m = len(xs)
    mx, my = sum(xs) / m, sum(ys) / m
    k = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    t0 = my - k * mx
    eff = 1 / (k * RAM_BW[HJE_RAM] * GB)
    res = {n: 1 / (t0 + x * k) for n, x in zip(SWEEP, xs)}
    return dict(t0=t0, k=k, eff=eff, pred=res)


def gpu_floor(build=None):
    """t0 里卡上部分的理论下限：全部稠密权重按显存带宽读一遍。剩下的是 CPU/GPU 往返和调度开销。"""
    p = active_bytes(build or 'IQ2_XXS', 0)
    return p['gpu'] / (GPU_BW[HJE_GPU] * GB)


def cpu_rate(fit=None):
    """拟合出的内存侧实际吞吐，字节/秒：HJE 机器 16 线程每秒能处理多少内存专家字节。"""
    fit = fit or fit_sweep()
    return 1 / fit['k']


def predict_tg(build, gpu_layers, ram_bw=None, fit=None):
    """decode 区间 (低, 高)，t/s。两种外推各给一个数，取小的当低端、大的当高端：
    A. eff 不变：内存侧吞吐按内存带宽等比缩放，适合 CPU 核数够、瓶颈在带宽的机器。
    B. 吞吐不变：内存侧吞吐等于 HJE 的拟合值，再被内存带宽封顶，适合带宽更高但核数没多的机器。
    依据：HJE 8 线程 11.8、16 线程 18.5、24/32 线程 15.3 t/s，那台机器的 CPU 侧既吃核数也吃带宽，
    单用一种外推都会偏。两端都是假设。"""
    fit = fit or fit_sweep()
    bw = (ram_bw if ram_bw else RAM_BW[HJE_RAM]) * GB
    cpu = active_bytes(build, gpu_layers)['cpu']
    a = 1 / (fit['t0'] + cpu / (bw * fit['eff']))
    b = 1 / (fit['t0'] + cpu / min(cpu_rate(fit), bw))
    return min(a, b), max(a, b)


def vram(build, gpu_layers, ctx, kv='f16', ub=512, n_seq=1, other=None, mmproj=False):
    """[(项, 字节)]，other 是 compute buffer 加 CUDA 上下文，缺省取 RECIPE 25 层那个点反推的值。"""
    p = placement(build, gpu_layers)
    if other is None:
        other = other_from_peak()[25]
    rows = [('稠密权重（注意力、共享专家、mHC、Engram 投影、输出头）', p['dense']),
            ('路由专家 %d 层' % gpu_layers, p['exp_gpu']),
            ('KV %s，%d x %d 槽' % (kv, ctx, n_seq), kv_total(ctx, kv, n_seq, ub)),
            ('compute buffer 等（RECIPE 峰值反推）', other)]
    if mmproj:
        rows.append(('mmproj BF16', MMPROJ_GB * GB))
    return rows


def fit_card(build, card_gib, ctx, kv='f16', ub=512, n_seq=1, margin_gib=1.0, other=None):
    """卡上最多放几层专家：从 0 层往上加，总量不超过 card - margin。显卡标的 24 GB、96 GB 实际是 GiB
    （nvidia-smi 显示 MiB），所以 card 按 GiB；margin 和 llama.cpp fit 缺省的 1024 MiB 一致。"""
    best = -1
    for n in range(CFG['n_layer'] + 1):
        tot = sum(b for _, b in vram(build, n, ctx, kv, ub, n_seq, other))
        if tot <= (card_gib - margin_gib) * GiB:
            best = n
    return best


def collect():
    D = {'bpw_types': BPW, 'published': PUBLISHED, 'engram_gb': ENGRAM_GB, 'ram_bw': RAM_BW, 'gpu_bw': GPU_BW,
         'sweep': SWEEP, 'peak': PEAK}
    D['summary'] = {b: {k: v for k, v in summary(b).items() if k != 'exp_layer'} for b in BUILDS}
    D['exp_layer'] = {b: summary(b)['exp_layer'] for b in BUILDS}
    D['place'] = {(b, n): placement(b, n) for b in BUILDS for n in (0, 10, 16, 22, 25, 28, 40)}
    D['kv'] = {(ctx, kv, ns): kv_total(ctx, kv, ns, 512) for ctx in (16384, 65536, 131072, 262144, 1048576)
               for kv in ('f16', 'q8_0') for ns in (1, 4)}
    D['kv_rows_64k'] = kv_ledger(65536, 'f16', 1, 2048)
    D['kv_rows_1m'] = kv_ledger(1048576, 'q8_0', 1, 512)
    D['kv_tok'] = {kv: kv_per_token(kv) for kv in KV_BYTES}
    D['other'] = {'GB': other_from_peak(GB), 'GiB': other_from_peak(GiB)}
    f = fit_sweep()
    D['fit'] = f
    D['gpu_floor'] = gpu_floor()
    D['active'] = {(b, n): active_bytes(b, n) for b in BUILDS for n in (0, 25, 40)}
    D['cpu_rate'] = cpu_rate(f)
    D['tg'] = {(b, n, bwk): predict_tg(b, n, RAM_BW[bwk], f)
               for b in BUILDS for n in (0, 3, 6, 11, 25, 29, 40) for bwk in RAM_BW}
    D['threads'] = {8: 11.8, 16: 18.5, 24: 15.3, 32: 15.3}
    D['vram_96'] = vram('IQ2_XXS', 25, 65536, 'f16', 2048)
    D['vram_24'] = vram('IQ2_XXS', 0, 16384, 'q8_0', 2048)
    D['fit_card'] = {(b, card, ctx, kv): fit_card(b, card, ctx, kv, 2048) for b in BUILDS for card in (24, 32, 48, 96)
                     for ctx, kv in ((16384, 'q8_0'), (65536, 'f16'))}
    D['ram_need'] = {(b, n): placement(b, n)['cpu'] for b in BUILDS for n in (0, 4, 25)}
    return D


def gb(x):
    return '%.2f GB' % (x / GB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', choices=BUILDS, default='IQ2_XXS')
    ap.add_argument('--gpu-layers', type=int, help='前 N 层的路由专家放卡上，其余留内存')
    ap.add_argument('--ctx', type=int, default=65536)
    ap.add_argument('--kv', default='f16', choices=KV_BYTES)
    ap.add_argument('--np', type=int, default=1)
    ap.add_argument('--ub', type=int, default=2048)
    ap.add_argument('--card', type=float, default=96, help='显存，GiB（卡上标的 GB 数）')
    ap.add_argument('--ram-bw', type=float, help='内存带宽，GB/s，缺省是 RECIPE 机器的 307.2')
    ap.add_argument('--other', type=float, help='compute buffer 等，GB，缺省按 RECIPE 峰值反推')
    ap.add_argument('--fit-card', type=float, metavar='GiB', help='给定显存最多放几层专家')
    ap.add_argument('--mmproj', action='store_true')
    ap.add_argument('--emit')
    a = ap.parse_args()
    other = a.other * GB if a.other is not None else None
    if a.emit:
        D = collect()
        with open(a.emit, 'w') as f:
            f.write('"""由 21-deepseek-v41/plan.py --emit 生成，不要手改。"""\n')
            for k, v in D.items():
                f.write('%s = %r\n' % (k.upper(), v))
        print('wrote', a.emit)
        return
    if a.fit_card:
        n = fit_card(a.build, a.fit_card, a.ctx, a.kv, a.ub, a.np, other=other)
        print('%s，%g GiB 卡，%d 上下文 %s KV：最多 %d 层专家上卡' % (a.build, a.fit_card, a.ctx, a.kv, n))
        if n >= 0:
            print('  内存要放 %s，decode 推算 %.1f-%.1f t/s' % (gb(placement(a.build, n)['cpu']),
                  *predict_tg(a.build, n, a.ram_bw)))
        return
    if a.gpu_layers is not None:
        rows = vram(a.build, a.gpu_layers, a.ctx, a.kv, a.ub, a.np, other, a.mmproj)
        tot = sum(b for _, b in rows)
        for n, b in rows:
            print('  %-44s %9s' % (n, gb(b)))
        print('  %-44s %9s（%.2f GiB）/ %g GiB  %s' % ('显存合计', gb(tot), tot / GiB, a.card,
              'OK' if tot <= (a.card - 1) * GiB else '超出（留 1 GiB 余量）'))
        p = placement(a.build, a.gpu_layers)
        print('  内存：token_embd + %d 层专家 %s；磁盘：Engram 表 %s' % (40 - a.gpu_layers, gb(p['cpu']), gb(p['disk'])))
        print('  decode 推算 %.1f-%.1f t/s（内存带宽 %.1f GB/s）' % (*predict_tg(a.build, a.gpu_layers, a.ram_bw), a.ram_bw or RAM_BW[HJE_RAM]))
        return
    for b in BUILDS:
        s = summary(b)
        print('%-8s 骨干 %.1fB 参数  %s（卡上写 %.2f GB）  位宽 %.4f（表头 %.4f）  专家 %s  稠密 %s  token_embd %s' % (
            b, s['params'] / 1e9, gb(s['bytes']), s['published'][1], s['bpw'], s['measured_bpw'],
            gb(s['by_kind']['exps']), gb(s['by_kind']['dense']), gb(s['by_kind']['tok_embd'])))
    print('Engram 表 %s（卡上写 %.2f GB）' % (gb(summary('IQ2_XXS')['engram_bytes']), ENGRAM_GB))
    print('\nKV（-np 1，ub 512）：')
    for ctx in (16384, 65536, 262144, 1048576):
        print('  %8d  f16 %s  q8_0 %s' % (ctx, gb(kv_total(ctx, 'f16')), gb(kv_total(ctx, 'q8_0'))))
    print('  每 token：f16 %.0f B，q8_0 %.0f B（模型卡 FP4 口径 890 B）' % (kv_per_token('f16'), kv_per_token('q8_0')))
    o = other_from_peak()
    print('\n峰值反推的其余项（峰值按 GiB 读）：' + '，'.join('%d 层 %s' % (n, gb(v)) for n, v in o.items()))
    f = fit_sweep()
    print('速度拟合：t0 %.1f ms（卡上读稠密权重的理论下限 %.1f ms），内存侧吞吐 %.1f GB/s，eff %.2f' % (
        f['t0'] * 1e3, gpu_floor() * 1e3, cpu_rate(f) / GB, f['eff']))
    for n, tg in SWEEP.items():
        print('  %2d 层上卡  实测 %.1f  拟合 %.1f t/s' % (n, tg, f['pred'][n]))


if __name__ == '__main__':
    sys.exit(main())
