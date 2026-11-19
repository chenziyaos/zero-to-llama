"""推理账本：启动前估算 decode（tg）、prompt 处理（pp）和并发 decode 的速度。

    python3 estimate_speed.py MODEL.gguf [--bw 35.5] [--depth 0] [--fa off] [--batch 1,2,4,8,16,32]
    python3 estimate_speed.py MODEL.gguf --raw          # 只给 Roofline 上限，不套校准系数
    python3 estimate_speed.py MODEL.gguf --ledger L.json --name q-Q4_K_M   # 没有 GGUF，用导出的账本

账本分两层。第一层是 Roofline 上限，全部从 GGUF 张量目录算出来：

  每 token 读的字节  W = 除 token_embd 以外所有张量的字节数（token_embd 走 GET_ROWS，只读一行）
                      + KV 读取 = 层数 x n_head_kv x head_dim x depth x (K + V 每元素字节)
  每 token 的运算量  F = 2 x 参与矩阵乘的参数量 + 注意力 4 x 层数 x n_head x head_dim x depth
                      pp 时 output 只算最后一个 token（llama-bench 只要最后一个 logits）
  上限              tg ≤ 带宽 / W；pp ≤ 峰值 / F

峰值按权重类型取：K-quant、legacy、i-quant 的 CPU 点积先把激活量化成 int8 再做整数点积
（ggml-cpu.c 里 Q4_K 的 vec_dot_type 是 Q8_K），对应 int8 峰值；BF16 对应 vdpbf16ps；F16、F32 对应 FP32 FMA。

第二层是本机校准，把上限换成预测，系数由 fit_ledger.py 从 logs/ 里的实测拟合出来：

  tg 一步  = max(W_权重 / bw, F / (eff_tg x 峰值)) + KV x depth / (bw x eff_attn) + t0
  pp       = eff_pp x 峰值 / F
  B 路并发 = 一步 tg 的时间 + (B - 1) x F / (eff_batch x 峰值)

t0 是每步固定开销（图构建、线程同步、采样），eff_* 是实际内核相对峰值的比例。
带宽和峰值默认读 logs/stream.jsonl 和 logs/peak.jsonl 的中位数，换机器先跑 measure.sh 和 fit_ledger.py。
"""
import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))

KV_BYTES = {'f32': 4.0, 'f16': 2.0, 'bf16': 2.0, 'q8_0': 34 / 32, 'q4_0': 18 / 32}
INT8_PATH = {'Q4_0', 'Q4_1', 'Q5_0', 'Q5_1', 'Q8_0', 'Q2_K', 'Q3_K', 'Q4_K', 'Q5_K', 'Q6_K',
             'IQ1_S', 'IQ1_M', 'IQ2_XXS', 'IQ2_XS', 'IQ2_S', 'IQ3_XXS', 'IQ3_S', 'IQ4_NL', 'IQ4_XS', 'TQ1_0', 'TQ2_0', 'MXFP4'}
CODEBOOK = {'IQ1_S', 'IQ1_M', 'IQ2_XXS', 'IQ2_XS', 'IQ2_S', 'IQ3_XXS', 'IQ3_S'}

# 本机校准（4 vCPU EPYC，llama.cpp 0ec4027，Qwen3-0.6B），fit_ledger.py 重新拟合后把结果抄回这里
CAL = {
    't0_ms': 3.6,          # 每步固定开销：sandwich.jsonl 联合拟合
    'eff_tg': 0.09,        # 单序列 decode 的整数点积效率，Q2_K 在快带宽时段封顶 86 t/s 定出
    'eff_tg_codebook': 0.044,  # IQ2、IQ3 查码本：IQ2_M 实测 60.7 t/s，按码本张量占 74% 反解
    'eff_pp': {'int8': 0.20, 'bf16': 0.42, 'fp32': 0.42},  # interleave.jsonl 的 pp512 中位数
    'eff_batch': 0.155,    # batched-q4km.md：每多一条序列一步多 4.66 ms
    'eff_attn': {'off': 0.50, 'on': 0.26},  # depth.jsonl：读 KV 的有效带宽比例，FA 开关分开
}


def machine(threads=4):
    """本机带宽和峰值：stream.jsonl 的 read 中位数（GB/s），peak.jsonl 最接近线程数那行的三种峰值（G op/s）。"""
    rows = [json.loads(x) for x in open(os.path.join(HERE, 'logs', 'stream.jsonl'))]
    bw = statistics.median(r['read_GBps'] for r in rows if r['threads'] == threads)
    pk = [json.loads(x) for x in open(os.path.join(HERE, 'logs', 'peak.jsonl'))]
    p = min(pk, key=lambda r: abs(r['threads'] - threads))
    return bw, {'int8': p['int8_vnni_GOPS'], 'bf16': p['bf16_dp_GFLOPS'], 'fp32': p['fp32_fma_GFLOPS']}


def path_of(type_name):
    if type_name in INT8_PATH:
        return 'int8'
    return 'bf16' if type_name == 'BF16' else 'fp32'


def ledger(g, depth=0, ctk='f16', ctv='f16'):
    """每 token 的字节和运算量，按 tg 和 pp 两种口径。g 是 gguf_lite.GGUF 或同接口的对象。"""
    L = g.hp('block_count')
    n_head = g.hp('attention.head_count')
    n_kv = g.hp('attention.head_count_kv')
    hd = g.hp('attention.key_length') or g.hp('embedding_length') // n_head
    mats = [t for t in g.tensors if len(t.shape) >= 2 and t.name != 'token_embd.weight']
    w_bytes = sum(t.nbytes for t in g.tensors if t.name != 'token_embd.weight')
    out = [t for t in mats if t.name == 'output.weight']
    if not out:  # 共享词表：output 复用 token_embd，tg 时要整张读
        out = [t for t in g.tensors if t.name == 'token_embd.weight']
        w_bytes += out[0].nbytes
        mats = mats + out
    p_all = sum(t.n_elems for t in mats)
    p_out = out[0].n_elems
    kv_tok = L * n_kv * hd * (KV_BYTES[ctk] + KV_BYTES[ctv])
    attn = 4 * L * n_head * hd * depth
    paths, codebook = {}, 0
    for t in mats:
        k = path_of(t.type_name)
        paths[k] = paths.get(k, 0) + t.n_elems
        codebook += t.n_elems if t.type_name in CODEBOOK else 0
    return {'w_bytes': w_bytes, 'kv_bytes': kv_tok * depth, 'tg_bytes': w_bytes + kv_tok * depth,
            'tg_flops': 2 * p_all + attn, 'pp_flops': 2 * (p_all - p_out) + attn,
            'path': max(paths, key=paths.get), 'codebook_frac': codebook / p_all,
            'params_mat': p_all, 'kv_per_token': kv_tok, 'depth': depth,
            'n_layer': L, 'n_head': n_head, 'head_dim': hd}


def at_depth(led, depth):
    """把 depth=0 的账本挪到另一个深度，供没有 GGUF 时用导出的账本。"""
    d = dict(led)
    attn = 4 * led['n_layer'] * led['n_head'] * led['head_dim']
    d['kv_bytes'] = led['kv_per_token'] * depth
    d['tg_bytes'] = led['w_bytes'] + d['kv_bytes']
    d['tg_flops'] = led['tg_flops'] - attn * led['depth'] + attn * depth
    d['pp_flops'] = led['pp_flops'] - attn * led['depth'] + attn * depth
    d['depth'] = depth
    return d


def bounds(led, bw, peak, batch=(1,)):
    """第一层：纯 Roofline 上限。"""
    pk = peak[led['path']] * 1e9
    bw = bw * 1e9
    r = {'tg_bw': bw / led['tg_bytes'], 'tg_compute': pk / led['tg_flops'], 'pp_compute': pk / led['pp_flops'],
         'intensity': led['tg_flops'] / led['tg_bytes'], 'ridge': pk / bw}
    r['tg'] = min(r['tg_bw'], r['tg_compute'])
    r['batch'] = []
    for B in batch:
        t_mem = (led['w_bytes'] + B * led['kv_bytes']) / bw
        t_cmp = B * led['tg_flops'] / pk
        r['batch'].append((B, B / max(t_mem, t_cmp), 'mem' if t_mem >= t_cmp else 'compute'))
    per = led['tg_flops'] / pk - led['kv_bytes'] / bw
    r['b_star'] = led['w_bytes'] / bw / per if per > 0 else float('inf')
    return r


def predict(led, bw, peak, batch=(1,), fa='off', cal=CAL):
    """第二层：套上本机校准系数的预测值。"""
    pk = peak[led['path']] * 1e9
    bw = bw * 1e9
    eff_tg = cal['eff_tg'] + (cal['eff_tg_codebook'] - cal['eff_tg']) * led['codebook_frac']
    t_w = led['w_bytes'] / bw
    t_c = led['tg_flops'] / (eff_tg * pk)
    t_kv = led['kv_bytes'] / (bw * cal['eff_attn'][fa])
    t0 = cal['t0_ms'] / 1e3
    step = max(t_w, t_c) + t_kv + t0
    r = {'tg': 1 / step, 'tg_limit': 'mem' if t_w >= t_c else 'compute',
         'parts_ms': {'weights': t_w * 1e3, 'compute_floor': t_c * 1e3, 'kv': t_kv * 1e3, 't0': t0 * 1e3},
         'pp': cal['eff_pp'][led['path']] * pk / led['pp_flops'], 'batch': []}
    per = led['tg_flops'] / (cal['eff_batch'] * pk)
    for B in batch:
        t = step + (B - 1) * (per + t_kv)
        r['batch'].append((B, B / t))
    r['batch_ceiling'] = 1 / (per + t_kv)
    r['b_half'] = step / (per + t_kv)  # 并发吞吐到达天花板一半时的 B
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model', nargs='?')
    ap.add_argument('--bw', type=float, help='可持续读带宽 GB/s，默认取 logs/stream.jsonl 中位数')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--depth', type=int, default=0)
    ap.add_argument('--fa', choices=['on', 'off'], default='off')
    ap.add_argument('--ctk', default='f16')
    ap.add_argument('--ctv', default='f16')
    ap.add_argument('--batch', default='1,2,4,8,16,32')
    ap.add_argument('--raw', action='store_true', help='只给 Roofline 上限')
    ap.add_argument('--ledger', help='fit_ledger.py --dump 导出的账本 JSON，没有 GGUF 时用')
    ap.add_argument('--name', help='--ledger 里的条目名')
    a = ap.parse_args()
    bw, peak = machine(a.threads)
    bw = a.bw or bw
    if a.ledger:
        led = at_depth(json.load(open(a.ledger))[a.name], a.depth)
        name = a.name
    else:
        from gguf_lite import GGUF
        led = ledger(GGUF(a.model), a.depth, a.ctk, a.ctv)
        name = os.path.basename(a.model)
    batch = [int(x) for x in a.batch.split(',')]
    r = bounds(led, bw, peak, batch)
    MB = 1e6
    print(f'model {name}  path {led["path"]}  bw {bw:.1f} GB/s  peak {peak[led["path"]]:.0f} G op/s')
    print(f'  bytes/token  weights {led["w_bytes"] / MB:.1f} MB + KV {led["kv_bytes"] / MB:.1f} MB (depth {a.depth})')
    print(f'  flops/token  tg {led["tg_flops"] / 1e9:.3f} G   pp {led["pp_flops"] / 1e9:.3f} G')
    print(f'  intensity {r["intensity"]:.2f} op/B   ridge {r["ridge"]:.1f} op/B')
    print(f'  roofline  tg ≤ {r["tg"]:.1f} t/s (bw {r["tg_bw"]:.1f}, compute {r["tg_compute"]:.0f})   pp ≤ {r["pp_compute"]:.0f} t/s')
    if a.raw:
        for B, tps, lim in r['batch']:
            print(f'    B={B:<3d} total {tps:7.1f} t/s  per-seq {tps / B:6.1f}  {lim}')
        return
    p = predict(led, bw, peak, batch, a.fa)
    pm = p['parts_ms']
    print(f'  predict   tg {p["tg"]:.1f} t/s ({p["tg_limit"]}-bound)   pp {p["pp"]:.0f} t/s')
    print(f'    step = max(weights {pm["weights"]:.2f}, compute {pm["compute_floor"]:.2f}) + kv {pm["kv"]:.2f} + t0 {pm["t0"]:.2f} ms')
    print(f'    batched ceiling {p["batch_ceiling"]:.0f} t/s, half of it at B ≈ {p["b_half"]:.1f}')
    for B, tps in p['batch']:
        print(f'    B={B:<3d} total {tps:7.1f} t/s  per-seq {tps / B:6.1f}')


if __name__ == '__main__':
    main()
