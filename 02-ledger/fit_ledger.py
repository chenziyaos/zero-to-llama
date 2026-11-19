"""从 logs/ 里的实测拟合 estimate_speed.py 的校准系数，并导出账本供测试和无 GGUF 环境使用。

    python3 fit_ledger.py                      # 用 logs/ledgers.json 里的账本拟合，打印系数和逐点对账
    python3 fit_ledger.py --models ../../models --dump logs/ledgers.json   # 有 GGUF 时重新导出账本

五组数据各定一个系数，互不耦合：
  sandwich.jsonl   tg 前后各测一次 STREAM，按当时带宽联合拟合 t0 和 eff_tg（网格搜索，最小化平均相对误差）
  interleave.jsonl pp512 的中位数，除以峰值和每 token 运算量得 eff_pp
  depth.jsonl      同一模型不同深度的 tg，每步多出的时间对 KV 字节 / 带宽做过原点回归，得 eff_attn
  batched-q4km.md  llama-batched-bench 的每步耗时对 B 做线性回归，斜率换算成 eff_batch
  codebook         IQ2_M 在 interleave 里的 tg 中位数，按码本张量占比反解 eff_tg_codebook
"""
import argparse
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from estimate_speed import CAL, at_depth, machine  # noqa: E402

LOGS = os.path.join(HERE, 'logs')


def jl(name):
    return [json.loads(x) for x in open(os.path.join(LOGS, name))]


def name_of(path):
    return os.path.basename(path).replace('.gguf', '')


def fit_sandwich(led, peak, rows, codebook_off=True):
    """返回 (t0_ms, eff_tg, 平均相对误差)。带宽取前后两次里较小的那次，保守。"""
    pk = peak['int8'] * 1e9

    def pred(r, t0, eff):
        l = led[r['model']]
        p = peak[l['path']] * 1e9
        bw = min(r['bw_before'], r['bw_after']) * 1e9
        return 1 / (max(l['w_bytes'] / bw, l['tg_flops'] / (eff * p)) + t0)

    best = None
    for t0 in [x / 10 for x in range(20, 60)]:
        for eff in [x / 1000 for x in range(40, 300, 5)]:
            e = statistics.mean(abs(pred(r, t0 / 1e3, eff) / r['tg'] - 1) for r in rows)
            if best is None or e < best[2]:
                best = (t0, eff, e)
    _ = pk
    return best, pred


def fit_pp(led, peak, rows):
    out = {}
    by = {}
    for r in rows:
        if r.get('kind') == 'bench' and r['n_prompt']:
            by.setdefault(name_of(r['model_filename']), []).append(r['avg_ts'])
    for m, v in by.items():
        l = led[m]
        out[m] = statistics.median(v) * l['pp_flops'] / (peak[l['path']] * 1e9)
    return out


def tg_median(rows, model):
    v = [r['avg_ts'] for r in rows if r.get('kind') == 'bench' and r['n_gen'] and name_of(r['model_filename']) == model]
    return statistics.median(v)


def fit_depth(led, rows):
    """每个 FA 设置：(depth 0 的步长, eff_attn, 逐点 [(depth, 实测额外 ms, KV/bw ms)])。"""
    l = led['q-Q4_K_M']
    out = {}
    for fa in (0, 1):
        pts = sorted((r['n_depth'], 1e3 / r['avg_ts'], r['bw']) for r in rows if r['flash_attn'] == fa)
        base = pts[0][1]
        xs, ys = [], []
        for d, ms, bw in pts[1:]:
            xs.append(l['kv_per_token'] * d / (bw * 1e9) * 1e3)
            ys.append(ms - base)
        slope = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
        out['on' if fa else 'off'] = (base, 1 / slope, [(d, y, x) for (d, _, _), x, y in zip(pts[1:], xs, ys)])
    return out


def read_batched(path):
    rows = []
    for line in open(path):
        c = [x.strip() for x in line.strip().strip('|').split('|')]
        if len(c) == 10 and c[0].isdigit():
            rows.append({'pp': int(c[0]), 'tg': int(c[1]), 'B': int(c[2]), 's_pp': float(c[5]), 's_tg': float(c[7])})
    return rows


def linfit(xs, ys):
    mx, my = statistics.mean(xs), statistics.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


def fit_batched(led, peak, rows):
    l = led['q-Q4_K_M']
    B = [r['B'] for r in rows]
    step = [r['B'] / r['s_tg'] * 1e3 for r in rows]
    a, b = linfit(B, step)
    # 一步多一条序列多 b 毫秒；KV 很短（128+32），这部分几乎全是矩阵乘
    eff = l['tg_flops'] / (b / 1e3) / (peak['int8'] * 1e9)
    return a, b, eff, list(zip(B, step))


def dump(models_dir, out):
    sys.path.insert(0, os.path.join(HERE, '..', 'common'))
    from gguf_lite import GGUF
    from estimate_speed import ledger
    res = {}
    for f in sorted(os.listdir(models_dir)):
        if f.endswith('.gguf') and f != 'imatrix.gguf':
            res[f[:-5]] = ledger(GGUF(os.path.join(models_dir, f)))
    json.dump(res, open(out, 'w'), indent=1)
    print('wrote', out, len(res), 'ledgers')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models')
    ap.add_argument('--dump')
    a = ap.parse_args()
    if a.dump:
        dump(a.models, a.dump)
        return
    led = json.load(open(os.path.join(LOGS, 'ledgers.json')))
    bw, peak = machine(4)
    print(f'machine  bw {bw:.1f} GB/s  int8 {peak["int8"]:.0f}  bf16 {peak["bf16"]:.0f}  fp32 {peak["fp32"]:.0f} G op/s')

    sw = jl('sandwich.jsonl')
    (t0, eff, err), pred = fit_sandwich(led, peak, sw)
    print(f'\n[sandwich] t0 {t0:.1f} ms  eff_tg {eff:.3f}  mean |err| {err * 100:.1f}%   (CAL: {CAL["t0_ms"]}, {CAL["eff_tg"]})')
    for r in sw:
        p = pred(r, t0 / 1e3, eff)
        print(f'  r{r["round"]} {r["model"]:<16} bw {min(r["bw_before"], r["bw_after"]):5.1f}  tg {r["tg"]:5.1f}  pred {p:5.1f}')

    il = jl('interleave.jsonl')
    pp = fit_pp(led, peak, il)
    print('\n[pp512] eff_pp per variant (CAL int8 %.2f, bf16 %.2f)' % (CAL['eff_pp']['int8'], CAL['eff_pp']['bf16']))
    for m, e in pp.items():
        print(f'  {m:<16} {e:.3f}')
    iq = tg_median(il, 'qi-IQ2_M')
    l = led['qi-IQ2_M']
    # predict 按 codebook_frac 在 eff_tg 和 eff_tg_codebook 之间插值，这里反解出纯码本张量的效率
    need = l['tg_flops'] / (1 / iq - CAL['t0_ms'] / 1e3) / (peak['int8'] * 1e9)
    eff_cb = CAL['eff_tg'] + (need - CAL['eff_tg']) / l['codebook_frac']
    print(f'\n[codebook] IQ2_M tg median {iq:.1f}  ->  eff_tg_codebook {eff_cb:.3f}  (CAL {CAL["eff_tg_codebook"]})')

    dp = fit_depth(led, jl('depth.jsonl'))
    for fa, (base, e, pts) in dp.items():
        print(f'\n[depth fa={fa}] step@0 {base:.2f} ms  eff_attn {e:.2f}  (CAL {CAL["eff_attn"][fa]})')
        for d, y, x in pts:
            print(f'  depth {d:5d}  extra {y:6.2f} ms   KV/bw {x:5.2f} ms')

    a0, b, eb, pts = fit_batched(led, peak, read_batched(os.path.join(LOGS, 'batched-q4km.md')))
    print(f'\n[batched] step = {a0:.1f} + {b:.2f} x B ms  ->  eff_batch {eb:.3f}  (CAL {CAL["eff_batch"]})')
    for B, s in pts:
        print(f'  B={B:<3d} step {s:6.1f} ms  fit {a0 + b * B:6.1f}')
    _ = at_depth


if __name__ == '__main__':
    main()
