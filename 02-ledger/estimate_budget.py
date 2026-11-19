"""内存预算和速度一起算：给定内存上限，每个量化文件最多能开多长上下文，开到一半深度时 decode 还剩多快。

    python3 estimate_budget.py --budget 2048 ../../models/*.gguf [--ctk f16 --ctv f16] [--fa on]
    python3 estimate_budget.py --budget 2048 --headroom 512 MODEL.gguf

内存账本复用 01-first-run/estimate_memory.py（权重 + KV + 计算缓冲，含 CPU_REPACK），
速度复用本目录 estimate_speed.py 的校准预测。上下文按 256 的槽位粒度二分查找。
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '01-first-run'))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
from estimate_memory import estimate  # noqa: E402
from estimate_speed import ledger, machine, predict  # noqa: E402

MiB = 2 ** 20


def total(g, ctx, a):
    e = estimate(g, ctx, a.ubatch, a.ctk, a.ctv, a.fa, 'avx512')
    return e['host'] + e['repack']


def max_ctx(g, budget, a):
    """预算内能开的最大上下文（256 的倍数），放不下返回 0。"""
    limit = g.hp('context_length') or 131072
    if total(g, 256, a) > budget:
        return 0
    lo, hi = 1, limit // 256
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if total(g, mid * 256, a) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo * 256


def plan(g, budget, a, bw, peak):
    c = max_ctx(g, budget, a)
    if not c:
        return {'ctx': 0}
    half = predict(ledger(g, c // 2, a.ctk, a.ctv), bw, peak, (1,), a.fa)['tg']
    zero = predict(ledger(g, 0, a.ctk, a.ctv), bw, peak, (1,), a.fa)['tg']
    return {'ctx': c, 'mem': total(g, c, a), 'tg0': zero, 'tg_half': half}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('models', nargs='+')
    ap.add_argument('--budget', type=float, required=True, help='可用内存 MiB')
    ap.add_argument('--headroom', type=float, default=0, help='给系统和其他进程留的 MiB，从预算里扣掉')
    ap.add_argument('--ubatch', type=int, default=512)
    ap.add_argument('--ctk', default='f16')
    ap.add_argument('--ctv', default='f16')
    ap.add_argument('--fa', choices=['on', 'off'], default='on')
    ap.add_argument('--bw', type=float)
    a = ap.parse_args()
    from gguf_lite import GGUF
    bw, peak = machine(4)
    bw = a.bw or bw
    budget = (a.budget - a.headroom) * MiB
    print(f'budget {budget / MiB:.0f} MiB  bw {bw:.1f} GB/s  K {a.ctk} V {a.ctv} FA {a.fa}')
    print(f'{"model":<24}{"max ctx":>9}{"mem MiB":>9}{"tg@0":>7}{"tg@ctx/2":>10}')
    for m in a.models:
        if os.path.basename(m) == 'imatrix.gguf':
            continue
        p = plan(GGUF(m), budget, a, bw, peak)
        name = os.path.basename(m)[:-5]
        if not p['ctx']:
            print(f'{name:<24}{"不够":>9}')
            continue
        print(f'{name:<24}{p["ctx"]:>9}{p["mem"] / MiB:>9.0f}{p["tg0"]:>7.1f}{p["tg_half"]:>10.1f}')


if __name__ == '__main__':
    main()
