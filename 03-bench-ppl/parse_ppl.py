#!/usr/bin/env python3
"""解析 llama-perplexity 的三种输出：普通 PPL、--kl-divergence、--hellaswag，并回答“差异算不算数”。

用法：
    python3 parse_ppl.py LOG [LOG ...]                 # 每个日志一行摘要
    python3 parse_ppl.py --curve LOG                    # 逐块的累计 PPL / KLD，看多少块以后稳定
    python3 parse_ppl.py --pair A.log B.log             # 两个 KLD 日志（同一份 base）比较：KLD 和 PPL 比值的差是否超出误差

口径（基线 0ec4027，tools/perplexity/perplexity.cpp）：
  * 普通 PPL：每块 n_ctx 个 token，只对后半块打分（first = n_ctx/2，第 540 行），每块贡献 n_ctx/2-1 个 token。
    逐块输出的 [i] 是到第 i 块为止的累计 PPL，不是第 i 块自己的 PPL。
  * Final estimate 的 ± 是 ln PPL 的标准误乘 PPL（第 641-647 行），把 token 当独立样本，偏乐观。
  * KLD 模式下 Mean PPL(Q) 和 Mean PPL(base) 在同一批 token 上算，二者高度相关；
    判断量化有没有变差要看 Mean ln(PPL(Q)/PPL(base)) 的 ±，它扣掉了协方差（第 1909-1911 行），
    比把两个 PPL 的 ± 直接叠加窄一个数量级。
  * Same top p 的 ± 是二项分布标准误 sqrt(p(1-p)/(n-1))（第 1982 行）。
  * HellaSwag 的最后一行是 task 数、acc_norm 和 95% 置信区间（第 988 行，Wilson 区间）。
"""
import argparse
import math
import re
import sys

NUM = r'([-\d.]+)'
PAT = {
    'n_ctx': r'n_ctx=(\d+)',
    'n_chunk': r'over (\d+) chunks',
    'final': r'Final estimate: PPL = ' + NUM + r' \+/- ' + NUM,
    'ppl_q': r'Mean PPL\(Q\)\s*:\s*' + NUM + r'\s*±\s*' + NUM,
    'ppl_base': r'Mean PPL\(base\)\s*:\s*' + NUM + r'\s*±\s*' + NUM,
    'cor': r'Cor\(ln\(PPL\(Q\)\), ln\(PPL\(base\)\)\):\s*' + NUM,
    'ln_ratio': r'Mean ln\(PPL\(Q\)/PPL\(base\)\)\s*:\s*' + NUM + r'\s*±\s*' + NUM,
    'kld': r'Mean\s+KLD:\s*' + NUM + r'\s*±\s*' + NUM,
    'kld99': r'99\.0%\s+KLD:\s*' + NUM,
    'kld_med': r'Median\s+KLD:\s*' + NUM,
    'rms_dp': r'RMS Δp\s*:\s*' + NUM + r'\s*±\s*' + NUM,
    'top1': r'Same top p:\s*' + NUM + r'\s*±\s*' + NUM,
    'spp': r'([\d.]+) seconds per pass',
}
HS_LINE = re.compile(r'^(\d+)\t([\d.]+)%\t\[([\d.]+)%, ([\d.]+)%\]', re.M)


def hs_outcomes(text):
    """从逐题累计准确率还原每一题的对错：第 i 题 = round(i*acc_i - (i-1)*acc_{i-1})。
    perplexity.cpp:775 用固定种子 mt19937(1) 抽题，同一份数据、同样 --hellaswag-tasks，
    不同模型做的是同一组题，可以逐题配对。"""
    rows = [(int(n), float(a) / 100) for n, a, _, _ in HS_LINE.findall(text)]
    out, prev = [], 0.0
    for n, acc in rows:
        out.append(int(round(n * acc - (n - 1) * prev)))
        prev = acc
    return out


def mcnemar(a, b):
    """配对二分类结果的 McNemar 检验（带连续性校正）。返回 (b 对 a 错, a 对 b 错, 卡方, 近似 p)。"""
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    if n01 + n10 == 0:
        return n01, n10, 0.0, 1.0
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p = math.erfc(math.sqrt(chi2 / 2))   # 自由度 1 的卡方分布上尾
    return n01, n10, chi2, p
CUM = re.compile(r'\[(\d+)\]([\d.]+),')
KLD_ROW = re.compile(r'^\s*(\d+)\s+([\d.]+)\s+±\s+([\d.]+)\s+([-\d.]+)\s+±\s+([\d.]+)\s+([\d.]+)\s+±\s+([\d.]+)\s+'
                     r'([\d.]+)\s+±\s+([\d.]+)\s*%\s+([\d.]+)\s+±\s+([\d.]+)\s*%', re.M)


def parse(text):
    out = {}
    for k, p in PAT.items():
        m = re.search(p, text)
        if not m:
            continue
        g = [float(x) for x in m.groups()]
        out[k] = g[0] if len(g) == 1 else tuple(g)
    out['cum'] = [(int(i), float(v)) for i, v in CUM.findall(text)]
    out['kld_rows'] = [tuple(float(x) for x in r) for r in KLD_ROW.findall(text)]
    hs = HS_LINE.findall(text)
    if hs:
        n, acc, lo, hi = hs[-1]
        out['hellaswag'] = (int(n), float(acc), float(lo), float(hi))
    return out


def load(path):
    with open(path, encoding='utf-8', errors='replace') as f:
        return parse(f.read())


def kind(d):
    if 'hellaswag' in d:
        return 'hellaswag'
    if 'kld' in d:
        return 'kld'
    if 'final' in d:
        return 'ppl'
    return 'unknown'


def summary(path, d):
    k = kind(d)
    if k == 'ppl':
        v, e = d['final']
        n_tok = int(d.get('n_chunk', 0) * (d.get('n_ctx', 0) // 2 - 1))
        return f'{path}: PPL {v:.4f} ± {e:.4f} ({100*e/v:.2f}%)  n_ctx={int(d.get("n_ctx", 0))} chunks={int(d.get("n_chunk", 0))} scored_tokens={n_tok}'
    if k == 'kld':
        q, qe = d['ppl_q']
        lr, lre = d['ln_ratio']
        kl, kle = d['kld']
        t1, t1e = d['top1']
        return (f'{path}: PPL(Q) {q:.3f}±{qe:.3f}  ln(Q/base) {lr:+.4f}±{lre:.4f}  '
                f'KLD {kl:.5f}±{kle:.5f}  p99 {d.get("kld99", float("nan")):.4f}  top1 {t1:.2f}±{t1e:.2f}%')
    if k == 'hellaswag':
        n, acc, lo, hi = d['hellaswag']
        return f'{path}: HellaSwag acc_norm {acc:.2f}% [{lo:.2f}, {hi:.2f}]  tasks={n}  halfwidth={(hi-lo)/2:.2f} pt'
    return f'{path}: 未识别'


def curve(d):
    lines = []
    if d['kld_rows']:
        lines.append('chunk   PPL(Q)    ln(Q/base)      KLD          top1')
        for r in d['kld_rows']:
            lines.append(f'{int(r[0]):5d}  {r[1]:8.3f}  {r[3]:+.4f}±{r[4]:.4f}  {r[5]:.5f}±{r[6]:.5f}  {r[9]:6.2f}±{r[10]:.2f}')
        return '\n'.join(lines)
    cum = d['cum']
    if not cum:
        return '没有逐块输出'
    final = cum[-1][1]
    lines.append('chunk  cum_PPL   距最终值')
    for i, v in cum:
        lines.append(f'{i:5d}  {v:8.4f}  {100*(v-final)/final:+6.2f}%')
    return '\n'.join(lines)


def per_chunk(d):
    """把累计值还原成每块自己的值。每块打分的 token 数相同，所以
    第 i 块的均值 = i*cum_i - (i-1)*cum_{i-1}；PPL 在对数域里做同样的事。
    返回 dict：ln_ppl / kld / ln_ratio / top1 各自的逐块列表（有哪个给哪个）。"""
    out = {}
    if d['kld_rows']:
        rows = d['kld_rows']
        cols = {'ln_ppl': [math.log(r[1]) for r in rows], 'ln_ratio': [r[3] for r in rows],
                'kld': [r[5] for r in rows], 'top1': [r[9] for r in rows]}
    elif d['cum']:
        cols = {'ln_ppl': [math.log(v) for _, v in d['cum']]}
    else:
        return out
    for k, cum in cols.items():
        vals, prev = [], 0.0
        for i, c in enumerate(cum, 1):
            vals.append(i * c - (i - 1) * prev)
            prev = c
        out[k] = vals
    return out


def chunk_se(vals):
    """块级标准误：把每块当一个样本。"""
    n = len(vals)
    if n < 2:
        return float('nan')
    m = sum(vals) / n
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1) / n)


def chunk_report(d):
    pc = per_chunk(d)
    if not pc:
        return '没有逐块输出'
    lines = [f'{"量":<10} {"均值":>10} {"块级SE":>10} {"日志SE":>10} {"倍数":>6}']
    tok = {}
    if 'kld' in d:
        tok = {'kld': d['kld'][1], 'ln_ratio': d['ln_ratio'][1], 'top1': d['top1'][1]}
        if 'ppl_q' in d:
            tok['ln_ppl'] = d['ppl_q'][1] / d['ppl_q'][0]
    elif 'final' in d:
        tok = {'ln_ppl': d['final'][1] / d['final'][0]}
    for k, vals in pc.items():
        se = chunk_se(vals)
        t = tok.get(k, float('nan'))
        ratio = se / t if t else float('nan')
        lines.append(f'{k:<10} {sum(vals)/len(vals):10.5f} {se:10.5f} {t:10.5f} {ratio:6.2f}')
    lines.append('倍数 > 1 说明 token 之间有相关，日志里的 ± 偏窄；块越少，块级 SE 本身越不稳。')
    return '\n'.join(lines)


def pair(a, b):
    """同一份 base 上的两个量化：差值的误差没法从日志精确得到（缺两者之间的协方差），
    这里给两个保守判断：区间不重叠才算显著；重叠时提示需要更多 token。"""
    out = []
    for key, name in (('kld', 'Mean KLD'), ('ln_ratio', 'ln(PPL(Q)/PPL(base))'), ('top1', 'Same top p')):
        if key not in a or key not in b:
            continue
        va, ea = a[key]
        vb, eb = b[key]
        lo_a, hi_a = va - 2 * ea, va + 2 * ea
        lo_b, hi_b = vb - 2 * eb, vb + 2 * eb
        sep = hi_a < lo_b or hi_b < lo_a
        z = abs(vb - va) / math.sqrt(ea ** 2 + eb ** 2) if ea or eb else float('inf')
        out.append(f'{name:<24} A {va:.5f}±{ea:.5f}  B {vb:.5f}±{eb:.5f}  diff {vb-va:+.5f}  '
                   f'z(独立假设)={z:.1f}  {"显著" if sep else "区间重叠，加数据再判断"}')
    return '\n'.join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('logs', nargs='*')
    ap.add_argument('--curve', metavar='LOG')
    ap.add_argument('--pair', nargs=2, metavar=('A', 'B'))
    ap.add_argument('--chunks', metavar='LOG', help='还原逐块值，比较块级标准误和日志里的 ±')
    ap.add_argument('--hs-pair', nargs=2, metavar=('A', 'B'), help='两个 HellaSwag 日志逐题配对，McNemar 检验')
    a = ap.parse_args(argv)
    if a.hs_pair:
        ta, tb = (open(p, encoding='utf-8', errors='replace').read() for p in a.hs_pair)
        oa, ob = hs_outcomes(ta), hs_outcomes(tb)
        n = min(len(oa), len(ob))
        n01, n10, chi2, p = mcnemar(oa[:n], ob[:n])
        print(f'tasks={n}  A {100*sum(oa[:n])/n:.2f}%  B {100*sum(ob[:n])/n:.2f}%  '
              f'B 对 A 错={n01}  A 对 B 错={n10}  chi2={chi2:.2f}  p={p:.3f}  '
              f'{"显著" if p < 0.05 else "不显著"}')
    if a.curve:
        print(curve(load(a.curve)))
    if a.chunks:
        print(chunk_report(load(a.chunks)))
    if a.pair:
        print(pair(load(a.pair[0]), load(a.pair[1])))
    for p in a.logs:
        print(summary(p, load(p)))
    if not (a.curve or a.pair or a.logs or a.chunks or a.hs_pair):
        ap.error('需要日志文件')
    return 0


if __name__ == '__main__':
    sys.exit(main())
