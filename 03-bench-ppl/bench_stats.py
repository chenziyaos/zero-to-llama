#!/usr/bin/env python3
"""把 llama-bench -o jsonl 的输出按配置分组，给出均值、置信区间和两组配置的差异检验。

用法：
    python3 bench_stats.py run1.jsonl [run2.jsonl ...]            # 每个配置一行
    python3 bench_stats.py --by-file noise-*.jsonl                # 每个文件（每个进程）单独一行，看进程间漂移
    python3 bench_stats.py --compare A.jsonl B.jsonl              # A、B 两组同名测试逐项比较
    python3 bench_stats.py --need 0.03 noise-*.jsonl              # 要分辨 3% 的差异，大概需要多少次重复

llama-bench 自己报的 avg_ts ± stddev_ts 只描述同一进程内 -r 次重复的离散程度（llama-bench.cpp:1564
get_ts 把每次的 ns 换成 t/s 再求均值和样本标准差）。云主机上进程之间的漂移往往比进程内大，
所以这里把每个文件当成一次独立实验：合并时既给“所有样本”的统计，也给“文件均值之间”的统计。

只用标准库。t 分布分位数用查表加插值，自由度 >30 时按正态近似。
"""
import argparse
import json
import math
import os
import sys
from collections import OrderedDict

# 双侧 95% 的 t 分位数，df=1..30
T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
       2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
       2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def t95(df):
    if df < 1:
        return float('nan')
    if df <= 30:
        return T95[int(df) - 1]
    return 1.96 + 2.4 / df   # 30 以上的近似，误差 < 0.005


def mean(xs):
    return sum(xs) / len(xs)


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def sd(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def ci95(xs):
    """均值的 95% 置信区间半宽。"""
    if len(xs) < 2:
        return float('nan')
    return t95(len(xs) - 1) * sd(xs) / math.sqrt(len(xs))


def test_name(r):
    """和 llama-bench markdown 输出里 test 列的写法一致：pp512、tg128、pp512 @ d4096。"""
    p, g, d = r['n_prompt'], r['n_gen'], r.get('n_depth', 0)
    if p and g:
        s = f'pp{p}+tg{g}'
    elif p:
        s = f'pp{p}'
    else:
        s = f'tg{g}'
    return s + (f' @ d{d}' if d else '')


# 区分配置的字段：同一组字段相同的行才合并
CONF_KEYS = ('model_filename', 'n_threads', 'n_batch', 'n_ubatch', 'type_k', 'type_v',
             'flash_attn', 'n_gpu_layers', 'repack', 'poll', 'cpu_strict')


def load(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('{'):
                rows.append(json.loads(line))
    return rows


def conf_of(r, short=True):
    parts = []
    for k in CONF_KEYS:
        if k not in r:
            continue
        v = r[k]
        if k == 'model_filename' and short:
            v = os.path.basename(v)
        parts.append(f'{k}={v}')
    return ' '.join(parts)


def samples_ts(r):
    if 'samples_ts' in r and r['samples_ts']:
        return list(r['samples_ts'])
    return [r['avg_ts']]


def group(paths, by_file=False):
    """返回 OrderedDict[(conf, test)] -> {'files': [[ts...], ...]}。"""
    g = OrderedDict()
    for path in paths:
        for r in load(path):
            key = (conf_of(r), test_name(r))
            if by_file:
                key = (os.path.basename(path), test_name(r))
            g.setdefault(key, {'files': [], 'model': os.path.basename(r['model_filename'])})
            g[key]['files'].append(samples_ts(r))
    return g


def summarize(files):
    """files：每个元素是一个进程（一行 jsonl）的样本列表。"""
    allv = [x for f in files for x in f]
    fmeans = [mean(f) for f in files]
    within = [sd(f) / mean(f) for f in files if len(f) > 1]
    out = {
        'n_files': len(files), 'n': len(allv), 'mean': mean(allv), 'sd': sd(allv),
        'ci': ci95(allv), 'min': min(allv), 'max': max(allv), 'median': median(allv),
        'cv_within': mean(within) if within else float('nan'),
    }
    if len(files) > 1:
        out['between_mean'] = mean(fmeans)
        out['cv_between'] = sd(fmeans) / mean(fmeans)
        out['ci_between'] = ci95(fmeans)
    # 掉速样本：比中位数低 10% 以上。云主机上多半是邻居抢核，不是被测配置本身
    out['n_slow'] = sum(1 for x in allv if x < 0.9 * out['median'])
    return out


def welch(a, b):
    """Welch t 检验：返回 (差值 b-a, 95% 区间半宽, t, df)。a、b 是样本列表。"""
    ma, mb, va, vb = mean(a), mean(b), sd(a) ** 2, sd(b) ** 2
    na, nb = len(a), len(b)
    se2 = va / na + vb / nb
    if se2 == 0:
        return mb - ma, 0.0, float('inf'), na + nb - 2
    df = se2 ** 2 / ((va / na) ** 2 / max(na - 1, 1) + (vb / nb) ** 2 / max(nb - 1, 1))
    se = math.sqrt(se2)
    return mb - ma, t95(df) * se, (mb - ma) / se, df


def need_reps(cv, rel, power_z=0.84):
    """两组各 n 次重复，要以 95% 置信、80% 功效分辨相对差 rel，需要的 n（正态近似）。"""
    if rel <= 0:
        return float('inf')
    return math.ceil(2 * ((1.96 + power_z) * cv / rel) ** 2)


def fmt_row(name, test, s):
    line = (f'{name:<20} {test:<14} mean {s["mean"]:8.2f} ± {s["ci"]:5.2f}  median {s["median"]:8.2f}  '
            f'min {s["min"]:7.1f}  n={s["n"]:<3d} slow={s["n_slow"]} cv_in={100*s["cv_within"]:5.2f}%')
    if 'cv_between' in s:
        line += f'  files={s["n_files"]} cv_between={100*s["cv_between"]:5.2f}%'
    return line


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='*')
    ap.add_argument('--by-file', action='store_true', help='每个文件单独一行')
    ap.add_argument('--compare', nargs=2, metavar=('A', 'B'), help='比较两个 jsonl 里同名测试')
    ap.add_argument('--need', type=float, help='目标相对差（如 0.03），按观测到的 cv 估算重复次数')
    a = ap.parse_args(argv)

    if a.compare:
        ga, gb = group([a.compare[0]]), group([a.compare[1]])
        ta = {k[1]: v for k, v in ga.items()}
        tb = {k[1]: v for k, v in gb.items()}
        print(f'{"test":<16} {"A":>9} {"B":>9} {"B/A":>7}  {"95% CI of diff":>18}  verdict')
        for t in ta:
            if t not in tb:
                continue
            xa = [x for f in ta[t]['files'] for x in f]
            xb = [x for f in tb[t]['files'] for x in f]
            d, h, tt, df = welch(xa, xb)
            verdict = 'same' if abs(d) <= h else ('B faster' if d > 0 else 'B slower')
            print(f'{t:<16} {mean(xa):9.2f} {mean(xb):9.2f} {mean(xb)/mean(xa):7.3f}  [{d-h:+8.2f}, {d+h:+8.2f}]  {verdict}')
        return 0

    if not a.files:
        ap.error('需要至少一个 jsonl 文件')
    g = group(a.files, by_file=a.by_file)
    for (name, test), v in g.items():
        label = v['model'] if not a.by_file else name
        s = summarize(v['files'])
        print(fmt_row(label, test, s))
        if a.need:
            cv = s.get('cv_between', s['cv_within'])
            print(f'    分辨 {100*a.need:.1f}% 的差异：按 cv={100*cv:.2f}% 每组约需 {need_reps(cv, a.need)} 次独立运行')
    return 0


if __name__ == '__main__':
    sys.exit(main())
