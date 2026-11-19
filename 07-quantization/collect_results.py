#!/usr/bin/env python3
"""把 measure.sh 跑出来的 llama-bench JSONL 和 llama-perplexity 日志合成一张表。

用法：
    python3 collect_results.py LOGDIR [--more DIR ...] [--json out.json] [--md]

--more 指定额外几轮 bench 的目录（同名 bench-<name>.jsonl），速度取所有轮次 avg_ts 的中位数。
CPU 云主机上同一配置两轮之间 tg 能差 30%，只跑一轮的速度数字不要信。

LOGDIR 里需要有：
    bench-<name>.jsonl   llama-bench -o jsonl 的输出，每个模型一份（pp 一行、tg 一行）
    ppl-<name>.log       llama-perplexity --kl-divergence 的输出
    ppl-<base>.log       基准模型（默认 qwen3-0.6b-bf16）只有 Final estimate 一行 PPL

只用标准库。字段口径：
    size_mib    llama-bench 报的 model_size，只算张量字节，不含 GGUF 头和元数据
    bpw         size * 8 / n_params
    pp, tg      各轮 avg_ts 的中位数，pp 是 -p 512，tg 是 -n 128
    ppl         Mean PPL(Q)，和 KLD 同一批 token 上算出来的，可以直接和 base 的 Mean PPL(base) 比
    kld         Mean KLD，kld99 是 99 分位
    top1        Same top p，量化模型和基准模型 argmax 相同的 token 比例
"""
import argparse
import glob
import json
import os
import re
import sys

BASE = 'qwen3-0.6b-bf16'

PAT = {
    'ppl': r'Mean PPL\(Q\)\s*:\s*([\d.]+)',
    'ppl_base': r'Mean PPL\(base\)\s*:\s*([\d.]+)',
    'kld': r'Mean\s+KLD:\s*([\d.]+)',
    'kld99': r'99\.0%\s+KLD:\s*([\d.]+)',
    'kld_max': r'Maximum KLD:\s*([\d.]+)',
    'rms_dp': r'RMS Δp\s*:\s*([\d.]+)',
    'top1': r'Same top p:\s*([\d.]+)',
    'final': r'Final estimate: PPL = ([\d.]+)',
}


def parse_ppl(path):
    with open(path, encoding='utf-8', errors='replace') as f:
        text = f.read()
    out = {}
    for k, p in PAT.items():
        m = re.search(p, text)
        if m:
            out[k] = float(m.group(1))
    return out


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def parse_bench(paths):
    out, runs = {}, {'pp': [], 'tg': []}
    for path in paths:
      for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out['size'] = r['model_size']
        out['n_params'] = r['model_n_params']
        out['type'] = r['model_type']
        key = 'pp' if r['n_prompt'] > 0 and r['n_gen'] == 0 else 'tg'
        runs[key].append(r['avg_ts'])
    for key, v in runs.items():
        if v:
            out[key] = median(v)
            out[key + '_runs'] = v
            out[key + '_spread'] = max(v) - min(v)
    return out


def collect(logdir, more=()):
    """bench-*.jsonl 和 ppl-*.log 取并集；只有 ppl 没有 bench 的变体（比如 --pure 实验）也出一行。"""
    names = set()
    for b in glob.glob(os.path.join(logdir, 'bench-*.jsonl')):
        names.add(os.path.basename(b)[6:-6])
    for p in glob.glob(os.path.join(logdir, 'ppl-*.log')):
        names.add(os.path.basename(p)[4:-4])
    rows = []
    for name in names - {'threads', 'depth'}:
        row = {'name': name, 'imatrix': name.startswith('qi-')}
        b = os.path.join(logdir, f'bench-{name}.jsonl')
        if os.path.exists(b):
            extra = [os.path.join(d, os.path.basename(b)) for d in more]
            row.update(parse_bench([b] + [e for e in extra if os.path.exists(e)]))
            row['size_mib'] = row['size'] / 2**20
            row['bpw'] = row['size'] * 8 / row['n_params']
        p = os.path.join(logdir, f'ppl-{name}.log')
        if os.path.exists(p):
            row.update(parse_ppl(p))
        rows.append(row)
    rows.sort(key=lambda r: (-r.get('size', 0), r['name']))
    return rows


def pareto(rows):
    """体积和 Mean KLD 两个维度上的帕累托前沿：不存在“更小且 KLD 更低”的变体。

    只看有 size 和 kld 的行。返回按体积从大到小排的名字列表。
    """
    pts = [r for r in rows if 'size' in r and 'kld' in r]
    front = [r for r in pts
             if not any(o['size'] <= r['size'] and o['kld'] <= r['kld'] and
                        (o['size'] < r['size'] or o['kld'] < r['kld']) for o in pts)]
    return [r['name'] for r in sorted(front, key=lambda r: -r['size'])]


def to_md(rows):
    head = '| 模型 | imatrix | MiB | bpw | pp512 t/s | tg128 t/s | PPL | Mean KLD | KLD p99 | top-1 一致 |'
    lines = [head, '|' + '---|' * 10]
    for r in rows:
        f = lambda k, fmt: (fmt % r[k]) if k in r else '—'
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            r['name'], '是' if r['imatrix'] else '否', f('size_mib', '%.0f'), f('bpw', '%.2f'),
            f('pp', '%.1f'), f('tg', '%.1f'), f('ppl', '%.2f'),
            f('kld', '%.4f'), f('kld99', '%.3f'), f('top1', '%.1f%%')))
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logdir')
    ap.add_argument('--more', action='append', default=[])
    ap.add_argument('--json')
    ap.add_argument('--md', action='store_true')
    ap.add_argument('--pareto', action='store_true', help='额外打印体积-KLD 帕累托前沿')
    a = ap.parse_args()
    rows = collect(a.logdir, a.more)
    if not rows:
        sys.exit(f'{a.logdir} 里没有 bench-*.jsonl 或 ppl-*.log')
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
    print(to_md(rows) if a.md else json.dumps(rows, ensure_ascii=False, indent=1))
    if a.pareto:
        print('\n体积-KLD 帕累托前沿：' + ' > '.join(pareto(rows)))


if __name__ == '__main__':
    main()
