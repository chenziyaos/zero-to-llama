"""汇总 llama-bench -o jsonl：按 KV 类型、FA、深度列出 pp 和 tg。
用法：python3 collect_bench.py logs/bench-*.jsonl"""
import json, sys, os

rows = []
for fn in sys.argv[1:]:
    for line in open(fn):
        line = line.strip()
        if not line.startswith('{'):
            continue
        d = json.loads(line)
        test = 'pp%d' % d['n_prompt'] if d['n_prompt'] else 'tg%d' % d['n_gen']
        fa = d.get('flash_attn')
        rows.append((d['type_k'], d['type_v'], str(fa), d['n_depth'], test, d['avg_ts'], d['stddev_ts']))
rows.sort(key=lambda r: (r[4], r[0], r[1], r[2], r[3]))
print('%-6s %-6s %-5s %6s %-6s %10s' % ('K', 'V', 'FA', 'depth', 'test', 't/s'))
for r in rows:
    print('%-6s %-6s %-5s %6d %-6s %7.2f ± %.2f' % r)
