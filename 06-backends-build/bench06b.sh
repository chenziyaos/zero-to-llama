#!/bin/bash
# 06 补充实验：云主机上不同时段测的数会漂，这里把各变体交替跑两轮，再在同一次 llama-bench 里对比 --repack 0,1
# 用法：bash bench06b.sh
set -u
OUT=${OUT:-$PWD/b06}
LOG=${LOG:-$PWD/logs}
M=${M:-$PWD/models}
cd /tmp
: > $LOG/interleave.txt
for round in 1 2; do
  for n in native generic blas nollamafile dl; do
    $OUT/$n/bin/llama-bench -m $M/q-Q4_K_M.gguf,$M/q-Q4_0.gguf,$M/q-Q8_0.gguf -p 512 -n 0 -r 3 -t 4 -o jsonl 2>/dev/null |
      python3 -c "
import json, sys, os
for l in sys.stdin:
    d = json.loads(l)
    print('$round', '$n', os.path.basename(d['model_filename']), 'pp512', '%.1f' % d['avg_ts'], '%.1f' % d['stddev_ts'])
" | tee -a $LOG/interleave.txt
  done
done
# repack 对照：同一次调用里 0 和 1 交替，-t 4 和 -t 1 各一次
: > $LOG/repack.txt
for n in native generic dl; do
  for t in 4 1; do
    $OUT/$n/bin/llama-bench -m $M/q-Q4_K_M.gguf,$M/q-Q4_0.gguf -p 512 -n 0 -r 3 -t $t --repack 0,1 -o jsonl 2>/dev/null |
      python3 -c "
import json, sys, os
for l in sys.stdin:
    d = json.loads(l)
    print('$n', 't$t', os.path.basename(d['model_filename']), 'repack', int(d['repack']), '%.1f' % d['avg_ts'], '%.1f' % d['stddev_ts'])
" | tee -a $LOG/repack.txt
  done
done
echo done
