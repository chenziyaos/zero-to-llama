#!/bin/bash
# 06 单线程对照：4 vCPU 云主机上 -t 4 的数受邻居干扰，单线程方差小得多，变体之间的差别看得更清楚
# 用法：bash bench06c.sh
set -u
OUT=${OUT:-$PWD/b06}
LOG=${LOG:-$PWD/logs}
M=${M:-$PWD/models}
cd /tmp
fmt() {
  python3 -c "
import json, sys, os
for l in sys.stdin:
    d = json.loads(l)
    t = 'pp%d' % d['n_prompt'] if d['n_prompt'] else 'tg%d' % d['n_gen']
    print('$1', os.path.basename(d['model_filename']), t, '%.1f' % d['avg_ts'], '%.1f' % d['stddev_ts'])
"
}
: > $LOG/t1.txt
for n in native generic sse42 blas nollamafile dl; do
  $OUT/$n/bin/llama-bench -m $M/q-Q4_K_M.gguf,$M/q-Q4_0.gguf,$M/q-Q8_0.gguf -p 256 -n 32 -r 3 -t 1 -o jsonl 2>/dev/null | fmt $n | tee -a $LOG/t1.txt
done
# dl 各变体单独放，只测 Q4_K_M；x64 太慢，只跑 1 轮
for v in sse42 sandybridge ivybridge piledriver haswell skylakex cannonlake cascadelake icelake cooperlake zen4 x64; do
  T=/tmp/dl1; rm -rf $T; mkdir -p $T
  cp $OUT/dl/bin/llama-bench $T/
  cp -P $OUT/dl/bin/libllama*.so* $OUT/dl/bin/libggml.so* $OUT/dl/bin/libggml-base.so* $T/ 2>/dev/null
  cp $OUT/dl/bin/libggml-cpu-$v.so $T/
  r=3; [ $v = x64 ] && r=1
  LD_LIBRARY_PATH=$T $T/llama-bench -m $M/q-Q4_K_M.gguf -p 256 -n 32 -r $r -t 1 -o jsonl 2>/dev/null | fmt dl:$v | tee -a $LOG/t1.txt
done
echo done
