#!/bin/bash
# 06 实验：对 exp06.sh 编出的每个变体跑 system_info 和 llama-bench；dl 变体再逐个 CPU 变体单独测
# 用法：MODEL=model.gguf bash bench06.sh [变体名...]
set -u
OUT=${OUT:-$PWD/b06}
LOG=${LOG:-$PWD/logs}
M=${M:-$PWD/models}
MODEL=${MODEL:-$M/q-Q4_K_M.gguf}
MODELS=${MODELS:-$M/q-Q4_K_M.gguf,$M/q-Q4_0.gguf,$M/q-Q8_0.gguf}
R=${R:-3}
mkdir -p $LOG
names=${*:-native generic sse42 blas nollamafile dl}
cd /tmp   # load_best 也会扫当前目录，放在空目录里跑，免得捡到别的 .so
for n in $names; do
  B=$OUT/$n/bin
  [ -x $B/llama-bench ] || { echo "skip $n"; continue; }
  $B/llama-completion -m $MODEL -n 1 -p hi -no-cnv -t 4 2>&1 | grep -E "system_info|load_backend|loaded .* backend" > $LOG/$n.sysinfo.txt
  # 逗号分隔多个模型，llama-bench 会逐个跑；Q8_0、Q4_0 走 llamafile sgemm，Q4_K 不走
  $B/llama-bench -m $MODELS -p 512 -n 64 -r $R -t 4 -o jsonl 2> /dev/null > $LOG/$n.bench.jsonl
  python3 - $n $LOG/$n.bench.jsonl <<'EOF'
import json, sys, os
for l in open(sys.argv[2]):
    d = json.loads(l)
    t = 'pp%d' % d['n_prompt'] if d['n_prompt'] else 'tg%d' % d['n_gen']
    print(sys.argv[1], os.path.basename(d['model_filename']), t, '%.1f ± %.1f' % (d['avg_ts'], d['stddev_ts']))
EOF
done

# dl：每次只放一个 CPU 变体，看各变体自己的速度
if [ -d $OUT/dl/bin ] && echo "$names" | grep -qw dl; then
  for v in $(ls $OUT/dl/bin | sed -n 's/^libggml-cpu-\(.*\)\.so$/\1/p'); do
    T=/tmp/dl1; rm -rf $T; mkdir -p $T
    cp $OUT/dl/bin/llama-bench $T/
    cp -P $OUT/dl/bin/libllama*.so* $OUT/dl/bin/libggml.so* $OUT/dl/bin/libggml-base.so* $T/ 2>/dev/null
    cp $OUT/dl/bin/libggml-cpu-$v.so $T/
    LD_LIBRARY_PATH=$T $T/llama-bench -m $MODEL -p 512 -n 64 -r $R -t 4 -o jsonl 2> $LOG/dl1_$v.err > $LOG/dl1_$v.jsonl
    python3 - dl:$v $LOG/dl1_$v.jsonl <<'EOF'
import json, sys
for l in open(sys.argv[2]):
    d = json.loads(l)
    t = 'pp%d' % d['n_prompt'] if d['n_prompt'] else 'tg%d' % d['n_gen']
    print(sys.argv[1], t, '%.1f ± %.1f' % (d['avg_ts'], d['stddev_ts']))
EOF
  done
  rm -rf /tmp/dl1
fi
