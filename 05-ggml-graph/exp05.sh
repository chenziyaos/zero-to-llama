#!/bin/bash
# 05 实验：ubatch、FA、上下文对计算缓冲和节点数的影响
B=${LLAMA_BIN:-llama.cpp/build/bin}
M=${MODEL:-models/q-Q4_K_M.gguf}
for c in 4096 16384; do for fa in on off; do for ub in 32 128 512 2048; do
  s=$($B/llama-completion -m $M -c $c -b 2048 -ub $ub -fa $fa -n 1 -p hi -no-cnv -t 4 -v 2>&1 | grep -E "CPU compute buffer size =|graph: nodes" | sed 's/^[0-9.]* I sched_reserve: *//' | tail -2 | tr '\n' ' ')
  echo "c=$c fa=$fa ub=$ub $s"
done; done; done
