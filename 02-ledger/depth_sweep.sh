#!/bin/bash
# 深度扫描：每个点前后各测一次带宽，FA 关和开各一遍，用来校准注意力那一项。
B=${LLAMA_BIN:-../../llama.cpp/build/bin}
M=${1:-../../models}
: > logs/depth.jsonl
for fa in 0 1; do
  for d in 0 512 1024 2048 4096; do
    bw=$(OMP_NUM_THREADS=4 ./stream 256 | sed 's/.*"read_GBps": \([0-9.]*\).*/\1/')
    $B/llama-bench -m $M/q-Q4_K_M.gguf -t 4 -p 0 -n 32 -d $d -fa $fa -r 2 -o jsonl 2>/dev/null \
      | sed "s/^{/{\"bw\": $bw, /" >> logs/depth.jsonl
  done
done
