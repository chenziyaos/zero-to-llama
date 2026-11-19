#!/bin/bash
# LCPP-02 的全部实测：本机带宽和峰值算力，再把 STREAM 和 llama-bench 交替跑，同一时段对账。
# 用法：bash measure.sh MODEL_DIR   （MODEL_DIR 里放 bf16 和各量化 GGUF）
set -e
B=${LLAMA_BIN:-../../llama.cpp/build/bin}
M=${1:-../../models}
mkdir -p logs
gcc -O3 -march=native -ffast-math -fopenmp stream.c -o stream
gcc -O3 -march=native -fopenmp peak.c -o peak
for t in 1 2 4; do OMP_NUM_THREADS=$t ./peak; done > logs/peak.jsonl
for r in $(seq 1 12); do for t in 1 2 3 4; do OMP_NUM_THREADS=$t ./stream 256; done; done > logs/stream.jsonl
# 交替：每轮先测一次带宽，再对每个变体测 pp512 和 tg64
: > logs/interleave.jsonl
for r in 1 2 3 4 5; do
  OMP_NUM_THREADS=4 ./stream 256 | sed "s/^{/{\"round\": $r, \"kind\": \"stream\", /" >> logs/interleave.jsonl
  for v in qwen3-0.6b-bf16 q-Q8_0 q-Q6_K q-Q4_K_M q-Q4_0 q-Q2_K qi-IQ2_M; do
    $B/llama-bench -m $M/$v.gguf -t 4 -p 512 -n 64 -r 2 -o jsonl 2>/dev/null | sed "s/^{/{\"round\": $r, \"kind\": \"bench\", /" >> logs/interleave.jsonl
  done
done
# 并发 decode：B 条序列一起生成，看吞吐什么时候从带宽墙转到算力墙
$B/llama-batched-bench -m $M/q-Q4_K_M.gguf -t 4 -c 8192 -b 2048 -ub 512 -npp 128 -ntg 32 -npl 1,2,4,8,16,32 2>&1 | grep -E '^\|' > logs/batched-q4km.md
# 夹心：每个 tg 前后各测一次带宽，按当时带宽对账，拟合 t0 和 eff_tg
: > logs/sandwich.jsonl
for r in 1 2 3 4 5 6; do
  for v in qwen3-0.6b-bf16 q-Q8_0 q-Q4_K_M q-Q2_K; do
    b0=$(OMP_NUM_THREADS=4 ./stream 256 | sed 's/.*"read_GBps": \([0-9.]*\).*/\1/')
    j=$($B/llama-bench -m $M/$v.gguf -t 4 -p 0 -n 64 -r 2 -o jsonl 2>/dev/null)
    b1=$(OMP_NUM_THREADS=4 ./stream 256 | sed 's/.*"read_GBps": \([0-9.]*\).*/\1/')
    echo "$j" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(json.dumps({'round': $r, 'model': '$v', 'bw_before': $b0, 'bw_after': $b1, 'tg': d['avg_ts'], 'samples': d['samples_ts'], 'model_size': d['model_size']}))" >> logs/sandwich.jsonl
  done
done
# 深度：FA 关和开各扫一遍
bash depth_sweep.sh $M
