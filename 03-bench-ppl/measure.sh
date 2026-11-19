#!/bin/bash
# LCPP-03 的全部实验：llama-bench 噪声、预热、生成长度；perplexity 上下文长度、确定性；KLD 自检；HellaSwag。
# 用法：LLAMA_BIN=../../llama.cpp/build/bin MODELS=../../models DATA=../../data ./measure.sh
# 4 核机器上总耗时约 40 分钟，HellaSwag 占一半。
set -eu
B=${LLAMA_BIN:-../../llama.cpp/build/bin}
M=${MODELS:-../../models}
D=${DATA:-../../data}
L=${OUT:-logs}
mkdir -p "$L"
Q=$M/q-Q4_K_M.gguf
BASE=$M/qwen3-0.6b-bf16.gguf

# A. 同一配置跑 5 个独立进程，每个进程 -r 10：看进程内和进程间的离散
for i in 1 2 3 4 5; do
  "$B/llama-bench" -m "$Q" -t 4 -p 512 -n 128 -r 10 -o jsonl > "$L/noise-$i.jsonl" 2>/dev/null
done

# B. 关掉预热，看第一个样本是不是更慢
"$B/llama-bench" -m "$Q" -t 4 -p 512 -n 128 -r 6 --no-warmup -o jsonl > "$L/nowarm.jsonl" 2>/dev/null

# C. 生成长度：tg16 / tg128 / tg512，越长越受 KV 增长影响
"$B/llama-bench" -m "$Q" -t 4 -p 0 -n 16,128,512 -r 3 -o jsonl > "$L/ngen.jsonl" 2>/dev/null

# D. 上下文长度对 PPL：每次总 token 数都是 10240，只改 -c
"$B/llama-perplexity" -m "$BASE" -f "$D/wiki.test.raw" -c 128  --chunks 80 -t 4 > "$L/ppl-c128.log"  2>&1
"$B/llama-perplexity" -m "$BASE" -f "$D/wiki.test.raw" -c 512  --chunks 40 -t 4 > "$L/ppl-c512.log"  2>&1
"$B/llama-perplexity" -m "$BASE" -f "$D/wiki.test.raw" -c 2048 --chunks 10 -t 4 > "$L/ppl-c2048.log" 2>&1

# E. HellaSwag 400 题：看置信区间有多宽
for m in qwen3-0.6b-bf16 q-Q4_K_M qi-Q4_K_M qi-Q2_K; do
  "$B/llama-perplexity" -m "$M/$m.gguf" -f "$D/hellaswag_val_full.txt" --hellaswag --hellaswag-tasks 400 -c 512 -t 4 > "$L/hs-$m.log" 2>&1
done

# F. KLD 自检：基准模型和自己比，KLD 应该接近 0，非 0 的部分是 uint16 存储误差
#    base.kld 由 07-quantization/measure.sh 生成（--kl-divergence-base 存的 bf16 logits）
"$B/llama-perplexity" -m "$BASE" -t 4 --chunks 20 --kl-divergence-base "$M/base.kld" --kl-divergence > "$L/kld-self.log" 2>&1

# G. 数值确定性：线程数和 batch 改变规约顺序，PPL 会不会变
"$B/llama-perplexity" -m "$Q" -f "$D/wiki.test.raw" -c 512 --chunks 20 -t 1        > "$L/ppl-t1.log"      2>&1
"$B/llama-perplexity" -m "$Q" -f "$D/wiki.test.raw" -c 512 --chunks 20 -t 4        > "$L/ppl-t4.log"      2>&1
"$B/llama-perplexity" -m "$Q" -f "$D/wiki.test.raw" -c 512 --chunks 20 -t 4 -b 128 > "$L/ppl-t4-b128.log" 2>&1

echo done
