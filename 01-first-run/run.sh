#!/bin/bash
# 跑一次 llama-completion，stderr 带 -v 落到 logs/，方便 parse_load_log.py 对账。
#   bash run.sh MODEL.gguf [CTX] [N_PREDICT]
# CTX 默认 4096。传 0 表示用 GGUF 里的训练长度（Qwen3-0.6B 是 40960，KV 会涨到 4480 MiB）。
set -euo pipefail
MODEL=${1:?用法: bash run.sh MODEL.gguf [CTX] [N_PREDICT]}
CTX=${2:-4096}
N=${3:-64}
BIN=${LLAMA_BIN:-../../llama.cpp/build/bin}
THREADS=${THREADS:-$(nproc)}
mkdir -p logs
TS=$(date +%Y%m%d-%H%M%S)
LOG=logs/run-c$CTX-$TS.log

# -no-cnv：不进对话模式，跑完 -n 个 token 就退出
# --temp 0：贪心解码，两次运行输出一致，方便比较
# -v：打开 fit 阶段那几行 TRC 日志
"$BIN/llama-completion" -m "$MODEL" -c "$CTX" -n "$N" -t "$THREADS" \
  -p "The capital of France is" -no-cnv --temp 0 -v \
  > "logs/run-c$CTX-$TS.out" 2> "$LOG"

echo "输出: logs/run-c$CTX-$TS.out"
echo "日志: $LOG"
grep -E "common_perf_print: +(load time|prompt eval time|eval time)" "$LOG" | sed 's/^[0-9. ]*I //'
