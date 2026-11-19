#!/usr/bin/env bash
# LCPP-20 第 10 章：Qwen3-32B IQ4_XS，32K 上下文，q8_0 KV，单槽，思考模式。
# 合计约 21.23 GiB（推算，python3 ledger.py --model 32b --quant IQ4_XS --ctx 32768 --kv q8_0）。
# 只写了命令，没有在 GPU 上跑过。上机前先跑一遍 llama-fit-params，确认 -ngl 保持 -1。
#
#   bash run_32b.sh            # 启动服务
#   bash run_32b.sh fit        # 只看 fit 的选择和三项 MiB，不加载权重
set -euo pipefail

LLAMA_BIN=${LLAMA_BIN:-../../llama.cpp/build/bin}
M=${M:-models/qwen3-32b/Qwen_Qwen3-32B-IQ4_XS.gguf}
COMMON=(-c 32768 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -np 1)

if [ "${1:-}" = fit ]; then
  exec "$LLAMA_BIN/llama-fit-params" -m "$M" "${COMMON[@]}" -fitp on
fi

exec "$LLAMA_BIN/llama-server" -m "$M" "${COMMON[@]}" \
  -rea on --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 "$@"
