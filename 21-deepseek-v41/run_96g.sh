#!/usr/bin/env bash
# LCPP-21 第 7 章：一张 96 GB 卡加内存，IQ2_XXS，64K 上下文，单槽，思考模式 effort 50。
# 前 25 层的路由专家上卡，25-39 层留内存。显存约 85.90 GiB，内存约 52.90 GB（推算，
#   python3 plan.py --build IQ2_XXS --gpu-layers 25 --ctx 65536 --ub 2048）。
# 同样的放法社区实测峰值 85.9（HumanjavaEnterprises/deepseek-v41-upstream RECIPE.md）。
# 只写了命令，没有在 GPU 上跑过。
#
#   bash run_96g.sh              # 启动服务
#   bash run_96g.sh fit          # 只看 llama-fit-params 的估算，不加载权重
#   GPU_LAYERS=27 bash run_96g.sh   # 多放两层，推算 92.56 GiB，峰值没人实测过
set -euo pipefail

LLAMA_BIN=${LLAMA_BIN:-../../llama.cpp/build/bin}
MD=${MD:-models/ds41}
M=${M:-$MD/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf}
GPU_LAYERS=${GPU_LAYERS:-25}
THREADS=${THREADS:-16}

# 第 GPU_LAYERS 到 39 层的专家留在内存：blk\.(25|26|...|39)\.ffn_(up|down|gate)_exps=CPU
cpu_layers=$(seq -s '|' "$GPU_LAYERS" 39)
OT="blk\\.(${cpu_layers})\\.ffn_(up|down|gate)_exps=CPU"

COMMON=(-c 65536 -np 1 -ngl 99 -t "$THREADS" -ub 2048 -ot "$OT")

if [ ! -e "$MD/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00006-of-00006.gguf" ]; then
  echo "缺第 6 片：在 $MD 里 ln -s DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00006-of-00006.gguf" >&2
  exit 1
fi

if [ "${1:-}" = fit ]; then
  exec "$LLAMA_BIN/llama-fit-params" -m "$M" "${COMMON[@]}" -fitp on
fi

exec "$LLAMA_BIN/llama-server" -m "$M" "${COMMON[@]}" \
  --spec-type ngram-mod \
  -rea on --reasoning-effort 50 --temp 1.0 --top-p 0.95 "$@"
