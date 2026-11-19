#!/usr/bin/env bash
# LCPP-21 第 7 章：一张 24 GB 卡加 128 GB 以上内存，IQ2_XXS，16K 上下文，q8_0 KV，单槽。
# 缺省全部路由专家留内存：显存约 11.5 GiB，内存约 133 GB（推算，
#   python3 plan.py --build IQ2_XXS --gpu-layers 0 --ctx 16384 --kv q8_0 --card 24）。
# GPU_LAYERS=4 把前 4 层专家放上卡，推算 22.61 GiB，贴着 24 GB，先跑 fit 看实际值。
# decode 快慢主要看内存带宽和物理核数，见 python3 plan.py --fit-card 24 --ctx 16384 --kv q8_0 --ram-bw <GB/s>。
# 只写了命令，没有在 GPU 上跑过。
#
#   THREADS=<物理核数> bash run_24g.sh
#   bash run_24g.sh fit
set -euo pipefail

LLAMA_BIN=${LLAMA_BIN:-../../llama.cpp/build/bin}
MD=${MD:-models/ds41}
M=${M:-$MD/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf}
GPU_LAYERS=${GPU_LAYERS:-0}
THREADS=${THREADS:-$(lscpu -p=CORE 2>/dev/null | grep -v '^#' | sort -u | wc -l)}

if [ "$GPU_LAYERS" -eq 0 ]; then
  OT='exps=CPU'
else
  cpu_layers=$(seq -s '|' "$GPU_LAYERS" 39)
  OT="blk\\.(${cpu_layers})\\.ffn_(up|down|gate)_exps=CPU"
fi

# DeepSeek-V4.1 要求 -ctk 和 -ctv 相同（llama-context.cpp），q8_0 需要 flash attention
COMMON=(-c 16384 -np 1 -ngl 99 -t "$THREADS" -ub 2048 -fa on -ctk q8_0 -ctv q8_0 -ot "$OT")

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
