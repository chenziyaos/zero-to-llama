#!/usr/bin/env bash
# LCPP-22 第 10 章：纯文本退路，用 lmstudio-community 的精简三档（b10430 转换，只有语言文件）。
# 24 GB 单卡，Q4_K_M（16.81 GB），32K 上下文，q8_0 KV，单槽，flash attention。
# 合计约 17.86 GiB（推算，python3 plan.py --source lmstudio --quant Q4_K_M --ctx 32768 --kv q8_0 --np 1）。
# 同仓另有 mmproj-Qwen3.8-27B-BF16.gguf，要看图就把它一起下载并加 -mm，或直接用 run_24g.sh（bartowski 多模态版）。
# lmstudio 模型卡没写 MTP 头怎么处理，本脚本不开 draft-mtp；要投机解码用 run_24g.sh 的 bartowski（MTP 内嵌主文件）。
# 只写了命令，没有在 GPU 上跑过。qwen35 架构 >= b7990；文件是 2026 年 8 月以后用 b10430 转的，实际建议不旧于 b10430。
#
#   bash run_text.sh            # 启动服务（默认思考采样 1.0/0.95/20）
#   THINK=0 bash run_text.sh    # 非思考采样：0.7/0.8/20 加 presence_penalty 1.5
#   bash run_text.sh fit        # 只看 fit 的模型/上下文/compute 三项 MiB，不加载权重
#
# 采样照官方两套口径：思考 temperature 1.0 / top_p 0.95 / top_k 20 / min_p 0；
# 非思考 temperature 0.7 / top_p 0.8 / top_k 20，presence_penalty 1.5。模板必须 --jinja，默认开思考。
set -euo pipefail

LLAMA_BIN=${LLAMA_BIN:-../../llama.cpp/build/bin}
MD=${MD:-models/qwen38-27b-lmstudio}
M=${M:-$MD/Qwen3.8-27B-Q4_K_M.gguf}
THINK=${THINK:-1}

if [ ! -e "$M" ]; then
  echo "缺权重：hf download lmstudio-community/Qwen3.8-27B-GGUF Qwen3.8-27B-Q4_K_M.gguf --local-dir $MD" >&2
  exit 1
fi

# 纯文本不挂 -mm；q8_0 V cache 需要 -fa on；chat template 已内嵌，仍要 --jinja 才按模板渲染。
COMMON=(-c 32768 -np 1 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 --jinja)

if [ "${1:-}" = fit ]; then
  exec "$LLAMA_BIN/llama-fit-params" -m "$M" "${COMMON[@]}" -fitp on
fi

# 思考默认（generation_config 与模型卡思考档）；THINK=0 切非思考档，另加 presence_penalty 1.5。
if [ "$THINK" = "0" ]; then
  SAMPLING=(--temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 --presence-penalty 1.5)
else
  SAMPLING=(--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0)
fi

exec "$LLAMA_BIN/llama-server" -m "$M" "${COMMON[@]}" "${SAMPLING[@]}" "$@"
