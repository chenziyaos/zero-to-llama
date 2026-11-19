#!/usr/bin/env bash
# LCPP-22 第 10 章：Qwen3.8-27B（GGUF 架构名 qwen35，混合 Gated DeltaNet 注意力），看图用 bartowski 多模态版。
# 24 GB 单卡，Q4_K_M（imatrix，主文件内嵌 MTP 头、imatrix 档里 MTP 存 Q4_0）+ mmproj bf16，
# 32K 上下文，q8_0 KV，单槽，flash attention。
# 合计约 19.32 GiB（推算，python3 plan.py --quant Q4_K_M --ctx 32768 --kv q8_0 --np 1 --mmproj）。
# 关键：64 层里只有 16 层存 softmax KV，其余 48 层（四分之三）是固定 149.6 MiB 的 DeltaNet 循环态（F32，单槽）。
# 只写了命令，没有在 GPU 上跑过。qwen35 架构 >= b7990，含 MTP 层的文件 >= b9180；
# 现行 bartowski 文件用 b10896 转，模型卡写明要用 b10896 或更新；lmstudio 文件用 b10430 转。
#
#   bash run_24g.sh            # 启动服务（看图 mmproj 已挂上；默认思考采样 1.0/0.95/20）
#   THINK=0 bash run_24g.sh    # 非思考采样：0.7/0.8/20 加 presence_penalty 1.5
#   MTP=3 bash run_24g.sh      # 打开 draft-mtp 投机解码，草稿最多 3 个 token（循环态 x4，约 19.76 GiB）
#   bash run_24g.sh fit        # 只看 fit 的模型/上下文/compute 三项 MiB，不加载权重
#
# 采样照官方两套口径：思考 temperature 1.0 / top_p 0.95 / top_k 20 / min_p 0；
# 非思考 temperature 0.7 / top_p 0.8 / top_k 20，presence_penalty 1.5。模板必须 --jinja，默认开思考。
set -euo pipefail

LLAMA_BIN=${LLAMA_BIN:-../../llama.cpp/build/bin}
MD=${MD:-models/qwen38-27b}
M=${M:-$MD/Qwen3.8-27B-Q4_K_M.gguf}
MM=${MM:-$MD/mmproj-Qwen3.8-27B-bf16.gguf}
MTP=${MTP:-0}
THINK=${THINK:-1}

if [ ! -e "$M" ]; then
  echo "缺权重：hf download bartowski/Qwen3.8-27B-GGUF Qwen3.8-27B-Q4_K_M.gguf mmproj-Qwen3.8-27B-bf16.gguf --local-dir $MD" >&2
  exit 1
fi
if [ ! -e "$MM" ]; then
  echo "缺 mmproj（看图才需要，纯文本可删掉 -mm 这一行）：hf download bartowski/Qwen3.8-27B-GGUF mmproj-Qwen3.8-27B-bf16.gguf --local-dir $MD" >&2
  exit 1
fi

# q8_0 的 V cache 需要 flash attention；-np 1 单槽，循环态只存一份，省 DeltaNet 状态。
COMMON=(-c 32768 -np 1 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 --jinja -mm "$MM")
if [ "$MTP" != "0" ]; then
  # bartowski 的 MTP 头已内嵌在主 GGUF（imatrix 档存 Q4_0），draft-mtp 直接用；
  # 循环态按 (1+草稿数) 倍增，需要 >= b9180。unsloth/ggml-org 的 MTP 是单独文件，本脚本不适用。
  COMMON+=(--spec-type draft-mtp --spec-draft-n-max "$MTP")
fi

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
