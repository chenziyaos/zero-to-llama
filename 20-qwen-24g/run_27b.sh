#!/usr/bin/env bash
# LCPP-20 第 10 章：Qwen3.8-27B UD-Q4_K_XL，64K 上下文，q8_0 KV，单槽，MTP 草稿 2，纯文本。
# 合计约 20.83 GiB（推算，python3 ledger.py --model 27b --quant UD-Q4_K_XL --ctx 65536 --kv q8_0 --np 1 --mtp 2）。
# 只写了命令，没有在 GPU 上跑过。MTP 文件在 unsloth 仓库的 MTP/ 子目录，-hf 找不到，所以用 -md 指定。
# 不加载 mmproj：用 -m 指向本地文件、不写 -mm 就不加载。
#
#   bash run_27b.sh            # 启动服务
#   bash run_27b.sh fit        # 只看 fit 的选择和三项 MiB，不加载权重
set -euo pipefail

LLAMA_BIN=${LLAMA_BIN:-../../llama.cpp/build/bin}
M=${M:-models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf}
MD=${MD:-models/qwen38-27b/MTP/mtp-Qwen3.8-27B-Q4_0.gguf}
COMMON=(-c 65536 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -np 1
        -md "$MD" --spec-type draft-mtp --spec-draft-n-max 2 -ngld all -ctkd q8_0 -ctvd q8_0)

if [ "${1:-}" = fit ]; then
  exec "$LLAMA_BIN/llama-fit-params" -m "$M" "${COMMON[@]}" -fitp on
fi

# 思考模式的采样参数；关掉思考时改成 --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 --presence-penalty 1.5
exec "$LLAMA_BIN/llama-server" -m "$M" "${COMMON[@]}" \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0 "$@"
