#!/bin/bash
# 从一份 bf16 GGUF 出发，量化出 LCPP-07 用到的全部变体：8 个普通 + imatrix + 6 个 imatrix 版本 + 4 个配方实验。
#
#   LLAMA_BIN=~/llama.cpp/build/bin ./quant.sh models/qwen3-0.6b-bf16.gguf data/wiki.train.raw
#
# 输出都放在 bf16 文件所在目录：q-<TYPE>.gguf、qi-<TYPE>.gguf、imatrix.gguf，日志在 logs/。
# 在 4 核 EPYC 上 Qwen3-0.6B 全部跑完约 10 分钟，imatrix 那一步约 3 分钟。
set -euo pipefail
B=${LLAMA_BIN:-../../llama.cpp/build/bin}
SRC=${1:?用法: quant.sh BF16.gguf CALIB.txt}
CALIB=${2:?需要校准文本，比如 wikitext-2 的 wiki.train.raw}
M=$(dirname "$SRC")
T=${THREADS:-$(nproc)}
mkdir -p logs

q() {  # q <输出名> <类型> [额外参数...]
  local out=$1 type=$2; shift 2
  [ -f "$M/$out.gguf" ] && return
  local t0=$(date +%s.%N)
  "$B/llama-quantize" "$@" "$SRC" "$M/$out.gguf" "$type" "$T" > "logs/quant-$out.log" 2>&1
  printf '%-18s %6.1f s\n' "$out" "$(echo "$(date +%s.%N) - $t0" | bc)"
}

# 1) 不用 imatrix
for t in Q8_0 Q6_K Q5_K_M Q4_K_M Q4_0 IQ4_XS Q3_K_M Q2_K; do q "q-$t" "$t"; done

# 2) imatrix：100 块 × 512 token，约 5 万 token，PR #4861 里作者用的也是这个量级
#    新版默认输出 GGUF 格式（PR #9400），里面是每个输入通道的 in_sum2 和 counts
if [ ! -f "$M/imatrix.gguf" ]; then
  "$B/llama-imatrix" -m "$SRC" -f "$CALIB" --chunks 100 -c 512 -t "$T" -o "$M/imatrix.gguf" > logs/imatrix.log 2>&1
fi

# 3) 用 imatrix。IQ3_XXS、IQ2_M 这类 i-quants 没有 imatrix 时 llama-quantize 会直接拒绝
for t in Q4_K_M IQ4_XS Q3_K_M Q2_K IQ3_XXS IQ2_M; do q "qi-$t" "$t" --imatrix "$M/imatrix.gguf"; done

# 4) 配方实验：输出层降到 Q4_K / 升到 Q8_0；--pure 去掉所有按层升级
q q-Q4_K_M-out4  Q4_K_M --output-tensor-type q4_k
q qi-Q4_K_M-out4 Q4_K_M --imatrix "$M/imatrix.gguf" --output-tensor-type q4_k
q qi-Q2_K-out8   Q2_K   --imatrix "$M/imatrix.gguf" --output-tensor-type q8_0
q q-Q4_K-pure    Q4_K_M --pure

ls -l "$M"/q*-*.gguf
