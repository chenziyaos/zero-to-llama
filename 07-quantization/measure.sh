#!/bin/bash
# 给 quant.sh 产出的每个变体测速度和质量，结果写进 logs/，再用 collect_results.py 汇总。
#
#   LLAMA_BIN=~/llama.cpp/build/bin ./measure.sh models/qwen3-0.6b-bf16.gguf data/wiki.test.raw
#   python3 collect_results.py logs --more logs/round2 --more logs/round3 --md
#
# 三件事，必须串行跑，不然几个进程抢核，速度数字全是噪声：
#   1) KLD 基线：bf16 在测试集上跑一遍，把每个位置的完整 log-prob 存进 base.kld
#      Qwen3 词表 151936，20 块 × 512 token 就有 1.5 GB，磁盘要留够
#   2) 每个量化变体读 base.kld，算 PPL、KLD 分布、top-1 一致率（PR #5076）
#   3) llama-bench pp512 / tg128，跑 ROUNDS 轮，每轮写到不同目录，汇总时取中位数
set -euo pipefail
B=${LLAMA_BIN:-../../llama.cpp/build/bin}
SRC=${1:?用法: measure.sh BF16.gguf TEST.txt}
TEST=${2:?需要测试文本，比如 wikitext-2 的 wiki.test.raw}
M=$(dirname "$SRC"); BASE=$(basename "$SRC" .gguf)
T=${THREADS:-$(nproc)}; CHUNKS=${CHUNKS:-20}; ROUNDS=${ROUNDS:-3}
mkdir -p logs

[ -f "$M/base.kld" ] || "$B/llama-perplexity" -m "$SRC" -f "$TEST" -c 512 --chunks "$CHUNKS" -t "$T" \
  --kl-divergence-base "$M/base.kld" > "logs/ppl-$BASE.log" 2>&1

VARIANTS=$(cd "$M" && ls q-*.gguf qi-*.gguf | sed 's/\.gguf$//')
for m in $VARIANTS; do
  # 有了 base.kld 就不用再给 -f，token 序列也存在里面，保证两边看的是同一批 token
  "$B/llama-perplexity" -m "$M/$m.gguf" -t "$T" --chunks "$CHUNKS" \
    --kl-divergence-base "$M/base.kld" --kl-divergence > "logs/ppl-$m.log" 2>&1
  echo "ppl $m: $(grep -o 'Mean    KLD: *[0-9.]*' "logs/ppl-$m.log")"
done

for r in $(seq 1 "$ROUNDS"); do
  d=logs; [ "$r" -gt 1 ] && d=logs/round$r && mkdir -p "$d"
  for m in $BASE $VARIANTS; do
    "$B/llama-bench" -m "$M/$m.gguf" -t "$T" -p 512 -n 128 -r 5 -o jsonl > "$d/bench-$m.jsonl" 2>/dev/null
  done
done
python3 "$(dirname "$0")/collect_results.py" logs $(for r in $(seq 2 "$ROUNDS"); do echo --more logs/round$r; done) --md
