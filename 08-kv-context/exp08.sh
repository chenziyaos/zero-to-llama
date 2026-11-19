#!/bin/bash
# 08 实验：KV 大小、FA 与 KV 量化的速度、KV 量化对 KLD 的影响、上下文长度与写满、FA 的 compute buffer
# 用法：bash exp08.sh [size|bench|kld|ctx|fa|all]
# 环境变量：LLAMA_BIN（缺省 ../../llama.cpp/build/bin），MODELS（模型目录），WIKI（wiki.test.raw）
set -u
HERE=$(cd $(dirname $0) && pwd)
B=${LLAMA_BIN:-$HERE/../../../llama.cpp/build/bin}
M=${MODELS:-$HERE/../../../models}
WIKI=${WIKI:-$HERE/../../../data/wiki.test.raw}
L=$HERE/logs
mkdir -p $L
what=${1:-all}
cd /tmp   # 避免捡到当前目录里的其他后端库

if [ $what = size ] || [ $what = all ]; then
  : > $L/kv_size.txt
  for c in 4096 40960; do for t in f16 q8_0 q4_0; do
    echo "== ctx $c type $t" >> $L/kv_size.txt
    $B/llama-completion -m $M/q-Q4_K_M.gguf -c $c -ctk $t -ctv $t -n 1 -p hi -no-cnv -t 4 -v 2>&1 |
      grep -E "llama_kv_cache: size|n_rot_|compute buffer size|Flash Attention|enabling flash" >> $L/kv_size.txt
  done; done
fi

if [ $what = bench ] || [ $what = all ]; then
  # 单线程，深度 0、2048、8192；同一深度内的多项测试复用同一份 KV，不重复预填充
  # q4_0 深度 8192 单线程要跑半小时以上，只测到 2048
  for cfg in "f16 f16 on 0,2048,8192" "f16 f16 off 0,2048,8192" "q8_0 q8_0 on 0,2048,8192" "q4_0 q4_0 on 0,2048"; do
    set -- $cfg
    $B/llama-bench -m $M/q-Q4_K_M.gguf -t 1 -r 3 -p 512 -n 32 -d $4 -ctk $1 -ctv $2 -fa $3 -o jsonl \
      2> /dev/null > $L/bench-$1-$2-fa$3.jsonl
  done
  python3 $HERE/collect_bench.py $L/bench-*.jsonl > $L/bench_summary.txt
fi

if [ $what = kld ] || [ $what = all ]; then
  # bf16 权重不变，只换 KV 类型，和 bf16 权重加 f16 KV 的 logits 比，单看 KV 量化的损失
  kld() { name=$1; shift; $B/llama-perplexity -m $M/qwen3-0.6b-bf16.gguf -t 4 --chunks 20 \
      --kl-divergence-base $M/base.kld --kl-divergence "$@" > $L/kld-$name.log 2>&1; }
  kld f16
  kld q8_0 -ctk q8_0 -ctv q8_0
  kld q4_0 -ctk q4_0 -ctv q4_0
  kld q8_0-q4_0 -ctk q8_0 -ctv q4_0
  kld q4_0-q8_0 -ctk q4_0 -ctv q8_0
  # Hadamard 旋转默认对量化 K、V 打开，用环境变量关掉看它挡住了多少误差
  LLAMA_ATTN_ROT_DISABLE=1 kld q8_0-norot -ctk q8_0 -ctv q8_0
  LLAMA_ATTN_ROT_DISABLE=1 kld q4_0-norot -ctk q4_0 -ctv q4_0
  python3 $HERE/collect_kld.py $L/kld-*.log > $L/kld_summary.txt
fi
if [ $what = ctx ] || [ $what = all ]; then
  # 同一份文本，每块长度不同：上下文越长，每个 token 看得到的前文越多
  # 8192 在 4 核 7 GiB 的机器上被 Bus error 杀掉，只测到 4096；4096 用 -b 1024、8 块
  $B/llama-perplexity -m $M/q-Q4_K_M.gguf -f $WIKI -c 512  -t 4 --chunks 64 > $L/ppl-c512.log 2>&1
  $B/llama-perplexity -m $M/q-Q4_K_M.gguf -f $WIKI -c 2048 -t 4 --chunks 16 > $L/ppl-c2048.log 2>&1
  $B/llama-perplexity -m $M/q-Q4_K_M.gguf -f $WIKI -c 4096 -b 1024 -t 4 --chunks 8 > $L/ppl-c4096.log 2>&1
  # 训练长度以内也打开静态缩放，量它对短文本的损失
  $B/llama-perplexity -m $M/q-Q4_K_M.gguf -f $WIKI -c 2048 -t 4 --chunks 16 \
      --rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 32768 > $L/ppl-c2048-yarn4.log 2>&1
  $B/llama-perplexity -m $M/q-Q4_K_M.gguf -f $WIKI -c 2048 -t 4 --chunks 16 \
      --rope-scaling linear --rope-scale 4 > $L/ppl-c2048-linear4.log 2>&1
  # 上下文写满以后：缺省停下，打开 --context-shift 才丢掉一半继续写
  $B/llama-completion -m $M/q-Q4_K_M.gguf -c 256 -n 600 -p "Once upon a time" -no-cnv -t 4 --temp 0 > $L/shift-off.log 2>&1
  $B/llama-completion -m $M/q-Q4_K_M.gguf -c 256 -n 600 -p "Once upon a time" -no-cnv -t 4 --temp 0 --context-shift --keep 8 > $L/shift-on.log 2>&1
fi

if [ $what = fa ] || [ $what = all ]; then
  # FA 开关对 compute buffer 的影响，要 -v 才打印
  : > $L/fa_buffer.txt
  for c in 4096 16384; do for fa in on off; do
    echo "== ctx $c fa $fa" >> $L/fa_buffer.txt
    timeout 120 $B/llama-completion -m $M/q-Q4_K_M.gguf -c $c -fa $fa -n 1 -p hi -no-cnv -t 4 -v 2>&1 |
      grep -E "llama_kv_cache: size|compute buffer size" >> $L/fa_buffer.txt
  done; done
fi
echo DONE $what
