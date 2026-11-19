#!/bin/bash
# 13 实验：投机解码的接受率、加速比和各类草稿来源
# 用法：bash exp13.sh [cost|base|draft|pmin|temp|ngram|lookup|granite|stoch|base2|noise|inter|unit|all]
# 环境变量：LLAMA_DIR（缺省 ../../../llama.cpp），MODELS（缺省 ../../../models）
#
# 只用不监听端口的程序：llama-bench、llama-completion、llama-speculative-simple、llama-speculative、
# llama-lookup、llama-lookahead 和单元测试。llama-server 的 --spec-synth-* 和 SPEED-Bench 不在这里跑。
# 本机 4 核 CPU，没有 GPU，数字只代表这台机器上的相对关系。
set -u
HERE=$(cd $(dirname $0) && pwd)
LD=${LLAMA_DIR:-$HERE/../../../llama.cpp}
B=$LD/build/bin
MD=${MODELS:-$HERE/../../../models}
P=$HERE/prompts
L=$HERE/logs
mkdir -p $L
what=${1:-all}
run() { [ $what = $1 ] || [ $what = all ]; }
T="-t 4 -c 4096"
N=256
K='encoded|decoded|n_draft|n_predict|n_drafted|n_accept|accept |statistics|checkpoints|error|failed|resetting|occupancy|W = |N = |G = |t_draft'
# -v 打开 statistics 行，但整份日志每百 token 上 MB，只留统计相关的行
sp() { f=$1; shift; $B/llama-speculative-simple "$@" -v 2>&1 > $L/$f.out | grep -aE "$K" > $L/$f.err; }

# 一次前向喂 k 个 token 的代价：验证 k 个草稿 token 要算 k+1 个位置
if run cost; then
  for m in q-Q8_0 qwen3-0.6b-bf16 q-Q4_0 q-Q2_K qi-IQ2_M; do
    $B/llama-bench -m $MD/$m.gguf -t 4 -r 3 -p 1,2,3,5,9,17 -n 64 -o csv > $L/cost-$m.csv 2> $L/cost-$m.err
  done
fi

# 不开投机的基线：同样的提示、同样的输出长度
if run base; then
  for m in q-Q8_0 qwen3-0.6b-bf16; do
    for p in code prose rewrite; do
      $B/llama-completion -m $MD/$m.gguf $T -n $N --temp 0 -f $P/$p.txt -no-cnv > $L/base-$m-$p.out 2> $L/base-$m-$p.err
    done
  done
fi

# 草稿模型：三种草稿量化 × n_max
if run draft; then
  for tgt in q-Q8_0 qwen3-0.6b-bf16; do
    for dft in q-Q4_0 q-Q2_K qi-IQ2_M; do
      for k in 1 2 4 8 16; do
        for p in code prose; do
          f=draft-$tgt-$dft-k$k-$p
          sp $f -m $MD/$tgt.gguf -md $MD/$dft.gguf --spec-type draft-simple \
            --spec-draft-n-max $k --temp 0 $T -n $N -f $P/$p.txt
        done
      done
    done
  done
fi

# p_min：草稿自己没把握就早停
if run pmin; then
  for pm in 0.0 0.3 0.6 0.8 0.9; do
    for p in code prose; do
      f=pmin-$pm-$p
      sp $f -m $MD/q-Q8_0.gguf -md $MD/q-Q4_0.gguf --spec-type draft-simple \
        --spec-draft-n-max 16 --spec-draft-p-min $pm --temp 0 $T -n $N -f $P/$p.txt
    done
  done
fi

# 温度大于 0：草稿取 argmax 再逐个比对，和草稿按分布采样再做拒绝采样
if run temp; then
  for mode in greedy probabilistic; do
    for s in 1 2 3 4 5; do
      for p in code prose; do
        f=temp-$mode-s$s-$p
        sp $f -m $MD/q-Q8_0.gguf -md $MD/q-Q4_0.gguf --spec-type draft-simple \
          --spec-draft-n-max 4 --spec-draft-sampling $mode --temp 0.8 --seed $s $T -n $N -f $P/$p.txt
      done
    done
  done
fi

# 不要草稿模型的 n-gram 类方法
if run ngram; then
  for t in ngram-simple ngram-map-k ngram-map-k4v ngram-mod ngram-cache; do
    for p in rewrite code prose; do
      f=ngram-$t-$p
      sp $f -m $MD/q-Q8_0.gguf --spec-type $t --temp 0 $T -n $N -f $P/$p.txt
    done
  done
  # ngram-mod 默认 n_min=48，草稿凑不满 48 个就作废；放宽后再测一次
  for nm in 4 16; do
    for p in rewrite code; do
      f=ngram-mod-min$nm-$p
      sp $f -m $MD/q-Q8_0.gguf --spec-type ngram-mod --spec-ngram-mod-n-min $nm \
        --spec-ngram-mod-n-match 8 --temp 0 $T -n $N -f $P/$p.txt
    done
  done
fi

# llama-lookup（prompt lookup）和 llama-lookahead（Jacobi 迭代）
if run lookup; then
  for p in rewrite code prose; do
    $B/llama-lookup -m $MD/q-Q8_0.gguf --spec-draft-n-max 8 --temp 0 $T -n $N -f $P/$p.txt > $L/lookup-$p.out 2> $L/lookup-$p.err
    $B/llama-lookahead -m $MD/q-Q8_0.gguf --temp 0 $T -n $N -f $P/$p.txt > $L/lookahead-$p.out 2> $L/lookahead-$p.err
  done
fi

# 另一组同族模型：granite MoE 3B 配 1B
if run granite; then
  G=$MD/moe
  $B/llama-completion -m $G/granite-3b-a800m-Q4_K_M.gguf $T -n $N --temp 0 -f $P/en.txt -no-cnv > $L/granite-base.out 2> $L/granite-base.err
  for k in 2 4 8; do
    sp granite-k$k -m $G/granite-3b-a800m-Q4_K_M.gguf -md $G/granite-1b-a400m-Q4_K_M.gguf \
      --spec-type draft-simple --spec-draft-n-max $k --temp 0 $T -n $N -f $P/en.txt
  done
fi

# llama-speculative-simple 只做逐个比对；老的 llama-speculative 在 temp>0 且单序列时做拒绝采样
# 同样 temp 0.8、k=4、5 个种子，对照 temp 段的 speculative-simple
if run stoch; then
  for s in 1 2 3 4 5; do
    for p in code prose; do
      $B/llama-speculative -m $MD/q-Q8_0.gguf -md $MD/q-Q4_0.gguf --spec-draft-n-max 4 \
        --temp 0.8 --seed $s $T -n $N -f $P/$p.txt > $L/stoch-s$s-$p.out 2> $L/stoch-s$s-$p.err.full
      grep -aE "$K" $L/stoch-s$s-$p.err.full > $L/stoch-s$s-$p.err; rm -f $L/stoch-s$s-$p.err.full
    done
  done
fi

# 同一个程序的“零草稿”基线：ngram-simple 的查找 n-gram 设成 1024，上下文里不可能命中，
# 一次草稿都不出，剩下的就是 speculative-simple 自己的解码循环。每组跑 2 次看噪声
if run base2; then
  for m in q-Q8_0 qwen3-0.6b-bf16; do
    for p in code prose rewrite; do
      for r in 1 2; do
        sp nodraft-$m-$p-r$r -m $MD/$m.gguf --spec-type ngram-simple --spec-ngram-simple-size-n 1024 \
          --temp 0 $T -n $N -f $P/$p.txt
      done
    done
  done
fi

# 噪声：llama-completion 和零草稿的 speculative-simple 交替各跑 3 次，看同一配置前后差多少
if run noise; then
  for p in code rewrite; do
    for r in 1 2 3; do
      $B/llama-completion -m $MD/q-Q8_0.gguf $T -n $N --temp 0 -f $P/$p.txt -no-cnv > /dev/null 2> $L/noise-comp-$p-r$r.err
      sp noise-nd-$p-r$r -m $MD/q-Q8_0.gguf --spec-type ngram-simple --spec-ngram-simple-size-n 1024 \
        --temp 0 $T -n $N -f $P/$p.txt
    done
  done
fi

# 机器速度会随时间漂，几个关键对比交替跑 3 轮，加速比按同一轮里的两个数算
if run inter; then
  for r in 1 2 3; do
    for tgt in q-Q8_0 qwen3-0.6b-bf16; do
      $B/llama-completion -m $MD/$tgt.gguf $T -n $N --temp 0 -f $P/code.txt -no-cnv > /dev/null 2> $L/inter-base-$tgt-r$r.err
      sp inter-k4-$tgt-r$r -m $MD/$tgt.gguf -md $MD/q-Q4_0.gguf --spec-type draft-simple \
        --spec-draft-n-max 4 --temp 0 $T -n $N -f $P/code.txt
    done
    sp inter-nd-rewrite-r$r -m $MD/q-Q8_0.gguf --spec-type ngram-simple --spec-ngram-simple-size-n 1024 \
      --temp 0 $T -n $N -f $P/rewrite.txt
    sp inter-cache-rewrite-r$r -m $MD/q-Q8_0.gguf --spec-type ngram-cache --temp 0 $T -n $N -f $P/rewrite.txt
  done
fi

if run unit; then
  # 随机生成整个词表（151936）上的 q 做草稿，拒绝采样后比对经验分布和目标分布，容差 6 倍标准误。
  # 目标只挂温度采样器，候选也是整个词表；拒绝时算残差要对每个候选在 q 里线性查找，一次是 V² 量级。
  # 本机一次试验约 2.7 秒，缺省 5000 次约 3.8 小时。这里只跑 5 次确认程序能走通：
  # 5 次时任何一个概率低于 0.5% 的 token 被抽中一次就超出容差，所以“失败”是预期的，不说明实现有错。
  s=$(date +%s)
  $B/test-speculative-rejection-sampling -m $MD/q-Q4_0.gguf -t 4 --n-trials 5 2>&1 | grep -av '^token' > $L/unit-rejection.txt
  echo "exit=${PIPESTATUS[0]} sec=$(( $(date +%s) - s ))" >> $L/unit-rejection.txt
fi
