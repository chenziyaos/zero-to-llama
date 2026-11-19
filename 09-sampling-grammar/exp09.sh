#!/bin/bash
# 09 实验：采样器耗时和候选数、语法约束的拒绝率、缺省参数从哪来、重复惩罚、后端采样、聊天模板
# 用法：bash exp09.sh [micro|gen|params|rep|bs|tmpl|all]
# 环境变量：LLAMA_DIR（缺省 ../../../llama.cpp），MODELS（模型目录）
# 只用 llama-completion 和本目录的 samp_bench；llama-cli 会在内部起一个 server，本系列不跑
set -u
HERE=$(cd $(dirname $0) && pwd)
LD=${LLAMA_DIR:-$HERE/../../../llama.cpp}
B=$LD/build/bin
M=${MODELS:-$HERE/../../../models}/q-Q4_K_M.gguf
G=$LD/grammars/json.gbnf
L=$HERE/logs
mkdir -p $L
what=${1:-all}
SB=$HERE/samp_bench
[ -x $SB ] || cc -O2 $HERE/samp_bench.c -I$LD/include -I$LD/ggml/include -L$B -lllama -lggml -lggml-base -lm \
  -Wl,-rpath,$B -o $SB
cd /tmp
P_FACT='The capital of France is'
P_STORY='Once upon a time, in a small village,'
P_CODE='def fibonacci(n):'
P_CAT='Return a JSON object describing a cat with fields name, age and color.
JSON:'
comp() { $B/llama-completion -m $M -no-cnv -t 4 "$@"; }

if [ $what = micro ] || [ $what = all ]; then
  # 三种分布：事实（中等熵）、故事（低熵）、代码；每个配置 200 次，语法 20 次
  $SB $M micro "$P_FACT"  200 $L/logits-fact.f32  $G > $L/micro-fact.txt
  $SB $M micro "$P_STORY" 200 $L/logits-story.f32 $G > $L/micro-story.txt
  $SB $M micro "$P_CODE"  200 $L/logits-code.f32  $G > $L/micro-code.txt
  python3 $HERE/samp_sim.py $L/logits-fact.f32 $L/logits-story.f32 $L/logits-code.f32 > $L/sim.txt
fi

if [ $what = gen ] || [ $what = all ]; then
  # 语法约束：先采样后检查（common 的做法）对比先套语法；换 5 个种子看拒绝率的波动
  : > $L/gen.txt
  for s in 1 2 3 4 5; do
    for mode in reject first; do
      echo "== seed $s mode $mode" >> $L/gen.txt
      SEED=$s $SB $M gen "$P_CAT" $G 120 0.8 $mode >> $L/gen.txt 2>&1
    done
  done
  echo "== greedy reject" >> $L/gen.txt
  $SB $M gen "$P_CAT" $G 120 0 reject >> $L/gen.txt 2>&1
  echo "== story reject" >> $L/gen.txt
  $SB $M gen "$P_STORY" $G 120 0.8 reject >> $L/gen.txt 2>&1
  echo "== story first" >> $L/gen.txt
  $SB $M gen "$P_STORY" $G 120 0.8 first >> $L/gen.txt 2>&1
  # JSON schema 走 llama-completion 的完整路径
  comp -n 80 -s 1 --temp 0.8 -p "$P_CAT" \
    -j '{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer","minimum":0},"color":{"enum":["black","white","orange","gray"]}},"required":["name","age","color"],"additionalProperties":false}' \
    > $L/schema.log 2>&1
fi

if [ $what = params ] || [ $what = all ]; then
  # GGUF 里的 general.sampling.* 只在用户没写对应参数时生效
  comp -n 1 -p hi > $L/params-gguf.log 2>&1
  comp -n 1 -p hi --temp 0.8 --top-k 40 > $L/params-user.log 2>&1
  comp -n 1 -p hi --temp 0 > $L/params-greedy.log 2>&1
  grep -hE "top_k = |sampler chain" $L/params-*.log > $L/params.txt
fi

if [ $what = rep ] || [ $what = all ]; then
  # greedy 下的重复：不惩罚、repeat 1.1、presence 1.5、DRY 0.8；输出单独存
  P_LIST='Here is a list of fruits:
1.'
  for cfg in "none" "repeat --repeat-penalty 1.1" "presence --presence-penalty 1.5" "dry --dry-multiplier 0.8"; do
    set -- $cfg; name=$1; shift
    comp -n 200 --temp 0 --ignore-eos --no-display-prompt -p "$P_LIST" "$@" > $L/rep-$name.txt 2> $L/rep-$name.log
  done
  python3 $HERE/collect_rep.py $L/rep-*.txt > $L/rep_summary.txt
fi

if [ $what = bs ] || [ $what = all ]; then
  # 后端采样：把截断和抽样放进计算图，CPU 上看采样时间
  for r in 1 2 3; do
    comp -n 128 -s 1 --temp 0.8 --ignore-eos --no-display-prompt -p 'Once upon a time' > /dev/null 2> $L/bs-off-$r.log
    comp -n 128 -s 1 --temp 0.8 --ignore-eos --no-display-prompt -p 'Once upon a time' -bs > /dev/null 2> $L/bs-on-$r.log
  done
  grep -E "sampling time|samplers time| eval time" $L/bs-*.log > $L/bs.txt
fi

if [ $what = tmpl ] || [ $what = all ]; then
  # 模板展开后的 token；再验证 llama-completion 里 --reasoning off 不改提示词
  for f in "" "--reasoning off" "--chat-template-kwargs {\"enable_thinking\":false}"; do
    echo "== flags: $f" >> $L/tmpl-flags.txt
    comp -cnv -st -p 'What is 2+3?' --verbose-prompt --temp 0 -n 1 $f 2>&1 | grep -E "prompt: |number of tokens|error|invalid" >> $L/tmpl-flags.txt
  done
  comp -cnv -st -p 'What is 2+3?' --verbose-prompt --temp 0 -n 200 > $L/tmpl-verbose.log 2>&1
  python3 $HERE/render_tmpl.py $M --gguf-py $LD/gguf-py > $L/tmpl-render.txt
fi
