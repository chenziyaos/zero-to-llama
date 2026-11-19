#!/bin/bash
# 17 实验：默认值、误配目录、按症状扫参数、诊断脚本的回放
# 用法：bash exp17.sh [defaults|misconfig|symptoms|diag|plan|unit|all]
# 环境变量：LLAMA_DIR（缺省 ../../../llama.cpp，用它的 build/bin），MODELS
# 只用 llama-completion、llama-bench 和自己的 Python；不起任何 server
set -u
HERE=$(cd $(dirname $0) && pwd)
LD=${LLAMA_DIR:-$HERE/../../../llama.cpp}
B=$LD/build/bin
MD=${MODELS:-$HERE/../../../models}
M=$MD/q-Q4_K_M.gguf
L=$HERE/logs
mkdir -p $L
what=${1:-all}
tag() { sed "s/^{/{\"$1\": \"$2\", /"; }
bench() { "$@" -o jsonl 2>/dev/null; }
run() { [ $what = $1 ] || [ $what = all ]; }
P='Explain in two sentences why the sky is blue.'
comp() { $B/llama-completion -m "$1" -p "$P" -no-cnv --seed 1 "${@:2}" < /dev/null; }

# 什么都不写时拿到的参数：n_ctx 取训练长度、KV 一次分配到位；再对照 -c 4096。rss.py 报峰值常驻内存
if run defaults; then
  : > $L/defaults.txt
  comp $M -n 32 > /dev/null 2> $L/default.log
  comp $M -n 32 -lv 4 > /dev/null 2> $L/default-v4.log
  comp $M -n 32 -c 4096 -lv 4 > /dev/null 2> $L/c4096-v4.log
  echo "noc   $(python3 $HERE/../16-router-ops/rss.py -- $B/llama-completion -m $M -p "$P" -no-cnv -n 8 --seed 1)" >> $L/defaults.txt
  echo "c4096 $(python3 $HERE/../16-router-ops/rss.py -- $B/llama-completion -m $M -p "$P" -no-cnv -n 8 --seed 1 -c 4096)" >> $L/defaults.txt
fi

# 误配目录：每种写法单独起一个进程，记退出码、墙钟和 stderr（-lv 4 才打 n_ubatch、n_seq_max），再交给 diagnose.py 归类
if run misconfig; then
  Q=$MD/q-Q4_0.gguf
  : > $L/misconfig.txt
  mis() {
    name=$1; shift
    t0=$(date +%s.%N)
    $B/llama-completion -m $Q -p "$P" -n 16 -no-cnv --seed 1 -lv 4 "$@" < /dev/null > /dev/null 2> $L/mis-$name.log
    rc=$?
    t1=$(date +%s.%N)
    echo "$name rc=$rc wall=$(awk "BEGIN{printf \"%.1f\", $t1 - $t0}") args=$*" >> $L/misconfig.txt
  }
  mis base -c 4096
  mis c100k -c 100000
  mis ctv-fa-off -c 4096 -ctv q8_0 -fa off
  mis ngl -c 4096 -ngl 10
  mis smrow -c 4096 -sm row
  mis mask -c 4096 -C 0x3 --cpu-strict 1 -t 4
  mis t8 -c 4096 -t 8
  mis b256 -c 4096 -b 256 -ub 512
  mis np4 -c 4096 -np 4
  mis ub16 -c 4096 -ub 16
  mis nofile -m $MD/does-not-exist.gguf
  for f in $L/mis-*.log; do
    echo "== $(basename $f)" >> $L/misconfig.txt
    python3 $HERE/diagnose.py log $f | grep -v '^生效参数' >> $L/misconfig.txt
  done
fi

# 按症状扫：decode 慢看线程和量化、prefill 慢看 repack 和 ubatch、长上下文看 FA 和 KV 类型
if run symptoms; then
  : > $L/symptoms.jsonl
  bench $B/llama-bench -m $M -t 1,2,3,4 -p 128 -n 32 -r 3 | tag grp threads >> $L/symptoms.jsonl
  bench $B/llama-bench -m $MD/q-Q4_0.gguf,$M,$MD/q-Q8_0.gguf -t 4 --repack 0,1 -p 128 -n 32 -r 3 | tag grp repack >> $L/symptoms.jsonl
  bench $B/llama-bench -m $M -t 4 -ub 32,128,512 -p 512 -n 0 -r 3 | tag grp ub >> $L/symptoms.jsonl
  bench $B/llama-bench -m $M -t 4 -fa 0,1 -d 0,4096 -p 0 -n 32 -r 2 | tag grp fa >> $L/symptoms.jsonl
  bench $B/llama-bench -m $M -t 4 -fa 1 -ctk f16,q8_0 -ctv f16,q8_0 -d 4096 -p 0 -n 32 -r 2 | tag grp kvtype >> $L/symptoms.jsonl
fi

# 诊断回放：正常、一个死循环占住 CPU 3、避开它只用 3 个线程，三份日志交给 diagnose.py 判速度
if run diag; then
  comp $M -n 64 -c 4096 -t 4 > /dev/null 2> $L/diag-t4.log
  taskset -c 3 bash -c 'while :; do :; done' & spin=$!
  comp $M -n 64 -c 4096 -t 4 > /dev/null 2> $L/diag-t4-spin.log
  comp $M -n 64 -c 4096 -t 3 -C 0x7 --cpu-strict 1 > /dev/null 2> $L/diag-t3-spin.log
  kill $spin
  : > $L/diag.txt
  for n in t4 t4-spin t3-spin; do
    echo "== $n" >> $L/diag.txt
    python3 $HERE/diagnose.py log $L/diag-$n.log --model $M | grep -v '^生效参数' >> $L/diag.txt
  done
fi

# 事前账：不跑模型，按几组参数算内存和 decode 预测
if run plan; then
  : > $L/plan.txt
  for args in "-c 0" "-c 4096" "-c 4096 -ctk q8_0 -ctv q8_0" "-c 16384" "-c 4096 -np 4"; do
    echo "== q-Q4_K_M $args" >> $L/plan.txt
    python3 $HERE/diagnose.py plan $M $args >> $L/plan.txt
  done
  for m in q-Q4_0 q-Q8_0 qwen3-0.6b-bf16; do
    echo "== $m -c 4096" >> $L/plan.txt
    python3 $HERE/diagnose.py plan $MD/$m.gguf -c 4096 >> $L/plan.txt
  done
fi

if run unit; then
  python3 $HERE/test_diagnose.py > $L/unit.out 2>&1; tail -3 $L/unit.out
fi
