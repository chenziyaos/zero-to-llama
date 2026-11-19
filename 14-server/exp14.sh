#!/bin/bash
# 14 实验：服务端的并发、批处理和共享前缀
# 用法：bash exp14.sh [bb|pps|wt|par|parpps|unit|rep|ledger|sim|all]
# 环境变量：LLAMA_DIR（缺省 ../../../llama.cpp），MODELS（缺省 ../../../models）
#
# 这里不启动 llama-server，也不启动任何监听端口的程序。
# 服务端的调度逻辑用三样东西逼近：
#   llama-batched-bench  同一个上下文里 B 条序列一起解码，测吞吐随 B 的变化
#   llama-parallel       模拟 N 个客户端轮流投递请求，连续批处理，带共享系统提示
#   test-server-component 服务端组件的单元测试，不加载模型、不开端口
# 槽位选择和提示缓存的淘汰规则用 slot_sim.py 按源码复刻。
# 本机 4 核 CPU，没有 GPU，数字只代表这台机器上的相对关系。
set -u
HERE=$(cd $(dirname $0) && pwd)
LD=${LLAMA_DIR:-$HERE/../../../llama.cpp}
B=$LD/build/bin
MD=${MODELS:-$HERE/../../../models}
L=$HERE/logs
mkdir -p $L
what=${1:-all}
run() { [ $what = $1 ] || [ $what = all ]; }
M=$MD/q-Q8_0.gguf
NPL=1,2,4,8,16

# 并发路数 B 对吞吐的影响：各序列独立提示，分开的 KV 和统一 KV 各跑一遍
if run bb; then
  for kv in nokvu kvu; do
    f=""; [ $kv = kvu ] && f="-kvu"
    $B/llama-batched-bench -m $M -t 4 -c 8192 -b 2048 -ub 512 $f \
      -npp 128,384 -ntg 64 -npl $NPL --output-format jsonl > $L/bb-$kv.jsonl 2> $L/bb-$kv.err
  done
fi

# 共享提示：只算一次前缀，再复制给 B 条序列
if run pps; then
  for kv in nokvu kvu; do
    f=""; [ $kv = kvu ] && f="-kvu"
    $B/llama-batched-bench -m $M -t 4 -c 8192 -b 2048 -ub 512 $f -pps \
      -npp 128,384 -ntg 64 -npl $NPL --output-format jsonl > $L/pps-$kv.jsonl 2> $L/pps-$kv.err
  done
fi

# 权重越大，解码越受内存带宽限制，并发摊薄权重读取的收益越大
if run wt; then
  for m in q-Q4_0 qwen3-0.6b-bf16; do
    $B/llama-batched-bench -m $MD/$m.gguf -t 4 -c 8192 -kvu \
      -npp 128 -ntg 64 -npl $NPL --output-format jsonl > $L/wt-$m.jsonl 2> $L/wt-$m.err
  done
fi

# 连续批处理下的多客户端：每个客户端轮流处理 16 条请求
K='n_parallel|Client|Total|Cache misses|clearing'
if run par; then
  for np in 1 2 4 8; do
    $B/llama-parallel -m $M -t 4 -c 8192 -np $np -ns 16 -n 64 --temp 0 --junk 1 -s 1 \
      > $L/par-np$np.out 2> $L/par-np$np.full
    grep -aE "$K" $L/par-np$np.full > $L/par-np$np.err; rm -f $L/par-np$np.full
  done
fi

# 系统提示共享与否，以及提示里夹带 junk 文本时
if run parpps; then
  for np in 4 8; do
    $B/llama-parallel -m $M -t 4 -c 8192 -np $np -ns 16 -n 64 --temp 0 --junk 1 -s 1 -pps \
      > $L/parpps-np$np.out 2> $L/parpps-np$np.full
    grep -aE "$K" $L/parpps-np$np.full > $L/parpps-np$np.err; rm -f $L/parpps-np$np.full
  done
  $B/llama-parallel -m $M -t 4 -c 8192 -np 4 -ns 16 -n 64 --temp 0 --junk 6 -s 1 \
    > $L/parjunk-np4.out 2> $L/parjunk-np4.full
  grep -aE "$K" $L/parjunk-np4.full > $L/parjunk-np4.err; rm -f $L/parjunk-np4.full
fi

# 服务端组件单元测试：不加载模型、不开端口
if run unit; then
  $B/test-server-component > $L/unit-server.out 2>&1
  python3 $HERE/test_server.py > $L/unit-sim.out 2>&1
fi

# 机器速度会漂：B=1 和 B=16、分开和统一 KV 交替跑 3 轮，比值按同一轮算
if run rep; then
  for r in 1 2 3; do
    for kv in nokvu kvu; do
      f=""; [ $kv = kvu ] && f="-kvu"
      $B/llama-batched-bench -m $M -t 4 -c 8192 -b 2048 -ub 512 $f \
        -npp 128 -ntg 64 -npl 1,16 --output-format jsonl > $L/rep$r-$kv.jsonl 2> /dev/null
    done
  done
fi

# KV 账本对账：用真实日志核对 n_ctx_seq 和 KV 大小，和 slot_sim.py 的 ctx_layout 对得上
if run ledger; then
  for c in 4096 10000; do
    for np in 3 4; do
      for kv in nokvu kvu; do
        f=""; [ $kv = kvu ] && f="-kvu"
        $B/llama-batched-bench -m $M -t 4 -c $c $f -npp 8 -ntg 1 -npl $np -v 2>&1 \
          | grep -aE "llama_context: (n_seq_max|n_ctx|n_ctx_seq|kv_unified) |llama_kv_cache: size" > $L/ledger-c$c-np$np-$kv.log
      done
    done
  done
fi

# 槽位选择、提示缓存淘汰和 KV 账本
if run sim; then
  python3 $HERE/slot_sim.py > $L/sim.out
fi
