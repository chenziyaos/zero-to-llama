#!/bin/bash
# 11 实验：MoE 路由抓取、专家缓存回放、显存账本、CPU 上的 -ncmoe/-cmoe/-ot 和 fit
# 用法：bash exp11.sh [trace|sim|ledger|cpu|fit|all]
# 环境变量：LLAMA_DIR（缺省 ../../../llama.cpp，用它的 build/bin 和头文件），MODELS，WIKI（wiki.test.raw）
# 只用 llama-bench、llama-fit-params 和自己编的 moe_trace；不起任何 server
#
# 本机没有 GPU。trace、sim、ledger 三组和设备无关；cpu 组只说明这几个参数在纯 CPU 构建里怎么落地，
# 不代表有卡时的速度，有卡的机器上把 cpu 组的命令加上 -ngl all 再跑一遍。
set -u
HERE=$(cd $(dirname $0) && pwd)
LD=${LLAMA_DIR:-$HERE/../../../llama.cpp}
B=$LD/build/bin
MD=${MODELS:-$HERE/../../../models}
W=${WIKI:-$HERE/../../../data/wiki.test.raw}
M1=$MD/moe/granite-1b-a400m-Q4_K_M.gguf
M3=$MD/moe/granite-3b-a800m-Q4_K_M.gguf
L=$HERE/logs
mkdir -p $L
what=${1:-all}
tag() { sed "s/^{/{\"$1\": \"$2\", /"; }
bench() { "$@" -o jsonl 2>/dev/null; }
run() { [ $what = $1 ] || [ $what = all ]; }

if run trace; then
  [ -x $HERE/moe_trace ] || cc -O2 -Wall $HERE/moe_trace.c -I$LD/include -I$LD/ggml/include -L$B \
      -lllama -lggml -lggml-base -lm -Wl,-rpath,$B -o $HERE/moe_trace
  cd $HERE
  : > $L/run-trace.log
  # wiki 一段 512 个 token 做 prefill；生成时只记生成出来的 256 个 token
  ./moe_trace $M1 text $W $L/trace-1b-wiki.bin 16384 512 >> $L/run-trace.log 2>&1
  ./moe_trace $M3 text $W $L/trace-3b-wiki.bin 8192 512 >> $L/run-trace.log 2>&1
  ./moe_trace $M1 gen 'Write a Python function that parses a CSV file and returns the average of each numeric column.' \
      $L/trace-1b-gen-code.bin 256 >> $L/run-trace.log 2>&1
  ./moe_trace $M1 gen 'The history of the Roman Empire began' $L/trace-1b-gen-prose.bin 256 >> $L/run-trace.log 2>&1
fi

if run sim; then
  cd $HERE
  # 槽位按单个 token 的用量（层数 × n_used）取倍数；生成的 trace 只有 256 个 token，前 64 个暖缓存，
  # hot 列用 wiki 的频次去钉，看换了领域之后静态钉法还剩多少
  python3 moe_cache_sim.py $L/trace-1b-wiki.bin --tokens 8192 --stats > $L/sim-1b-wiki.txt
  python3 moe_cache_sim.py $L/trace-3b-wiki.bin --tokens 8192 --stats > $L/sim-3b-wiki.txt
  python3 moe_cache_sim.py $L/trace-1b-gen-code.bin --warm 64 --stats --profile $L/trace-1b-wiki.bin > $L/sim-1b-code.txt
  python3 moe_cache_sim.py $L/trace-1b-gen-prose.bin --warm 64 --stats --profile $L/trace-1b-wiki.bin > $L/sim-1b-prose.txt
fi

if run ledger; then
  cd $HERE
  H=$MD/hdr
  {
    for m in $M1 $M3; do
      python3 vram_ledger.py $m --ctx 4096
      python3 vram_ledger.py $m --ctx 4096 --cmoe
    done
    # 只有张量头的截断 GGUF 就够算账
    Q=$H/qwen3-30b-a3b-Q4_K_M.head
    python3 vram_ledger.py $Q --ctx 4096
    python3 vram_ledger.py $Q --ctx 4096 --cmoe
    for b in 24576 16384 12288 8192; do python3 vram_ledger.py $Q --ctx 4096 --budget-mib $b | tail -1; done
    # 和 fit 的 README 对账：那里的数字是 F16 权重、旧的 -ngl 计数，-ngl 48 加 -ot 把 14 到 47 层的专家留在 CPU
    python3 vram_ledger.py $Q --ctx 4096 --as-type F16 --ngl 48 --cpu-exps 14-47 --legacy
    python3 vram_ledger.py $Q --ctx 4096 --as-type F16 --ngl 48 --cpu-exps 14-47
    python3 vram_ledger.py $H/Qwen3-14B-Q4_K_M.head --ctx 4096
  } > $L/ledger.txt 2>&1
fi

if run cpu; then
  # llama-bench 没有 -cmoe，用 -ncmoe 99 代替，效果相同
  # 纯 CPU 构建：-ngl 被忽略，-cmoe/-ot 只是把专家张量的缓冲类型改成 CPU，q4_K 的专家照样被 repack
  : > $L/cpu.jsonl
  for r in 1 2; do
    for m in 1b:$M1 3b:$M3; do
      n=${m%%:*}; f=${m#*:}
      bench $B/llama-bench -m $f -p 128 -n 32 -r 3 -t 4 | tag cfg default | tag model $n | tag round $r >> $L/cpu.jsonl
      bench $B/llama-bench -m $f -p 128 -n 32 -r 3 -t 4 -ncmoe 12 | tag cfg ncmoe12 | tag model $n | tag round $r >> $L/cpu.jsonl
      bench $B/llama-bench -m $f -p 128 -n 32 -r 3 -t 4 -ncmoe 99 | tag cfg cmoe | tag model $n | tag round $r >> $L/cpu.jsonl
      bench $B/llama-bench -m $f -p 128 -n 32 -r 3 -t 4 -ot 'exps=CPU' | tag cfg ot-exps | tag model $n | tag round $r >> $L/cpu.jsonl
    done
  done
  # 加载日志里看缓冲类型被改成了什么
  $B/llama-bench -m $M1 -p 8 -n 0 -r 1 -t 4 -ncmoe 99 -v 2>&1 | grep -E 'overridden|cannot be used|CPU_REPACK|no usable GPU' \
      | sort -u > $L/cpu-override.txt
fi

if run fit; then
  # 没有 GPU 时 fit 只看主存
  for m in $M1 $M3; do
    echo "== $(basename $m)"
    # -v 才会打出 memory breakdown；CPU_REPACK 那一行不进 Host 的合计
    $B/llama-fit-params -m $m -c 4096 -v 2>&1 | grep -E 'common_params_fit_impl|common_fit_params|memory_breakdown|^-c' \
        | sed -E 's/^[0-9.]+ [A-Z] //'
  done > $L/fit.txt
fi
