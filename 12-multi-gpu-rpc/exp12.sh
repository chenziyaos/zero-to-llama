#!/bin/bash
# 12 实验：RPC 消息尺寸、-sm layer 切层账本、假多卡上的调度和显存、三种并行的通信量
# 用法：bash exp12.sh [msg|split|fake|comm|all]
# 环境变量：LLAMA_DIR（缺省 ../../../llama.cpp，用它的 build/bin 和头文件），MODELS
#
# 本机没有 GPU，也不允许起监听端口的进程，所以这里不跑 rpc-server、不跑 llama-server。
# msg 只用编译器算结构体大小；fake 在进程里注册几张"假 GPU"（权重不分配），
# 让 llama.cpp 真的走一遍选设备、切层、流水线开关和调度器切图，数字都来自 llama.cpp 自己的日志和账本。
# 速度相关的结论只能引用 PR 里的实测，见文章正文。
set -u
HERE=$(cd $(dirname $0) && pwd)
LD=${LLAMA_DIR:-$HERE/../../../llama.cpp}
B=$LD/build/bin
MD=${MODELS:-$HERE/../../../models}
Q=$MD/q-Q4_0.gguf
H=$MD/hdr
L=$HERE/logs
mkdir -p $L
what=${1:-all}
run() { [ $what = $1 ] || [ $what = all ]; }

if run msg; then
  [ -x $HERE/rpc_msg ] || cc -O2 -Wall $HERE/rpc_msg.c -o $HERE/rpc_msg
  # 三档图：0.6B 一次 decode 的图大约 1000 个节点；后两档是 7B、70B 量级的节点数
  $HERE/rpc_msg 1000 1500 3000 4500 8000 12000 > $L/rpc_msg.txt
fi

if run split; then
  cd $HERE
  {
    python3 split_plan.py $Q --free 8000,8000 --layers
    python3 split_plan.py $Q --free 8000,8000 --ts 3,1
    python3 split_plan.py $Q --free 8000,8000 --ngl 20
    python3 split_plan.py $H/Qwen3-14B-Q4_K_M.head --free 24000,12000 --rpc 1
    python3 split_plan.py $H/Qwen3-14B-Q4_K_M.head --free 12000,12000,12000
    python3 split_plan.py $H/qwen3-30b-a3b-Q4_K_M.head --free 24000,24000 --ctx 32768
    python3 split_plan.py $H/qwen3-30b-a3b-Q4_K_M.head --free 24000,24000 --ctx 32768 --ts 1,1
  } > $L/split.txt 2>&1
fi

if run fake; then
  [ -x $HERE/fake_devs ] || g++ -O2 -std=c++17 -Wall $HERE/fake_devs.cpp -o $HERE/fake_devs \
      -I$LD/include -I$LD/ggml/include -I$LD/src -I$LD/ggml/src \
      -L$B -lllama -lggml -lggml-base -lggml-cpu -Wl,-rpath,$B
  F=$HERE/fake_devs
  $F $Q --n 2                              > $L/fake-n2.txt 2>&1
  $F $Q --n 2 --events 1                   > $L/fake-n2-events.txt 2>&1
  $F $Q --n 3                              > $L/fake-n3.txt 2>&1
  $F $Q --n 1 --rpc 1 --free 24000,12000   > $L/fake-rpc.txt 2>&1
  $F $Q --n 2 --ts 3,1                     > $L/fake-ts31.txt 2>&1
  $F $Q --n 2 --free 12000,4000            > $L/fake-free124.txt 2>&1
  $F $Q --n 2 --ngl 20                     > $L/fake-ngl20.txt 2>&1
  $F $Q --n 2 --ub 2048                    > $L/fake-ub2048.txt 2>&1
  $F $Q --n 2 --ctx 32768                  > $L/fake-ctx32k.txt 2>&1
  $F $Q --n 2 --sm tensor                  > $L/fake-tp2.txt 2>&1
  $F $Q --n 4 --sm tensor                  > $L/fake-tp4.txt 2>&1
  # 调度器切出来的每一段和它的输入张量
  GGML_SCHED_DEBUG=1 $F $Q --n 3           > $L/fake-sched.txt 2>&1
fi

if run comm; then
  cd $HERE
  {
    python3 comm_volume.py $Q --dev 2,3,4 --tokens 1,512 --rpc-graph 1000,1500
    # 下面两组的 n_embd、n_layer 对应 PR #19378 里测过的 llama 8B 和 70B
    python3 comm_volume.py --n-embd 4096 --n-layer 32 --name llama-8B --dev 2,4 --tokens 1,512,2048 --gbps 32
    python3 comm_volume.py --n-embd 8192 --n-layer 80 --name llama-70B --dev 4 --tokens 1,512 --gbps 32
  } > $L/comm.txt 2>&1
fi
