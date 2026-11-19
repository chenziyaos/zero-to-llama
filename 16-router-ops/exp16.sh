#!/bin/bash
# 16 实验：router 多模型的内存账、配置优先级、加载耗时和调度规则
# 用法：bash exp16.sh [fit|auto|sysconf|load|rss|metrics|preset|sim|unit|all]
# 环境变量：LLAMA_DIR（缺省 ../../../llama.cpp），MODELS（缺省 ../../../models）
#
# 这里不启动 llama-server，也不启动任何监听端口的程序。router 的行为用三样东西逼近：
#   llama-fit-params   和 router 的测量子进程调同一套 --fit 代码，算 min 和 max 两档内存
#   llama-completion   加载一次模型就退出，量冷启动和热启动的加载耗时
#   preset_tool.py / router_sim.py  按源码复刻预设合并和 LRU 卸载
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
MODELS3="q-Q4_0 q-Q8_0 qwen3-0.6b-bf16"

# 内存三项：权重、KV、计算缓冲，单位 MiB。-c 4096 对应测量子进程的 min，40960 是训练长度
if run fit; then
  : > $L/fit_print.txt
  for m in $MODELS3; do
    for np in 1 4; do
      for c in 4096 16384 40960; do
        echo "$m np=$np c=$c $($B/llama-fit-params -m $MD/$m.gguf -c $c -np $np -fitp on 2>/dev/null)" >> $L/fit_print.txt
      done
    done
  done
fi

# 不写 -c 时 --fit 自己挑上下文：留给系统的余量 -fitt 从 1 GiB 加到 4 GiB
if run auto; then
  : > $L/fit_auto.txt
  for m in q-Q8_0 qwen3-0.6b-bf16; do
    for np in 1 4; do
      for t in 1024 2048 3072 4096; do
        out=$($B/llama-fit-params -m $MD/$m.gguf -np $np -fitt $t 2>/dev/null)
        echo "$m np=$np fitt=$t ${out:-FAIL}" >> $L/fit_auto.txt
      done
    done
  done
  grep MemTotal /proc/meminfo >> $L/fit_auto.txt
fi

# 系统级配置文件、环境变量、命令行三者的优先级，用 fit-params 打印出来的 KV 大小判断谁生效
if run sysconf; then
  X=$(mktemp -d)
  mkdir -p $X/llama.cpp
  printf '[*]\nctx-size = 8192\n' > $X/llama.cpp/config.ini
  M=$MD/q-Q8_0.gguf
  {
    echo "none    $($B/llama-fit-params -m $M -c 4096 -fitp on 2>/dev/null)"
    echo "config  $(XDG_CONFIG_HOME=$X $B/llama-fit-params -m $M -fitp on 2>/dev/null)"
    echo "env     $(XDG_CONFIG_HOME=$X LLAMA_ARG_CTX_SIZE=2048 $B/llama-fit-params -m $M -fitp on 2>/dev/null)"
    echo "cli     $(XDG_CONFIG_HOME=$X LLAMA_ARG_CTX_SIZE=2048 $B/llama-fit-params -m $M -c 1024 -fitp on 2>/dev/null)"
    XDG_CONFIG_HOME=$X $B/llama-fit-params -m $M -fitp on 2>&1 >/dev/null | grep -a "config file" | sed "s#$X#\$XDG_CONFIG_HOME#"
  } > $L/sysconf.txt
  rm -rf $X
fi

# 加载耗时：每个模型连跑 3 次，第 1 次可能是冷的（页缓存里没有），后两次是热的
if run load; then
  : > $L/load.txt
  for m in $MODELS3; do
    for r in 1 2 3; do
      t=$($B/llama-completion -m $MD/$m.gguf -p hi -n 1 -c 4096 -t 4 -no-cnv 2>&1 | grep -a "load time" | sed 's/.*= *\([0-9.]*\) ms.*/\1/')
      echo "$m run=$r load_ms=$t" >> $L/load.txt
    done
  done
fi

# 进程峰值常驻内存：-c 4096 和 8192 各一次，KV 分配后清零，页是真占的
if run rss; then
  : > $L/rss.txt
  for c in 4096 8192 16384; do
    echo "q-Q8_0 c=$c $(python3 $HERE/rss.py -- $B/llama-completion -m $MD/q-Q8_0.gguf -p hi -n 1 -c $c -t 4 -no-cnv)" >> $L/rss.txt
  done
  V=$MD/vl/SmolVLM2-256M-Video-Instruct-Q8_0.gguf
  echo "smolvlm c=4096 text $(python3 $HERE/rss.py -- $B/llama-completion -m $V -p hi -n 1 -c 4096 -t 4 -no-cnv)" >> $L/rss.txt
  echo "smolvlm c=4096 mmproj $(python3 $HERE/rss.py -- $B/llama-mtmd-cli -m $V --mmproj $MD/vl/mmproj-SmolVLM2-256M-Video-Instruct-Q8_0.gguf \
    -c 4096 -t 4 --temp 0 -n 1 --image $LD/tools/mtmd/test-1.jpeg -p hi)" >> $L/rss.txt
  # 测量子进程看得到的三项（不含 mmproj），和 mtmd 日志里 clip 自己报的大小
  echo "smolvlm fit $($B/llama-fit-params -m $V -c 4096 -fitp on 2>/dev/null)" >> $L/rss.txt
  $B/llama-mtmd-cli -m $V --mmproj $MD/vl/mmproj-SmolVLM2-256M-Video-Instruct-Q8_0.gguf -c 4096 -t 4 --temp 0 -n 1 \
    --image $LD/tools/mtmd/test-1.jpeg -p hi -v 2>&1 >/dev/null | grep -aE "clip_model_loader: model size|clip.*compute buffer size|CPU_Mapped model buffer|CPU KV buffer|CPU compute buffer" >> $L/rss.txt
  grep -E "MemTotal|MemAvailable" /proc/meminfo >> $L/rss.txt
fi

# /metrics 的口径：按源码规则重放请求，速度取 14-server 的 batched-bench 实测
if run metrics; then
  python3 $HERE/metrics_parse.py --sample $L/metrics_sample.txt $HERE/../14-server/logs > $L/metrics.out
fi

# 预设合并：示例 presets.ini 加一条 router 命令行，打印每个模型最后拿到的子进程参数
if run preset; then
  python3 $HERE/preset_tool.py $HERE/presets.ini --router-args "--temp 0 --api-key secret -c 4096" > $L/preset.out 2>&1
fi

# LRU 卸载的模拟
if run sim; then
  python3 $HERE/router_sim.py $L > $L/sim.out
fi

# 单元测试：参数解析、服务端组件，加本目录三个复刻的回归用例
if run unit; then
  { time $B/test-arg-parser > /dev/null 2>&1; echo "rc=$?"; } > $L/test-arg-parser.log 2>&1
  $B/test-server-component > $L/test-server-component.log 2>&1
  python3 $HERE/test_router.py > $L/unit-router.out 2>&1
fi
