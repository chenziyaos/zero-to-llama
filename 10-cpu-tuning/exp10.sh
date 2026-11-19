#!/bin/bash
# 10 实验：线程数和绑核、掩码残留、OpenMP 与自带线程池、poll、优先级、ubatch、repack、tiled、-t 和 -tb、
# OpenMP 等待策略、邻居干扰、带宽上限、加载方式、THP
# 用法：bash exp10.sh [threads|sticky|poll|prio|ub|repack|tiled|tb|omp|noisy|ceiling|load|thp|all]
# 环境变量：LLAMA_DIR（缺省 ../../../llama.cpp，用它的 build/bin），NOOMP_BIN（GGML_OPENMP=OFF 编出的 bin 目录），MODELS
# 只用 llama-bench、llama-completion 和 02 章的 stream；不起任何 server
#
# 注意：llama-bench 在同一个进程里跑完所有组合，线程池每个测试重建，但上一个测试设的线程亲和性会留在
# 主线程和 OpenMP 的常驻线程上。比较绑核效果时，每个组合必须单独起一个进程，sticky 组专门演示这个坑。
set -u
HERE=$(cd $(dirname $0) && pwd)
LD=${LLAMA_DIR:-$HERE/../../../llama.cpp}
B=$LD/build/bin
NB=${NOOMP_BIN:-$HERE/../../../b10/noomp/bin}
MD=${MODELS:-$HERE/../../../models}
M=$MD/q-Q4_K_M.gguf
ST=$HERE/../02-ledger/stream
L=$HERE/logs
mkdir -p $L
what=${1:-all}
[ -x $ST ] || gcc -O3 -march=native -ffast-math -fopenmp $HERE/../02-ledger/stream.c -o $ST
cd /tmp
# tag 把一个 JSON 字段塞进 llama-bench 的每一行 jsonl
tag() { sed "s/^{/{\"$1\": \"$2\", /"; }
bench() { "$@" -o jsonl 2>/dev/null; }
bw() { OMP_NUM_THREADS=${1:-4} $ST 256 | sed 's/.*"read_GBps": \([0-9.]*\).*/\1/'; }
run() { [ $what = $1 ] || [ $what = all ]; }

if run threads; then
  # 每个组合一个进程，三轮交错。0x3 是同一物理核的两个超线程，0x5 是两个物理核各一个
  : > $L/threads.jsonl
  for r in 1 2 3; do
    for build in omp noomp; do
      BB=$B; [ $build = noomp ] && BB=$NB
      for t in 1 2 3 4; do
        bench $BB/llama-bench -m $M -p 128 -n 32 -r 2 -t $t | tag cfg none | tag build $build | tag round $r >> $L/threads.jsonl
        bench $BB/llama-bench -m $M -p 128 -n 32 -r 2 -t $t -C 0xf | tag cfg f-loose | tag build $build | tag round $r >> $L/threads.jsonl
      done
      for t in 1 2; do
        bench $BB/llama-bench -m $M -p 128 -n 32 -r 2 -t $t -C 0x3 --cpu-strict 1 | tag cfg 3-strict | tag build $build | tag round $r >> $L/threads.jsonl
        bench $BB/llama-bench -m $M -p 128 -n 32 -r 2 -t $t -C 0x5 --cpu-strict 1 | tag cfg 5-strict | tag build $build | tag round $r >> $L/threads.jsonl
      done
      # 线程比掩码里的 CPU 多：strict 轮转，两个线程落到同一个 CPU 上
      # 这一组很慢（个位数到十几 t/s），只测一次短 prefill
      [ $r = 1 ] && bench $BB/llama-bench -m $M -p 32 -n 0 -r 1 -t 3 -C 0x5 --cpu-strict 1 | tag cfg 5-strict-over | tag build $build | tag round $r >> $L/threads.jsonl
    done
  done
fi

if run sticky; then
  # 同一进程里先跑一个超订的绑核测试，再跑不绑核的测试，看亲和性有没有残留
  : > $L/sticky.jsonl
  for build in omp noomp; do
    BB=$B; [ $build = noomp ] && BB=$NB
    bench $BB/llama-bench -m $M -p 64 -n 0 -r 2 -t 3 | tag order fresh | tag build $build >> $L/sticky.jsonl
    bench $BB/llama-bench -m $M -p 64 -n 0 -r 1 -t 3 -C 0x5,0x0 --cpu-strict 1 | tag order after | tag build $build >> $L/sticky.jsonl
  done
fi

if run poll; then
  # poll 只在自带线程池里起作用；OpenMP 构建对照一遍
  : > $L/poll.jsonl
  for r in 1 2; do
    for build in omp noomp; do
      BB=$B; [ $build = noomp ] && BB=$NB
      for p in 0 50 100; do
        bench $BB/llama-bench -m $M -p 64 -n 32 -r 3 -t 4 --poll $p | tag build $build | tag round $r >> $L/poll.jsonl
      done
    done
  done
fi

if run prio; then
  # 优先级是全局参数；2 是 SCHED_FIFO 80，要 root 或 CAP_SYS_NICE
  : > $L/prio.jsonl
  for r in 1 2; do
    for build in omp noomp; do
      BB=$B; [ $build = noomp ] && BB=$NB
      for p in 0 2; do
        bench $BB/llama-bench -m $M -p 128 -n 32 -r 3 -t 4 --prio $p | tag build $build | tag prio $p | tag round $r >> $L/prio.jsonl
      done
    done
  done
fi

if run ub; then
  # ubatch：prefill 一次算多少 token
  bench $B/llama-bench -m $M -p 512 -n 0 -r 2 -t 4 -ub 16,32,64,128,256,512 > $L/ub.jsonl
fi

if run repack; then
  # repack 只认部分格式：AVX2 下 Q4_0、Q4_K、IQ4_NL、MXFP4，Q2_K 要 AVX512
  : > $L/repack.jsonl
  for v in q-Q4_0 q-Q4_K-pure q-Q4_K_M q-Q2_K q-Q6_K q-Q8_0; do
    bench $B/llama-bench -m $MD/$v.gguf -p 128 -n 32 -r 2 -t 4 --repack 0,1 | tag variant $v >> $L/repack.jsonl
  done
fi

if run tiled; then
  # repack 关掉以后 K 系列走 tiled，GGML_CPU_TILED_MM=0 再退回 vec_dot；三条路径对比
  : > $L/tiled.jsonl
  for v in q-Q4_K-pure q-Q6_K q-Q4_0; do
    bench $B/llama-bench -m $MD/$v.gguf -p 8,32,128 -n 16 -r 2 -t 4 --repack 1 | tag variant $v | tag path repack >> $L/tiled.jsonl
    bench $B/llama-bench -m $MD/$v.gguf -p 8,32,128 -n 16 -r 2 -t 4 --repack 0 | tag variant $v | tag path tiled >> $L/tiled.jsonl
    GGML_CPU_TILED_MM=0 bench $B/llama-bench -m $MD/$v.gguf -p 8,32,128 -n 16 -r 2 -t 4 --repack 0 | tag variant $v | tag path vecdot >> $L/tiled.jsonl
  done
fi

if run tb; then
  # -t 管 decode，-tb 管 prefill；llama-bench 只有一个 -t，所以用 llama-completion 看两段各自的速度
  PR=$(python3 -c "print(' '.join(['The quick brown fox jumps over the lazy dog.']*40))")
  : > $L/tb.txt
  for r in 1 2; do
    for tt in "2 4" "4 4" "4 2" "2 2"; do
      set -- $tt
      echo "== t $1 tb $2 r $r" >> $L/tb.txt
      $B/llama-completion -m $M -no-cnv -t $1 -tb $2 -n 48 --temp 0 -p "$PR" 2>&1 | grep -E "prompt eval time|       eval time" >> $L/tb.txt
    done
  done
fi

if run omp; then
  # llama.cpp 设的是 KMP_BLOCKTIME，那是 LLVM/Intel OpenMP 的变量；GCC 的 libgomp 认 OMP_WAIT_POLICY 和 GOMP_SPINCOUNT
  : > $L/omp.jsonl
  for r in 1 2; do
    for e in "X=1" "OMP_WAIT_POLICY=active" "OMP_WAIT_POLICY=passive" "GOMP_SPINCOUNT=0" "KMP_BLOCKTIME=0"; do
      env $e $B/llama-bench -m $M -p 64 -n 32 -r 3 -t 4 -o jsonl 2>/dev/null | tag env "$e" | tag round $r >> $L/omp.jsonl
    done
  done
fi

if run noisy; then
  # 邻居干扰：一个死循环占住 CPU 3，或者 stream 在另一个物理核上抢带宽
  : > $L/noisy.jsonl
  for t in 3 4; do bench $B/llama-bench -m $M -p 128 -n 32 -r 2 -t $t | tag noise none >> $L/noisy.jsonl; done
  taskset -c 3 bash -c 'while :; do :; done' & spin=$!
  for t in 3 4; do bench $B/llama-bench -m $M -p 128 -n 32 -r 2 -t $t | tag noise spin3 >> $L/noisy.jsonl; done
  bench $B/llama-bench -m $M -p 128 -n 32 -r 2 -t 3 -C 0x7 --cpu-strict 1 | tag noise spin3-pinned >> $L/noisy.jsonl
  kill $spin
  bench $B/llama-bench -m $M -p 128 -n 32 -r 2 -t 2 -C 0x3 | tag noise none-c3 >> $L/noisy.jsonl
  # 带宽邻居：llama 在核 0（CPU 0,1），stream 在核 1（CPU 2,3）循环跑
  ( while :; do OMP_NUM_THREADS=2 taskset -c 2,3 $ST 256 > /dev/null; done ) & bwn=$!
  bench $B/llama-bench -m $M -p 128 -n 32 -r 2 -t 2 -C 0x3 | tag noise stream23 >> $L/noisy.jsonl
  kill $bwn; sleep 2
  for p in $(ps -eo pid,comm | awk '$2=="stream"{print $1}'); do kill $p; done
fi

if run ceiling; then
  # 带宽上限：同一轮里按线程数各测一次 STREAM 和 tg，两种量化
  : > $L/ceiling.jsonl
  for r in 1 2; do
    for t in 1 2 3 4; do
      for v in q-Q4_K_M q-Q8_0; do
        b0=$(bw $t)
        j=$($B/llama-bench -m $MD/$v.gguf -t $t -p 0 -n 32 -r 2 -o jsonl 2>/dev/null)
        echo "$j" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(json.dumps({'round': $r, 'threads': $t, 'model': '$v', 'bw': $b0, 'tg': d['avg_ts'], 'model_size': d['model_size']}))" >> $L/ceiling.jsonl
      done
    done
  done
fi

if run load; then
  # 冷启动和热启动：每种加载方式先清页缓存再跑一次，紧接着热跑一次。要 root 才能写 drop_caches
  : > $L/load.txt
  # -c 512：缺省上下文 40960 时 KV 有 4.5 GB，分配和清零会盖过权重加载本身
  for r in 1 2; do
    for v in q-Q4_K_M q-Q8_0; do
      for lm in mmap none mmap+mlock dio; do
        sync; echo 3 > /proc/sys/vm/drop_caches
        echo "== $v $lm cold r$r $(python3 $HERE/../../../rss.py $B/llama-completion -m $MD/$v.gguf -no-cnv -c 512 -t 4 -n 4 -p hi -lm $lm)" >> $L/load.txt
        echo "== $v $lm warm r$r $(python3 $HERE/../../../rss.py $B/llama-completion -m $MD/$v.gguf -no-cnv -c 512 -t 4 -n 4 -p hi -lm $lm)" >> $L/load.txt
      done
    done
  done
  # 加载方式对稳态速度有没有影响
  [ -s $L/load-speed.jsonl ] || bench $B/llama-bench -m $M -p 128 -n 32 -r 2 -t 4 -lm mmap,none,mmap+mlock > $L/load-speed.jsonl
fi

if run thp; then
  # 透明大页：mmap 的权重是文件页，THP 管不到；-lm none 时权重在匿名内存里，always 才可能生效
  T=/sys/kernel/mm/transparent_hugepage/enabled
  old=$(sed 's/.*\[\(.*\)\].*/\1/' $T)
  : > $L/thp.jsonl; : > $L/thp-anon.txt
  for mode in madvise always; do
    echo $mode > $T
    for lm in mmap none; do
      $B/llama-bench -m $M -p 512 -n 64 -r 3 -t 4 -lm $lm -o jsonl 2>/dev/null | tag thp $mode >> $L/thp.jsonl &
      pid=$!; sleep 6
      echo "$mode $lm $(grep AnonHugePages /proc/meminfo)" >> $L/thp-anon.txt
      wait $pid
    done
  done
  echo $old > $T
fi
