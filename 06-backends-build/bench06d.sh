#!/bin/bash
# 06 补充实验：把正文第 5、6 章里手动跑过的几组命令收在一起
#   1. repack 单线程对照：pp256 和 tg32，同一次调用里 --repack 0,1 交替
#   2. BLAS 构建在三种线程设置下的 Q8_0 pp256（-t 1、2、4）
#   3. 4 线程 pp512 下 OMP_WAIT_POLICY=passive 对 blas 和 native 的影响
#   4. GGML_SCHED_DEBUG=1 数 BLAS 构建的 split 数；--repack 0 时 Q4_K_M 被 BLAS 接走
#   5. 加载日志里 CPU_REPACK 和 CPU_Mapped 的缓冲大小
# 用法：OUT=b06 M=models LOG=logs bash bench06d.sh
set -u
OUT=${OUT:-$PWD/b06}
LOG=${LOG:-$PWD/logs}
M=${M:-$PWD/models}
mkdir -p $LOG
cd /tmp   # load_best 也扫当前目录，放在不含 .so 的目录里跑

j2line() {  # jsonl -> 一行一条
  python3 -c "
import json, sys, os
for l in sys.stdin:
    d = json.loads(l)
    print('$1', os.path.basename(d['model_filename']), d['n_prompt'], d['n_gen'], d.get('repack', ''), '%.1f' % d['avg_ts'], '%.1f' % d['stddev_ts'])
"
}

# 1 repack，单线程
$OUT/native/bin/llama-bench -m $M/q-Q4_K_M.gguf,$M/q-Q4_0.gguf -p 256 -n 32 -r 3 -t 1 --repack 0,1 -o jsonl 2>/dev/null |
  j2line native | tee $LOG/repack_t1_tg.txt

# 2 BLAS 线程设置
: > $LOG/blas_threads.txt
for env in "X=1" "OMP_WAIT_POLICY=passive" "OPENBLAS_NUM_THREADS=1"; do
  for t in 1 2 4; do
    env $env $OUT/blas/bin/llama-bench -m $M/q-Q8_0.gguf -p 256 -n 0 -r 3 -t $t -o jsonl 2>/dev/null |
      j2line "$env t$t" | tee -a $LOG/blas_threads.txt
  done
done

# 3 passive 对 blas 和 native，4 线程 pp512
: > $LOG/blas_omp.txt
for env in "X=1" "OMP_WAIT_POLICY=passive"; do
  for n in blas native; do
    echo "[$env $n]" | tee -a $LOG/blas_omp.txt
    env $env $OUT/$n/bin/llama-bench -m $M/q-Q4_K_M.gguf,$M/q-Q8_0.gguf -p 512 -n 0 -r 3 -t 4 -o jsonl 2>/dev/null |
      python3 -c "
import json, sys, os
for l in sys.stdin:
    d = json.loads(l); print(os.path.basename(d['model_filename']), '%.1f' % d['avg_ts'])
" | tee -a $LOG/blas_omp.txt
  done
done

# 4 split 数：每个 split 在调试日志里有一行 ## SPLIT
: > $LOG/blas_splits.txt
for q in Q4_0 Q4_K_M Q8_0; do
  n=$(GGML_SCHED_DEBUG=1 $OUT/blas/bin/llama-completion -m $M/q-$q.gguf -n 1 -p hi -no-cnv -t 1 -v 2>&1 | grep -c '## SPLIT')
  echo "blas $q splits $n" | tee -a $LOG/blas_splits.txt
done
$OUT/blas/bin/llama-bench -m $M/q-Q4_K_M.gguf -p 256 -n 0 -r 3 -t 1 --repack 0,1 -o jsonl 2>/dev/null |
  j2line blas | tee -a $LOG/blas_splits.txt

# 5 repack 的内存代价
: > $LOG/repack_mem.txt
for nr in "" "-nr"; do
  echo "[native $nr]" >> $LOG/repack_mem.txt
  $OUT/native/bin/llama-completion -m $M/q-Q4_K_M.gguf -n 1 -p hi -no-cnv -t 1 $nr 2>&1 |
    grep -E 'model buffer size' | tee -a $LOG/repack_mem.txt
done
echo done
