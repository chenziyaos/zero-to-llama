#!/usr/bin/env bash
# LCPP-04 的加载实验：三种加载方式 × repack 开关，热缓存和冷缓存，以及分卷对比。
# 用法：B=llama.cpp/build/bin M=models ./load_bench.sh [本地盘目录，默认 /tmp/lcpp04]
# 结果追加到 logs/，每行一次运行：maxrss、墙钟、缺页、load time。
set -u
B=${B:-llama.cpp/build/bin}; M=${M:-models}; L=${1:-/tmp/lcpp04}
H=$(cd "$(dirname "$0")" && pwd); mkdir -p "$H/logs" "$L"
run() { python3 "$H/rss.py" "$B/llama-completion" -m "$1" -lm "$2" $3 -c 512 -n 1 -t 4 -p hi -no-cnv; }

# 1. 模型放在家目录（网络盘）上
for r in 1 2 3; do for lm in mmap none dio; do for rp in 1 0; do
  echo "r$r q4km $lm repack=$rp $(run "$M/q-Q4_K_M.gguf" $lm "$( [ $rp = 0 ] && echo --no-repack )" 2>&1 | tail -1)"
done; done; done | tee -a "$H/logs/load_q4km.txt"

# 2. 拷到本地盘，切成三卷，热缓存下对比原文件和分卷
cp "$M/q-Q4_K_M.gguf" "$L/orig.gguf"
"$B/llama-gguf-split" --split --split-max-size 200M "$L/orig.gguf" "$L/q4" | tee "$H/logs/split.txt"
"$B/llama-gguf-split" --merge "$L/q4-00001-of-00003.gguf" "$L/merged.gguf" | tee -a "$H/logs/split.txt"
for r in 1 2; do for lm in mmap none dio mmap+mlock; do for rp in 1 0; do
  echo "r$r tmp $lm repack=$rp $(run "$L/orig.gguf" $lm "$( [ $rp = 0 ] && echo --no-repack )" 2>&1 | tail -1)"
done; done
  echo "split-tmp $(run "$L/q4-00001-of-00003.gguf" mmap "" 2>&1 | tail -1)"
done | tee -a "$H/logs/load_tmp.txt"

# 3. 冷缓存：每次先把文件踢出页缓存
for r in 1 2 3; do for lm in mmap none dio; do
  python3 "$H/evict.py" "$L/orig.gguf"
  echo "r$r cold-tmp $lm $(run "$L/orig.gguf" $lm "" 2>&1 | tail -1)"
done; done | tee -a "$H/logs/load_cold.txt"

# 4. 合并前后张量数据一致
"$B/llama-gguf-hash" --xxh64 --no-layer "$L/orig.gguf" "$L/merged.gguf" 2>/dev/null | tail -2
