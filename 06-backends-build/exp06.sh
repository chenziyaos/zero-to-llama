#!/bin/bash
# 06 实验：同一份源码按不同编译开关各编一份，记编译时间、库大小、system_info 和 llama-bench
# 用法：bash exp06.sh NAME [cmake 选项...]
set -u
SRC=${SRC:-../../llama.cpp}
OUT=${OUT:-$PWD/b06}
LOG=${LOG:-$PWD/logs}
mkdir -p $OUT $LOG
name=$1; shift
B=$OUT/$name
t0=$(date +%s)
cmake -S $SRC -B $B -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF \
      -DLLAMA_CURL=OFF -DGGML_CCACHE=OFF "$@" > $LOG/$name.cmake.txt 2>&1 || { echo "$name cmake failed"; exit 1; }
t1=$(date +%s)
cmake --build $B -j 4 --target llama-bench llama-completion > $LOG/$name.build.txt 2>&1 || { echo "$name build failed"; tail -20 $LOG/$name.build.txt; exit 1; }
# GGML_BACKEND_DL 时后端是单独的 .so，也要编出来
if grep -q "GGML_BACKEND_DL:BOOL=ON" $B/CMakeCache.txt; then
  cmake --build $B -j 4 >> $LOG/$name.build.txt 2>&1
fi
t2=$(date +%s)
echo "$name configure $((t1-t0)) s build $((t2-t1)) s" | tee $LOG/$name.time.txt
ls -l $B/bin/*.so >> $LOG/$name.time.txt 2>/dev/null
find $B/bin -name 'libggml*.so*' -type f -printf '%s %f\n' >> $LOG/$name.time.txt
find $B/bin -name 'libggml*.so*' -type f -printf '%s\n' | awk '{s+=$1} END {printf "ggml_total %.2f MiB\n", s/1048576}' | tee -a $LOG/$name.time.txt
find $B/bin -name '*.so*' -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {printf "so_total %.1f MiB\n", s/1048576}' | tee -a $LOG/$name.time.txt
