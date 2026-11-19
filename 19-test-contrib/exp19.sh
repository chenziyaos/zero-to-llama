#!/usr/bin/env bash
# LCPP-19 实验驱动。不启动任何监听端口的进程；同一时刻只跑一个重任务。
# 用法：bash exp19.sh [ctest|behavior|vocabs|tbo|bug|nofuse|fix|perf|bench|all]
# 环境变量：R 指向放 llama.cpp、models 的根目录，缺省是本仓库的上上级。
set -euo pipefail
cd "$(dirname "$0")"
R=${R:-$(cd ../../.. && pwd)}
L=$R/llama.cpp
B=$L/build/bin
BUG=$R/bug19
mkdir -p logs

# test-model-resolution 会绑一个随机端口，本机不跑
NOPORT="-E test-model-resolution"

step_ctest() {
  # CI 的 build-cpu 跑的就是 -L main
  (cd "$L/build" && python3 "$OLDPWD/runlog.py" "$OLDPWD/logs/ctest-main.log" -- \
      ctest -L main $NOPORT --timeout 900)
  tail -3 logs/ctest-main.log
}

step_behavior() {
  # 照抄 behavior-check.yml 的环境变量；model 和 backend 标签的测试另有工作流
  (cd "$L/build" && python3 "$OLDPWD/runlog.py" "$OLDPWD/logs/ctest-behavior.log" -- \
      env GGML_NLOOP=3 GGML_N_THREADS=1 GGML_SCHED_DEBUG_REALLOC=1 \
      ctest -L behavior -LE 'model|backend' $NOPORT --timeout 900) || true
  tail -3 logs/ctest-behavior.log
}

step_vocabs() {
  # 没装 git-lfs 时 clone 下来的是指针文件，改用 resolve 地址把真文件拉下来
  local V=$L/models/ggml-vocabs
  for f in "$V"/*/*.gguf; do
    if head -c 7 "$f" | grep -q version; then
      local rel=${f#"$V"/}
      curl -sfL -o "$f.tmp" "https://huggingface.co/ggml-org/vocabs/resolve/main/$rel" && mv "$f.tmp" "$f"
    fi
  done
  (cd "$L/build" && python3 "$OLDPWD/runlog.py" "$OLDPWD/logs/ctest-vocabs.log" -- \
      ctest -R test-tokenizers-ggml-vocabs --output-on-failure --timeout 900)
  tail -2 logs/ctest-vocabs.log
}

step_tbo() {
  # 全量：CPU 优化路径对 CPU 参考实现，-j 2 和 ci/run.sh 一样
  python3 runlog.py logs/tbo-cpu-full.log -- "$B/test-backend-ops" test -b CPU -j 2 > logs/tbo-cpu-full.rc 2>&1
  grep "tests passed" logs/tbo-cpu-full.log | tail -1
}

build_bug() {
  [ -x "$BUG/build/bin/test-backend-ops" ] && return
  mkdir -p "$BUG"
  tar -C "$L" --exclude=./build --exclude=./docs --exclude=./.git -cf - . | tar -C "$BUG" -xf -
  (cd "$BUG" && patch -p1 < "$OLDPWD/patches/fused-tail.patch")
  (cd "$BUG" && cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON \
      -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF \
      > "$OLDPWD/logs/bug19-cmake.log" 2>&1 && cmake --build build -t test-backend-ops -j 4 \
      > "$OLDPWD/logs/bug19-build.log" 2>&1)
}

step_bug() {
  # 单算子测试看不见融合路径里的错，融合测试才看得见
  build_bug
  for op in RMS_NORM_MUL_ADD RMS_NORM_MUL_ROPE RMS_NORM MUL; do
    python3 runlog.py "logs/bug-$op.log" -- "$BUG/build/bin/test-backend-ops" test -b CPU -o $op \
        > /dev/null 2>&1 || true
    grep "tests passed" "logs/bug-$op.log" | tail -1
  done
}

step_nofuse() {
  # 关掉融合，错就没了：二分时先用这个开关判断是不是融合路径
  build_bug
  python3 runlog.py logs/bugnofuse-RMS_NORM_MUL_ADD.log -- env GGML_CPU_DISABLE_FUSION=1 \
      "$BUG/build/bin/test-backend-ops" test -b CPU -o RMS_NORM_MUL_ADD > /dev/null 2>&1
  grep "tests passed" logs/bugnofuse-RMS_NORM_MUL_ADD.log | tail -1
}

step_fix() {
  for op in RMS_NORM_MUL_ADD RMS_NORM_MUL_ROPE; do
    python3 runlog.py "logs/fix-$op.log" -- "$B/test-backend-ops" test -b CPU -o $op > /dev/null 2>&1
    grep "tests passed" "logs/fix-$op.log" | tail -1
  done
}

step_perf() {
  python3 runlog.py logs/perf-fuse.log -- ./perf_fuse.sh > /dev/null 2>&1
  python3 "$L/scripts/compare-llama-bench.py" --tool test-backend-ops -i logs/fuse.sqlite \
      -b fuse0 -c fuse1 | tee logs/cmp-fuse.txt
}

step_bench() {
  # 要两份构建：$R/b06/native（默认，llamafile sgemm 开）和 $R/b06/nollamafile（-DGGML_LLAMAFILE=OFF）
  python3 runlog.py logs/bench-ab.log -- ./bench_ab.sh > /dev/null 2>&1
  python3 "$L/scripts/compare-llama-bench.py" -i logs/bench.sqlite -b sgemm0 -c sgemm1 | tee logs/cmp-bench.txt
}

case "${1:-all}" in
  ctest) step_ctest ;;
  behavior) step_behavior ;;
  vocabs) step_vocabs ;;
  tbo) step_tbo ;;
  bug) step_bug ;;
  nofuse) step_nofuse ;;
  fix) step_fix ;;
  perf) step_perf ;;
  bench) step_bench ;;
  all) step_ctest; step_behavior; step_vocabs; step_tbo; step_bug; step_nofuse; step_fix; step_perf; step_bench ;;
  *) echo "unknown step: $1" >&2; exit 2 ;;
esac
