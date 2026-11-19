#!/usr/bin/env bash
# 融合开和关各跑两轮，交替执行，结果灌进 logs/fuse.sqlite
set -euo pipefail
cd "$(dirname "$0")"
R=${R:-$(cd ../../.. && pwd)}
B=$R/llama.cpp/build/bin
rm -f logs/fuse.sqlite
for rep in 1 2; do
    for f in 0 1; do
        if [ $f = 1 ]; then lab=fuse0; env_v=1; else lab=fuse1; env_v=0; fi
        GGML_CPU_DISABLE_FUSION=$env_v $B/test-backend-ops perf -b CPU -o RMS_NORM_MUL_ROPE --output sql 2>/dev/null > logs/perf-$lab-$rep.sql
        python3 load_sql.py logs/fuse.sqlite logs/perf-$lab-$rep.sql --label $lab
    done
done
