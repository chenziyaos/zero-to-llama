#!/usr/bin/env bash
# 同一 commit 的两种构建（llamafile 开、关）交替跑 llama-bench，结果灌进 logs/bench.sqlite
set -euo pipefail
cd "$(dirname "$0")"
R=${R:-$(cd ../../.. && pwd)}
M=${M:-$R/models/q-Q8_0.gguf}
rm -f logs/bench.sqlite
for rep in 1 2; do
    for v in native nollamafile; do
        if [ $v = native ]; then lab=sgemm1; else lab=sgemm0; fi
        $R/b06/$v/bin/llama-bench -m $M -p 64,512 -n 32 -r 3 -t 4 -o sql 2>/dev/null > logs/bench-$lab-$rep.sql
        python3 load_sql.py logs/bench.sqlite logs/bench-$lab-$rep.sql --label $lab
    done
done
