#!/bin/bash
# 05 实验：CPU 融合开关和图复用开关的对照，两组都交替跑 3 轮
B=${LLAMA_BIN:-llama.cpp/build/bin}
M=${MODEL:-models/q-Q4_K_M.gguf}

echo "== fusion: llama-bench tg64，GGML_CPU_DISABLE_FUSION 0/1 交替"
for i in 1 2 3; do for f in 0 1; do
  echo -n "i=$i fusion_off=$f "
  GGML_CPU_DISABLE_FUSION=$f $B/llama-bench -m $M -p 0 -n 64 -r 2 -t 4 -fa 1 -o jsonl 2>/dev/null |
    python3 -c "import sys,json;[print(round(json.loads(l)['avg_ts'],1)) for l in sys.stdin]"
done; done

echo "== graph reuse: llama-completion --perf，LLAMA_GRAPH_REUSE_DISABLE 0/1 交替"
for i in 1 2 3; do for dis in 0 1; do
  LLAMA_GRAPH_REUSE_DISABLE=$dis $B/llama-completion -m $M -c 4096 -n 64 \
      -p "Explain how a compiler works." -no-cnv -t 4 --perf 2>&1 |
    grep -E "eval time|graphs reused" | tr "\n" " "; echo " dis=$dis"
done; done
