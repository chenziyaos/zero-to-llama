#!/usr/bin/env bash
# LCPP-18 实验驱动。不启动任何监听端口的进程；同一时刻只跑一个重任务。
# 用法：bash exp18.sh [convert|ref|base|kv|inject|tok|archs|debug|short|all]
# 环境变量：R 指向放 llama.cpp、models 的根目录，缺省是本仓库的上上级。
set -euo pipefail
cd "$(dirname "$0")"
R=${R:-$(cd ../../.. && pwd)}
L=$R/llama.cpp
HF=$R/models/Qwen3-0.6B
G=$R/models/conv/qwen3-bf16.gguf
REF=$R/ref18
OUT=$R/out18
BIN=$R/bin18
J=logs/compare.jsonl
mkdir -p logs "$BIN" "$OUT" "$R/models/conv"

build_dump() {
  [ -x "$BIN/dump_logits" ] && return
  g++ -O2 -std=c++17 -I "$L/include" -I "$L/ggml/include" dump_logits.cpp \
      -L "$L/build/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$L/build/bin" -o "$BIN/dump_logits"
}

dump() {  # dump 名字 gguf [额外参数...]
  local n=$1 g=$2; shift 2
  python3 runlog.py "logs/dump-$n.log" -- "$BIN/dump_logits" -m "$g" -i "$REF/f32/tokens.bin" \
      -o "$OUT/$n" -t 4 --keep l_out,result_norm "$@"
  python3 compare.py "$REF/f32" "$OUT/$n" --layers --tag "$n" --json $J > "logs/cmp-$n.txt"
  head -1 "logs/cmp-$n.txt"
}

step_convert() {
  python3 runlog.py logs/convert-bf16.log -- python3 "$L/convert_hf_to_gguf.py" "$HF" \
      --outtype bf16 --outfile "$G"
  python3 runlog.py logs/convert-dry.log -- python3 "$L/convert_hf_to_gguf.py" "$HF" --outtype bf16 --dry-run
}

step_ref() {
  python3 runlog.py logs/hf-f32.log -- python3 hf_ref.py --model "$HF" --text prompt.txt --out "$REF/f32" --dtype f32
  python3 runlog.py logs/hf-bf16.log -- python3 hf_ref.py --model "$HF" --text prompt.txt --out "$REF/bf16" --dtype bf16
  python3 compare.py "$REF/f32" "$REF/bf16" --layers --tag hf-bf16 --json $J > logs/cmp-hf-bf16.txt
  head -1 logs/cmp-hf-bf16.txt
  python3 massive.py "$REF/f32" > logs/massive.txt
}

step_base() {
  build_dump
  dump bf16 "$G"
  dump bf16-fa0 "$G" --fa 0
  dump q8_0 "$R/models/q-Q8_0.gguf"
  dump q4_k_m "$R/models/q-Q4_K_M.gguf"
  # 位置 0 单独看：量化后的最大误差为什么都在这里
  python3 pos0.py "$REF/f32" "$OUT" bf16 q8_0 q4_k_m > logs/pos0.txt
}

step_kv() {
  build_dump
  dump kv-theta1e4 "$G" --kv qwen3.rope.freq_base=float:10000
  dump kv-eps1e-5 "$G" --kv qwen3.attention.layer_norm_rms_epsilon=float:0.00001
  dump kv-ctx512 "$G" --kv qwen3.context_length=int:512
}

step_inject() {
  build_dump
  for b in qk-permute norm-plus1 qk-norm-one swap-gate-up; do
    local g=$R/models/conv/bug-$b.gguf
    python3 runlog.py "logs/inject-$b.log" -- python3 inject.py "$G" "$g" "$b"
    dump "bug-$b" "$g"
    rm -f "$g"
  done
  # 只坏一层：逐层对拍应该在第 14 层第一次跳出来
  local g=$R/models/conv/bug-qk-permute-l14.gguf
  python3 runlog.py logs/inject-qk-permute-l14.log -- python3 inject.py "$G" "$g" qk-permute --layer 14
  dump bug-qk-permute-l14 "$g"
  rm -f "$g"
}

step_tok() {
  # 只转词表，再改 tokenizer.ggml.pre 做对照：qwen2 是转换器写的值，另外三个是常见的错选
  local V=$R/models/conv/vocab
  mkdir -p "$V"
  python3 runlog.py logs/convert-vocab.log -- python3 "$L/convert_hf_to_gguf.py" "$HF" --vocab-only --outfile "$V/qwen2.gguf"
  for p in llama3 gpt2 deepseek-llm; do
    python3 "$L/gguf-py/gguf/scripts/gguf_new_metadata.py" --pre-tokenizer $p "$V/qwen2.gguf" "$V/$p.gguf" --force > /dev/null
  done
  for p in qwen2 llama3 gpt2 deepseek-llm; do
    python3 runlog.py "logs/tok-$p.log" -- python3 tok_compare.py --hf "$HF" --gguf "$V/$p.gguf" \
        --tokenize "$L/build/bin/llama-tokenize" --tag $p --json "logs/tok-$p.json" > /dev/null
    grep tok_compare "logs/tok-$p.log"
  done
  # gpt2 那份根本加载不了，留一份报错原文
  "$L/build/bin/llama-tokenize" -m "$V/gpt2.gguf" -p hi > logs/tok-gpt2-load.log 2>&1 || true
  python3 runlog.py logs/test-tokenizer-0.log -- "$L/build/bin/test-tokenizer-0" "$L/models/ggml-vocab-qwen2.gguf"
}

step_archs() {
  python3 runlog.py logs/test-llama-archs.log -- "$L/build/bin/test-llama-archs" -a qwen3
}

step_debug() {
  # 官方 model-conversion 流程里的那一步：只存最后一个位置的 logits
  python3 runlog.py logs/llama-debug.log -- "$L/build/bin/llama-debug" -m "$G" -p "Hello, my name is" \
      --save-logits --logits-output-dir "$OUT/debug" -t 4 --tensor-filter result_output
  python3 runlog.py logs/llama-debug-c512.log -- "$L/build/bin/llama-debug" -m "$G" -p "Hello, my name is" \
      --save-logits --logits-output-dir "$OUT/debug512" -t 4 -c 512 --tensor-filter result_output
  python3 runlog.py logs/hf-last.log -- python3 hf_last.py --model "$HF" --text "Hello, my name is" \
      --llama "$OUT/debug/llamacpp-qwen3-bf16.bin"
}

step_short() {
  # 同样几种错放到 5 个 token 的短提示上，只看最后一个位置：哪些能被官方口径抓到
  build_dump
  local T=$OUT/short-tokens.bin
  python3 runlog.py logs/hf-short.log -- python3 hf_last.py --model "$HF" --text "Hello, my name is" --tokens-out "$T"
  local ds=()
  sdump() { local n=$1 g=$2; shift 2
    python3 runlog.py "logs/sdump-$n.log" -- "$BIN/dump_logits" -m "$g" -i "$T" -o "$OUT/s-$n" -t 4 "$@"
    ds+=("$OUT/s-$n"); }
  sdump bf16 "$G"
  sdump kv-theta1e4 "$G" --kv qwen3.rope.freq_base=float:10000
  sdump kv-eps1e-5 "$G" --kv qwen3.attention.layer_norm_rms_epsilon=float:0.00001
  for b in qk-permute qk-permute-l14 qk-norm-one; do
    local g=$R/models/conv/bug-$b.gguf
    case $b in
      qk-permute-l14) python3 inject.py "$G" "$g" qk-permute --layer 14 > /dev/null ;;
      *) python3 inject.py "$G" "$g" "$b" > /dev/null ;;
    esac
    sdump "bug-$b" "$g"
    rm -f "$g"
  done
  python3 runlog.py logs/hf-short-cmp.log -- python3 hf_last.py --model "$HF" --text "Hello, my name is" --dump "${ds[@]}"
}

case "${1:-all}" in
  convert) step_convert ;;
  ref) step_ref ;;
  base) step_base ;;
  kv) step_kv ;;
  inject) step_inject ;;
  tok) step_tok ;;
  archs) step_archs ;;
  debug) step_debug ;;
  short) step_short ;;
  all) step_convert; step_ref; step_base; step_kv; step_inject; step_tok; step_archs; step_debug; step_short ;;
  *) echo "unknown step $1"; exit 2 ;;
esac
