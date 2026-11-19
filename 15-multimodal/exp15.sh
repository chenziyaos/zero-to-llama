#!/bin/bash
# 15 实验：一张图、一段音频在 libmtmd 里变成多少 token，编码花多少时间
# 用法：bash exp15.sh [build|count|time|threads|batch|cli|climax|asr|unit|model|all]
# 环境变量：LLAMA_DIR（缺省 ../../../llama.cpp），MODELS（缺省 ../../../models/vl）
#
# 不启动 llama-server，不开任何端口。用到的程序：
#   mm_probe        本目录的小程序，直接调 libmtmd：count 只切块数 token，time 测编码和解码，batch 比较逐块和批量编码
#   llama-mtmd-cli  命令行跑一遍图片描述和语音转写，看真实日志
#   test-mtmd-impl、test-mtmd-c-api  libmtmd 自带的单元测试
#   mm_tokens.py    按源码复刻的 token 计数，test_mm.py 拿它和 count 日志对账
# 本机 4 核 CPU，没有 GPU，数字只代表这台机器上的相对关系。
set -u
HERE=$(cd $(dirname $0) && pwd)
LD=${LLAMA_DIR:-$HERE/../../../llama.cpp}
B=$LD/build/bin
MD=${MODELS:-$HERE/../../../models/vl}
L=$HERE/logs
P=$HERE/mm_probe
mkdir -p $L
what=${1:-all}
run() { [ $what = $1 ] || [ $what = all ]; }

SM="-m $MD/SmolVLM2-256M-Video-Instruct-Q8_0.gguf --mmproj $MD/mmproj-SmolVLM2-256M-Video-Instruct-Q8_0.gguf"
SF="-m $MD/SmolVLM2-256M-Video-Instruct-Q8_0.gguf --mmproj $MD/mmproj-SmolVLM2-256M-Video-Instruct-f16.gguf"
LF="-m $MD/LFM2-VL-450M-Q8_0.gguf --mmproj $MD/mmproj-LFM2-VL-450M-Q8_0.gguf"
QA="-m $MD/Qwen3-ASR-0.6B-Q8_0.gguf --mmproj $MD/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf"
SIZES="256x256 512x512 640x488 1024x768 1536x1536 2048x1024 4000x3000"

if run build; then
  g++ -O2 -std=c++17 -Wall $HERE/mm_probe.cpp -o $P \
    -I$LD/include -I$LD/ggml/include -I$LD/tools/mtmd \
    -L$B -lmtmd -lllama -lggml -lggml-base -Wl,-rpath,$B
fi

# 只切块、不编码：各种尺寸的块数和 token 数，max-tok 调上限
if run count; then
  $P count $SM $SIZES > $L/count-smol.txt 2>/dev/null
  $P count $SF $SIZES > $L/count-smol-f16.txt 2>/dev/null
  $P count $LF $SIZES > $L/count-lfm2.txt 2>/dev/null
  for mt in 64 256 1024; do
    $P count $SM --max-tok $mt $SIZES > $L/count-smol-max$mt.txt 2>/dev/null
  done
  for mt in 64 128; do
    $P count $LF --max-tok $mt $SIZES > $L/count-lfm2-max$mt.txt 2>/dev/null
  done
fi

# 编码、把嵌入喂进语言模型、同样多文本 token 的预填充，各取 3 次中位数
if run time; then
  $P time $LF --reps 3 256x256 512x512 640x488 1024x768 2048x1024 > $L/time-lfm2.txt 2> $L/time-lfm2.err
  $P time $SM --reps 3 256x256 640x488 1024x768 2048x1024 > $L/time-smol.txt 2> $L/time-smol.err
  $P time $SM --reps 3 --max-tok 64 640x488 2048x1024 > $L/time-smol-max64.txt 2>/dev/null
  $P time $SM --reps 3 --max-tok 256 640x488 2048x1024 > $L/time-smol-max256.txt 2>/dev/null
  $P time $SF --reps 3 640x488 > $L/time-smol-f16.txt 2>/dev/null
  $P time $LF --reps 3 --max-tok 64 640x488 > $L/time-lfm2-max64.txt 2>/dev/null
fi

# 编码器的线程扩展
if run threads; then
  for t in 1 2 4; do
    $P time $LF -t $t --reps 3 640x488 > $L/time-lfm2-t$t.txt 2>/dev/null
  done
fi

# 多块图逐块编码和交给 mtmd 批量编码的对比
if run batch; then
  $P batch $LF --reps 2 1024x768 > $L/batch-lfm2.txt 2>&1
  $P batch $SM --reps 2 640x488 > $L/batch-smol.txt 2>&1
fi

# 日志只留编码、解码和性能统计
K='mtmd batch|decoded|n_tokens_batch|image slice|llama_perf|load_hparams|experimental|cap image'

# 命令行：仓库自带的 test-1.jpeg，贪心解码 64 个 token
if run cli; then
  for m in smol lfm2; do
    a="$SM"; [ $m = lfm2 ] && a="$LF"
    $B/llama-mtmd-cli $a -t 4 --temp 0 -n 64 --image $LD/tools/mtmd/test-1.jpeg \
      -p "Describe this image in one sentence." -v > $L/cli-$m.txt 2> $L/cli-$m.full
    grep -aE "$K" $L/cli-$m.full > $L/cli-$m.err; rm -f $L/cli-$m.full
  done
fi

# 同一张图压低上限再说一遍，看描述还对不对
if run climax; then
  for c in "smol 64" "smol 256" "lfm2 64"; do
    set -- $c; a="$SM"; [ $1 = lfm2 ] && a="$LF"
    $B/llama-mtmd-cli $a -t 4 --temp 0 -n 64 --image-max-tokens $2 --image $LD/tools/mtmd/test-1.jpeg \
      -p "Describe this image in one sentence." -v > $L/cli-$1-max$2.txt 2> $L/cli-$1-max$2.full
    grep -aE "$K" $L/cli-$1-max$2.full > $L/cli-$1-max$2.err; rm -f $L/cli-$1-max$2.full
  done
fi

# 语音转写：仓库自带的 test-2.mp3
if run asr; then
  $B/llama-mtmd-cli $QA -t 4 --temp 0 -n 96 --audio $LD/tools/mtmd/test-2.mp3 \
    -p "Transcribe the audio." -v > $L/asr.txt 2> $L/asr.full
  grep -aE "$K" $L/asr.full > $L/asr.err; rm -f $L/asr.full
fi

if run unit; then
  $B/test-mtmd-impl > $L/unit-impl.txt 2>&1
  $B/test-mtmd-c-api 2>&1 | tail -8 > $L/unit-capi.txt
  python3 $HERE/test_mm.py > $L/unit-py.txt 2>&1
fi

if run model; then
  python3 $HERE/mm_tokens.py > $L/model.txt
fi
