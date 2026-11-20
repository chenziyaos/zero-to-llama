# 07-quantization：量化

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-llama/blob/main/colab/07-quantization.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

LCPP-07 的配套代码。要回答四个问题：一个量化块里到底存了什么；Q4_K_M 这类“配方”在不同张量上用了哪些类型；imatrix 改的是量化里的哪一步；同样体积下怎么用 KLD 选类型。

## 文件

| 文件 | 作用 | 需要什么 |
|---|---|---|
| `quant.sh` | 从 bf16 GGUF 出发，量化出 8 个普通变体、imatrix、6 个 imatrix 变体、4 个配方实验 | 编好的 llama.cpp、校准文本 |
| `measure.sh` | 存 KLD 基线，逐个跑 PPL/KLD，再跑 3 轮 llama-bench | 同上，测试文本 |
| `collect_results.py` | 把 `logs/` 里的 bench JSONL 和 perplexity 日志合成一张表，速度取多轮中位数；`--pareto` 打印体积-KLD 帕累托前沿 | 只要日志 |
| `dequant.py` | 纯 Python 反量化 F32/F16/BF16/Q8_0/Q4_0/Q4_K/Q6_K；拆开 Q4_K 超级块；和 bf16 比误差；按 imatrix 加权算误差 | GGUF 文件 |
| `type_mix.py` | 按“角色”汇总一个 GGUF 里每类张量的类型和字节；按 `use_more_bits` 规则推导 Q4_K_M 配方并逐个比对 | GGUF 文件 |
| `test_quantization.py` | 15 个用例：块大小、nibble 顺序、6 bit scale 打包往返、Q4_K/Q6_K 反量化、配方规则、日志汇总、帕累托前沿 | 只要仓库 |
| `logs/` | Qwen3-0.6B 在 4 核 EPYC 上的实测日志：15 个变体 + 4 个配方实验的 perplexity 日志，第一轮 bench | — |
| `logs/round2/` | 第二、三轮 bench（同一文件里两轮，每轮 `-r 5`） | — |

## 快速试跑

不需要模型文件：

```bash
python3 test_quantization.py
python3 collect_results.py logs --more logs/round2 --md --pareto
```

有模型文件时：

```bash
# 1. 看配方
python3 type_mix.py models/q-Q4_K_M.gguf --check-q4km     # 311/311 个张量和规则一致
python3 type_mix.py models/q-Q4_K_M.gguf --layers         # 哪 14 层的 attn_v/ffn_down 升到了 Q6_K

# 2. 拆开一个 Q4_K 超级块，和 bf16 比误差
python3 dequant.py models/q-Q4_K_M.gguf blk.0.ffn_up.weight --ref models/qwen3-0.6b-bf16.gguf

# 3. imatrix 到底优化了什么：普通误差和按激活能量加权的误差
python3 dequant.py models/q-Q4_K_M.gguf  blk.13.attn_q.weight --rows 16 --ref models/qwen3-0.6b-bf16.gguf --imatrix models/imatrix.gguf
python3 dequant.py models/qi-Q4_K_M.gguf blk.13.attn_q.weight --rows 16 --ref models/qwen3-0.6b-bf16.gguf --imatrix models/imatrix.gguf
```

完整复现（约 40 分钟，其中 KLD 基线文件 1.5 GB）：

```bash
export LLAMA_BIN=~/llama.cpp/build/bin
./quant.sh   models/qwen3-0.6b-bf16.gguf data/wiki.train.raw
./measure.sh models/qwen3-0.6b-bf16.gguf data/wiki.test.raw
```

## 实测结果（Qwen3-0.6B，wikitext-2 test 前 20 块 × 512 token）

基准 bf16 在同一批 token 上的 PPL 是 23.08。速度是 3 轮中位数，pp512 / tg128，4 线程。

| 变体 | MiB | BPW | PPL | Mean KLD | KLD p99 | top-1 一致 | pp t/s | tg t/s |
|---|---|---|---|---|---|---|---|---|
| bf16 | 1434 | 16.00 | 23.08 | — | — | — | 430 | 39.8 |
| Q8_0 | 762 | 8.50 | 23.32 | 0.0026 | 0.015 | 97.1% | 339 | 61.2 |
| Q6_K | 588 | 6.56 | 23.47 | 0.0196 | 0.117 | 92.3% | 443 | 72.0 |
| Q5_K_M | 520 | 5.81 | 24.26 | 0.0391 | 0.255 | 88.6% | 516 | 62.3 |
| Q4_K_M | 456 | 5.09 | 25.62 | 0.1138 | 0.673 | 82.9% | 414 | 70.1 |
| Q4_K_M + imatrix | 456 | 5.09 | 24.19 | 0.0722 | 0.457 | 85.7% | 378 | 83.0 |
| Q4_0 | 442 | 4.93 | 26.27 | 0.2138 | 1.285 | 76.5% | 408 | 76.9 |
| IQ4_XS | 425 | 4.75 | 26.75 | 0.1603 | 1.015 | 79.3% | 490 | 71.8 |
| IQ4_XS + imatrix | 424 | 4.73 | 25.17 | 0.1069 | 0.643 | 82.7% | 513 | 68.0 |
| Q3_K_M | 389 | 4.34 | 33.82 | 0.4427 | 2.580 | 67.3% | 406 | 72.3 |
| Q3_K_M + imatrix | 389 | 4.34 | 27.73 | 0.2561 | 1.643 | 75.2% | 438 | 73.1 |
| Q2_K | 326 | 3.63 | 291.09 | 2.8556 | 9.760 | 29.8% | 310 | 79.4 |
| Q2_K + imatrix | 326 | 3.63 | 45.50 | 0.8463 | 4.672 | 55.5% | 313 | 80.6 |
| IQ3_XXS + imatrix | 324 | 3.62 | 43.91 | 0.7824 | 4.029 | 57.9% | 475 | 51.3 |
| IQ2_M + imatrix | 311 | 3.47 | 73.18 | 1.2337 | 6.065 | 49.6% | 420 | 58.2 |

配方实验（只测了质量）：

| 变体 | MiB | PPL | Mean KLD | top-1 | 说明 |
|---|---|---|---|---|---|
| Q4_K_M | 456 | 25.62 | 0.1138 | 82.9% | 对照组：output Q6_K，14 层 attn_v/ffn_down Q6_K |
| Q4_K_M，`--output-tensor-type q4_k` | 418 | 26.29 | 0.1301 | 80.8% | 只把输出层降到 Q4_K，省 38 MiB，KLD +0.016 |
| Q4_K_M `--pure` | 403 | 27.68 | 0.2051 | 77.3% | 再去掉 attn_v/ffn_down 的升级，又省 14.5 MiB，KLD +0.075；和 Q4_0 比 KLD 略低、PPL 反而更高 |
| Q4_K_M + imatrix，`--output-tensor-type q4_k` | 418 | 24.81 | 0.0878 | 83.9% | 有 imatrix 时同样的操作 |
| Q2_K + imatrix，`--output-tensor-type q8_0` | 361 | 45.28 | 0.8446 | 55.8% | 输出层升到 Q8_0，多花 36 MiB，几乎没收益 |

帕累托前沿（体积更小且 KLD 更低的变体不存在）：Q8_0 > Q6_K > Q5_K_M > Q4_K_M+imatrix > IQ4_XS+imatrix > Q3_K_M+imatrix > IQ3_XXS+imatrix > IQ2_M+imatrix。Q4_K_M 及更小的变体里，不带 imatrix 的一个都不在前沿上。

按每 MiB 的 KLD 算：attn_v/ffn_down 升级是 0.0052/MiB，输出层升级是 0.00043/MiB，前者每 MiB 的收益高 12 倍。

速度噪声很大：同一个 Q4_K_M 三轮 tg 在 62 到 88 t/s 之间。tg 列的差异小于 15% 时不要下结论，测量方法见 LCPP-03。

## 对照源码（llama.cpp 0ec4027）

- `ggml/src/ggml-common.h`：各 `block_*` 结构体
- `ggml/src/ggml-quants.c`：`quantize_row_q4_K_ref`（1467 行）、`quantize_row_q4_K_impl`（1563 行，imatrix 权重在 1586 行）、`dequantize_row_q4_K`（1539 行）、`get_scale_min_k4`（889 行）
- `src/llama-quant.cpp`：`llama_tensor_get_type_impl`（484 行）、`use_more_bits`（490 行）、`tensor_requires_imatrix`（902 行）
- `tools/imatrix/imatrix.cpp`：660 行累加 `x_j^2`；579 行只收集 `blk.*`，`output.weight` 要 `--process-output`
- `tools/perplexity/perplexity.cpp`：`log_softmax` 的 KLD 版本在 191 行，230 行跳过 `log p_base <= -16` 的项
