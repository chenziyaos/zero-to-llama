# 02-ledger：推理账本

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-llama/blob/main/colab/02-ledger.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

LCPP-02 的配套代码，回答三个问题：
- 一台机器的内存带宽和峰值算力到底是多少？
- 一个 GGUF 每生成一个 token 要读多少字节、做多少运算？
- 不启动模型，怎么估出 decode、prompt 处理和并发 decode 的速度，以及给定内存下能开多长的上下文？

账本分两层：
- 第一层是 Roofline 上限，全部从 GGUF 的张量目录算出来。
- 第二层是本机校准，把上限换成预测。所需的五个系数由 `fit_ledger.py` 从 `logs/` 里的实测拟合。

## 文件

| 文件 | 作用 | 需要什么 |
|---|---|---|
| `stream.c` | 最小 STREAM：copy、scale、add、triad 加只读 read，输出一行 JSON | gcc、OpenMP |
| `peak.c` | AVX512 FP32 FMA、BF16 点积（vdpbf16ps）、VNNI int8 点积（vpdpbusd）的峰值，每线程 12 条依赖链，必须 `-O3` | 支持 AVX512_VNNI、AVX512_BF16 的 CPU |
| `measure.sh` | 全部实测：峰值、12 轮带宽、带宽与 llama-bench 交替、llama-batched-bench、夹心对账，最后调 `depth_sweep.sh` | 编好的 llama.cpp、GGUF 目录 |
| `depth_sweep.sh` | Q4_K_M 在深度 0 至 4096 的 tg，FA 关和开各一遍，每个点先测带宽 | 同上 |
| `estimate_speed.py` | 账本和预测：`ledger()` 从 GGUF 算字节和运算量，`bounds()` 给上限，`predict()` 套校准系数 | GGUF 或导出的账本 |
| `fit_ledger.py` | 从 `logs/` 拟合 `t0`、`eff_tg`、`eff_tg_codebook`、`eff_pp`、`eff_batch`、`eff_attn`，打印逐点对账；`--dump` 导出账本 | 只要日志 |
| `estimate_budget.py` | 给定内存上限，二分查找每个量化文件能开的最大上下文，再算深度 0 和半程深度的 tg | GGUF，复用 `01-first-run/estimate_memory.py` |
| `kv_cache_timing.py` | 开场实验：随机权重的小模型带 / 不带 KV cache 各生成一遍，比时间和运算量，并核对两边 token 相同 | numpy |
| `test_kv_cache_timing.py` | 6 个用例：两种做法 token 相同、增量 logits 等于整段重算、运算量公式 | numpy |
| `test_ledger.py` | 17 个用例：账本字节和运算量、KV 每 token、Roofline、预测对实测、四组拟合；设 `MODELS` 时再核对导出和预算 | 只要仓库 |
| `logs/` | 4 vCPU EPYC 上的实测，见下 | — |

`logs/` 里的文件：

| 文件 | 内容 |
|---|---|
| `peak.jsonl`、`stream.jsonl` | 1 至 4 线程的峰值；1 至 4 线程各 12 轮带宽 |
| `stream-ws.jsonl` | 4 线程下每个数组 1 至 256 MiB 的带宽，看工作集超过 32 MiB L3 之后的真实带宽 |
| `interleave.jsonl` | 5 轮，每轮先测带宽，再测 7 个变体的 pp512 和 tg64 |
| `sandwich.jsonl` | 6 轮 × 4 个变体，每次 tg 前后各测一次带宽 |
| `depth.jsonl` | 深度扫描，每行带当时的带宽 `bw` |
| `batched-q4km.md` | llama-batched-bench，B = 1 至 32 |
| `ledgers.json` | 19 个 Qwen3-0.6B GGUF 的账本，测试夹具，也供没有 GGUF 时用 |
| `bench-*.jsonl` | 线程数、FA、深度、变体的单项 llama-bench |

## 快速试跑

不需要模型文件：

```bash
python3 test_ledger.py
python3 fit_ledger.py                                                  # 拟合系数和逐点对账
python3 estimate_speed.py --ledger logs/ledgers.json --name q-Q4_K_M   # 用导出的账本预测
python3 estimate_speed.py --ledger logs/ledgers.json --name q-Q4_K_M --depth 4096 --fa on
```

有模型文件时：

```bash
python3 estimate_speed.py models/q-Q4_K_M.gguf --raw                   # 只看 Roofline 上限
python3 estimate_speed.py models/q-Q4_K_M.gguf --batch 1,4,16,32       # 预测加并发
python3 estimate_budget.py --budget 2048 models/*.gguf                 # 2 GiB 内每个文件最多开多长
python3 estimate_budget.py --budget 2048 --ctk q8_0 --ctv q8_0 models/*.gguf
MODELS=models python3 test_ledger.py                                   # 多跑两个用例
```

## 开场实验：KV cache 开和关

```bash
python3 kv_cache_timing.py                       # 随机权重的小模型，带 / 不带 KV cache 各生成一遍
python3 kv_cache_timing.py --prompt 128 --gen 256
python3 kv_cache_timing.py --params-b 0.6 --layers 28 --d-model 1024 --prompt 512 --gen 256   # 只算账：Qwen3-0.6B 尺寸
python3 test_kv_cache_timing.py
```

不带 cache 时，每生成一个 token 都要把整段序列重新过一遍模型；带 cache 时只算新 token 一行，K、V 从缓存里取。两种做法用同一份 float64 权重做 greedy 解码，生成的 token 必须完全相同，脚本会检查。下面的输出在 Apple M4 Pro 上测得，时间随机器变，运算量和 token 一致性不随机器变：

```
模型：4 层，d=128，4 头，d_ff=512，vocab=512，float64，随机权重
prompt 32 个 token，greedy 生成 96 个，每种做法跑 3 遍取最短
               时间 (s)     GFLOP   每 token (ms)
带 cache         0.018      0.23            0.18
不带 cache       0.114     12.72            1.19
不带 / 带：时间 6.5x，运算量 55.6x
两种做法生成的 token 完全相同
```

运算量差 55.6 倍，时间只差 6.5 倍左右（本机多次运行在 6.5 到 6.9 倍之间）。原因是带 cache 的每一步只算一行，矩阵很小，时间主要花在 Python 和 numpy 的调用开销上，算力吃不满；把 prompt 和生成长度调大，两种做法的时间差会随之拉大。换成真实模型尺寸只算账：

```
模型 0.6B，28 层，d=1024，prompt 512，生成 256
  带 cache         0.95 TFLOP
  不带 cache     202.55 TFLOP
  不带 / 带 = 212.3x（只算运算量；带 cache 的 decode 实际受带宽限制，见本章账本）
```

真实部署里没有人关掉 KV cache。这个实验只是说明：从「每步只算一个 token」开始，decode 的瓶颈就从算力转到了读权重和读 KV 的带宽上，本章后面的账本都建立在这个前提上。

## 换一台机器

校准系数只对测它的那台机器有效。换机器时按下面四步重新校准：

1. 跑 `bash measure.sh MODEL_DIR`，4 核机器大约要 40 分钟。
2. 跑 `python3 fit_ledger.py`，看逐点对账，确认平均误差在 10% 以内。
3. 把打印出的系数抄回 `estimate_speed.py` 的 `CAL`。
4. 跑 `python3 fit_ledger.py --models MODEL_DIR --dump logs/ledgers.json` 重新导出账本，再跑测试。

云主机的带宽会分时段跳变，本机就在 35、46、65 GB/s 三档之间切换。所以每个 tg 前后都要测一次带宽，拟合时取两次里较小的那个值。只拿一次带宽去配所有 tg，误差会翻倍。

## 实测结果（Qwen3-0.6B，4 vCPU EPYC，llama.cpp 0ec4027）

| 项目 | 数值 |
|---|---|
| 读带宽 1 / 2 / 3 / 4 线程（中位数） | 30.3 / 33.7 / 34.1 / 35.5 GB/s |
| 峰值 4 线程 fp32 / bf16 / int8 | 454 / 887 / 1662 G op/s |
| int8 屋脊点 | 47 op/B；单序列 decode 只有 3 op/B |
| Q4_K_M 每 token 读权重 | 390.8 MB，KV 每 token 112 KiB |

以下预测都在 35.5 GB/s 带宽下：

| 变体 | Roofline tg | 预测 tg | 慢带宽时段实测 | B=32 预测总吞吐 |
|---|---|---|---|---|
| bf16 | 29.8 | 26.9 | 27.5 至 28.0 | 105 |
| Q8_0 | 56.0 | 46.6 | 47.0 至 48.9 | 194 |
| Q4_K_M | 90.9 | 68.5 | 72.0 至 72.1 | 203（实测 199） |
| Q2_K | 122.3 | 84.9 | 85.2 至 87.9 | 206 |
| IQ2_M | 137.1 | 61.0（算力受限） | 中位数 60.7 | 200 |

校准系数：

| 系数 | 数值 | 含义 |
|---|---|---|
| `t0_ms` | 3.6 | 每步固定开销 |
| `eff_tg` | 0.09 | decode 整数点积效率 |
| `eff_tg_codebook` | 0.044 | IQ2、IQ3 查码本 |
| `eff_pp` | int8 0.20，bf16 0.42 | prompt 处理效率 |
| `eff_batch` | 0.155 | 每多一条序列一步多 4.62 ms |
| `eff_attn` | FA 关 0.50，开 0.26 | 读 KV 的有效带宽比例 |

本机 CPU 上开 FA 反而更慢：深度 4096 时 FA 关 24.9 t/s，开 17.1 t/s。2 GiB 预算下（f16 KV），bf16 最多开 2560 的上下文，Q4_K_M 能开到 11520；KV 改成 q8_0 后，Q4_K_M 能开到 21760。
