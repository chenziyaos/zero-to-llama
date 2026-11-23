# 21-deepseek-v41：把 DeepSeek-V4.1-Flash 装进一台机器

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zyaochen/zero-to-llama/blob/main/colab/21-deepseek-v41.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

LCPP-21 的配套代码。这一篇要回答的是：ISTA-DASLab 的 GSQ-RCO GGUF（IQ1_M、IQ2_XXS、IQ2_XS 三档）在一张显卡加一台机器的内存上怎么放，Engram 表为什么不占内存，KV 要多少，路由专家放几层上卡，decode 大概多快。

我手上没有 GPU 机器，这个目录里的字节、显存和速度数字**全部是推算**。字节账按 llama.cpp 0ec4027 的张量形状和发布者附带的 RCO 分配表逐个张量算，速度模型拿 HumanjavaEnterprises/deepseek-v41-upstream 的社区实测扫描做拟合。两条启动命令只写了，没有在 GPU 上跑过。**不启动 llama-server，也不启动任何监听端口的程序**；`run_*.sh` 是给上机时用的。

## 文件

| 文件 | 作用 | 需要什么 |
|---|---|---|
| `plan.py` | 字节账、放置、KV 和速度：逐张量算三档量化的骨干字节并和模型卡核对；`placement` 按前 N 层专家上卡分卡上、内存、磁盘三处；`kv_ledger` 列原始 KV、CSA、HCA 压缩 KV 和压缩器状态；`fit_sweep` 拟合社区扫描，`predict_tg` 给 decode 区间；`fit_card` 算给定显存最多放几层专家；`--emit` 把文中用到的数字写成模块 | Python 3 标准库 |
| `test_plan.py` | 回归用例：把正文引用的推算值钉住，改公式、分配表或实测数字时会报出来 | Python 3 标准库 |
| `run_96g.sh` | 第 7 章的 96 GB 配置：IQ2_XXS，64K，单槽，前 25 层专家上卡，ngram-mod，思考模式 effort 50；`fit` 子命令只跑 llama-fit-params | 编好的 llama.cpp 和 GGUF |
| `run_24g.sh` | 第 7 章的 24 GB 配置：IQ2_XXS，16K，q8_0 KV，单槽，专家全留内存；`GPU_LAYERS` 可改；`fit` 子命令同上 | 编好的 llama.cpp、GGUF 和 128 GB 以上内存 |
| `alloc/` | 三档量化的 RCO 分配表，发布者随模型附带，每个张量一行量化类型，表头有 measured bpw | 无 |

## 快速试跑

不需要 GPU：

```bash
python3 test_plan.py                     # 12 个用例
python3 plan.py                          # 三档量化的字节核对、KV 表、峰值反推、速度拟合
python3 plan.py --build IQ2_XXS --gpu-layers 25 --ctx 65536 --kv f16 --ub 2048 --card 96
python3 plan.py --build IQ2_XXS --gpu-layers 0 --ctx 16384 --kv q8_0 --card 24 --ram-bw 89.6
python3 plan.py --fit-card 24 --build IQ2_XXS --ctx 16384 --kv q8_0 --ram-bw 89.6
python3 plan.py --fit-card 96 --build IQ1_M --ctx 65536 --kv f16
```

上机以后：

```bash
bash run_96g.sh fit                      # 看 compute buffer 的实际 MiB
python3 plan.py --build IQ2_XXS --gpu-layers 25 --ctx 65536 --ub 2048 --other 6.6   # 用实测值重算
bash run_96g.sh
```

几点注意：

- 社区报的显存峰值 85.9 和 96.1 我按 GiB 读。按 GB 读时 28 层那个点的“其余”项是负数，按 GiB 读两个点只差 0.19 GB，`other_from_peak` 的注释里有推导。显卡标的 24 GB、96 GB 也按 GiB 算，`--card`、`--fit-card` 都是 GiB，余量 1 GiB。
- “其余”项（compute buffer 加 CUDA 上下文）是从 25 层那个峰值反推的 6.60 GB，换上下文长度和 `-ub` 后会变，上机后用 `llama-fit-params -fitp on` 读实际值，填进 `--other`。
- 速度给的是区间，两端是两种外推假设：一端假设内存侧吞吐跟内存带宽等比缩放，另一端假设吞吐不变、只受带宽封顶。社区机器 16 线程比 24、32 线程快，说明那台机器的 CPU 侧既吃核数也吃带宽，单用一种外推都会偏。
- 模型要 llama.cpp b11870 或更新。共享的 Engram 文件要链接成每档的第 6 片，两个脚本缺这个链接时会提示命令后退出。
- `LLAMA_BIN`、`MD`、`M`、`GPU_LAYERS`、`THREADS` 环境变量改二进制、模型目录、模型路径、上卡层数和线程数。
