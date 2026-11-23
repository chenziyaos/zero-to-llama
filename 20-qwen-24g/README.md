# 20-qwen-24g：24 GB 单卡跑 Qwen3-32B 和 Qwen3.8-27B

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zyaochen/zero-to-llama/blob/main/colab/20-qwen-24g.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

LCPP-20 的配套代码。这一篇要回答的是：手上一张 RTX 4090 或 3090，该下哪档量化，上下文能开多长，-np 和投机解码怎么配，decode 大概多快。

我手上没有 GPU 机器，这个目录里的显存和速度数字**全部是推算**，按 llama.cpp 0ec4027 的分配逻辑写成公式，放在 `ledger.py` 里。两条启动命令只写了，没有在 GPU 上跑过。**不启动 llama-server，也不启动任何监听端口的程序**；`run_*.sh` 是给上机时用的。

## 文件

| 文件 | 作用 | 需要什么 |
|---|---|---|
| `ledger.py` | 显存账和速度区间：权重、KV、DeltaNet 状态、草稿模型或 MTP 头、mmproj、compute buffer 逐项列出，和 22.5 GiB 预算比；`max_ctx` 二分最长上下文；`cache_entry` 算一条提示缓存多大；`tg_bound`、`tg_split`、`spec_bound` 给 decode 上限；`--emit` 把文中用到的数字写成模块 | Python 3 标准库 |
| `test_ledger.py` | 回归用例：把正文引用的推算值钉住，改公式或文件大小时会报出来 | Python 3 标准库 |
| `run_32b.sh` | 第 10 章的 32B 配置：IQ4_XS，32K，q8_0 KV，单槽，思考模式；`fit` 子命令只跑 llama-fit-params | 编好的 llama.cpp 和 GGUF |
| `run_27b.sh` | 第 10 章的 27B 配置：UD-Q4_K_XL，64K，q8_0 KV，单槽，MTP 草稿 2，纯文本；`fit` 子命令同上 | 编好的 llama.cpp、GGUF 和 MTP 附带文件 |

## 快速试跑

不需要 GPU：

```bash
python3 test_ledger.py                 # 18 个用例
python3 ledger.py                      # 各档量化在 24 GB 上的最长上下文和 4090 速度区间
python3 ledger.py --model 32b --quant IQ4_XS --ctx 32768 --kv q8_0
python3 ledger.py --model 27b --quant UD-Q4_K_XL --ctx 65536 --kv q8_0 --np 1 --mtp 2
python3 ledger.py --model 32b --quant IQ4_XS --ctx 65536 --kv q4_0 --yarn 2
python3 ledger.py --model 32b --quant IQ4_XS --ctx 16384 --kv q8_0 --draft Qwen3-0.6B-Q8_0 --kvd q8_0
python3 ledger.py --model 27b --quant UD-Q4_K_XL --ctx 65536 --kv q8_0 --mtp 2 --cache 32
```

上机以后：

```bash
bash run_32b.sh fit      # 看 compute buffer 的实际 MiB
python3 ledger.py --model 32b --quant IQ4_XS --ctx 32768 --kv q8_0 --compute 0.88   # 用实测值重算
bash run_32b.sh
```

几点注意：

- 预算 22.5 GiB = 24 GiB 减 CUDA 上下文 0.5 减 fit 余量 1.0。CUDA 上下文和 compute buffer 是假设值，`--cuda-ctx`、`--compute` 可改，上机后用 `llama-fit-params -fitp on` 读实际值。
- 27B 的几何（层数、KV 头、head_dim、DeltaNet 状态）和 unsloth 标准档精确字节以 `common/dual_models.json` 为唯一来源，经 `common/hybrid_ledger.py` 读取，本篇不再手抄；`QUANTS['27b']` 里的 UD-* 动态量化档不在 manifest，保留两位小数的历史展示值。32B 是全 64 层历史对照，几何和字节仍在本篇本地。
- 文件大小抄自 Hugging Face 文件列表，单位 GB（10 的 9 次方），脚本里统一换成字节再和 GiB 比。unsloth 重新上传过文件，按 commit 下载，大小变了就改 `QUANTS`（标准档改 manifest，UD-* 档改本篇）。
- 速度区间的效率系数下限来自 Discussion #15013 的 Llama 2 7B 实测，上限来自 Kingy AI 的 27B 社区实测，都写在 `eff()` 里。
- 两个脚本都把 `-c`、`-ngl 99`、`-np 1` 写死，放不下时在加载阶段直接报错，fit 不会在背后把层挪到内存。
- `LLAMA_BIN`、`M`、`MD` 环境变量改二进制和模型路径。
