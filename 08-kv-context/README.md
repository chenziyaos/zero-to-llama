# 08-kv-context：KV cache 与长上下文

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zyaochen/zero-to-llama/blob/main/colab/08-kv-context.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

LCPP-08 的配套代码。要回答几个问题：KV cache 按什么公式分配、启动前能不能算出来；上下文变深以后预填充和解码各慢多少；CPU 上的 Flash Attention 什么时候帮忙；KV 量化和 Hadamard 旋转换来多少误差；GQA、MLA、滑动窗口各省多少；上下文写满、超出训练长度以后会发生什么。

## 文件

| 文件 | 作用 | 需要什么 |
|---|---|---|
| `kv_ledger.py` | 按 llama.cpp 0ec4027 的规则算 K、V 各占多少字节：读 GGUF 或用 9 个内置预设，支持 KV 量化、`-np`、unified、SWA 和 `--swa-full` | GGUF 文件，或只用预设 |
| `exp08.sh` | 五组实验：`size` KV 大小日志，`bench` 深度测速，`kld` KV 量化和旋转的 KLD，`ctx` 上下文长度、YaRN/linear 和写满，`fa` FA 对 compute buffer 的影响 | 编好的 llama.cpp、Q4_K_M 和 bf16 两个 GGUF、wiki.test.raw |
| `collect_bench.py` | 把 llama-bench 的 JSONL 合成一张表 | 只要日志 |
| `collect_kld.py` | 从 perplexity 日志里抽 PPL、平均 KLD、99% KLD 和 top-1 相同比例 | 只要日志 |
| `test_kv.py` | 8 个用例：实测 KV 大小回归、缺省取训练长度、256 取整、-np 平分、K/V 混合类型、块大小整除、SWA 层模式、MLA 不建 V | 只要仓库 |
| `logs/` | Qwen3-0.6B 在 4 核 EPYC 上的实测日志和两份汇总 `bench_summary.txt`、`kld_summary.txt` | — |

## 快速试跑

不需要模型文件：

```bash
python3 test_kv.py
python3 kv_ledger.py --list
python3 kv_ledger.py --preset gemma3-27b -c 131072 --np 4 --kvu
python3 kv_ledger.py --preset gemma3-27b -c 131072 --swa-full
```

有模型文件时：

```bash
python3 kv_ledger.py --gguf models/q-Q4_K_M.gguf -c 4096 40960 -t f16 q8_0 q8_0/q4_0
bash exp08.sh size      # 一分钟，和上面的账本对一遍
bash exp08.sh ctx       # 约 15 分钟
bash exp08.sh bench     # 单线程深度测速，q8_0 和 q4_0 很慢，约两小时
```

本机没有 GPU，所有数字都是 CPU 单机实测；8192 上下文的 PPL 在这台 7 GiB 内存的机器上被 Bus error 杀掉，只测到 4096。
