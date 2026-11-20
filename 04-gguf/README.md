# 04-gguf：把 GGUF 拆开看

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-llama/blob/main/colab/04-gguf.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

LCPP-04 的配套代码，回答三个问题：
- 一个 GGUF 文件里每一段各占多少字节？
- 一个文件坏在哪里时 llama.cpp 会拒绝加载？
- mmap、普通读、O_DIRECT 三种加载方式和 repack 开关，各自慢在哪里、多占多少内存？

全部只用 Python 标准库，不需要 gguf-py，也不读权重本身。

## 文件

| 文件 | 作用 |
|---|---|
| `gguf_check.py` | 分段账本加合法性检查。逐条对照 `ggml/src/gguf.cpp` 的 `gguf_init_from_file_impl`，`--json` 出一行 JSON，`--strict` 有 ERROR 时退出码 1 |
| `gguf_diff.py` | 比较两个 GGUF：KV 增删改、张量类型和形状变化、各段大小差 |
| `gguf_write.py` | 最小写入器，测试用它造合法文件和各种坏文件 |
| `test_gguf.py` | 26 个用例：3 个合法文件，19 种坏法，分卷键，mmap 对齐；设 `MODELS` 时再查真实模型和只有文件头的夹具 |
| `load_bench.sh` | 加载实验：网络盘和本地盘，三种加载方式 × repack，冷缓存，分卷和合并 |
| `rss.py`、`evict.py` | 量峰值 RSS 和缺页；把文件踢出页缓存 |
| `logs/` | 4 vCPU EPYC、llama.cpp 主干 0ec4027 上的实测 |

`logs/` 里的文件：

| 文件 | 内容 |
|---|---|
| `load_q4km.txt` | 模型在家目录（网络盘）上，三种方式 × repack 开关，3 轮 |
| `load_tmp.txt` | 拷到本地 /tmp，热缓存，四种方式 × repack 开关，加分卷对比 |
| `load_cold.txt` | 本地 /tmp，每次运行前用 `evict.py` 清掉页缓存 |
| `split.txt` | `llama-gguf-split` 按 200M 切三卷和合并的输出 |
| `load_v.log.gz` | `-v` 加载日志全文，看缓冲区大小和每个张量放在哪 |

## 快速试跑

```bash
python3 test_gguf.py
python3 gguf_write.py tiny.gguf && python3 gguf_check.py tiny.gguf
```

有模型文件时：

```bash
python3 gguf_check.py models/q-Q4_K_M.gguf
python3 gguf_diff.py models/qwen3-0.6b-bf16.gguf models/q-Q4_K_M.gguf
MODELS=models python3 test_gguf.py
B=llama.cpp/build/bin M=models ./load_bench.sh
```

## 本机结果摘要

Qwen3-0.6B Q4_K_M，484 MB：

- 文件头 24 B，KV 5.93 MB（词表和 merges 占 99.9%），张量信息 18 KB，数据段从 5,951,872 开始，模 4096 余 384。
- 本地盘热缓存下，mmap 加 repack 加载 370 ms，峰值 RSS 782 MiB；关 repack 后 175 至 192 ms，581 MiB。
- 多出的约 200 MiB 是 CPU_REPACK 缓冲（204.75 MiB 匿名内存）。
- 同样的文件放在网络盘上，mmap 要 1.6 s，O_DIRECT 要 7 s。切成三卷和不切差不到 2%。
- 本地盘冷缓存（evict.py 清页缓存）：mmap 515 至 560 ms，普通读 607 至 617 ms，O_DIRECT 824 至 895 ms。
- Q4_K_M 里 output.weight 是 Q6_K，121.7 MiB，约占文件 26%；Qwen3 共享输入输出词表，加载代码在缺这个张量时会退回 token_embd。
- `gguf_diff.py` 比 Q4_K_M 和 IQ2_M：KV 多 4 个 imatrix 键，126 个张量从 Q4_K 变成 IQ2_S，output 从 Q6_K 变成 Q5_K。
