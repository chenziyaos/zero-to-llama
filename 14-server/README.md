# 14-server：服务端的槽位、连续批处理和提示缓存

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-llama/blob/main/colab/14-server.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

LCPP-14 的配套代码。要回答几个问题：多开几路并发，总吞吐能涨多少、每条请求慢多少；统一 KV（`-kvu`）和分开的 KV 各在什么场景下划算；`-c` 和 `-np` 怎么分账，每个槽位最后拿到多少上下文；共享系统提示能省多少预填充；槽位选择和 `--cache-ram` 的 RAM 缓存在用户比槽位多时能复用多少；上下文撑满以后会怎样。

本机是 4 核 CPU，没有 GPU，数字只代表这台机器上的相对关系。**不启动 llama-server，也不启动任何监听端口的程序**。服务端的调度用三样东西逼近：`llama-batched-bench`（同一上下文里 B 条序列一起解码）、`llama-parallel`（N 个客户端轮流投递请求，带共享系统提示）和 `test-server-component`（服务端组件的单元测试，不加载模型）。槽位选择和 RAM 缓存的淘汰规则用 `slot_sim.py` 按源码复刻。`scripts/server-bench.py` 端到端压测、router 多模型、server 里的投机解码 **都未在本机实测**。

机器速度会随时间漂。`rep` 段把分开的 KV 和统一 KV 交替跑 3 轮，比较时只拿同一轮里的两个数相除。

## 文件

| 文件 | 作用 | 需要什么 |
|---|---|---|
| `exp14.sh` | 九组实验：`bb`（并发路数 × 统一 KV）、`pps`（共享提示）、`wt`（Q4_0 和 bf16 权重）、`par`（llama-parallel -np 1/2/4/8）、`parpps`（共享前缀和长系统提示）、`unit`（单元测试）、`rep`（交替 3 轮）、`ledger`（-c、-np、-kvu 的上下文分账）、`sim` | 编好的 llama.cpp、Qwen3-0.6B 的 Q8_0/Q4_0/bf16 |
| `slot_sim.py` | 每 token KV 的账本、RAM 缓存能装多少 token、按源码复刻的选槽（相似度阈值 `--slot-prompt-similarity`）和 RAM 缓存淘汰，扫缓存大小和用户数看前缀复用率 | 只要 Python 3 |
| `test_server.py` | 16 个用例：上下文按 256 对齐的分账、每 token KV 字节数、公共前缀、RAM 缓存的去重、超限跳过和淘汰最旧、恢复缓存的阈值、选槽顺序（空槽优先、前缀相似度阈值、跳过忙槽），以及模拟器里缓存在用户多时有用、用户少时无用 | 只要 Python 3 |
| `logs/` | 上面各组的输出；`.err` 只留统计相关的行 | — |

## 快速试跑

不需要模型文件：

```bash
python3 slot_sim.py          # 账本、缓存容量、选槽与 RAM 缓存的复用率、用户数扫描
python3 test_server.py
```

有模型和编好的 llama.cpp 时：

```bash
bash exp14.sh bb        # 并发 1 到 16 路，分开的 KV 和统一 KV 各一遍
bash exp14.sh par       # llama-parallel 在 -np 1/2/4/8 下各跑 16 条请求
bash exp14.sh ledger    # 看 -c 和 -np 怎么分成每个槽位的 n_ctx_seq
bash exp14.sh unit      # test-server-component 加 test_server.py
```

几点注意：

- 不写 `-np` 时 llama-batched-bench 和 server 会走统一 KV；要对比分开的 KV，必须显式写 `-np` 并且不加 `-kvu`。
- `llama-parallel` 的日志很长，`exp14.sh` 用 grep 只留每条请求的耗时和汇总行。
- 加载时提示 `n_ctx_seq` 小于 `n_ctx_train` 只是提醒，不是错误。
- `slot_sim.py` 复刻的是源码里的选槽和淘汰规则，不跑模型；它给的是复用率的趋势，不是延迟。
- 不跑 llama-server、llama-cli、rpc-server、`scripts/server-bench.py`。
