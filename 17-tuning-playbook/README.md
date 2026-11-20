# 17-tuning-playbook：按症状查参数

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-llama/blob/main/colab/17-tuning-playbook.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

LCPP-17 的配套代码。要回答几个问题：什么都不写时 llama.cpp 给了你哪些参数，上下文为什么开到了训练长度；哪些误配会直接退出、哪些只打一行警告、哪些连警告都没有；decode 慢、prefill 慢、内存爆、长上下文变慢时先动哪个参数；实测速度离带宽上限还差多少，差出来的那部分是谁吃掉的。

本机是 4 核 CPU（2 个物理核、每核 2 个超线程），没有 GPU，数字只代表这台机器上的相对关系。**不启动 llama-server，也不启动任何监听端口的程序**。所有实验只用 `llama-completion` 和 `llama-bench`，诊断和事前账用自己的 Python。

## 文件

| 文件 | 作用 | 需要什么 |
|---|---|---|
| `exp17.sh` | 六组实验：`defaults`（不写 `-c` 和写 `-c 4096` 的日志与峰值内存）、`misconfig`（11 种写法各起一个进程，记退出码、墙钟和 stderr）、`symptoms`（线程、repack、ubatch、FA、KV 类型五组 llama-bench）、`diag`（正常、一个死循环占住 CPU 3、避开它只用 3 个线程）、`plan`、`unit` | 编好的 llama.cpp，Qwen3-0.6B 的 Q4_0、Q4_K_M、Q8_0、bf16 |
| `diagnose.py` | `log`：解析一次运行的 stderr，列出生效参数、命中的已知误配（error、warn、silent 三档，带源码位置）和速度离预测的比值；`plan`：不跑模型，按参数算内存四项、decode 预测和 `--fit` 会不会动它 | Python 3；`plan` 和 `--model` 要 GGUF 文件，借 `02-ledger` 的账本 |
| `test_diagnose.py` | 回归用例：手写的最小日志加 `logs/` 下的真实日志 | Python 3；没有日志或模型时对应用例跳过 |
| `logs/` | 各组输出：`mis-*.log`、`misconfig.txt`、`symptoms.jsonl`、`diag.txt`、`plan.txt` | — |

## 快速试跑

不需要模型文件：

```bash
python3 test_diagnose.py                 # 只跑手写日志的用例
python3 diagnose.py log logs/mis-c100k.log
python3 diagnose.py log logs/mis-t8.log --json
```

有模型和编好的 llama.cpp 时：

```bash
python3 diagnose.py plan ../../../models/q-Q4_K_M.gguf            # 不写 -c 会拿到什么
python3 diagnose.py plan ../../../models/q-Q4_K_M.gguf -c 4096 -np 4
bash exp17.sh misconfig    # 11 个进程，大约 2 分钟（mask 那一条要 100 秒）
bash exp17.sh symptoms     # 五组 llama-bench，大约 15 分钟
```

自己的日志：运行时加 `-lv 4` 再交给 `diagnose.py log`。缺省日志级别下只能拿到 system_info、sampler、generate 和计时，`n_ubatch`、`n_seq_max`、KV 大小这些要 `-lv 4` 才打。

几点注意：

- `misconfig` 用 Q4_0 跑，`defaults`、`symptoms`、`diag` 用 Q4_K_M。Q4_K_M 开 repack 以后 prefill 反而变慢，见 `symptoms.jsonl` 的 repack 组。
- `diag` 会在 CPU 3 上起一个 `while :; do :; done` 的死循环，跑完按 PID 杀掉。中途打断要自己 `kill` 它。
- `symptoms` 里每组之间不能和别的负载重叠，两个 llama-bench 同时跑会把两组数字都污染。
- 规则表 `RULES` 对齐 llama.cpp 0ec4027 的日志文本，换版本后先跑 `test_diagnose.py`。
- 不跑 llama-server、llama-cli、rpc-server、`scripts/server-bench.py`。
