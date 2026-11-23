# 05-ggml-graph：计算图、内存规划和调度

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zyaochen/zero-to-llama/blob/main/colab/05-ggml-graph.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

LCPP-05 的配套代码，回答四个问题：
- 一层 Qwen3 在 ggml 的计算图里是多少个节点，哪些真正要算？
- gallocr 怎么让中间结果的内存和层数无关？
- compute buffer 为什么在缺省参数下有 300 MiB，n_outputs_max 和 Flash Attention 各影响多少？
- CPU 融合和图复用各自省了多少时间？

## 文件

| 文件 | 作用 |
|---|---|
| `toy_graph.c` | 用 ggml 搭 L 层小网络，对比 `ggml_gallocr_reserve` 规划的缓冲和所有中间张量之和 |
| `nout.c` | 通过 C API 设 `n_outputs_max`，打印 CPU compute buffer |
| `sched_parse.py` | 统计 `GGML_SCHED_DEBUG=2` 日志里的节点数、视图数和算子分布，支持 `.gz` |
| `exp05.sh` | ubatch × FA × 上下文的 compute buffer 和节点数扫表 |
| `bench05.sh` | 融合开关和图复用开关的对照，两组都交替跑 3 轮 |
| `logs/` | 4 vCPU EPYC、Qwen3-0.6B Q4_K_M、llama.cpp 主干 0ec4027 上的实测 |

`logs/` 里的文件：

| 文件 | 内容 |
|---|---|
| `sched.log.gz`、`sched_nofa.log.gz` | FA 开和关时 `GGML_SCHED_DEBUG=2 -v` 的全文 |
| `buf.txt` | `exp05.sh` 和 `nout` 的 compute buffer 表 |
| `fusion.txt` | 融合开关，`llama-bench -r 3` 单次 |
| `fusion_alt.txt` | 融合开关交替 3 次的 tg64，差别在噪声里 |
| `reuse_rep.txt` | 图复用开关交替 3 轮的 `--perf` 输出 |
| `ub_pp.jsonl`、`ub_pp2.jsonl` | pp512 随 ubatch 的两次独立测量 |

## 编译和运行

```bash
L=path/to/llama.cpp            # 已用 cmake 编译到 $L/build
cc -O2 toy_graph.c -I$L/ggml/include -L$L/build/bin -lggml -lggml-base -lggml-cpu -Wl,-rpath,$L/build/bin -o toy_graph
cc -O2 nout.c -I$L/include -I$L/ggml/include -L$L/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$L/build/bin -o nout

./toy_graph 28 1024 512        # planned=12.00 MiB，中间张量合计 616 MiB
./nout model.gguf 512 1        # n_outputs_max=1 时的 compute buffer

LLAMA_BIN=$L/build/bin MODEL=model.gguf bash exp05.sh
LLAMA_BIN=$L/build/bin MODEL=model.gguf bash bench05.sh

GGML_SCHED_DEBUG=2 $L/build/bin/llama-completion -m model.gguf -c 4096 -ub 512 -fa on \
    -n 1 -p hi -no-cnv -t 4 -v 2> sched.log
python3 sched_parse.py sched.log
```

## 主要结论

- FA 开时一次前向 989 个节点，653 个要算，336 个是视图；FA 关时 1129 个，737 个要算。
- toy_graph 从 1 层到 28 层，规划后的缓冲都是 12 MiB，只随 token 数线性涨。
- ub=512 时 compute buffer 312.77 MiB，其中 296.75 MiB 是 logits；`n_outputs_max=1` 降到 32.02 MiB。
- 16k 上下文、ub=512、`n_outputs_max=1`：非 FA 566.01 MiB，FA 44.02 MiB。
- 图复用：64 个 token 解码 63 次 eval 复用 62 次，每 token 12.54 对 13.42 ms。
- RMS_NORM 加 MUL 融合：交替 3 次的 tg64 均值 73.8 对 73.4 t/s，在噪声里。
