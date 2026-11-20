# 11-gpu-offload-moe：GPU 卸载与 MoE

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-llama/blob/main/colab/11-gpu-offload-moe.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

LCPP-11 的配套代码。要回答几个问题：-ngl N 到底把哪几层放上卡，输出层算不算；-cmoe、-ncmoe、-ot 各把哪些张量留在主存；一个 MoE 模型要多少显存，给定一块卡 -ncmoe 至少设多少；--fit 怎么替你选参数，哪些内存它看不见；MoE 的路由有多偏、多有局部性，按层放、钉热专家、LRU、SLRU 四种放法在同样槽位下各命中多少；`--moe-cache-mib` 给多少才不白给。

本机没有 GPU。路由抓取、缓存回放和显存账本和设备无关；`cpu` 组只说明这几个参数在纯 CPU 构建里怎么落地，不代表有卡时的速度。

## 文件

| 文件 | 作用 | 需要什么 |
|---|---|---|
| `moe_trace.c` | 用 cb_eval 抓 `ffn_moe_topk-N`，记下每个 token 在每层选了哪几个专家；`text` 模式把一段文字分块 prefill，`gen` 模式贪心生成 | 编好的 llama.cpp（头文件和 libllama） |
| `moe_cache_sim.py` | 回放路由，比较按层放、钉热专家、oracle、LRU、SLRU 五种放法的命中率；`--stats` 报每层偏斜、相邻 token 重合和连续 n 个 token 用到的专家比例 | 只要 Python 标准库 |
| `vram_ledger.py` | 按 GGUF 张量表算 `-ngl`、`-ncmoe`、`-cmoe`、`-ot` 下显卡和主存各放多少；`--budget-mib` 倒推 `-ncmoe`；`--cache-mib` 按 0ec4027 的分组规则估每组槽位 | GGUF 文件，或只含张量头的截断文件 |
| `exp11.sh` | 五组实验：`trace`、`sim`、`ledger`、`cpu`、`fit` | 编好的 llama.cpp、两个 granite MoE 的 GGUF、wiki.test.raw |
| `test_offload.py` | 15 个用例：缓存语义用手写的小路由测，账本用张量头测，实测结论用 logs 核对 | 只要仓库；没有模型的用例跳过 |
| `logs/` | granite-1b-a400m、granite-3b-a800m 的路由 trace、回放结果、账本、fit 输出和纯 CPU 的 llama-bench | — |

## 快速试跑

不需要模型文件：

```bash
python3 test_offload.py
python3 moe_cache_sim.py logs/trace-1b-wiki.bin --tokens 8192 --stats
```

有 GGUF 时：

```bash
python3 vram_ledger.py model.gguf --ctx 4096 --budget-mib 16384
python3 vram_ledger.py model.gguf --cmoe --cache-mib 3400
bash exp11.sh trace     # 编 moe_trace，抓四份路由
bash exp11.sh sim
bash exp11.sh fit
```

几点注意：

- `-ngl N` 从后往前数，输出层算第 n_layer 层，所以 `-ngl 1` 只放输出层；2025 年底以前的资料用的是旧计数，`vram_ledger.py --legacy` 可以对账。
- llama-bench 没有 `-cmoe`，用 `-ncmoe 99` 代替。
- 纯 CPU 构建里 `llama-fit-params` 的 Host 合计不含 CPU_REPACK 缓冲，repack 过的权重不进账，主存估少一半以上。
- `--moe-cache-mib` 的槽位按专家张量的类型和形状分组，Q4_K_M 通常分成两组，每组槽位要和这组层数乘 n_used 比；低于 2 倍时 SLRU 的保护段装不下一个 token，回放里命中率比 LRU 还低。
- 只用 `llama-bench`、`llama-fit-params` 和自己编的 `moe_trace`，不起 server。有卡的机器上把 `cpu` 组的命令加上 `-ngl all` 再跑一遍。
