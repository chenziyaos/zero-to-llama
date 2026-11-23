# 12-multi-gpu-rpc：多卡与 RPC

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zyaochen/zero-to-llama/blob/main/colab/12-multi-gpu-rpc.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

LCPP-12 的配套代码。要回答几个问题：`-sm layer` 把哪几层切给哪张卡，`-ts` 和空闲显存怎么换算；RPC 远端设备为什么总排在本机卡前面；流水线并行什么时候真的打开，打开后多花多少显存；`-sm tensor` 每层要做几次 AllReduce，每次搬多少字节；RPC 每次图计算往网线上送多少字节，图缓存和权重哈希各省掉多少。

本机没有 GPU，也不起任何监听端口的进程，所以**这里的结论都未在真实多卡或真实 RPC 集群上实测**。`fake_devs` 在进程里注册几张"假 GPU"，权重不分配（`no_alloc`），让 llama.cpp 自己走一遍选设备、切层、流水线开关和调度器切图；它报出来的层分配、显存账本和切图都是 llama.cpp 原样的输出，速度只能引用 PR 里的实测。

## 文件

| 文件 | 作用 | 需要什么 |
|---|---|---|
| `rpc_msg.c` | 照抄 `ggml-rpc.cpp` 里 `pack(1)` 的消息结构，打印每条命令上线的字节数、GRAPH_COMPUTE 和 GRAPH_RECOMPUTE 的差距、10 MiB 权重哈希命中和未命中的差距 | 只要 C 编译器 |
| `split_plan.py` | 照 `src/llama-model.cpp` 的 `-sm layer` 规则算每层落在哪张卡、每张卡分到多少权重和 KV；`--rpc K` 把前 K 张当远端 | GGUF 文件，或只含张量头的截断文件 |
| `fake_devs.cpp` | 注册 N 张假 GPU（可选 K 张名叫 RPC 的），不分配权重，打印 llama.cpp 的设备选择、逐层落点、流水线开关、调度器切分和显存账本 | 编好的 llama.cpp（头文件、libllama、libggml） |
| `comm_volume.py` | 按实现算三种方式每个 ubatch 的通信量：流水线的边界激活、张量并行的 AllReduce（NCCL 的 F32/BF16 门槛、butterfly 兜底）、RPC 的图序列化 | GGUF，或手给 `--n-embd`、`--n-layer` |
| `exp12.sh` | 四组实验：`msg`、`split`、`fake`、`comm` | 编好的 llama.cpp、q-Q4_0.gguf、两个张量头 |
| `test_split.py` | 17 个用例：切层规则和通信量用公式测，假多卡的结论用 logs 核对 | 只要仓库；没有模型或 logs 的用例跳过 |
| `logs/` | 上面四组的输出 | — |

## 快速试跑

不需要模型文件：

```bash
cc -O2 rpc_msg.c -o rpc_msg && ./rpc_msg 1000 1500
python3 comm_volume.py --n-embd 4096 --n-layer 32 --name llama-8B --dev 2,4 --tokens 1,512
python3 test_split.py
```

有 GGUF 和编好的 llama.cpp 时：

```bash
python3 split_plan.py model.gguf --free 24000,12000 --rpc 1
bash exp12.sh fake      # 编 fake_devs，跑 12 组假多卡
GGML_SCHED_DEBUG=1 ./fake_devs model.gguf --n 3     # 看调度器切出的每一段和它的输入
./fake_devs model.gguf --n 2 --events 1             # 报告支持 event，看流水线并行打开后多出的显存
./fake_devs model.gguf --n 4 --sm tensor            # 看 Meta 设备
```

几点注意：

- `fake_devs` 的设备报 `caps.events`，但 `event_new` 是空的。它只够让 llama.cpp 做规划和 reserve，不能真的跑 decode。
- `-sm tensor` 下 `llama_get_memory_breakdown` 把所有卡合成一行 `Meta()`，看不到每张卡各分多少；这一行的数字是所有卡的合计。
- 最后一张卡的 compute buffer 最大，因为 logits 落在它上面：n_vocab × n_ubatch × 4 B，Qwen3 的词表在 ubatch 512 时就是 296.75 MiB。
- 截断的张量头（`.head`）只能给 `split_plan.py` 用，`fake_devs` 要完整的 GGUF。
- 不跑 rpc-server、llama-server，也不跑 `tests/test-rpc-multi-server.sh`。要在自己的集群上验证，按正文里的命令在可信的内网起 rpc-server，再用 `llama-bench --rpc` 测。
