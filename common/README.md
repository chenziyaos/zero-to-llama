# common：全系列共用的只读工具和双模型基线

这个目录放各章都能 import 的东西，不绑定任何单一章节。所有脚本只用 Python 3.9+ 标准库，不需要装 `gguf` 包，也不需要 GPU。

## 文件

| 文件 | 作用 |
|---|---|
| `gguf_lite.py` | 只读 GGUF 文件头：KV 元数据、张量目录、量化类型分布，不读权重本身 |
| `dual_models.json` | 双模型通用基线的唯一事实源（SSOT）：官方几何、revision、GGUF 发布者和精确字节 |
| `hybrid_ledger.py` | 混合架构（Gated DeltaNet + full attention）统一显存账本，从 manifest 读几何 |
| `test_hybrid_ledger.py` | 账本回归测试，固定 KV 每 token、循环状态和制品字节 |
| `results_qwen3_0.6b.json` | Qwen3-0.6B 历史量化实测，降级为 smoke test 和回归夹具 |

## 双模型基线

通用主线统一两个模型，几何和制品口径只在 `dual_models.json` 里登记一次：

| key | 模型 | FFN | llama.cpp 架构 | 角色 |
|---|---|---|---|---|
| `qwen3.8-27b` | Qwen3.8-27B | 稠密 | `qwen35` | 稠密主案例 |
| `qwen3.5-35b-a3b` | Qwen3.5-35B-A3B | MoE | `qwen35moe` | MoE 主案例 |

两个模型都不是“所有层都是传统全注意力”。它们都混合了 Gated DeltaNet 线性注意力和少量 full attention：

- 只有 full attention 层存随上下文增长的 softmax KV；
- DeltaNet 层只存固定大小的循环状态（F32），不随上下文变长；
- “稠密 / MoE”说的是 FFN 形态，和注意力形态是两件事。

MoE 的每 token 激活参数（35B-A3B 是 3B）只决定计算量，不决定常驻显存。权重存储按整份量化文件的字节算，不能拿 3B 去估显存。

## 数据状态

`dual_models.json` 里每个模型带 `measurement_status`。当前都是 `pending_gpu_measurement`：

- 已核实的是官方 `config.json` 几何、官方和 GGUF 仓库的 revision、发布者文件的精确字节；
- 没有真实 GPU 实测，`hybrid_ledger.py` 只产出可复算的字节账本和纯带宽上限，不产出 pp/tg、峰值显存这类实测数；
- CUDA 上下文和 compute buffer 是沿用的假设值（默认各 0.5 GiB），上机后要用 `llama-fit-params -fitp on` 核对，再回填。

不要把 Qwen3-0.6B、Qwen3-30B-A3B 的旧实测改个模型名当成新模型实测。新性能数字必须有原始日志、命令、文件 revision 和机器环境。

## 跑测试

```bash
cd common
python3 test_hybrid_ledger.py
```

## 命令行看账

```bash
# 27B 单卡 24 GiB，bartowski Q4_K_M，32k q8 KV，单槽挂 mmproj
python3 hybrid_ledger.py --model qwen3.8-27b --publisher bartowski \
    --quant Q4_K_M --ctx 32768 --kv q8_0 --np 1 --mmproj --card 24

# 35B-A3B 同样口径
python3 hybrid_ledger.py --model qwen3.5-35b-a3b --publisher bartowski \
    --quant Q4_K_M --ctx 32768 --kv q8_0 --np 1 --mmproj --card 24
```
