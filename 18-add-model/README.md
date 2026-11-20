# 18-add-model：新模型接入和逐层对拍

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-llama/blob/main/colab/18-add-model.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

LCPP-18 的配套代码。要回答几个问题：把一个 HF 模型接进 llama.cpp 要改哪三层代码；转换出来的 GGUF 怎么证明和原始实现一致；转换器常见的几种错各自留下什么指纹；官方 model-conversion 流程只比短提示最后一个位置，哪些错它抓不到；分词器的 pre-tokenizer 选错会错成什么样。

样本模型是 Qwen3-0.6B（llama.cpp 已支持，拿来当“已知正确答案”），对照参考是 HF transformers 的 f32 前向。代码对齐 llama.cpp 0ec4027。本机 4 核 CPU，没有 GPU。**不启动 llama-server，也不启动任何监听端口的程序**，只用 `llama-tokenize`、`llama-debug`、测试程序和自己写的 `dump_logits`。

## 文件

| 文件 | 作用 | 需要什么 |
|---|---|---|
| `exp18.sh` | 九组实验：`convert`（转 bf16 GGUF，外加一次 `--dry-run`）、`ref`（HF f32 和 bf16 参考）、`base`（bf16、关 FA、Q8_0、Q4_K_M 对拍）、`kv`（改 rope theta、rms eps、context 三个元数据）、`inject`（四种转换错和只坏第 14 层）、`tok`（四种 pre-tokenizer 对拍加 `test-tokenizer-0`）、`archs`（`test-llama-archs -a qwen3`）、`debug`（官方 `llama-debug --save-logits`）、`short`（同几种错放到 5 个 token 的短提示上） | 编好的 llama.cpp，`models/Qwen3-0.6B`（HF 权重），transformers 和 torch |
| `dump_logits.cpp` | 用 libllama 跑一遍 token 序列，把每个位置的 logits 和指定中间张量（`l_out-N`、`result_norm`）写成 float32；`--kv` 在加载时改元数据 | libllama 头文件和动态库 |
| `hf_ref.py` | HF 参考前向：每个位置的 logits、每层 decoder 输出（forward hook 取）、最后的 norm 输出 | transformers |
| `compare.py` | 对拍：max_abs、nmse、nmse_var（check-nmse.py 口径）、位置 0 误差、KL 散度、top-1 一致率、第一次翻转位置，`--layers` 再给每层 nmse 和最差位置余弦 | numpy |
| `inject.py` | 往 GGUF 里注入四种错：`qk-permute`、`norm-plus1`、`qk-norm-one`、`swap-gate-up`，`--layer N` 只改一层 | `llama.cpp/gguf-py` |
| `hf_last.py` | 官方口径：短提示只比最后一个位置，报 nmse_var 分档、top-1、top-10 重合 | transformers |
| `massive.py` | 参考输出里每层最大激活的位置和通道，解释 bf16 基线逐层 max_abs 为什么从第 2 层起跳到 3 左右 | numpy |
| `pos0.py` | 位置 0 和其余位置分开算逐层余弦，并给出位置 0 最大通道的参考值和实测值，解释量化后最大误差为什么在位置 0 | numpy |
| `tok_compare.py` | 25 条样本分别过 HF tokenizer 和 `llama-tokenize`，逐条比 id | transformers，只含词表的 GGUF |
| `test_add_model.py` | 回归用例：permute 和两种 RoPE 布局的关系、统计口径、`logs/` 里的几条实验结论 | numpy；没有 gguf-py 或日志时对应用例跳过 |
| `runlog.py` | 跑一个命令，记墙钟、退出码和峰值 RSS | Python 3 |
| `prompt.txt` | 212 个 token 的中英混合提示 | — |
| `logs/` | `compare.jsonl`（每次对拍一行）、`cmp-*.txt`、`tok-*.json`、`hf-short-cmp.log` 和各步的运行日志 | — |

## 快速试跑

不需要模型文件：

```bash
python3 test_add_model.py                     # 10 个用例，零点几秒
python3 -c "import json;[print(x['tag'],x['logits']['kld_mean']) for x in map(json.loads,open('logs/compare.jsonl'))]"
```

有 HF 权重和编好的 llama.cpp 时：

```bash
bash exp18.sh convert   # 约 40 秒，1.5 GB 输出
bash exp18.sh ref       # HF f32 和 bf16 两遍前向
bash exp18.sh base      # 第一次会编 dump_logits
bash exp18.sh inject    # 每个坏 GGUF 用完就删，磁盘只多占 1.5 GB
bash exp18.sh short     # 和官方口径对照
```

几点注意：

- `R` 环境变量指向放 `llama.cpp`、`models` 的根目录，缺省是本仓库的上上级；坏模型写到 `$R/models/conv`，对拍输出写到 `$R/out18`。
- `llama-tokenize` 只加载词表，不吃 `--override-kv`，所以 pre-tokenizer 对照组用 `gguf_new_metadata.py --pre-tokenizer` 生成单独的文件。
- `llama-debug` 缺省按训练长度 40960 开上下文，峰值近 6 GB；加 `-c 512` 只要 1.5 GB，logits 完全相同。
- 每一步只跑一个重任务。HF 前向和 `dump_logits` 同时跑会把两边的墙钟和内存都弄乱。
- 不跑 llama-server、llama-cli、rpc-server。
