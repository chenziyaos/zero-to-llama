# zero-to-llama

“从 0 开始 llama.cpp”飞书系列的配套代码。每章一个目录，目录名和文档章节号一一对应，索引页是 LCPP-00。

系列文档的 Markdown 版放在 [`book/`](book/README.md)，从索引页 LCPP-00 读起；各章末尾有上一章、下一章的链接。

在线阅读：<https://chenziyaos.github.io/zero-to-llama/>，带侧边目录和全文搜索。它由 `mkdocs.yml` 从 `book/` 生成，推送到 main 后自动发布；本地预览：`pip install -r docs-site/requirements.txt && mkdocs serve`。

## 在 Colab 上跑

[`colab/`](colab/) 下每章一个 notebook，打开后依次执行：克隆本仓库、装 `requirements-cpu.txt`、跑本章单元测试、跑正文里的 CPU 命令。notebook 里的命令都在 Python 3.11 的 CPU 环境下实跑过，退出码为 0；少数演示脚本按设计以非 0 退出（样例本身不达标），已在命令后注明。需要 GPU、模型文件或运行中服务的命令没有放进去。

入口示例：[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-llama/blob/main/colab/02-ledger.ipynb)

## 版本基线

- llama.cpp 主干提交 `0ec4027`（2026-10-30），发行版 `v0.8.0`（2026-10-28），ggml `0.28.0`
- 实测机器：4 vCPU（AMD EPYC，AVX512、AVX512_VNNI、AVX512_BF16），8 GB 内存，无 GPU
- 编译：`cmake -B build -DCMAKE_BUILD_TYPE=Release`，默认 `GGML_NATIVE=ON`、`GGML_OPENMP=ON`、`GGML_CPU_REPACK=ON`
- 实测模型：Qwen3-0.6B，从 HF 的 bf16 权重自己转 GGUF 再量化，15 个变体
- 脚本只依赖 Python 3.9 及以上的标准库，GGUF 读取用 `common/gguf_lite.py`，不装 `gguf` 包也能跑

## 两层运行环境

无 GPU 的小机器和大模型部署案例是两回事，不要混为一谈：

- **CPU 层（本仓库默认）**：预测器、解析器、模拟器和回归测试只吃标准库，在 8 GB 无 GPU 机器上直接跑，用 Qwen3-0.6B 做 smoke test 和历史夹具。
- **GPU 部署层（双模型基线）**：通用主线是稠密 `Qwen3.8-27B`（llama.cpp 架构 `qwen35`）和 MoE `Qwen3.5-35B-A3B`（`qwen35moe`）。它们的几何、revision 和 GGUF 制品字节登记在 `common/dual_models.json`，显存账走 `common/hybrid_ledger.py`。这一层只做启动前的字节推算，真实 pp/tg、峰值显存必须上 GPU 实测后回填，`measurement_status` 在实测前保持 `pending_gpu_measurement`。

## 文件命名约定

| 前缀或后缀 | 含义 | 运行环境 |
|---|---|---|
| `estimate_*.py`、`sim_*.py` | 启动前的预测器或模拟器，复现主干里的关键账本 | CPU，不需要 llama.cpp |
| `parse_*.py`、`collect_*.py` | 从 llama.cpp 的日志和 `llama-bench -o jsonl` 里抽数字，和预测对账 | CPU |
| `*.sh` | 编译、下载、运行、测量脚本，默认 `LLAMA_BIN=../../llama.cpp/build/bin` | 装了 llama.cpp 的机器 |
| `test_*.py` | 用仓库里的日志夹具测解析器和预测器，`python3 test_xxx.py` 直接运行 | CPU |
| `sample_*.log` | 实机录制的日志夹具，来源写在各章 README 里 | — |

## 目录

| 目录 | 章节 | 内容 |
|---|---|---|
| `common/` | 全系列 | GGUF 头读取器、双模型 manifest（`dual_models.json`）、混合架构显存账本（`hybrid_ledger.py`）、Qwen3-0.6B 量化实测结果 |
| `01-first-run/` | LCPP-01 第一次运行 | 内存预测器、加载日志解析与对账、运行脚本 |
| `02-ledger/` | LCPP-02 推理账本 | 内存、带宽和 Roofline 计算器 |
| `03-bench-ppl/` | LCPP-03 测量方法 | llama-bench、perplexity、KLD 的跑法和统计 |
| `04-gguf/` | LCPP-04 GGUF | 文件格式拆解、元数据和张量目录检查 |
| `05-ggml-graph/` | LCPP-05 ggml 与计算图 | toy_graph 验证 gallocr、n_outputs_max 对照、调度器日志统计、融合和图复用对照 |
| `06-backends-build/` | LCPP-06 后端与编译 | 6 种编译开关和 14 个 CPU 变体的对照、check_build.py 自检、BLAS 和 repack 实验 |
| `07-quantization/` | LCPP-07 量化 | type_mix 读配方、dequant 拆超级块和按激活加权的误差、quant.sh 与 measure.sh 跑 19 个文件的 KLD、collect_results 出帕累托前沿、15 个回归用例 |
| `08-kv-context/` | LCPP-08 KV cache 与长上下文 | KV 账本（含双模型混合架构的 full-attention KV 与 DeltaNet 循环状态分账）、FA 与 KV 量化的速度对比 |
| `09-sampling-grammar/` | LCPP-09 采样、语法与聊天模板 | 采样链、GBNF、JSON Schema |
| `10-cpu-tuning/` | LCPP-10 CPU 调优 | cpu_probe 拓扑探测、十三组线程/绑核/等待/repack/tiled/加载/带宽实验、11 个回归用例 |
| `11-gpu-offload-moe/` | LCPP-11 GPU 卸载与 MoE | moe_trace 抓路由、moe_cache_sim 五种放法回放、vram_ledger 显存账本和缓存分组、15 个回归用例 |
| `12-multi-gpu-rpc/` | LCPP-12 多卡与 RPC | fake_devs 假多卡跑切层和显存账本、split_plan 复算、comm_volume 通信量、rpc_msg 消息尺寸、17 个回归用例 |
| `13-speculative/` | LCPP-13 投机解码 | spec_model 收益模型和 α 反解、草稿量化与 p_min 实测、逐位比对对拒绝采样、n-gram 与 lookahead、交替跑加速比、17 个回归用例 |
| `14-server/` | LCPP-14 服务端 | batched-bench 并发与统一 KV、llama-parallel 共享前缀、slot_sim 选槽与 RAM 缓存复刻、上下文分账对账、16 个回归用例 |
| `15-multimodal/` | LCPP-15 多模态 | mm_probe 计块与编码计时、mm_tokens 复刻三种模型的 token 计数、线程与批量编码、llama-mtmd-cli 看图和转写、11 个回归用例 |
| `16-router-ops/` | LCPP-16 路由与运维 | preset_tool 复刻预设合并、router_sim 七个调度场景（LRU、忙模型、CPU/GPU 余量、mmproj、--fit 抖动）、metrics_parse 看 gauge 口径、加载耗时与峰值内存、20 个回归用例 |
| `17-tuning-playbook/` | LCPP-17 调参手册 | diagnose.py 解析 stderr 给误配分 error/warn/silent 三档并算实测比预测、plan 不跑模型算内存四项；11 种误配、五组症状 benchmark、抢核诊断、13 个回归用例 |
| `18-add-model/` | LCPP-18 新模型接入 | dump_logits 用 eval 回调取每层出口、hf_ref 出 f32 参考、compare 算 KLD 和逐层余弦、inject 注入四种转换错、massive 和 pos0 查超大激活、tok_compare 比分词、10 个回归用例 |
| `19-test-contrib/` | LCPP-19 测试与贡献 | ctest 标签与 fixture、test-backend-ops 注入融合内核错误、parse_tbo 逐条预测、perf_fuse 和 bench_ab 交替 A/B、load_sql 入库、13 个回归用例 |
| `20-qwen-24g/` | LCPP-20 实战：24 GB 单卡 | ledger.py 推算 Qwen3-32B 和 Qwen3.8-27B 的显存账（KV、DeltaNet 状态、草稿和 MTP、mmproj、提示缓存）和 decode 区间，27B 几何与字节走共享账本，两套启动命令，18 个回归用例 |
| `21-deepseek-v41/` | LCPP-21 实战：DeepSeek-V4.1-Flash 装进一台机器 | plan.py 逐张量核对 ISTA GSQ-RCO 三档量化的字节，推算 Engram 懒加载、压缩 KV、路由专家上卡层数和 decode 区间（拟合社区实测扫描），96 GB 和 24 GB 两套启动命令，12 个回归用例 |
| `22-qwen38-27b/` | LCPP-22 实战：一张卡跑 Qwen3.8-27B | plan.py 推算 16 层 softmax KV 加 48 层 Gated DeltaNet 的显存账、24 GB 单卡的量化档位与上下文上限、MTP 投机解码和 mmproj（全部是推算，没有上 GPU），两套启动命令，8 个回归用例 |

## 安装依赖

依赖分两档，版本都锁定了：

```bash
pip install -r requirements-cpu.txt          # 模拟器和单元测试
pip install -r requirements-full.txt         # 要跑 18 章真实模型对拍时
```

## 快速试跑

不装 llama.cpp 也能跑：

```bash
cd zero-to-llama/01-first-run
python3 parse_load_log.py sample_load.log
python3 test_first_run.py
```

有 GGUF 文件时，先预测再对账：

```bash
python3 estimate_memory.py ~/models/Qwen3-0.6B-Q4_K_M.gguf --ctx 4096
bash run.sh ~/models/Qwen3-0.6B-Q4_K_M.gguf
python3 parse_load_log.py logs/run-*.log --model ~/models/Qwen3-0.6B-Q4_K_M.gguf
```

## 参考

- llama.cpp：https://github.com/ggml-org/llama.cpp
- ggml：https://github.com/ggml-org/ggml
- GGUF 规范：https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

## 引用

如果本系列对你的工作有帮助，可以这样引用：

```bibtex
@misc{zero-to-llama,
  author       = {chenziyaos},
  title        = {从 0 开始 llama.cpp（zero-to-llama）},
  year         = {2026},
  howpublished = {\url{https://github.com/chenziyaos/zero-to-llama}},
  note         = {GitHub repository}
}
```
