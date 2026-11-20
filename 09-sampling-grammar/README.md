# 09-sampling-grammar：采样、语法与聊天模板

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-llama/blob/main/colab/09-sampling-grammar.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

LCPP-09 的配套代码。要回答几个问题：缺省采样链每一步留下多少候选、花多少微秒；温度放在截断前还是截断后有什么差别；GGUF 里的采样参数什么时候生效；重复惩罚和 DRY 各压住多少重复；GBNF 语法约束的两种用法差多少；后端采样省下什么；聊天模板到底拼出了哪些 token。

## 文件

| 文件 | 作用 | 需要什么 |
|---|---|---|
| `samp_bench.c` | 直接调 libllama：`micro` 在最后一个位置的 logits 上测 14 种采样器配置的耗时和候选数，并把 logits 存成 `.f32`；`gen` 带 GBNF 生成，对比先套语法和先采样后检查两种做法的拒绝率和每 token 耗时 | 编好的 llama.cpp、一个 GGUF |
| `samp_sim.py` | 按 llama.cpp 0ec4027 的规则用 numpy 重算 top_k、top_p、min_p、top-nσ、typical、penalties，报候选数、概率质量、熵，以及温度位置的影响 | numpy，`.f32` logits 或合成数据 |
| `render_tmpl.py` | 从 GGUF 取出 `tokenizer.chat_template`，用 jinja2 渲染五种消息（含 enable_thinking=false、多轮、工具） | jinja2、llama.cpp 的 gguf-py |
| `collect_rep.py` | 统计输出里重复 4-gram 占比、重复行数和最长重复片段 | 只要输出文本 |
| `exp09.sh` | 六组实验：`micro`、`gen`、`params`、`rep`、`bs`、`tmpl` | 编好的 llama.cpp、Q4_K_M GGUF |
| `test_samp.py` | 11 个用例：numpy 重算和 C 采样器对拍、min_p 的等价形式、惩罚的符号、温度位置、top-nσ 与温度无关、日志里的关键数字 | numpy，只要仓库 |
| `logs/` | Qwen3-0.6B 在 4 核 EPYC 上的实测日志、三份 logits 和汇总 `sim.txt`、`rep_summary.txt` | — |

## 快速试跑

不需要模型文件：

```bash
python3 test_samp.py
python3 samp_sim.py                       # 合成 logits
python3 samp_sim.py logs/logits-fact.f32  # 实测 logits
```

有模型文件时：

```bash
bash exp09.sh micro    # 约两分钟
bash exp09.sh gen      # 约一分钟
bash exp09.sh params
bash exp09.sh rep
bash exp09.sh bs
bash exp09.sh tmpl
```

只用 `llama-completion` 和本目录的 `samp_bench`。`llama-cli` 在 0ec4027 里会在进程内起一个 server，本系列不跑它；`--chat-template-kwargs` 只注册给 server 和 cli，所以 enable_thinking 的效果用 `render_tmpl.py` 渲染出来看。本机没有 GPU，所有数字都是 CPU 单机实测。
