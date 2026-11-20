# 13-speculative：投机解码

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenziyaos/zero-to-llama/blob/main/colab/13-speculative.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

LCPP-13 的配套代码。要回答几个问题：草稿模型配多快、猜多长才回本；CPU 上目标模型一次验 k+1 个位置到底贵多少；p_min 早停为什么能把 k = 16 救回来；温度大于 0 时逐位比对和拒绝采样差多少接受率；n-gram、lookup、lookahead 这些不要草稿模型的方法在什么任务上赚。

本机是 4 核 CPU，没有 GPU，数字只代表这台机器上的相对关系。只用不监听端口的程序：`llama-bench`、`llama-completion`、`llama-speculative-simple`、`llama-speculative`、`llama-lookup`、`llama-lookahead` 和 `test-speculative-rejection-sampling`。llama-server 的 `--spec-synth-*`、SPEED-Bench、EAGLE-3、MTP、DFlash、DSpark **都未在本机实测**。

机器速度会随时间漂，同一配置前后两次能差 10% 以上。`inter` 段把几个关键对比交替跑 3 轮，加速比只拿同一轮里的两个数相除。

## 文件

| 文件 | 作用 | 需要什么 |
|---|---|---|
| `spec_model.py` | Leviathan 式 (1) 的期望产出、加速比 E / (γc + v(γ))、由平均接受长度反解 α、按实测代价曲线预测加速比、合成分布上两种验证的接受率 | 只要 Python 3 |
| `exp13.sh` | 十三组实验：`cost`、`base`、`draft`、`pmin`、`temp`、`ngram`、`lookup`、`granite`、`stoch`、`base2`、`noise`、`inter`、`unit` | 编好的 llama.cpp、Qwen3-0.6B 的 Q8_0/bf16/Q4_0/Q2_K/IQ2_M，granite 一对 MoE |
| `test_spec.py` | 17 个用例：收益模型用公式测，实验结论（温度 0 无损、接受率随位置递减、p_min 的趋势、拒绝采样多接受、交替跑的加速比）用 logs 核对 | 只要仓库；没有 logs 的用例跳过 |
| `prompts/` | code、prose、rewrite（提示里带原文）、en 四段提示 | — |
| `logs/` | 上面各组的输出；`.err` 只留统计相关的行 | — |

## 快速试跑

不需要模型文件：

```bash
python3 spec_model.py table             # α × γ 的期望接受长度
python3 spec_model.py synth 3.0 4       # 平均接受长度 3、n_max 4 时每位置的接受率
python3 spec_model.py temp              # 温度从 0 到 1.2，两种验证的期望接受率
python3 test_spec.py
```

有模型和编好的 llama.cpp 时：

```bash
bash exp13.sh cost      # 先看目标一次喂 1、2、3、5、9、17 个 token 的耗时
bash exp13.sh draft     # 草稿量化 × n_max，约 1 小时
bash exp13.sh inter     # 交替跑，算同一轮的加速比
python3 spec_model.py fit logs
```

几点注意：

- `llama-speculative-simple` 不给草稿填 `result_q`，`--spec-draft-sampling probabilistic` 在这里不起作用，温度大于 0 时也只做逐位比对。拒绝采样要用老的 `llama-speculative`（单序列、温度大于 0）或 server 路径。
- 统计行要加 `-v` 才打印，整份日志很大，`exp13.sh` 用 grep 只留相关的行。
- `ngram-mod` 缺省 `n_min = 48`，草稿凑不满 48 个就作废；短输出里一次草稿都不出是正常的。
- `test-speculative-rejection-sampling` 缺省 5000 次试验，本机约 3.8 小时；这里只跑 5 次确认程序能走通，“失败”是样本太少，不说明实现有错。
- 不跑 llama-server、llama-cli、rpc-server。
