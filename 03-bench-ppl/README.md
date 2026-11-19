# 03-bench-ppl：测量方法

LCPP-03 的配套代码。要回答的问题只有一个：两个数不一样，这个差异算不算数。速度用 `llama-bench`，质量用 `llama-perplexity` 的三种模式（普通 PPL、`--kl-divergence`、`--hellaswag`），两边都先量噪声，再下结论。

## 文件

| 文件 | 作用 | 需要什么 |
|---|---|---|
| `measure.sh` | 本章全部实验：5 进程噪声、关预热、生成长度、上下文长度对 PPL、HellaSwag 400 题、KLD 自检、线程数和 batch 对 PPL 的影响 | 编好的 llama.cpp、bf16 和量化 GGUF、wiki.test.raw、hellaswag_val_full.txt |
| `bench_stats.py` | 读 `llama-bench -o jsonl`：按配置分组给均值、95% CI、中位数、进程内和进程间 cv、掉速样本数；`--compare` 做 Welch 检验；`--need` 估算要分辨某个相对差需要多少次独立运行 | 只要日志 |
| `parse_ppl.py` | 读 perplexity 日志：一行摘要；`--curve` 看累计值怎么收敛；`--chunks` 把累计值还原成逐块值，比较块级标准误和日志里的 ±；`--pair` 比较两个 KLD 日志；`--hs-pair` 把两个 HellaSwag 日志逐题配对做 McNemar 检验 | 只要日志 |
| `test_bench_ppl.py` | 16 个用例，夹具是本目录 `logs/` 和 `../07-quantization/logs/` | 只要仓库 |
| `logs/` | Qwen3-0.6B 在 4 核 EPYC 上的实测输出，见下表 | — |

## 快速试跑

不需要模型文件：

```bash
python3 test_bench_ppl.py

# 速度：同一配置 5 个进程，每个 -r 10
python3 bench_stats.py logs/noise-*.jsonl --need 0.03
python3 bench_stats.py --by-file logs/noise-*.jsonl
python3 bench_stats.py --compare logs/noise-2.jsonl logs/noise-3.jsonl   # A/A 测试，同一配置也会判“B faster”

# 质量：PPL 的累计曲线、块级标准误
python3 parse_ppl.py logs/ppl-c128.log logs/ppl-c512.log logs/ppl-c2048.log
python3 parse_ppl.py --curve  logs/ppl-c512.log
python3 parse_ppl.py --chunks logs/ppl-c512.log

# KLD：自检、两个量化的比较
python3 parse_ppl.py logs/kld-self.log
python3 parse_ppl.py --chunks ../07-quantization/logs/ppl-q-Q4_K_M.log
python3 parse_ppl.py --pair ../07-quantization/logs/ppl-q-Q4_K_M.log ../07-quantization/logs/ppl-qi-Q4_K_M.log

# HellaSwag：区间宽度和逐题配对
python3 parse_ppl.py logs/hs-*.log
python3 parse_ppl.py --hs-pair logs/hs-qwen3-0.6b-bf16.log logs/hs-q-Q4_K_M.log
python3 parse_ppl.py --hs-pair logs/hs-qwen3-0.6b-bf16.log logs/hs-qi-Q2_K.log
```

完整复现，4 核约 40 分钟：

```bash
LLAMA_BIN=../../llama.cpp/build/bin MODELS=../../models DATA=../../data ./measure.sh
```

`base.kld` 由 `../07-quantization/measure.sh` 生成（wiki.test 前 20 块，c=512，约 1.5 GB）。

## 日志和关键数字

| 日志 | 内容 | 结果 |
|---|---|---|
| `noise-1..5.jsonl` | Q4_K_M，`-t 4 -p 512 -n 128 -r 10`，5 个独立进程 | pp512 进程均值 399 到 459，进程间 cv 5.9%；tg128 75.5 到 84.7，进程间 cv 4.8%，50 个样本里 6 个比中位数慢 10% 以上 |
| `nowarm.jsonl` | `--no-warmup -r 6` | 第一个样本 pp 500、tg 71.7，不比后面慢；CPU 上预热影响小于噪声 |
| `ngen.jsonl` | `-p 0 -n 16,128,512 -r 3` | tg16 三次 56/42/72，太短不稳；tg512 59.7，比 tg128 的 71.7 低 17%，KV 变长的代价 |
| `ppl-c128/512/2048.log` | bf16，wiki.test，总 token 数相同只改 `-c` | PPL 35.56 / 19.45 / 16.37；块级标准误是日志 ± 的 1.4 / 2.5 / 4.7 倍 |
| `kld-self.log` | bf16 和自己存的 base logits 比 | KLD 最大 5.5e-5，top-1 100%；但 ln(PPL(Q)/PPL(base)) = +0.0128 ± 0.0038：存 base 时把比最大值低 16 nat 以下的 logit 截断（perplexity.cpp:86），极少数“意外” token 的损失被压低，base 的 PPL 因此偏低 1.3% |
| `ppl-t1/t4/t4-b128.log` | Q4_K_M，`-t 1`、`-t 4`、`-t 4 -b 128` | 三次 PPL 都是 25.6211 ± 1.25012，逐块累计值 4 位小数完全一致；耗时 89 s / 27 s / 31 s |
| `hs-*.log` | HellaSwag 400 题，`-c 512` | bf16 46.75%、Q4_K_M 46.50%、qi-Q4_K_M 46.50%、qi-Q2_K 36.50%；区间半宽约 4.9 个百分点 |

## 计算口径

- `llama-bench` 的 `stddev_ts` 是同一进程内 `-r` 次重复的样本标准差（llama-bench.cpp:110、1564），不包含进程之间的漂移。`bench_stats.py` 把每个文件当一次独立实验，`cv_between` 是文件均值之间的变异系数。
- perplexity 每块只对后半块打分（perplexity.cpp:540），逐块输出 `[i]` 是累计值。`parse_ppl.py --chunks` 用 `x_i = i·c_i − (i−1)·c_{i−1}` 还原逐块值（PPL 在对数域里还原），每块当一个样本求标准误。
- `Final estimate` 的 ± 是把 token 当独立样本的标准误（perplexity.cpp:641-647），偏窄。KLD 模式的 `Mean ln(PPL(Q)/PPL(base))` 的 ± 扣掉了协方差（perplexity.cpp:1909-1911）。
- HellaSwag 用固定种子 `mt19937(1)` 抽题（perplexity.cpp:775），同一份数据、同样的 `--hellaswag-tasks`，不同模型做的是同一组题。`hs_outcomes` 从累计准确率还原每一题的对错，`mcnemar` 只看两个模型答案不一致的题。
