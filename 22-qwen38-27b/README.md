# 22-qwen38-27b：一张卡跑 Qwen3.8-27B

LCPP-22 的配套代码。这一篇要回答的是：这个 64 层的稠密模型里只有 16 层存 softmax KV，其余 48 层是 Gated DeltaNet 线性注意力，那长上下文的显存账和普通 Transformer 差在哪；24 GB 单卡选哪档量化、上下文能开多长；自带的 1 个 MTP 头怎么开 draft-mtp 投机解码；看图为什么要挂 mmproj，纯文本 GGUF 的坑在哪。模型是 Qwen3.8-27B，几何和 Qwen3.5-27B 完全一样：官方 config 的 model_type 仍是 qwen3_5，GGUF 元数据里的架构名仍是 qwen35，所以公式一行没改，只换文件口径和采样参数。

我手上没有 GPU 机器，这个目录里的字节、显存和速度数字**全部是推算**，按 llama.cpp 0ec4027 的张量形状和分配逻辑写成公式，放在 `plan.py` 里。两条启动命令只写了，没有在 GPU 上跑过。**不启动 llama-server，也不启动任何监听端口的程序**；`run_*.sh` 是给上机时用的。

## 文件

| 文件 | 作用 | 需要什么 |
|---|---|---|
| `plan.py` | 显存账和速度：16 个 full-attn 层每 token 的 KV、48 个 DeltaNet 层固定大小的 F32 循环态、各档量化在 16/24/32/48 GB 上的最长上下文、draft-mtp 让循环态按 (1+草稿数) 倍增、RTX 4090 等卡的 decode 带宽上限；`--emit` 把文中数字写成模块 | Python 3 标准库 |
| `test_plan.py` | 回归用例：把正文引用的推算值钉住，改几何、公式或发布者文件大小时会报出来 | Python 3 标准库 |
| `run_24g.sh` | 第 10 章的看图配置：bartowski Q4_K_M（imatrix，主文件内嵌 MTP 头）+ mmproj bf16，32K，q8_0 KV，单槽，`-fa on`，思考采样 1.0/0.95/20；`THINK=0` 切非思考档 0.7/0.8/20 加 presence 1.5；`MTP=3` 打开 draft-mtp；`fit` 子命令只跑 llama-fit-params | 编好的 llama.cpp（不旧于 b10896）和 GGUF |
| `run_text.sh` | 第 10 章的纯文本退路：lmstudio-community Q4_K_M（16.81 GB，仓里另有 mmproj，本脚本不挂，模型卡没写 MTP 处理），32K，q8_0 KV，单槽；同样支持 `THINK=0` 和 `fit` | 编好的 llama.cpp（不旧于 b10430，建议直接上 b10896 或更新）和 GGUF |

## 快速试跑

不需要 GPU：

```bash
python3 test_plan.py                 # 7 个用例
python3 plan.py                      # 各档量化在 16/24/32/48 GB 上的最长上下文、带宽上限
python3 plan.py --quant Q4_K_M --ctx 32768 --kv q8_0 --np 1 --mmproj
python3 plan.py --quant Q4_K_M --ctx 32768 --kv q8_0 --np 1 --mtp 3 --mmproj
python3 plan.py --source lmstudio --quant Q4_K_M --ctx 32768 --kv q8_0 --np 1
python3 plan.py --fit-card 24 --quant Q4_K_M --kv q8_0 --np 1 --mmproj
```

上机以后：

```bash
hf download bartowski/Qwen3.8-27B-GGUF Qwen3.8-27B-Q4_K_M.gguf \
  mmproj-Qwen3.8-27B-bf16.gguf --local-dir models/qwen38-27b
bash run_24g.sh fit                  # 看 compute buffer 的实际 MiB
python3 plan.py --quant Q4_K_M --ctx 32768 --kv q8_0 --np 1 --mmproj --compute 0.8   # 用实测值重算
bash run_24g.sh                      # 或 MTP=3 bash run_24g.sh
```

几点注意：

- 模型几何取自官方 config.json（Qwen3.8 revision 1d4bf0f）：64 层，`full_attention_interval=4`，层号 3、7、…、63 是 full attention，共 16 层存 KV；其余 48 层是 Gated DeltaNet，循环态写死 F32，不可量化，每条序列固定 149.63 MiB，和上下文长度无关。
- KV 每 token 只有 16 层：f16 64 KiB，q8_0 约 34 KiB。32K 单池 f16 2.0 GiB、q8_0 1.06 GiB；开到训练长度 256K，f16 也才 16 GiB。这是它和普通 27B Transformer 最大的不同。
- 文件大小抄自 Hugging Face 文件列表（2026-11-04 复核），单位 GB（10 的 9 次方），脚本统一换成字节再和 GiB 比。bartowski 的 Q4_K_M 17.44 GB（imatrix，MTP 头内嵌主文件，imatrix 档里 MTP 存 Q4_0），mmproj bf16 另算 0.93 GB；lmstudio-community 的 Q4_K_M 16.81 GB（精简三档，模型卡没写 MTP 处理）。不同发布者不直接比字节。
- 预算按 GiB，每张卡留 1 GiB fit 余量；CUDA 上下文和 compute buffer 各按 0.5 GiB 假设，`--cuda-ctx`、`--compute` 可改，上机后用 `llama-fit-params -fitp on` 读实际值。循环态按槽数乘，开 draft-mtp 再乘 (1+草稿数)。bartowski 的 MTP 头在主文件里，开 draft-mtp 不增加权重，只倍增循环态。
- 16 GB 卡连 Q4_K_M 权重都放不下任何主推量化，27B 的甜点是 24 GB（Q4_K_M）和 32 GB（Q5/Q6）。decode 数字是带宽上限乘校准效率区间，不是实测；唯一的社区实测锚点是 kingy 在 4090 上跑 unsloth 旧修订 Q4_K_M 的 49.1 t/s。
- 两个脚本都写死 `-c 32768`、`-np 1`、`-ngl 99`，放不下时在加载阶段直接报错，fit 不会在背后把层挪到内存。`LLAMA_BIN`、`MD`、`M`、`MM`、`MTP`、`THINK` 环境变量改二进制、模型目录和模型路径。
