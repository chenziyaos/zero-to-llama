# 15-multimodal：mmproj、图片切块和音频

LCPP-15 的配套代码。要回答几个问题：一张图在 libmtmd 里会切成几块、变成多少 token；`--image-max-tokens` 调高调低各会怎样；同样多的 token，图片比文本贵多少，贵在哪；编码器吃不吃线程、多块图能不能批量编码；一段音频按什么规则变成 token；换一份 mmproj 会不会让 token 数变掉。

本机是 4 核 CPU，没有 GPU，数字只代表这台机器上的相对关系。**不启动 llama-server，也不启动任何监听端口的程序**。用到的模型和前面几篇的 Qwen3-0.6B 不同：视觉用 SmolVLM2-256M 和 LFM2-VL-450M，音频用 Qwen3-ASR-0.6B，都从 Hugging Face 的 ggml-org 下载 Q8_0，SmolVLM2 另下一份 f16 的 mmproj 做对照。视频输入要外部 ffmpeg，本机没装，**未实测**；server 里的多模态行为只读了源码，**未实测**。

## 文件

| 文件 | 作用 | 需要什么 |
|---|---|---|
| `exp15.sh` | 十段：`build`（编 mm_probe）、`count`（各尺寸切块数和 token 数）、`time`（编码、喂进语言模型、同长度文本预填充）、`threads`（编码器 1/2/4 线程）、`batch`（逐块编码和 mtmd_batch 批量编码）、`cli`（llama-mtmd-cli 看图说话）、`climax`（调低上限后再看图说话）、`asr`（语音转写）、`unit`、`model` | 编好的 llama.cpp（含 libmtmd）、上面三个模型和它们的 mmproj |
| `mm_probe.cpp` | 直接调 libmtmd 的小程序：`count` 用占位位图只走预处理和切块，`time` 真跑编码和解码并做文本对照，`batch` 比较两种编码方式；合成图是确定性的渐变加噪点 | libmtmd、libllama |
| `mm_tokens.py` | 按 0ec4027 源码复刻的 token 计数：SmolVLM2（IDEFICS3）的缩放和切块、LFM2 的 smart resize 和切块、Qwen3-ASR 的 800 帧窗口；连 float 精度造成的多切一圈也复刻了 | 只要 Python 3 |
| `test_mm.py` | 11 个用例：两种视觉模型的缺省计数和 count 日志逐条对账、小图被放大、上限调低、旧 mmproj、LFM2 缩略图和切块门槛、像素上限、音频 token 公式和 CLI 日志对账 | 只要 Python 3 |
| `logs/` | 上面各段的输出；`.err` 只留编码、解码和性能统计的行 | — |

## 快速试跑

不需要模型文件：

```bash
python3 mm_tokens.py         # 各尺寸在两种视觉模型下的块数和 token 数，音频 1 到 60 秒的 token 数
python3 test_mm.py
```

有模型和编好的 llama.cpp 时：

```bash
bash exp15.sh build     # 编 mm_probe
bash exp15.sh count     # 7 种尺寸，缺省和各档 --image-max-tokens
bash exp15.sh time      # 编码和预填充，3 次取中位数
bash exp15.sh cli       # test-1.jpeg 让两个模型各说一句话
bash exp15.sh asr       # test-2.mp3 转写
bash exp15.sh unit      # test-mtmd-impl、test-mtmd-c-api 加 test_mm.py
```

几点注意：

- SmolVLM2 的 Q8_0 和 f16 两份 mmproj 元数据不同（f16 那份没带 `preproc_image_size`），同一张图的 token 数差好几倍。比较之前先确认用的是哪一份。
- `--image-max-tokens` 只对动态分辨率的模型生效；语言模型要对图片 token 做非因果注意力的模型（Gemma 3 等），llama-mtmd-cli 和 server 都会把它压到 `-ub`；SmolVLM2 和 LFM2 不受这条影响。
- 两种视觉编码器都不支持批量，`batch` 段里 `n_batches` 等于块数，批量编码不会更快。
- Qwen3-ASR 不写 `-c` 时按训练长度 65536 开 KV，内存紧时会被自动降，日志里有提示。
- 不跑 llama-server、llama-cli、rpc-server、`scripts/server-bench.py`。
