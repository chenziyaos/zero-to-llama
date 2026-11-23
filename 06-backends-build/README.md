# 06-backends-build：后端与编译

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zyaochen/zero-to-llama/blob/main/colab/06-backends-build.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

LCPP-06 的配套代码，回答五个问题：
- 同一份源码换几个 CMake 开关，预填充和解码各差多少？
- GGML_NATIVE=OFF 以后丢了哪些指令集，怎么自检？
- GGML_BACKEND_DL 加 GGML_CPU_ALL_VARIANTS 编出的 14 个 CPU 库，各自多快，启动时挑哪个？
- 开 OpenBLAS 为什么没变快，4 线程为什么反而更慢？
- llamafile_sgemm 和 repack 各接管了哪些量化类型，代价是什么？

GPU 部分（CUDA 架构列表、HIP、Vulkan、SYCL、Metal）按源码和官方文档整理，未在本机实测，这里没有对应脚本。

## 文件

| 文件 | 作用 |
|---|---|
| `exp06.sh NAME [cmake 选项]` | 按给定开关编一份，记 configure、build 时间和 ggml 相关 .so 的体积 |
| `run06.sh` | 调 `exp06.sh` 编 6 种对照：native、generic、sse42、blas、nollamafile、dl |
| `bench06.sh` | 每个构建抓 system_info，跑 4 线程 pp512 和 tg64；dl 构建逐个 CPU 变体单独测 |
| `bench06b.sh` | 4 线程下各构建交替跑两轮，以及 `--repack 0,1` 对照 |
| `bench06c.sh` | 正文主证据：单线程 pp256、tg32，6 种构建加 12 个可加载的 CPU 变体 |
| `bench06d.sh` | 第 5、6 章的补充：repack 单线程、BLAS 线程设置、OMP_WAIT_POLICY、split 数、repack 内存 |
| `check_build.py` | 对比 system_info、CMakeCache 和 `/proc/cpuinfo`，报 MISSING、UNUSED 和 NOTE |

脚本的路径都走环境变量：`SRC`（llama.cpp 源码，缺省 `../../llama.cpp`）、`OUT`（构建输出）、`LOG`（日志）、`M`（模型目录，里面放 `q-Q4_K_M.gguf`、`q-Q4_0.gguf`、`q-Q8_0.gguf`）。

```bash
SRC=~/llama.cpp OUT=$PWD/b06 LOG=$PWD/logs bash run06.sh   # 4 核约 30 分钟
OUT=$PWD/b06 LOG=$PWD/logs M=~/models bash bench06c.sh
python3 check_build.py b06/generic ~/models/q-Q4_K_M.gguf
```

所有跑 llama-bench 的脚本都先 `cd /tmp`。dl 构建会把当前目录也加进 CPU 库的搜索路径，在别的构建目录下运行会把那里的 `libggml-cpu-*.so` 一起拿来打分。

## logs/

4 vCPU EPYC（AVX512、AVX512_VNNI、AVX512_BF16）、gcc 12.2、cmake 3.25、Qwen3-0.6B、llama.cpp 主干 0ec4027 上的实测。

| 文件 | 内容 |
|---|---|
| `t1.txt` | 单线程主表，正文第 1、3、4、5、6 章的数都出自这里 |
| `NAME.time.txt`、`sizes.txt` | 编译耗时和各 .so 体积 |
| `NAME.cmake.txt`、`NAME.build_tail.txt` | 配置输出全文和编译日志末尾（编译全文太大，没放进来） |
| `NAME.sysinfo.txt` | 每个构建的 system_info 和 load_backend 行 |
| `NAME.bench.jsonl`、`bench_summary.txt` | 4 线程 pp512、tg64 |
| `dl1_VARIANT.jsonl`、`.err` | 单个 CPU 变体的 4 线程结果；alderlake、sapphirerapids 的 .err 是 CPU backend is not loaded |
| `interleave.txt` | 4 线程交替两轮，同一配置两轮能差一倍，所以正文以单线程为准 |
| `repack.txt`、`repack.jsonl`、`repack_t1_tg.txt` | `--repack 0,1` 对照 |
| `blas_omp.txt` | OMP_WAIT_POLICY=passive 前后 blas 和 native 的 4 线程 pp512 |
