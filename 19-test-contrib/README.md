# 19-test-contrib：测试、性能对比和贡献

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zyaochen/zero-to-llama/blob/main/colab/19-test-contrib.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

LCPP-19 的配套代码。这一篇要回答几个问题：改了 llama.cpp 以后本地要跑哪些测试，和 CI 怎么对应；test-backend-ops 在没有 GPU 的机器上比的是什么，阈值怎么定；融合路径里的错为什么单算子测试抓不到；性能改动怎么用 compare-llama-bench.py 拿出可信的对比；PR 要满足哪些硬性要求。

代码对齐 llama.cpp 0ec4027。本机 4 核 CPU，没有 GPU。**不启动 llama-server，也不启动任何监听端口的程序**。ctest 一律带上 `-E test-model-resolution`，因为这个测试会绑一个端口。

## 文件

| 文件 | 作用 | 需要什么 |
|---|---|---|
| `exp19.sh` | 九组实验，按顺序是：`ctest`（`ctest -L main`，和 build-cpu 工作流相同）、`behavior`（照 behavior-check.yml 的环境变量跑 `-L behavior`）、`vocabs`（没装 git-lfs 时补下真的词表文件再重跑）、`tbo`（`test-backend-ops test -b CPU` 全量）、`bug`（打上 `patches/fused-tail.patch` 另编一份，跑两个融合测试和两个单算子测试）、`nofuse`（同一份带错构建加 `GGML_CPU_DISABLE_FUSION=1`）、`fix`（干净构建重跑融合测试）、`perf`（融合开、关的性能对比）、`bench`（llamafile sgemm 开、关两份构建的 llama-bench 对比） | 编好的 llama.cpp（`LLAMA_BUILD_TESTS=ON`）；`bench` 还要 `$R/b06/native`、`$R/b06/nollamafile` 两份构建和 `models/q-Q8_0.gguf` |
| `patches/fused-tail.patch` | 故意引入的错：融合后的 RMS_NORM 加 MUL 循环只算到 ne0 向下取 16 的倍数，余下的列不写 | `patch` |
| `parse_tbo.py` | 解析 test-backend-ops 和 ctest 的日志：逐用例结果、出错节点、误差和阈值、总数、墙钟；`fusable()` 判断 CPU 会不会融合，`tail_bug_predicts()` 预测哪些用例会错 | Python 3 |
| `load_sql.py` | 把 `--output sql`（`-o sql`）的输出灌进 SQLite，`--label` 把这批记录的 build_commit 改成固定标签，同一个 commit 的两种构建也能交给 compare-llama-bench.py 比 | Python 自带 sqlite3 |
| `perf_fuse.sh` | `GGML_CPU_DISABLE_FUSION` 取 1 和 0，交替各跑两轮 `test-backend-ops perf -o RMS_NORM_MUL_ROPE` | 编好的 test-backend-ops |
| `bench_ab.sh` | 两份构建交替各跑两轮 `llama-bench -p 64,512 -n 32 -r 3` | 两份构建和一个 GGUF |
| `test_test_contrib.py` | 回归用例：nmse 口径、尾部漏算的误差规律、日志解析、改标签只动新插入的行，以及 `logs/` 里的实验结论 | numpy；没有日志时对应用例跳过 |
| `runlog.py` | 跑一个命令，记墙钟、退出码和峰值 RSS | Python 3 |
| `logs/` | 各步的运行日志、`cmp-fuse.txt`、`cmp-bench.txt`、两份 `.sqlite` 和原始 `.sql` | — |

## 快速试跑

不需要编译：

```bash
python3 test_test_contrib.py          # 13 个用例，零点几秒
python3 parse_tbo.py logs/bug-RMS_NORM_MUL_ADD.log
cat logs/cmp-fuse.txt logs/cmp-bench.txt
```

有编好的 llama.cpp 时：

```bash
bash exp19.sh ctest     # 约 110 秒，第一次会下载 19 MB 的 stories15M
bash exp19.sh tbo       # 约 350 秒，峰值 5.7 GB
bash exp19.sh bug       # 第一次会把源码复制到 $R/bug19 并只编 test-backend-ops
bash exp19.sh perf      # 约 30 秒
```

几点注意：

- `R` 环境变量指向放 `llama.cpp`、`models` 的根目录，缺省是本仓库的上上级。
- test-tokenizers-ggml-vocabs 会 git clone 一个 Hugging Face 仓库。没装 git-lfs 时拿到的是指针文件，测试报 `invalid magic characters: 'vers'`；`exp19.sh vocabs` 用 resolve 地址补下真文件。
- test-fusion 在 CPU 上直接退出，因为 CPU 后端没有导出 `ggml_backend_fusion_*` 调试接口。CPU 的融合只能靠 test-backend-ops 里的 RMS_NORM_MUL_ADD 和 RMS_NORM_MUL_ROPE 覆盖。
- compare-llama-bench.py 遇到长度不同的 build_commit 会按最短的截断，所以标签长度要一致（`fuse0`、`fuse1`）。
- 每一步只跑一个重任务。test-backend-ops 全量和 llama-bench 同时跑，两边的数字都会失真。
- 不跑 llama-server、llama-cli、rpc-server，也不跑 test-model-resolution。
