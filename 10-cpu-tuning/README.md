# 10-cpu-tuning：CPU 调优

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zyaochen/zero-to-llama/blob/main/colab/10-cpu-tuning.ipynb)　CPU 练习和单元测试可以直接在 Colab 里跑。

LCPP-10 的配套代码。要回答几个问题：-t 该设多少，超线程算不算核；-C 和 --cpu-strict 什么时候有用、什么时候坑人；OpenMP 和自带线程池差在哪，--poll、--prio、KMP_BLOCKTIME 管不管用；-tb 和 ubatch 怎么配；哪些量化类型走 repack、哪些走 tiled、哪些退回 vec_dot；mmap、mlock、direct io 冷启动和稳态各差多少；THP 和 NUMA 在这台机器上能不能帮忙；怎么判断解码已经贴着内存带宽。

## 文件

| 文件 | 作用 | 需要什么 |
|---|---|---|
| `cpu_probe.py` | 读 /proc 和 /sys：物理核与超线程分组、NUMA 节点、THP、numa_balancing、mlock 上限、实时调度配额、指令集；按 0ec4027 的源码规则列出走 repack 和走 tiled 的类型；给出 `-t`、`-C` 的起点和解码上限 | 只要 Python 标准库 |
| `exp10.sh` | 十三组实验：`threads`、`sticky`、`poll`、`prio`、`ub`、`repack`、`tiled`、`tb`、`omp`、`noisy`、`ceiling`、`load`、`thp` | 编好的 llama.cpp（OpenMP 和 `GGML_OPENMP=OFF` 两份）、几种量化的 GGUF、02 章的 `stream` |
| `test_cpu.py` | 11 个用例：拓扑解析和掩码在假的 /sys 目录树上测，实测结论用 logs 核对 | 只要仓库 |
| `logs/` | Qwen3-0.6B 在 4 vCPU EPYC 上的实测：jsonl 是 llama-bench 输出加上实验标签，txt 是 llama-completion 和 rss.py 的输出 | — |

## 快速试跑

不需要模型文件：

```bash
python3 test_cpu.py
python3 cpu_probe.py --model-gib 0.45 --bw-gbps 35
```

有模型文件时：

```bash
bash exp10.sh threads   # 约六分钟，每个组合单独一个进程
bash exp10.sh repack
bash exp10.sh tiled
bash exp10.sh ceiling
bash exp10.sh load      # 要 root，写 /proc/sys/vm/drop_caches
bash exp10.sh thp       # 要 root，临时改 THP 模式，跑完恢复
```

几点注意：

- 比较绑核效果时每个组合要单独起进程。llama-bench 在同一个进程里跑完所有组合，上一个测试设的亲和性会留在主线程和 OpenMP 的常驻线程上，`sticky` 组专门复现这个问题。
- 线程数多于掩码里的 CPU 时，`--cpu-strict 1` 会把两个线程绑到同一个 CPU 上，barrier 空转，速度掉到个位数，这组只跑一次短 prefill。
- `--prio 2` 以上用 SCHED_FIFO，要 root 或 CAP_SYS_NICE。
- 只用 `llama-bench`、`llama-completion` 和 02 章的 `stream`，不起 server。本机没有 GPU，所有数字都是 CPU 单机实测。
