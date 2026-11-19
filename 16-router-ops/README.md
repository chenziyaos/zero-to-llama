# 16-router-ops：router 多模型、预设、内存余量和监控

LCPP-16 的配套代码。要回答几个问题：router 模式下几个模型同时常驻，谁先被卸载、排队的请求要等多久；`--models-max` 和 `--models-memory-margin` 各管什么，纯 CPU 机器上余量有没有用；预设 INI、`[*]` 全局段、router 命令行三者谁盖谁；不钉 `-c` 时 `--fit` 会把内存吃到多少；mmproj 为什么不进内存账；`/metrics` 上的速率怎么读才不被骗。

本机是 4 核 CPU，没有 GPU，数字只代表这台机器上的相对关系。**不启动 llama-server（包括 router 模式），也不启动任何监听端口的程序**。router 的行为用三样东西逼近：`llama-fit-params`（和 router 的测量子进程走同一套 `--fit` 代码，算每个模型的内存三项）、`llama-completion` 和 `llama-mtmd-cli`（加载一次就退出，量加载耗时和进程峰值常驻内存），以及按 llama.cpp 0ec4027 源码复刻的三个 Python 程序。GPU 场景里的设备是假设的，只用来走通余量逻辑。

## 文件

| 文件 | 作用 | 需要什么 |
|---|---|---|
| `exp16.sh` | 九组实验：`fit`（三个模型在 np 1/4、c 4096/16384/40960 下的内存三项）、`auto`（不写 `-c` 时 `--fit` 按 `-fitt` 挑的上下文）、`sysconf`（系统配置文件、环境变量、命令行的优先级）、`load`（冷、热加载耗时）、`rss`（进程峰值常驻内存，含 mmproj）、`metrics`、`preset`、`sim`、`unit` | 编好的 llama.cpp、Qwen3-0.6B 的 Q4_0/Q8_0/bf16、SmolVLM2-256M 和它的 mmproj |
| `preset_tool.py` | 复刻 `common/preset.cpp` 和 `server-models.cpp` 的预设合并：INI 解析、标签规范化、取反布尔键、`--models-dir` 扫描、cache/models-dir/preset 的覆盖顺序、router 命令行盖到每个模型、保留键删除，渲染成子进程命令行 | 只要 Python 3 |
| `router_sim.py` | 复刻 `server-models-sched.cpp` 的排队、LRU 卸载和设备内存账，七个场景：models_max × 访问模式、忙的模型不卸、纯 CPU 余量无效、GPU 余量、什么都没加载时照样加载、mmproj 不进账、不钉 `-c` 的抖动 | 只要 Python 3；有 `logs/` 时读实测的加载耗时和内存 |
| `metrics_parse.py` | 解析 `/metrics` 的 Prometheus 文本，按源码规则重放请求，看 gauge 的清零、首 token 不计步数和长请求在结束前不可见 | 只要 Python 3；有 `14-server/logs` 时读实测速度 |
| `rss.py` | 跑一条命令，报子进程的 ru_maxrss 和 load time | 只要 Python 3 |
| `test_router.py` | 20 个用例覆盖上面三个复刻的规则 | 只要 Python 3 |
| `presets.ini` | 示例预设：`version`、`[*]`、带量化标签的段名、取反键、`load-on-startup`、写坏的 `stop-timeout` | — |
| `logs/` | 各组输出 | — |

## 快速试跑

不需要模型文件：

```bash
python3 preset_tool.py presets.ini --router-args "--temp 0 --api-key secret -c 4096"
python3 router_sim.py          # 用缺省的加载耗时和内存
python3 metrics_parse.py
python3 test_router.py
```

有模型和编好的 llama.cpp 时：

```bash
bash exp16.sh fit       # 内存三项
bash exp16.sh load      # 冷、热加载各量一次
bash exp16.sh rss       # 进程峰值常驻内存
bash exp16.sh sim       # 用 logs/ 里的实测重跑模拟
```

几点注意：

- `load` 段第一次跑可能是冷的，要看冷加载就先清掉页缓存，或者换一个刚下载的模型。
- `llama-fit-params` 加 `-mm` 会失败，mmproj 的大小只能从 `llama-mtmd-cli -v` 的日志和 `rss` 段里拿。
- `router_sim.py` 复刻的是调度规则，不跑模型，加载耗时和内存是输入，排队等待是输出。
- `metrics_parse.py` 不连任何服务，它渲染的文本和服务端同格式，可以拿真实抓到的文本替换 `logs/metrics_sample.txt` 再用 `parse`、`derive` 读。
- 不跑 llama-server、llama-cli、rpc-server、`scripts/server-bench.py`。
