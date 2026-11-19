"""从 llama.cpp 的启动和运行日志里抽关键数字，可选和 estimate_memory.py 的预测对账。

    python3 parse_load_log.py LOG [--model MODEL.gguf] [--json]

LOG 是 llama-completion / llama-cli / llama-server 的 stderr。加 -v 的日志行更多，
但这里用到的行在默认级别都会打印（fit 那几行除外，它们是 TRC 级，只有 -v 才有）。

抽取的行和源码位置（基线 0ec4027）：
  fit       common/fit.cpp:402-405        projected to use / will leave ... no changes needed
  meta      src/llama-model-loader.cpp    loaded meta data with N key-value pairs and M tensors
            print_info: file type / file size / arch / n_layer / n_head_kv / n_ctx_train ...
  weights   src/llama-model.cpp:1640      load_tensors: (load_mode = ...)，随后各 buffer 的 size
  context   src/llama-context.cpp         n_ctx / n_batch / n_ubatch / flash_attn
  kv        src/llama-kv-cache.cpp:286    llama_kv_cache: size = ... K (f16): ... V (f16): ...
  compute   src/llama-context.cpp:859     sched_reserve: CPU compute buffer size
  graph     sched_reserve: graph: nodes = N, splits = M
  perf      common_perf_print             load / prompt eval / eval 时间
  breakdown common_memory_breakdown_print 退出前的 | memory breakdown [MiB] | 表

fit 阶段会用 no_alloc 先把模型和上下文"空载"一遍（load_mode = none，所有 buffer 0.00 MiB），
所以日志里 load_tensors、llama_context 会出现两次。这里只取最后一次，也就是真正加载的那次。
"""
import argparse
import json
import os
import re
import sys

PAT = {
    'fit_used': r'projected to use (\d+) MiB of host memory vs\. (\d+) MiB',
    'fit_leave': r'will leave (-?\d+) >= (\d+) MiB of system memory',
    'fit_time': r'fitting params to free memory took ([\d.]+) seconds',
    'n_kv': r'loaded meta data with (\d+) key-value pairs and (\d+) tensors',
    'file_type': r'print_info: file type\s+=\s+(.+?)\s*$',
    'file_size': r'print_info: file size\s+=\s+([\d.]+) MiB \(([\d.]+) BPW\)',
    'arch': r'print_info: arch\s+=\s+(\S+)',
    'n_layer': r'print_info: n_layer\s+=\s+(\d+)',
    'n_head': r'print_info: n_head\s+=\s+(\d+)',
    'n_head_kv': r'print_info: n_head_kv\s+=\s+(\d+)',
    'head_dim': r'print_info: n_embd_head_k\s+=\s+(\d+)',
    'n_ctx_train': r'print_info: n_ctx_train\s+=\s+(\d+)',
    'n_vocab': r'print_info: n_vocab\s+=\s+(\d+)',
    'params_m': r'print_info: model params\s+=\s+([\d.]+) M',
    'load_mode': r'load_tensors: loading model tensors.*\(load_mode = (\S+)\)',
    'n_ctx': r'llama_context: n_ctx\s+=\s+(\d+)',
    'n_batch': r'llama_context: n_batch\s+=\s+(\d+)',
    'n_ubatch': r'llama_context: n_ubatch\s+=\s+(\d+)',
    'flash_attn': r'llama_context: flash_attn\s+=\s+(\S+)',
    'fa_resolved': r'resolve_fused_ops: Flash Attention (enabled|disabled)',
    'kv': r'llama_kv_cache: size =\s+([\d.]+) MiB \(\s*(\d+) cells,\s+(\d+) layers.*K \((\w+)\):\s+([\d.]+) MiB, V \((\w+)\):\s+([\d.]+) MiB',
    'compute': r'sched_reserve:\s+(\S+) compute buffer size =\s+([\d.]+) MiB',
    'graph': r'graph: nodes = (\d+), splits = (\d+)',
    'load_ms': r'common_perf_print:\s+load time =\s+([\d.]+) ms',
    'pp': r'prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens.*?([\d.]+) tokens per second',
    'tg': r'common_perf_print:\s+eval time =\s+([\d.]+) ms /\s+(\d+) runs.*?([\d.]+) tokens per second',
    'reused': r'graphs reused =\s+(\d+)',
    'repack_skip': r"tensor '(\S+)' \((\w+)\) \(and (\d+) others\) cannot be used with preferred buffer type (\w+)",
}
BUF = re.compile(r'load_tensors:\s+(\S+) model buffer size =\s+([\d.]+) MiB')
BRK = re.compile(r'\|\s+-\s+(\S+)\s+\|\s+(?:(\d+)\s+(\d+)\s+)?(\d+) =\s+(\d+) \+\s+(\d+) \+\s+(\d+)\s+(\d+)?\s*\|')


def last(pat, text):
    m = None
    for m in re.finditer(pat, text, re.M):
        pass
    return m


def parse(text):
    r = {}
    for k, p in PAT.items():
        m = last(p, text)
        if not m:
            continue
        g = m.groups()
        r[k] = g[0] if len(g) == 1 else list(g)

    # 权重 buffer：只累加最后一次 load_tensors 之后的行
    start = 0
    for m in re.finditer(PAT['load_mode'], text, re.M):
        start = m.end()
    bufs = {}
    for m in BUF.finditer(text, start):
        bufs[m.group(1)] = round(bufs.get(m.group(1), 0.0) + float(m.group(2)), 2)
    r['weight_buffers'] = bufs

    # breakdown 表：取最后一张
    tables = re.split(r'\| memory breakdown \[MiB\]', text)
    if len(tables) > 1:
        rows = {}
        for line in tables[-1].splitlines()[1:]:
            m = BRK.search(line)
            if not m:
                if rows:
                    break
                continue
            rows[m.group(1)] = dict(self=int(m.group(4)), model=int(m.group(5)),
                                    context=int(m.group(6)), compute=int(m.group(7)))
        r['breakdown'] = rows
    return r


def show(r, est=None):
    MiB = 2 ** 20
    say = print
    say('[模型]')
    if 'arch' in r:
        say(f'  {r["arch"]}，{r.get("n_layer")} 层，头 {r.get("n_head")} / KV 头 {r.get("n_head_kv")}，'
            f'head_dim {r.get("head_dim")}，词表 {r.get("n_vocab")}，训练长度 {r.get("n_ctx_train")}')
    if 'file_size' in r:
        say(f'  {r.get("file_type")}，{r["file_size"][0]} MiB，{r["file_size"][1]} BPW，'
            f'{r.get("params_m")} M 参数')
    if 'n_kv' in r:
        say(f'  GGUF：{r["n_kv"][0]} 个 KV，{r["n_kv"][1]} 个张量')
    if 'fit_used' in r:
        say('[fit]')
        say(f'  预计占用 {r["fit_used"][0]} MiB / 总内存 {r["fit_used"][1]} MiB')
        if 'fit_leave' in r:
            say(f'  剩余 {r["fit_leave"][0]} MiB，要求余量 {r["fit_leave"][1]} MiB')
        if 'fit_time' in r:
            say(f'  fit 耗时 {r["fit_time"]} s')
    say('[加载]')
    say(f'  load_mode {r.get("load_mode")}，权重 buffer ' +
        '，'.join(f'{k} {v:.2f} MiB' for k, v in r.get('weight_buffers', {}).items()))
    if 'repack_skip' in r:
        n, t, k, buft = r['repack_skip'][0], r['repack_skip'][1], int(r['repack_skip'][2]), r['repack_skip'][3]
        say(f'  {n}（{t}）和另外 {k} 个张量不能放进 {buft}')
    say('[上下文]')
    say(f'  n_ctx {r.get("n_ctx")}，n_batch {r.get("n_batch")}，n_ubatch {r.get("n_ubatch")}，'
        f'flash_attn {r.get("flash_attn")} -> {r.get("fa_resolved", "?")}')
    if 'kv' in r:
        kv = r['kv']
        say(f'  KV {kv[0]} MiB = {kv[1]} cells x {kv[2]} 层，K({kv[3]}) {kv[4]} + V({kv[5]}) {kv[6]}')
    if 'compute' in r:
        say(f'  compute buffer（{r["compute"][0]}）{r["compute"][1]} MiB，'
            f'graph nodes {r.get("graph", ["?"])[0]}，splits {r.get("graph", ["?", "?"])[1]}')
    if 'breakdown' in r:
        say('[memory breakdown]')
        for k, v in r['breakdown'].items():
            say(f'  {k:<12} {v["self"]:>6} = {v["model"]} + {v["context"]} + {v["compute"]}')
    if 'load_ms' in r or 'tg' in r:
        say('[性能]')
        if 'load_ms' in r:
            say(f'  load {float(r["load_ms"]) / 1000:.2f} s')
        if 'pp' in r:
            say(f'  prompt {r["pp"][1]} tokens，{r["pp"][2]} t/s')
        if 'tg' in r:
            say(f'  eval {r["tg"][1]} runs，{r["tg"][2]} t/s，graphs reused {r.get("reused")}')
    if est:
        say('[对账] 预测 vs 日志（MiB）')
        host = r.get('breakdown', {}).get('Host')
        rows = [('model', est['model'], host and host['model']),
                ('KV', est['k'] + est['v'], float(r['kv'][0]) if 'kv' in r else None),
                ('compute', est['compute'], float(r['compute'][1]) if 'compute' in r else None),
                ('Host 合计', est['host'], host and host['self']),
                ('CPU_REPACK', est['repack'], r.get('weight_buffers', {}).get('CPU_REPACK'))]
        for name, p, a in rows:
            p /= MiB
            if a is None:
                say(f'  {name:<10} 预测 {p:8.2f}  日志 —')
            else:
                say(f'  {name:<10} 预测 {p:8.2f}  日志 {float(a):8.2f}  差 {float(a) - p:+.2f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('--model', help='同一个 GGUF，给了就和 estimate_memory 的预测对账')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args()
    text = open(a.log, encoding='utf-8', errors='replace').read()
    r = parse(text)
    if a.json:
        print(json.dumps(r, ensure_ascii=False, indent=1))
        return
    est = None
    if a.model:
        from estimate_memory import estimate
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
        from gguf_lite import GGUF
        kv = r.get('kv') or [0, 4096, 0, 'f16', 0, 'f16', 0]
        est = estimate(GGUF(a.model), int(r.get('n_ctx', 4096)), int(r.get('n_ubatch', 512)),
                       kv[3], kv[5], 'on' if r.get('fa_resolved') == 'enabled' else 'off', 'avx512')
    show(r, est)


if __name__ == '__main__':
    main()
