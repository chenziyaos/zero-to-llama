"""按症状查参数：读 llama-completion 的日志，告诉你实际生效的参数、命中了哪条已知误配、速度离上限还差多少。

    python3 diagnose.py log RUN.log [--model M.gguf]      # 事后：解析一次运行的 stderr
    python3 diagnose.py plan M.gguf [-c 0] [-np 1] [-ctk f16] [-ctv f16] [-t 2]   # 事前：不跑模型，先算默认值会给你什么

日志口径对齐 llama.cpp 0ec4027。默认日志级别下能拿到 system_info、sampler、generate 和 common_perf_print；
加 -lv 4 以后多出 common_fit、print_info、llama_context、llama_kv_cache、sched_reserve 这几段，诊断会更全。

规则表 RULES 里每一条是（正则, 级别, 一句话结论, 怎么改, 源码位置）。级别分三档：
  error  进程退出或拒绝启动
  warn   打了警告但照常跑，参数被忽略或被改掉
  silent 没有任何警告，但从解析出来的参数能看出问题（比如线程数超过核数）
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
for sub in ('common', '01-first-run', '02-ledger'):
    sys.path.insert(0, os.path.join(ROOT, sub))

MiB = 2 ** 20

# (正则, 级别, 结论, 怎么改, 源码位置)
RULES = [
    (r'failed to fit params to free device memory', 'error',
     '--fit 放弃了：按你给的 -c 怎么压都放不下', '把 -c 改小，或者 -c 0 交给 --fit 按内存挑', 'common/fit.cpp:976'),
    (r'n_ctx_seq \((\d+)\) > n_ctx_train \((\d+)\)', 'warn',
     '每条序列的上下文超过训练长度', '-c 不要超过 n_ctx_train 乘 -np，要更长先确认模型的 RoPE 缩放', 'src/llama-context.cpp:425'),
    (r'failed to allocate buffer of size (\d+)', 'error',
     '分配内存失败', '看紧跟着的是 kv cache 还是 compute，前者改 -c 或 -ctk/-ctv，后者改 -ub', 'ggml/src/ggml-backend.cpp:2831'),
    (r'quantized V cache requires flash_attn', 'error',
     '量化 V cache 必须开 Flash Attention', '去掉 -fa off，或者 -ctv 改回 f16', 'src/llama-context.cpp:4137'),
    (r'enabling flash_attn since it is required for quantized V cache', 'info',
     '因为 V cache 量化，FA 被自动打开', '无需处理', 'src/llama-context.cpp:4133'),
    (r'no usable GPU found, --gpu-layers option will be ignored', 'warn',
     '没有可用 GPU，-ngl 被忽略', '纯 CPU 构建不用写 -ngl；有卡时检查构建选项和驱动', 'common/arg.cpp:2761'),
    (r'compiled without support for GPU offload\. Setting the split mode', 'warn',
     '没有 GPU 后端，-sm 不起作用', '删掉 -sm', 'common/arg.cpp:2787'),
    (r'Not enough set bits in CPU mask \((\d+)\) to satisfy requested thread count: (\d+)', 'warn',
     'CPU 掩码里的核比线程少，多出来的线程会挤在同几个核上', '-t 不超过掩码里的核数，或者去掉 --cpu-strict', 'common/common.cpp:329'),
    (r'unable to create context', 'error',
     '上下文没建起来', '往上找第一条 E 级日志，多半是内存或参数冲突', 'tools/completion/completion.cpp:150'),
    (r'failed to load model', 'error',
     '模型没加载起来', '检查路径、文件是否完整、GGUF 版本', 'src/llama.cpp'),
]

PATS = {
    'n_threads': r'system_info: n_threads = (\d+) \(n_threads_batch = (\d+)\) / (\d+)',
    'generate': r'generate: n_ctx = (\d+), n_batch = (\d+), n_predict = (-?\d+)',
    'sampler': r'top_k = (\d+), top_p = ([\d.]+), min_p = ([\d.]+).*?temp = ([\d.]+)',
    'load_ms': r'common_perf_print:\s+load time =\s+([\d.]+) ms',
    'pp': r'prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens.*?([\d.]+) tokens per second',
    'tg': r'common_perf_print:\s+eval time =\s+([\d.]+) ms /\s+(\d+) runs.*?([\d.]+) tokens per second',
    'file_type': r'print_info: file type\s+= (.+)',
    'file_size': r'print_info: file size\s+= ([\d.]+) MiB \(([\d.]+) BPW\)',
    'n_ctx_train': r'print_info: n_ctx_train\s+= (\d+)',
    'ctx': r'llama_context: n_ctx\s+= (\d+)',
    'n_ctx_seq': r'llama_context: n_ctx_seq\s+= (\d+)',
    'n_ubatch': r'llama_context: n_ubatch\s+= (\d+)',
    'n_seq_max': r'llama_context: n_seq_max\s+= (\d+)',
    'flash_attn': r'llama_context: flash_attn\s+= (\w+)',
    'kv_unified': r'llama_context: kv_unified\s+= (\w+)',
    'kv': r'llama_kv_cache: size =\s+([\d.]+) MiB \(\s*(\d+) cells.*?K \((\w+)\):\s+([\d.]+) MiB, V \((\w+)\):\s+([\d.]+) MiB',
    'repack': r'CPU_REPACK model buffer size =\s+([\d.]+) MiB',
    'compute': r'CPU compute buffer size =\s+([\d.]+) MiB',
    'load_mode': r'load_mode = (\w+)',
    'fit_host': r'projected to use (\d+) MiB of host memory vs\. (\d+) MiB of total host memory',
}


def parse(text):
    """把一次运行的 stderr 解析成扁平的字典，没出现的字段不给。"""
    out = {}
    for k, p in PATS.items():
        m = re.search(p, text)
        if not m:
            continue
        g = m.groups()
        if k == 'n_threads':
            out.update(threads=int(g[0]), threads_batch=int(g[1]), hw_threads=int(g[2]))
        elif k == 'generate':
            out.update(n_ctx=int(g[0]), n_batch=int(g[1]), n_predict=int(g[2]))
        elif k == 'sampler':
            out.update(top_k=int(g[0]), top_p=float(g[1]), min_p=float(g[2]), temp=float(g[3]))
        elif k == 'pp':
            out.update(pp_ms=float(g[0]), pp_tokens=int(g[1]), pp_ts=float(g[2]))
        elif k == 'tg':
            out.update(tg_ms=float(g[0]), tg_runs=int(g[1]), tg_ts=float(g[2]))
        elif k == 'file_size':
            out.update(file_mib=float(g[0]), bpw=float(g[1]))
        elif k == 'kv':
            out.update(kv_mib=float(g[0]), kv_cells=int(g[1]), ctk=g[2], k_mib=float(g[3]), ctv=g[4], v_mib=float(g[5]))
        elif k == 'fit_host':
            out.update(fit_host_mib=int(g[0]), host_total_mib=int(g[1]))
        elif k in ('flash_attn', 'kv_unified', 'load_mode', 'file_type'):
            out[k] = g[0].strip()
        elif k in ('repack', 'compute', 'load_ms'):
            out[k + ('_mib' if k != 'load_ms' else '')] = float(g[0])
        else:
            out[k] = int(g[0])
    return out


def match_rules(text):
    """命中的规则，按出现顺序，同一条只报一次。"""
    hits = []
    for pat, level, what, fix, src in RULES:
        m = re.search(pat, text)
        if m:
            hits.append({'level': level, 'what': what, 'fix': fix, 'src': src, 'pos': m.start(), 'groups': m.groups()})
    return sorted(hits, key=lambda h: h['pos'])


def silent_checks(p):
    """没有警告、但参数本身有问题的情况。p 是 parse() 的结果。"""
    out = []
    if p.get('threads') and p.get('hw_threads') and p['threads'] > p['hw_threads']:
        out.append(('silent', f'-t {p["threads"]} 超过机器的 {p["hw_threads"]} 个逻辑核，线程会互相抢核',
                    '-t 不超过物理核数，decode 一般 2 到 3 个就饱和'))
    if p.get('n_ctx_train') and p.get('ctx') == p.get('n_ctx_train') and p.get('n_seq_max', 1) == 1:
        out.append(('silent', f'上下文开到了训练长度 {p["ctx"]}，KV {p.get("kv_mib", 0):.0f} MiB 一次分配到位',
                    '按业务的最长提示加最长输出写 -c'))
    if p.get('n_batch') and p.get('n_ubatch') and p['n_ubatch'] < 64:
        out.append(('silent', f'n_ubatch 只有 {p["n_ubatch"]}，prefill 每步太小', '-ub 回到 512，或至少 128'))
    if p.get('n_batch') and p.get('n_ubatch') and p['n_batch'] < 512 and p['n_ubatch'] == p['n_batch']:
        out.append(('silent', f'n_batch {p["n_batch"]} 比 n_ubatch 的缺省 512 小，-ub 被悄悄截到 {p["n_ubatch"]}',
                    '-b 不小于 -ub；只想调物理批大小就只写 -ub'))
    if p.get('n_seq_max', 1) > 1 and p.get('n_ctx_seq') and p.get('n_ctx'):
        out.append(('silent', f'-np {p["n_seq_max"]}：-c {p["n_ctx"]} 是总量，每条序列只有 {p["n_ctx_seq"]}',
                    '-c 写成每条序列要的长度乘 -np'))
    if p.get('kv_unified') == 'true' and p.get('n_seq_max', 1) > 1:
        out.append(('silent', '多序列共用一个 KV 池，并发 decode 会变慢', '没有共享前缀时去掉 -kvu'))
    return out


def ceiling(model, threads, depth=0, ctk='f16', ctv='f16'):
    """带宽上限和校准后的预测，借 02-ledger 的账本。返回 (上限 t/s, 预测 t/s, 带宽 GB/s)。"""
    from gguf_lite import GGUF
    import estimate_speed as es
    led = es.ledger(GGUF(model), depth, ctk, ctv)
    bw, peak = es.machine(min(max(threads, 1), 4))
    b = es.bounds(led, bw, peak)
    pr = es.predict(led, bw, peak)
    return b['tg'], pr['tg'], bw


def speed_verdict(tg, pred):
    """实测 tg 和预测比：低于 0.75 倍说明线程、邻居或者内存带宽出了问题。"""
    r = tg / pred
    if r < 0.5:
        return r, 'far', '远低于预测：先看有没有别的进程抢核（top 里的 %st 和邻居负载），再看 -t 是否超过物理核'
    if r < 0.75:
        return r, 'low', '低于预测：可能是线程数不对、换页或者 depth 很深'
    return r, 'ok', '在预测范围内，想再快只能换更小的量化或者更少的字节'


def plan(model, ctx=0, np_=1, ctk='f16', ctv='f16', ub=512, fa='on', threads=2, host_mib=None):
    """事前账：不跑模型，算这组参数会占多少内存、decode 大概多快、--fit 会不会动它。"""
    from gguf_lite import GGUF
    import estimate_memory as em
    g = GGUF(model)
    e = em.estimate(g, ctx, ub, ctk, ctv, fa, 'avx512')
    total = (e['host'] + e['repack']) / MiB
    notes = []
    if ctx == 0:
        notes.append(f'没写 -c：取训练长度 {e["n_ctx_train"]}，KV {(e["k"] + e["v"]) / MiB:.0f} MiB；'
                     '--fit 只在放不下时才往下压，内存够就照单全收')
    if np_ > 1:
        notes.append(f'-np {np_}：-c 是总量，每条序列只分到 {e["n_ctx"] // np_}')
    if host_mib and total + 1024 > host_mib:
        notes.append(f'合计 {total:.0f} MiB 加 --fit 余量 1024 超过本机 {host_mib} MiB，--fit 会把上下文压小')
    out = {'n_ctx': e['n_ctx'], 'kv_mib': (e['k'] + e['v']) / MiB, 'model_mib': e['model'] / MiB,
           'repack_mib': e['repack'] / MiB, 'compute_mib': e['compute'] / MiB, 'total_mib': total, 'notes': notes}
    try:
        up, pred, bw = ceiling(model, threads, 0, ctk, ctv)
        out.update(tg_ceiling=up, tg_pred=pred, bw=bw)
    except (OSError, KeyError, StopIteration):
        pass
    return out


def host_mib():
    try:
        for line in open('/proc/meminfo'):
            if line.startswith('MemTotal:'):
                return int(line.split()[1]) // 1024
    except OSError:
        return None


def report_log(path, model=None):
    with open(path, errors='replace') as f:
        text = f.read()
    p = parse(text)
    hits = match_rules(text)
    rep = {'params': p, 'rules': [{k: h[k] for k in ('level', 'what', 'fix', 'src')} for h in hits],
           'silent': [{'level': a, 'what': b, 'fix': c} for a, b, c in silent_checks(p)]}
    if model and p.get('tg_ts'):
        up, pred, bw = ceiling(model, p.get('threads', 2))
        r, tag, msg = speed_verdict(p['tg_ts'], pred)
        rep['speed'] = {'tg': p['tg_ts'], 'pred': round(pred, 1), 'ceiling': round(up, 1), 'bw': bw,
                        'ratio': round(r, 2), 'verdict': tag, 'advice': msg}
    return rep


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    a1 = sp.add_parser('log')
    a1.add_argument('path')
    a1.add_argument('--model')
    a1.add_argument('--json', action='store_true')
    a2 = sp.add_parser('plan')
    a2.add_argument('model')
    a2.add_argument('-c', type=int, default=0)
    a2.add_argument('-np', type=int, default=1)
    a2.add_argument('-ctk', default='f16')
    a2.add_argument('-ctv', default='f16')
    a2.add_argument('-ub', type=int, default=512)
    a2.add_argument('-fa', default='on')
    a2.add_argument('-t', type=int, default=2)
    a = ap.parse_args()
    if a.cmd == 'log':
        rep = report_log(a.path, a.model)
        if a.json:
            print(json.dumps(rep, ensure_ascii=False, indent=1))
            return
        p = rep['params']
        print('生效参数：' + '，'.join(f'{k}={v}' for k, v in p.items()))
        for h in rep['rules'] + rep['silent']:
            src = f'（{h["src"]}）' if 'src' in h else ''
            print(f'[{h["level"]}] {h["what"]}{src}\n        改法：{h["fix"]}')
        if 'speed' in rep:
            s = rep['speed']
            print(f'decode {s["tg"]:.1f} t/s，预测 {s["pred"]}，带宽上限 {s["ceiling"]}（{s["bw"]} GB/s），'
                  f'比值 {s["ratio"]}：{s["advice"]}')
    else:
        r = plan(a.model, a.c, a.np, a.ctk, a.ctv, a.ub, a.fa, a.t, host_mib())
        print(f'n_ctx {r["n_ctx"]}，模型 {r["model_mib"]:.0f} + repack {r["repack_mib"]:.0f} + KV {r["kv_mib"]:.0f} '
              f'+ 计算 {r["compute_mib"]:.0f} = {r["total_mib"]:.0f} MiB')
        if 'tg_pred' in r:
            print(f'decode 预测 {r["tg_pred"]:.1f} t/s，带宽上限 {r["tg_ceiling"]:.1f} t/s（{r["bw"]} GB/s）')
        for n in r['notes']:
            print('注意：' + n)


if __name__ == '__main__':
    main()
