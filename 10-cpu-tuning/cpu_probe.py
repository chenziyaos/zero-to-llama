#!/usr/bin/env python3
"""cpu_probe.py：读 /proc 和 /sys，给 llama.cpp 的 CPU 参数一个起点。

报告物理核和超线程分组、NUMA 节点、THP、numa_balancing、mlock 上限、实时调度配额、
指令集，以及按 llama.cpp 0ec4027 的源码规则，哪些量化类型在这台机器上走 repack、哪些走 tiled。
最后给出 -t 的起点：先用物理核数（llama.cpp 的缺省），再试全部逻辑核，两者都要实测；绑核掩码只取每个核的第一个超线程。

用法：python3 cpu_probe.py [--root /] [--model-gib 0.45 --bw-gbps 35]
--root 用来对着一份拷出来的 /proc、/sys 目录树测试。
"""
import argparse, os, re, resource

# repack.cpp 第 4979 行附近：x86 上哪些类型有重排格式，要什么指令集
REPACK_X86 = {'Q4_0': 'avx2', 'Q4_K': 'avx2', 'IQ4_NL': 'avx2', 'MXFP4': 'avx2', 'Q2_K': 'avx512f'}
# tiled.cpp ggml_tiled_supported：K 系列和大部分 IQ 类型；被 repack 过的权重不走 tiled
TILED = ['Q6_K', 'Q5_K', 'Q4_K', 'Q3_K', 'Q2_K', 'IQ4_XS', 'IQ2_XXS', 'IQ2_XS', 'IQ2_S', 'IQ3_XXS', 'IQ3_S', 'IQ1_S', 'IQ1_M']


def rd(root, p, default=''):
    try:
        with open(os.path.join(root, p.lstrip('/'))) as f:
            return f.read().strip()
    except OSError:
        return default


def parse_list(s):
    """'0-1,4' -> [0, 1, 4]"""
    out = []
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-')
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def topology(root):
    base = 'sys/devices/system/cpu'
    online = parse_list(rd(root, f'{base}/online', '0'))
    groups = {}
    for c in online:
        sib = rd(root, f'{base}/cpu{c}/topology/thread_siblings_list', str(c))
        groups.setdefault(tuple(parse_list(sib)), None)
    cores = sorted(groups)
    nodes = []
    nd = os.path.join(root, 'sys/devices/system/node')
    if os.path.isdir(nd):
        for n in sorted(os.listdir(nd)):
            if re.fullmatch(r'node\d+', n):
                nodes.append((n, parse_list(rd(root, f'sys/devices/system/node/{n}/cpulist', ''))))
    return online, cores, nodes


def bracket(s):
    m = re.search(r'\[(\w+)\]', s)
    return m.group(1) if m else s


def flags(root):
    for line in rd(root, '/proc/cpuinfo').splitlines():
        if line.startswith('flags'):
            return set(line.split(':', 1)[1].split())
    return set()


def mask_of(cpus):
    return hex(sum(1 << c for c in cpus))


def advise(online, cores, nodes):
    first = [g[0] for g in cores]
    out = {'threads': len(cores), 'mask': mask_of(first), 'smt': any(len(g) > 1 for g in cores)}
    if len(nodes) > 1:
        out['numa'] = '--numa distribute，或用 numactl 把进程和内存放在同一个节点上再加 --numa numactl'
    else:
        out['numa'] = '只有一个节点，--numa 是空操作'
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/')
    ap.add_argument('--model-gib', type=float, default=0)
    ap.add_argument('--bw-gbps', type=float, default=0)
    a = ap.parse_args()
    online, cores, nodes = topology(a.root)
    f = flags(a.root)
    print(f'逻辑 CPU {len(online)}，物理核 {len(cores)}，分组 {[list(g) for g in cores]}')
    print(f'NUMA 节点 {len(nodes) or 1}：' + '，'.join(f'{n} {c}' for n, c in nodes))
    print(f"THP enabled={bracket(rd(a.root, '/sys/kernel/mm/transparent_hugepage/enabled', '?'))}"
          f"（mmap 的权重是文件页，THP 管不到；-lm none 时权重在匿名内存里才可能用上）")
    nb = rd(a.root, '/proc/sys/kernel/numa_balancing', '')
    if nb:
        print(f'numa_balancing={nb}' + ('（非 0 时 ggml 会打印警告）' if nb != '0' else ''))
    if a.root == '/':
        soft, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        print('RLIMIT_MEMLOCK=' + ('unlimited' if soft == resource.RLIM_INFINITY else f'{soft // 1024} KiB')
              + '，mlock 失败时 llama.cpp 只打印警告，接着跑')
    rt = rd(a.root, '/proc/sys/kernel/sched_rt_runtime_us', '')
    if rt:
        print(f'sched_rt_runtime_us={rt}，--prio 1 以上用 SCHED_FIFO，要 root 或 CAP_SYS_NICE')
    isa = [x for x in ('avx2', 'avx512f', 'avx512_vnni', 'avx512vl', 'f16c', 'fma') if x in f]
    print('指令集：' + ' '.join(isa))
    rp = [t for t, need in REPACK_X86.items() if need in f]
    print('走 repack 的类型：' + ('、'.join(rp) if rp else '无'))
    print('repack 开着时走 tiled 的类型：' + '、'.join(t for t in TILED if t not in rp))
    print('--repack 0 时走 tiled 的类型：' + '、'.join(TILED) + '；GGML_CPU_TILED_MM=0 再退回 vec_dot')
    adv = advise(online, cores, nodes)
    print(f"起点：-t {adv['threads']}（物理核数，llama.cpp 的缺省）" + (f"；再试 -t {len(online)}（全部逻辑核），两个都实测；要绑核用 -C {adv['mask']} --cpu-strict 1，线程数不能多于掩码里的 CPU" if adv['smt'] else ''))
    print(adv['numa'])
    if a.model_gib and a.bw_gbps:
        ceil = a.bw_gbps / (a.model_gib * 1.073741824)
        print(f'解码上限约 {ceil:.0f} t/s（带宽 {a.bw_gbps} GB/s 除以每 token 读的权重 {a.model_gib} GiB）')


if __name__ == '__main__':
    main()
