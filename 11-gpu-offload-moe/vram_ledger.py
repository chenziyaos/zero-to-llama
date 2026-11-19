#!/usr/bin/env python3
"""vram_ledger.py：按 GGUF 张量表算 -ngl、-ncmoe、-cmoe、-ot 把多少权重放到显卡上。

放置规则照 llama.cpp 0ec4027：
- llama-model.cpp：n_gpu_layers 把输出层算作第 n_layer 层，i_gpu_start = max(n_layer + 1 - ngl, 0)，
  编号不小于 i_gpu_start 的层上卡，也就是从后往前卸；-ngl -1（all）等于 n_layer + 1；输入层 token_embd 永远留在 CPU。
- common.h：--cpu-moe 把所有 ffn_(up|down|gate|gate_up)_(ch|)exps 留在 CPU，--n-cpu-moe N 只留前 N 层的。
- llama-kv-cache.cpp：offload_kqv 打开（缺省）时每层的 KV 跟着这一层的设备走。
compute buffer 和 CUDA 上下文没算，和 fit 一样另外留余量。

用法：
  python3 vram_ledger.py MODEL.gguf [--ngl 99] [--ncmoe N | --cmoe] [--ctx 4096] [--kv-bytes 2]
                         [--as-type F16] [--budget-mib 24077 --margin-mib 1024]
  --as-type 把所有 2 维以上的权重按某种类型重算字节数，用来对照只有 F16 数字的资料；只读张量头，可以用截断的 GGUF。
  --budget-mib 给了就倒推：在这块卡上，-ncmoe 至少要多少。
"""
import argparse, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from gguf_lite import GGUF, type_bytes, TYPE_ID

MiB = 2 ** 20
EXPS = re.compile(r'\.ffn_(up|down|gate|gate_up)_(ch|)exps')
BLK = re.compile(r'^blk\.(\d+)\.')


def tensor_table(g, as_type=None):
    """[(name, layer, is_exps, bytes)]，layer 是 -1 表示输入，n_layer 表示输出"""
    n_layer = g.hp('block_count')
    out = []
    for t in g.tensors:
        m = BLK.match(t.name)
        if m:
            il = int(m.group(1))
        elif t.name.startswith('token_embd'):
            il = -1
        else:
            il = n_layer
        nb = t.nbytes
        # 转换脚本通常让路由矩阵 ffn_gate_inp 保持 F32，重算时也不动它
        if as_type and len(t.shape) >= 2 and t.type_name != 'F32':
            nb = type_bytes(TYPE_ID[as_type], t.n_elems)
        out.append((t.name, il, bool(EXPS.search(t.name)), nb))
    if not any(n.startswith('output.weight') for n, *_ in out):
        # 没有单独的 output.weight 时和 token_embd 共享（TENSOR_DUPLICATED），输出层要再放一份
        emb = [x for x in out if x[0].startswith('token_embd')][0]
        out.append(('output.weight(dup)', n_layer, False, emb[3]))
    return out


def kv_bytes_per_layer(g, ctx, kv_bytes):
    n_kv = g.hp('attention.head_count_kv') or g.hp('attention.head_count')
    if isinstance(n_kv, list):
        n_kv = max(n_kv)
    head = g.hp('embedding_length') // g.hp('attention.head_count')
    k = g.hp('attention.key_length', head)
    v = g.hp('attention.value_length', head)
    return ctx * n_kv * (k + v) * kv_bytes


def ledger(g, ngl, ncmoe, ctx, kv_bytes, as_type=None, cpu_exps=(), legacy=False):
    """cpu_exps：另外用 -ot 留在 CPU 的专家层号；legacy：2025 年 b6895 前后的旧规则，
    i_gpu_start = n_layer - ngl，-ngl N 是 N 个重复层，输出层要 N 大于 n_layer 才上卡"""
    n_layer = g.hp('block_count')
    if ngl < 0:
        ngl = n_layer + 1
    start = max(n_layer - ngl if legacy else n_layer + 1 - ngl, 0)
    act = min(ngl, n_layer + 1)
    on_gpu = lambda il: il >= 0 and il >= start and il - start < act
    dev = {'dense': 0, 'exps': 0, 'kv': 0, 'output': 0}
    host = {'dense': 0, 'exps': 0, 'kv': 0, 'embd': 0}
    for name, il, ex, nb in tensor_table(g, as_type):
        if il == -1:
            host['embd'] += nb
        elif il == n_layer:
            (dev if on_gpu(il) else host)['output' if on_gpu(il) else 'dense'] += nb
        elif ex:
            (dev if on_gpu(il) and il >= ncmoe and il not in cpu_exps else host)['exps'] += nb
        else:
            (dev if on_gpu(il) else host)['dense'] += nb
    kvl = kv_bytes_per_layer(g, ctx, kv_bytes)
    for il in range(n_layer):
        (dev if on_gpu(il) else host)['kv'] += kvl
    return start, dev, host


def per_layer(g, as_type=None):
    n_layer = g.hp('block_count')
    dense = [0] * n_layer
    exps = [0] * n_layer
    for name, il, ex, nb in tensor_table(g, as_type):
        if 0 <= il < n_layer:
            (exps if ex else dense)[il] += nb
    return dense, exps


def min_ncmoe(g, budget, margin, ctx, kv_bytes, as_type=None):
    n_layer = g.hp('block_count')
    for n in range(n_layer + 1):
        _, dev, _ = ledger(g, -1, n, ctx, kv_bytes, as_type)
        if sum(dev.values()) + margin * MiB <= budget * MiB:
            return n, dev
    return None, None


def cache_groups(g, ncmoe, cache_mib, align=128):
    """照 llama-moe-cache.cpp 估 --moe-cache-mib 的槽位：
    专家全在主存的层，按专家张量的类型和形状分组，同组共用一个 SLRU；
    预算按各组主存专家字节数分；槽位是满足 sum(pad(每专家字节 × (n + 1))) <= 预算的最大 n，多的一个槽给 MMQ 越界读；
    槽位少于 n_used 的组不缓存。单卡，align 取 CUDA 缓冲的 128 字节对齐。"""
    n_layer = g.hp('block_count')
    n_used = g.hp('expert_used_count')
    by_layer = {}
    for t in g.tensors:
        m = BLK.match(t.name)
        if m and EXPS.search(t.name) and int(m.group(1)) < ncmoe:
            by_layer.setdefault(int(m.group(1)), []).append(t)
    groups = {}
    for il, ts in sorted(by_layer.items()):
        ts.sort(key=lambda t: t.name)
        key = tuple((t.name.split('.', 2)[2], t.type_name, tuple(t.shape)) for t in ts)
        grp = groups.setdefault(key, {'layers': [], 'per': [t.nbytes // t.shape[2] for t in ts],
                                      'n_expert': ts[0].shape[2], 'types': '/'.join(t.type_name for t in ts)})
        grp['layers'].append(il)
    host = sum(sum(grp['per']) * grp['n_expert'] * len(grp['layers']) for grp in groups.values())
    pad = lambda x: (x + align - 1) // align * align
    out = []
    for grp in groups.values():
        hb = sum(grp['per']) * grp['n_expert'] * len(grp['layers'])
        budget = cache_mib * MiB * hb / host
        n, top = 0, len(grp['layers']) * grp['n_expert']
        while n < top and sum(pad(b * (n + 2)) for b in grp['per']) <= budget:
            n += 1
        if n < n_used:
            n = 0
        out.append((grp['layers'], grp['types'], n, top, len(grp['layers']) * n_used))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--ngl', type=int, default=-1)
    ap.add_argument('--ncmoe', type=int, default=0)
    ap.add_argument('--cmoe', action='store_true')
    ap.add_argument('--ctx', type=int, default=4096)
    ap.add_argument('--kv-bytes', type=float, default=2)
    ap.add_argument('--as-type', default=None)
    ap.add_argument('--cpu-exps', default='', help='再用 -ot 留在 CPU 的专家层，比如 14-47')
    ap.add_argument('--legacy', action='store_true', help='用旧的 -ngl 计数规则（输出层不算在 N 里）')
    ap.add_argument('--budget-mib', type=float, default=0)
    ap.add_argument('--margin-mib', type=float, default=1024)
    ap.add_argument('--cache-mib', type=float, default=0, help='--moe-cache-mib，估每组槽位和单 token 用量的倍数')
    a = ap.parse_args()
    g = GGUF(a.model)
    n_layer = g.hp('block_count')
    ncmoe = n_layer if a.cmoe else a.ncmoe
    dense, exps = per_layer(g, a.as_type)
    print(f'{os.path.basename(a.model)}  arch={g.arch}  {n_layer} 层  专家 {g.hp("expert_count")} 选 {g.hp("expert_used_count")}'
          + (f'  按 {a.as_type} 重算' if a.as_type else ''))
    print(f'每层非专家 {sum(dense) / n_layer / MiB:.1f} MiB，专家 {sum(exps) / n_layer / MiB:.1f} MiB；'
          f'专家合计 {sum(exps) / MiB:.0f} MiB，占权重 {sum(exps) / sum(nb for *_, nb in tensor_table(g, a.as_type)) * 100:.1f}%')
    ce = set()
    for part in filter(None, a.cpu_exps.split(',')):
        x, _, y = part.partition('-')
        ce |= set(range(int(x), int(y or x) + 1))
    start, dev, host = ledger(g, a.ngl, ncmoe, a.ctx, a.kv_bytes, a.as_type, ce, a.legacy)
    print(f'-ngl {a.ngl} -ncmoe {ncmoe} -c {a.ctx}：第 {start} 层起上卡')
    print('  显卡  ' + '  '.join(f'{k} {v / MiB:.0f}' for k, v in dev.items()) + f'  合计 {sum(dev.values()) / MiB:.0f} MiB')
    print('  主存  ' + '  '.join(f'{k} {v / MiB:.0f}' for k, v in host.items()) + f'  合计 {sum(host.values()) / MiB:.0f} MiB')
    if a.budget_mib:
        n, d = min_ncmoe(g, a.budget_mib, a.margin_mib, a.ctx, a.kv_bytes, a.as_type)
        if n is None:
            print(f'{a.budget_mib:.0f} MiB 放不下，-cmoe 之后还要减 -ngl 或 -c')
        else:
            print(f'{a.budget_mib:.0f} MiB 留 {a.margin_mib:.0f}：-ngl all -ncmoe {n}，显卡 {sum(d.values()) / MiB:.0f} MiB')
    if a.cache_mib:
        print(f'--moe-cache-mib {a.cache_mib:.0f}（专家在主存的层：前 {ncmoe} 层）')
        for layers, types, n, top, foot in cache_groups(g, ncmoe, a.cache_mib):
            ls = ','.join(map(str, layers[:4])) + (',…' if len(layers) > 4 else '')
            note = '不缓存' if n == 0 else f'{n / foot:.2f} 倍单 token 用量' + ('，低于 2 倍，SLRU 的保护段装不下一个 token' if n < 2 * foot else '')
            print(f'  {len(layers):2d} 层（{ls}）{types}：{n} 槽 / {top} 个专家（{n / top * 100:.1f}%），{note}')


if __name__ == '__main__':
    main()
