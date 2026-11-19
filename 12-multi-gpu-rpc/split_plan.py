#!/usr/bin/env python3
"""split_plan.py：照 llama.cpp 的 -sm layer 规则，算每一层落在哪张卡上、每张卡分到多少权重和 KV。

规则照 llama.cpp 0ec4027 src/llama-model.cpp 1657-1712 行：
- -ts 全是 0（缺省）时按每张卡的空闲显存加权；设备报 free=0、total=0 时改用主存大小。
- 权重先累加再归一化成切分点 splits[i]，第 il 层落在 upper_bound(splits, (il - i_gpu_start) / act) 那张卡。
- i_gpu_start = max(n_layer + 1 - ngl, 0)，act = min(ngl, n_layer + 1)，输出层按第 n_layer 层参与切分，输入层永远留在 CPU。
- 切分点是 float，这里用 struct 把累加和比值都截成 float32，边界和 C++ 一致。
设备顺序照 src/llama.cpp：RPC 设备排在本机 GPU 前面，--rpc 给几个就把前几张当远端。

用法：
  python3 split_plan.py MODEL.gguf --free 24000,24000 [--ts 3,1] [--ngl -1] [--ctx 4096] [--rpc 1] [--layers]
"""
import argparse, os, struct, sys
from bisect import bisect_right
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
sys.path.insert(0, os.path.join(HERE, '..', '11-gpu-offload-moe'))
from gguf_lite import GGUF
from vram_ledger import tensor_table, kv_bytes_per_layer

MiB = 2 ** 20
f32 = lambda x: struct.unpack('<f', struct.pack('<f', x))[0]


def split_points(weights):
    acc, out = f32(0.0), []
    for w in weights:
        acc = f32(acc + f32(w))
        out.append(acc)
    n = len(weights)
    return [f32(x / acc) if acc != 0 else f32((i + 1) / n) for i, x in enumerate(out)]


def assign(n_layer, ngl, splits):
    """返回长度 n_layer + 1 的设备号列表，-1 表示 CPU；最后一项是输出层"""
    if ngl < 0:
        ngl = n_layer + 1
    start = max(n_layer + 1 - ngl, 0)
    act = min(ngl, n_layer + 1) if splits else 0
    out = []
    for il in range(n_layer + 1):
        if il < start or il - start >= act:
            out.append(-1)
        else:
            out.append(bisect_right(splits, f32(f32(il - start) / act)))
    return out


def plan(g, weights, ngl=-1, ctx=4096, kv_bytes=2):
    n_layer = g.hp('block_count')
    splits = split_points(weights)
    dev_of = assign(n_layer, ngl, splits)
    nd = len(weights)
    w = [0] * (nd + 1)      # 最后一格是 CPU
    kv = [0] * (nd + 1)
    layers = [[] for _ in range(nd + 1)]
    for name, il, ex, nb in tensor_table(g):
        d = -1 if il == -1 else dev_of[il]
        w[d] += nb
    kvl = kv_bytes_per_layer(g, ctx, kv_bytes)
    for il in range(n_layer):
        kv[dev_of[il]] += kvl
        layers[dev_of[il]].append(il)
    if dev_of[n_layer] >= 0:
        layers[dev_of[n_layer]].append('out')
    return splits, dev_of, w, kv, layers


def fmt_layers(ls):
    if not ls:
        return '-'
    nums = [x for x in ls if x != 'out']
    s = f'{nums[0]}-{nums[-1]}' if nums else ''
    if 'out' in ls:
        s += (' + ' if s else '') + '输出层'
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--free', default='24000,24000', help='每张卡的空闲显存 MiB，逗号分隔，顺序就是设备顺序')
    ap.add_argument('--ts', default='', help='-ts 比例；不给或全 0 时按 --free 加权')
    ap.add_argument('--ngl', type=int, default=-1)
    ap.add_argument('--ctx', type=int, default=4096)
    ap.add_argument('--kv-bytes', type=float, default=2)
    ap.add_argument('--rpc', type=int, default=0, help='前几张是 RPC 远端设备，只影响名字')
    ap.add_argument('--layers', action='store_true', help='逐层打印落点')
    a = ap.parse_args()
    g = GGUF(a.model)
    n_layer = g.hp('block_count')
    free = [float(x) for x in a.free.split(',')]
    ts = [float(x) for x in a.ts.split(',')] if a.ts else []
    if ts and len(ts) != len(free):
        sys.exit('--ts 和 --free 的个数要一样')
    weights = ts if ts and any(ts) else free
    names = [f'RPC{i}' if i < a.rpc else f'GPU{i - a.rpc}' for i in range(len(free))]
    splits, dev_of, w, kv, layers = plan(g, weights, a.ngl, a.ctx, a.kv_bytes)
    print(f'{os.path.basename(a.model)}  {n_layer} 层  -ngl {a.ngl}  -c {a.ctx}  '
          f'按 {"-ts " + a.ts if ts and any(ts) else "空闲显存"} 切')
    print('切分点 ' + ' '.join(f'{x:.4f}' for x in splits))
    for i, nm in enumerate(names):
        tot = w[i] + kv[i]
        print(f'  {nm:5s} 层 {fmt_layers(layers[i]):14s} {len([x for x in layers[i] if x != "out"]):3d} 层  '
              f'权重 {w[i] / MiB:8.1f}  KV {kv[i] / MiB:7.1f}  合计 {tot / MiB:8.1f} MiB  '
              f'占空闲 {tot / MiB / free[i] * 100:5.1f}%')
    print(f'  CPU   输入层{"和 " + fmt_layers(layers[-1]) if layers[-1] else ""}  权重 {w[-1] / MiB:8.1f}  KV {kv[-1] / MiB:7.1f} MiB')
    if a.layers:
        print('逐层：' + ' '.join(f'{il if il < n_layer else "out"}:{names[d] if d >= 0 else "CPU"}'
                                for il, d in enumerate(dev_of)))


if __name__ == '__main__':
    main()
