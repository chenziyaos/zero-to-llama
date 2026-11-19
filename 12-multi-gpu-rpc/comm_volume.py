#!/usr/bin/env python3
"""comm_volume.py：按 llama.cpp 的实现，算三种多卡方式每个 ubatch 要在设备之间搬多少字节、同步多少轮。

- -sm layer（流水线）：相邻两段之间只传残差流 l_out，n_embd × n_tokens × 4 B（F32），每个 ubatch 跨 n_dev - 1 道边界。
  GGML_SCHED_DEBUG=1 的 SPLIT 输入里能看到 l_out-9 (2M)，和这里的公式一致（见 logs/fake-sched.txt）。
- -sm tensor（张量并行）：Meta 后端在 split axis 为 PARTIAL 的节点处切子图，每段之后做一次 AllReduce。
  attn_output 和 ffn_down 按输入维切（AXIS_0，src/llama-model.cpp 575、610 行），输出是部分和，所以每层 2 次。
  AllReduce 有三条路（ggml-cuda.cu 1135-1380 行、ggml-backend-meta.cpp 2355 行）：
    nccl     小张量按 F32、大张量压成 BF16 再归约；门槛 n≤2: 32768，n=3: 131072，n≥4: 262144 个元素
    butterfly 没有 comm 接口时的兜底：先把超出 2 的幂的设备折进来，再做 ceil(log2 n) 步两两交换
  每设备发送字节按教科书公式算：ring 2(n-1)/n × S（Patarasuk & Yuan 2009），butterfly 每步发整块 S。
- RPC：每次图计算发一份序列化的图，4 + 8 + 4 + 8 × n_nodes + 4 + 296 × n_tensors + 9 B；
  同一张图第二次起只发 GRAPH_RECOMPUTE（4 + 8 + 9 B），见 rpc_msg.c。

用法：
  python3 comm_volume.py MODEL.gguf [--dev 2,4] [--tokens 1,512]
  python3 comm_volume.py --n-embd 8192 --n-layer 80 --name llama-70B [--dev 4] [--tokens 1]
  加 --gbps 31.5 会按给定带宽把字节折成传输时间；同步轮数和延迟另算，脚本不替你假设单轮延迟。
"""
import argparse, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))

KiB, MiB = 1024, 1024 ** 2
RPC_TENSOR = 296
HDR_REQ = 9


def nccl_bf16(n_dev, ne):
    """ggml-cuda.cu 1155 行的门槛：小于它按 F32 归约，否则压成 BF16"""
    th = 32768 if n_dev <= 2 else 131072 if n_dev == 3 else 262144
    return ne >= th


def butterfly_steps(n_dev):
    return math.ceil(math.log2(n_dev)) if n_dev > 1 else 0


def butterfly_rounds(n_dev):
    """折叠 1 轮（不是 2 的幂时）+ ceil(log2 p2) 步交换 + 拷回 1 轮"""
    p2 = 1 << (n_dev.bit_length() - 1)
    extra = 2 if p2 != n_dev else 0
    return butterfly_steps(p2) + extra


def tp(n_embd, n_layer, n_dev, n_tok):
    ne = n_embd * n_tok
    s32 = ne * 4
    bf16 = nccl_bf16(n_dev, ne)
    s_nccl = ne * 2 if bf16 else s32
    n_ar = 2 * n_layer
    ring = 2 * (n_dev - 1) / n_dev * s_nccl
    bfly = butterfly_steps(1 << (n_dev.bit_length() - 1)) * s32
    if n_dev & (n_dev - 1):
        bfly += s32          # 折叠或拷回那一轮，参与的设备多发一次
    return dict(ne=ne, s32=s32, bf16=bf16, n_ar=n_ar, ring=ring * n_ar, bfly=bfly * n_ar,
                rounds_ring=2 * (n_dev - 1) * n_ar, rounds_bfly=butterfly_rounds(n_dev) * n_ar)


def pp(n_embd, n_dev, n_tok):
    s = n_embd * n_tok * 4
    return dict(per_hop=s, total=s * (n_dev - 1), hops=n_dev - 1)


def rpc_graph(n_nodes, n_tensors):
    return 4 + 8 + 4 + 8 * n_nodes + 4 + RPC_TENSOR * n_tensors + HDR_REQ, 4 + 8 + HDR_REQ


def fmt(b):
    if b >= MiB:
        return f'{b / MiB:.2f} MiB'
    if b >= KiB:
        return f'{b / KiB:.1f} KiB'
    return f'{b:.0f} B'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model', nargs='?')
    ap.add_argument('--n-embd', type=int)
    ap.add_argument('--n-layer', type=int)
    ap.add_argument('--name', default='')
    ap.add_argument('--dev', default='2,4')
    ap.add_argument('--tokens', default='1,512')
    ap.add_argument('--gbps', type=float, default=0, help='链路单向带宽 GB/s，给了才折算时间')
    ap.add_argument('--rpc-graph', default='', help='n_nodes,n_tensors：顺带算 RPC 每次图计算的字节')
    a = ap.parse_args()
    if a.model:
        from gguf_lite import GGUF
        g = GGUF(a.model)
        n_embd, n_layer = g.hp('embedding_length'), g.hp('block_count')
        name = a.name or os.path.basename(a.model)
    else:
        if not (a.n_embd and a.n_layer):
            sys.exit('给 GGUF，或者给 --n-embd 和 --n-layer')
        n_embd, n_layer, name = a.n_embd, a.n_layer, a.name or 'model'
    devs = [int(x) for x in a.dev.split(',')]
    toks = [int(x) for x in a.tokens.split(',')]
    print(f'{name}  n_embd {n_embd}  n_layer {n_layer}')
    print(f'{"dev":>3} {"tok":>5} | {"PP 每跳":>10} {"PP 合计":>10} | {"TP 次数":>6} {"单次 F32":>10} {"NCCL":>5} '
          f'{"ring 每卡":>11} {"bfly 每卡":>11} {"ring 轮":>7} {"bfly 轮":>7}')
    for n in devs:
        for t in toks:
            p, q = pp(n_embd, n, t), tp(n_embd, n_layer, n, t)
            line = (f'{n:3d} {t:5d} | {fmt(p["per_hop"]):>10} {fmt(p["total"]):>10} | {q["n_ar"]:6d} {fmt(q["s32"]):>10} '
                    f'{"BF16" if q["bf16"] else "F32":>5} {fmt(q["ring"]):>11} {fmt(q["bfly"]):>11} '
                    f'{q["rounds_ring"]:7d} {q["rounds_bfly"]:7d}')
            if a.gbps:
                us = lambda b: b / (a.gbps * 1e9) * 1e6
                line += f' | 纯带宽 PP {us(p["total"]):.1f} us  ring {us(q["ring"]):.1f} us  bfly {us(q["bfly"]):.1f} us'
            print(line)
    if a.rpc_graph:
        nn, nt = (int(x) for x in a.rpc_graph.split(','))
        full, re_ = rpc_graph(nn, nt)
        print(f'RPC 图 n_nodes={nn} n_tensors={nt}：首次 {fmt(full)}（{full} B），重放 {re_} B')


if __name__ == '__main__':
    main()
