#!/usr/bin/env python3
"""纯 Python 反量化几种常用块格式，用来看清一个块里到底存了什么。

对照源码（llama.cpp 0ec4027）：
  ggml/src/ggml-common.h   block_q4_0 / block_q8_0 / block_q4_K / block_q6_K 的结构体
  ggml/src/ggml-quants.c   dequantize_row_q4_0 / q8_0 / q4_K（1539 行）/ q6_K（1949 行）
                           get_scale_min_k4（889 行）

用法：
  python3 dequant.py MODEL.gguf blk.0.ffn_up.weight            # 打印第一个块的拆解
  python3 dequant.py MODEL.gguf blk.0.ffn_up.weight --ref BF16.gguf   # 和 bf16 原始权重比误差

只支持 F32、F16、BF16、Q8_0、Q4_0、Q4_K、Q6_K，够看清 legacy 和 k-quants 两类结构。
"""
import argparse
import math
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from gguf_lite import GGUF, GGML_TYPES  # noqa: E402

QK_K = 256


def f16(b, off=0):
    return struct.unpack_from('<e', b, off)[0]


def bf16_row(b):
    n = len(b) // 2
    u = struct.unpack(f'<{n}H', b)
    return [struct.unpack('<f', struct.pack('<I', x << 16))[0] for x in u]


def deq_q8_0(b):
    # 34 字节：d(f16) + 32 个 int8
    out = []
    for o in range(0, len(b), 34):
        d = f16(b, o)
        out += [d * q for q in struct.unpack_from('<32b', b, o + 2)]
    return out


def deq_q4_0(b):
    # 18 字节：d(f16) + 16 字节，低 4 位是前 16 个，高 4 位是后 16 个，值域 -8..7
    out = []
    for o in range(0, len(b), 18):
        d = f16(b, o)
        qs = b[o + 2:o + 18]
        out += [d * ((x & 0xF) - 8) for x in qs] + [d * ((x >> 4) - 8) for x in qs]
    return out


def scale_min_k4(j, s):
    # get_scale_min_k4：12 字节里塞了 8 个 6 bit 的 scale 和 8 个 6 bit 的 min
    if j < 4:
        return s[j] & 63, s[j + 4] & 63
    return (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4), (s[j + 4] >> 4) | ((s[j] >> 6) << 4)


def q4_k_block(b, o):
    """拆一个 144 字节的 Q4_K 超级块，返回 (d, dmin, [(sc, m)]*8, 256 个反量化值)。"""
    d, dmin = f16(b, o), f16(b, o + 2)
    s = b[o + 4:o + 16]
    q = b[o + 16:o + 144]
    sm = [scale_min_k4(j, s) for j in range(8)]
    y = []
    for j in range(4):           # 4 组，每组 64 个值共享 32 字节
        qq = q[32 * j:32 * j + 32]
        sc1, m1 = sm[2 * j]
        sc2, m2 = sm[2 * j + 1]
        y += [d * sc1 * (x & 0xF) - dmin * m1 for x in qq]
        y += [d * sc2 * (x >> 4) - dmin * m2 for x in qq]
    return d, dmin, sm, y


def deq_q4_k(b):
    out = []
    for o in range(0, len(b), 144):
        out += q4_k_block(b, o)[3]
    return out


def deq_q6_k(b):
    # 210 字节：ql[128] 低 4 位 + qh[64] 高 2 位 + scales[16] int8 + d(f16)
    out = []
    for o in range(0, len(b), 210):
        ql, qh = b[o:o + 128], b[o + 128:o + 192]
        sc = struct.unpack_from('<16b', b, o + 192)
        d = f16(b, o + 208)
        y = [0.0] * 256
        for n in range(2):
            L, H, S, Y = ql[64 * n:], qh[32 * n:], sc[8 * n:], 128 * n
            for l in range(32):
                i = l // 16
                q1 = ((L[l] & 0xF) | (((H[l] >> 0) & 3) << 4)) - 32
                q2 = ((L[l + 32] & 0xF) | (((H[l] >> 2) & 3) << 4)) - 32
                q3 = ((L[l] >> 4) | (((H[l] >> 4) & 3) << 4)) - 32
                q4 = ((L[l + 32] >> 4) | (((H[l] >> 6) & 3) << 4)) - 32
                y[Y + l] = d * S[i] * q1
                y[Y + l + 32] = d * S[i + 2] * q2
                y[Y + l + 64] = d * S[i + 4] * q3
                y[Y + l + 96] = d * S[i + 6] * q4
        out += y
    return out


DEQ = {
    'F32': lambda b: list(struct.unpack(f'<{len(b) // 4}f', b)),
    'F16': lambda b: list(struct.unpack(f'<{len(b) // 2}e', b)),
    'BF16': bf16_row,
    'Q8_0': deq_q8_0, 'Q4_0': deq_q4_0, 'Q4_K': deq_q4_k, 'Q6_K': deq_q6_k,
}


def read_rows(g, name, rows):
    """读张量的前 rows 行（ne[0] 是行长），返回 (类型名, 行长, 原始字节)。"""
    t = next((t for t in g.tensors if t.name == name), None)
    if t is None:
        raise KeyError(f'{g.path} 里没有 {name}')
    _, blk, nb = GGML_TYPES[t.type_id]
    row_bytes = t.shape[0] // blk * nb
    rows = min(rows, t.n_elems // t.shape[0])
    with open(g.path, 'rb') as f:
        f.seek(g.data_offset + t.offset)
        raw = f.read(row_bytes * rows)
    return t.type_name, t.shape[0], raw


def dequant(g, name, rows=4):
    tn, n0, raw = read_rows(g, name, rows)
    if tn not in DEQ:
        raise NotImplementedError(f'{tn} 不在本脚本支持的类型里')
    return tn, n0, DEQ[tn](raw)


def error_stats(x, ref):
    n = len(x)
    se = sum((a - b) ** 2 for a, b in zip(x, ref))
    ss = sum(b * b for b in ref)
    return {'n': n, 'rmse': math.sqrt(se / n), 'rel': math.sqrt(se / ss) if ss else 0.0,
            'maxabs': max(abs(a - b) for a, b in zip(x, ref))}


def imatrix_weights(path, name):
    """从 imatrix.gguf 取某个权重的 in_sum2 / counts，即每个输入通道的平均激活平方 E[x_j^2]。

    llama-imatrix 在每次矩阵乘时累加 x_j^2（tools/imatrix/imatrix.cpp 660 行），
    GGUF 格式里存成 <name>.in_sum2 和 <name>.counts 两个 F32 张量（PR #9400）。
    """
    g = GGUF(path)
    by = {t.name: t for t in g.tensors}
    s2, cnt = by.get(name + '.in_sum2'), by.get(name + '.counts')
    if s2 is None or cnt is None:
        return None
    with open(path, 'rb') as f:
        f.seek(g.data_offset + s2.offset)
        v = struct.unpack(f'<{s2.shape[0]}f', f.read(4 * s2.shape[0]))
        f.seek(g.data_offset + cnt.offset)
        c = struct.unpack('<f', f.read(4))[0]
    return [x / c for x in v]


def weighted_error(x, ref, n0, w):
    """sum_j w_j (q_ij - x_ij)^2 / sum_j w_j x_ij^2：输出误差的期望，正是 imatrix 量化在最小化的东西。"""
    num = den = 0.0
    for i, (a, b) in enumerate(zip(x, ref)):
        wj = w[i % n0]
        num += wj * (a - b) ** 2
        den += wj * b * b
    return math.sqrt(num / den) if den else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('tensor')
    ap.add_argument('--rows', type=int, default=4)
    ap.add_argument('--ref', help='bf16/f16 GGUF，用来算量化误差')
    ap.add_argument('--imatrix', help='imatrix.gguf，额外报告按激活能量加权的误差')
    a = ap.parse_args()
    g = GGUF(a.model)
    tn, n0, raw = read_rows(g, a.tensor, a.rows)
    print(f'{a.tensor}: {tn}, 行长 {n0}, 读了 {len(raw)} 字节')
    if tn == 'Q4_K':
        d, dmin, sm, y = q4_k_block(raw, 0)
        print(f'第一个超级块 144 B = d {d:.6g}（2 B）+ dmin {dmin:.6g}（2 B）'
              f'+ 8 组 6 bit scale/min（12 B）+ 256 个 4 bit 量化值（128 B）')
        for j, (sc, m) in enumerate(sm):
            print(f'  子块 {j}: scale={sc:2d} min={m:2d} -> 实际步长 {d * sc:.5g}，偏移 {dmin * m:.5g}')
        print('  前 8 个值', ' '.join(f'{v:+.4f}' for v in y[:8]))
    x = DEQ[tn](raw)
    if a.ref:
        rt, rn, rraw = read_rows(GGUF(a.ref), a.tensor, a.rows)
        ref = DEQ[rt](rraw)
        s = error_stats(x, ref)
        print(f'和 {rt} 原始权重比（{s["n"]} 个值）：RMSE {s["rmse"]:.3e}，'
              f'相对误差 {s["rel"] * 100:.2f}%，最大绝对误差 {s["maxabs"]:.3e}')
        if a.imatrix:
            w = imatrix_weights(a.imatrix, a.tensor)
            if w:
                print(f'按 imatrix 激活能量加权的相对误差 {weighted_error(x, ref, n0, w) * 100:.2f}%')
            else:
                print(f'{a.imatrix} 里没有 {a.tensor} 的统计（output.weight 默认不收集）')


if __name__ == '__main__':
    main()
