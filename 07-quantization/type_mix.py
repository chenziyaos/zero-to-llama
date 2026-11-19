#!/usr/bin/env python3
"""看一个量化文件里每类张量实际用了什么类型，并和“按规则推出来的配方”对照。

llama-quantize 的 Q4_K_M 不是“全部 Q4_K”。llama_tensor_get_type_impl（src/llama-quant.cpp 484 行）
按张量名和层号逐个改类型，核心是 490 行的 use_more_bits：

    use_more_bits(i, n) = i < n/8 || i >= 7n/8 || (i - n/8) % 3 == 2

Q4_K_M / Q5_K_M 下，attn_v 和 ffn_down 在 use_more_bits 为真的层升到 Q6_K，
output.weight 升到 Q6_K。这个脚本把 GGUF 里的真实类型按“角色”汇总，
再用 predict_q4_k_m 推一遍，两边对得上，说明你读懂了配方。

用法：
    python3 type_mix.py MODEL.gguf              # 按角色汇总类型和字节
    python3 type_mix.py MODEL.gguf --layers     # 逐层打印 attn_v / ffn_down 的类型
    python3 type_mix.py MODEL.gguf --check-q4km # 和 Q4_K_M 规则推导结果逐个比
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from gguf_lite import GGUF  # noqa: E402

MiB = 2 ** 20


def role(name):
    """blk.12.attn_v.weight -> ('attn_v', 12)；output.weight -> ('output', None)。"""
    m = re.match(r'blk\.(\d+)\.(.+?)\.(weight|bias)$', name)
    if m:
        return m.group(2), int(m.group(1))
    return name.rsplit('.', 1)[0], None


def use_more_bits(i, n):
    return i < n // 8 or i >= 7 * n // 8 or (i - n // 8) % 3 == 2


def predict_q4_k_m(name, n_layer, ne0=None):
    """只覆盖 Qwen/Llama 这类稠密模型的常见张量，规则摘自 llama_tensor_get_type。

    - output.weight：Q6_K（行长不是 256 的倍数时退到 Q8_0）
    - token_embd：跟随基础类型 Q4_K
    - attn_v、ffn_down：use_more_bits 的层 Q6_K，其余 Q4_K
    - 1 维张量（norm、bias）保持 F32，不量化
    """
    r, i = role(name)
    if r.endswith('norm'):
        return 'F32'
    if r == 'output':
        return 'Q8_0' if ne0 is not None and ne0 % 256 else 'Q6_K'
    if r in ('attn_v', 'ffn_down') and i is not None:
        return 'Q6_K' if use_more_bits(i, n_layer) else 'Q4_K'
    return 'Q4_K'


def summarize(g):
    rows = {}
    for t in g.tensors:
        r, _ = role(t.name)
        d = rows.setdefault(r, {})
        n, b = d.get(t.type_name, (0, 0))
        d[t.type_name] = (n + 1, b + t.nbytes)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--layers', action='store_true')
    ap.add_argument('--check-q4km', action='store_true')
    a = ap.parse_args()
    g = GGUF(a.model)
    n_layer = g.hp('block_count')
    print(f'{os.path.basename(a.model)}: {n_layer} 层, 张量 {g.tensor_bytes / MiB:.1f} MiB, '
          f'{g.tensor_bytes * 8 / g.n_params:.2f} BPW')
    by_type = {}
    for r, d in sorted(summarize(g).items(), key=lambda kv: -sum(b for _, b in kv[1].values())):
        parts = ', '.join(f'{n}×{tn}' for tn, (n, _) in sorted(d.items()))
        mib = sum(b for _, b in d.values()) / MiB
        print(f'  {r:14s} {mib:8.1f} MiB  {parts}')
        for tn, (_, b) in d.items():
            by_type[tn] = by_type.get(tn, 0) + b
    print('  按类型合计', ', '.join(f'{k} {v / MiB:.1f} MiB' for k, v in sorted(by_type.items(), key=lambda kv: -kv[1])))
    if a.layers:
        by = {t.name: t.type_name for t in g.tensors}
        for i in range(n_layer):
            mark = '*' if use_more_bits(i, n_layer) else ' '
            print(f'  {mark} blk.{i:<3d} attn_v {by.get(f"blk.{i}.attn_v.weight", "-"):7s} '
                  f'ffn_down {by.get(f"blk.{i}.ffn_down.weight", "-")}')
        print('  * = use_more_bits 为真的层')
    if a.check_q4km:
        bad = [(t.name, t.type_name, predict_q4_k_m(t.name, n_layer, t.shape[0])) for t in g.tensors
               if t.type_name != predict_q4_k_m(t.name, n_layer, t.shape[0])]
        print(f'Q4_K_M 规则推导：{len(g.tensors) - len(bad)}/{len(g.tensors)} 个张量一致')
        for n, got, want in bad[:20]:
            print(f'  {n}: 文件里 {got}，规则推出 {want}')
        sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
