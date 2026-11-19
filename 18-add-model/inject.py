"""往一份 GGUF 里注入转换器常见的错，生成新的 GGUF，给 compare.py 看每种错的样子。

用法：python3 inject.py SRC.gguf DST.gguf BUG [--layer N]
--layer N 只改第 N 层，用来演示逐层对拍怎么把问题定位到具体某一层。
BUG：
  qk-permute   对 attn_q、attn_k 按 LLaMA 转换器的 permute 重排行（把 NORM rope 的布局套到 NEOX 架构上）
  norm-plus1   所有 RMSNorm 权重加 1（把 Gemma 的 (1+w) 约定误用到 Qwen3）
  qk-norm-one  attn_q_norm、attn_k_norm 权重全置 1（相当于转换时漏了这两个张量、用缺省值顶上）
  swap-gate-up ffn_gate 和 ffn_up 对调（映射表把两个名字写反）
先整文件拷贝，再用 GGUFReader 以 r+ 打开原位改字节，元数据和张量顺序都不动。
"""
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'llama.cpp', 'gguf-py'))
from gguf import GGUFReader  # noqa: E402

N_HEAD, N_HEAD_KV = 16, 8


def permute_rows(w, n_head):
    # 和 conversion/llama.py 的 permute 一样，只是作用在按行存的字节上：每个头的前一半行和后一半行交错
    rows = w.shape[0]
    return (w.reshape(n_head, 2, rows // n_head // 2, *w.shape[1:])
             .swapaxes(1, 2).reshape(w.shape))


def main():
    src, dst, bug = sys.argv[1:4]
    only = int(sys.argv[sys.argv.index('--layer') + 1]) if '--layer' in sys.argv else None
    shutil.copyfile(src, dst)
    r = GGUFReader(dst, 'r+')
    n = 0
    for t in r.tensors:
        nm = t.name
        if only is not None and not nm.startswith(f'blk.{only}.'):
            continue
        if bug == 'qk-permute' and nm.endswith(('attn_q.weight', 'attn_k.weight')):
            h = N_HEAD if 'attn_q' in nm else N_HEAD_KV
            t.data[...] = permute_rows(np.array(t.data), h)
            n += 1
        elif bug == 'norm-plus1' and nm.endswith('norm.weight'):
            t.data[...] = t.data + 1.0
            n += 1
        elif bug == 'qk-norm-one' and nm.endswith(('attn_q_norm.weight', 'attn_k_norm.weight')):
            t.data[...] = 1.0
            n += 1
        elif bug == 'swap-gate-up' and nm.endswith('ffn_gate.weight'):
            up = next(x for x in r.tensors if x.name == nm.replace('ffn_gate', 'ffn_up'))
            g = np.array(t.data)
            t.data[...] = up.data
            up.data[...] = g
            n += 2
    r.data.flush()
    print(f'inject {bug}: {n} tensors modified -> {dst}')


if __name__ == '__main__':
    main()
