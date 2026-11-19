"""看参考输出里的超大激活：每层最大绝对值在哪个位置、哪个通道，和其余位置差多少。
用法：python3 massive.py REF_DIR > logs/massive.txt
REF_DIR 是 hf_ref.py 的输出目录，读 layers.bin 和 meta.txt。
bf16 基线逐层 max_abs 从第 2 层起跳到 3 左右，就是这个位置上的数太大，bf16 的相对误差乘上去变成了绝对误差。
"""
import os
import sys

import numpy as np


def main():
    d = sys.argv[1]
    meta = dict(l.split() for l in open(os.path.join(d, 'meta.txt')) if len(l.split()) == 2)
    n_layer, n_tok, n_embd = int(meta['n_layer']), int(meta['n_tokens']), int(meta['n_embd'])
    L = np.fromfile(os.path.join(d, 'layers.bin'), dtype=np.float32).reshape(n_layer, n_tok, n_embd)
    print('layer max_abs pos chan median rest_max')
    for i in range(n_layer):
        a = np.abs(L[i])
        p, c = np.unravel_index(a.argmax(), a.shape)
        rest = np.delete(a, p, axis=0).max()
        print(f'{i} {a.max():.2f} {p} {c} {np.median(a):.3f} {rest:.2f}')


if __name__ == '__main__':
    main()
