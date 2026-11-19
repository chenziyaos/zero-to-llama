"""位置 0 和其余位置分开看逐层余弦：量化后的最大误差为什么集中在位置 0。
用法：python3 pos0.py REF_DIR OUT_ROOT tag1 tag2 ... > logs/pos0.txt
REF_DIR 是 hf_ref.py 的输出，OUT_ROOT/tag 是 dump_logits --keep l_out 的输出。
每行：tag 层号 位置0余弦 位置0去掉最大通道后的余弦 其余位置余弦中位数 最大通道号 该通道参考值 该通道实测值
"""
import os
import sys

import numpy as np


def cos(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    ref, root, tags = sys.argv[1], sys.argv[2], sys.argv[3:]
    meta = dict(l.split() for l in open(os.path.join(ref, 'meta.txt')) if len(l.split()) == 2)
    nl, n, e = int(meta['n_layer']), int(meta['n_tokens']), int(meta['n_embd'])
    R = np.fromfile(os.path.join(ref, 'layers.bin'), dtype=np.float32).reshape(nl, n, e)
    print('tag layer cos_pos0 cos_pos0_wo_max cos_rest_median chan ref_val test_val')
    for tag in tags:
        for i in range(nl):
            p = os.path.join(root, tag, f'l_out-{i}.bin')
            if not os.path.exists(p):
                continue
            t = np.fromfile(p, dtype='<f4').reshape(n, e)
            ch = int(np.abs(R[i][0]).argmax())
            m = np.ones(e, bool)
            m[ch] = False
            rest = np.median([cos(t[j], R[i][j]) for j in range(1, n)])
            print(f'{tag} {i} {cos(t[0], R[i][0]):.5f} {cos(t[0][m], R[i][0][m]):.5f} {rest:.5f} {ch} {R[i][0][ch]:.1f} {t[0][ch]:.1f}')


if __name__ == '__main__':
    main()
