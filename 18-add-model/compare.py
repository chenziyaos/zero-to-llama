"""对拍两份 logits（和可选的逐层输出）：参考来自 hf_ref.py，待测来自 dump_logits 或另一份 hf_ref。

用法：python3 compare.py REF_DIR TEST_DIR [--layers] [--json out.json] [--tag 名字]
logits 指标逐位置算再汇总：
  max_abs   所有位置所有词表项里最大的绝对差
  nmse      sum((t-r)^2) / sum(r^2)，全体元素一起算
  nmse_var  mean((t-r)^2) / var(r)，和 examples/model-conversion 的 check-nmse.py 同一口径
  pos0      第 0 个位置的 max_abs；RoPE 在位置 0 转角为 0，只错在 RoPE 的 bug 这里和基线一样
  kld       KL(softmax(r) || softmax(t))，逐位置算后取均值和最大值
  top1      两边 argmax 一致的位置比例
逐层（--layers）：每层 nmse 和最差位置的余弦相似度。参考的 layers.bin 是 n_layer x n_tok x n_embd，
待测是 l_out-<i>.bin（ggml 形状 n_embd x n_tok，内存顺序和 numpy 的 n_tok x n_embd 相同）。
"""
import argparse
import json
import os

import numpy as np


def meta(d):
    m = {}
    for line in open(os.path.join(d, 'meta.txt')):
        k, *v = line.split()
        m[k] = v[0] if len(v) == 1 else v
    return m


def logsoftmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def logits_stats(r, t):
    d = t.astype(np.float64) - r
    nmse = float((d * d).sum() / (r.astype(np.float64) ** 2).sum())
    lr, lt = logsoftmax(r.astype(np.float64)), logsoftmax(t.astype(np.float64))
    kld = (np.exp(lr) * (lr - lt)).sum(-1)
    top1 = r.argmax(-1) == t.argmax(-1)
    return {
        'max_abs': float(np.abs(d).max()),
        'nmse': nmse,
        'nmse_var': float((d * d).mean() / r.astype(np.float64).var()),
        'pos0': float(np.abs(d[0]).max()),
        'kld_mean': float(kld.mean()),
        'kld_max': float(kld.max()),
        'kld_argmax': int(kld.argmax()),
        'top1': float(top1.mean()),
        'first_flip': int(np.argmin(top1)) if not top1.all() else -1,
    }


def load_test_layer(d, i, n, e):
    p = os.path.join(d, f'l_out-{i}.bin')
    if not os.path.exists(p):
        return None
    return np.fromfile(p, dtype='<f4').reshape(n, e)


def layer_stats(ref_dir, test_dir, n, e, nl):
    ref = np.fromfile(os.path.join(ref_dir, 'layers.bin'), dtype='<f4').reshape(nl, n, e)
    out = []
    for i in range(nl):
        t = load_test_layer(test_dir, i, n, e)
        if t is None:
            t = np.fromfile(os.path.join(test_dir, 'layers.bin'), dtype='<f4').reshape(nl, n, e)[i]
        r = ref[i].astype(np.float64)
        t = t.astype(np.float64)
        d = t - r
        cos = (r * t).sum(-1) / (np.linalg.norm(r, axis=-1) * np.linalg.norm(t, axis=-1) + 1e-30)
        out.append({'layer': i, 'nmse': float((d * d).sum() / (r * r).sum()),
                    'cos_min': float(cos.min()), 'cos_min_pos': int(cos.argmin()),
                    'max_abs': float(np.abs(d).max())})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ref')
    ap.add_argument('test')
    ap.add_argument('--layers', action='store_true')
    ap.add_argument('--json')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    mr, mt = meta(a.ref), meta(a.test)
    n, v = int(mr['n_tokens']), int(mr['n_vocab'])
    assert int(mt['n_tokens']) == n and int(mt['n_vocab']) == v, 'n_tokens / n_vocab 对不上'
    r = np.fromfile(os.path.join(a.ref, 'logits.bin'), dtype='<f4').reshape(n, v)
    t = np.fromfile(os.path.join(a.test, 'logits.bin'), dtype='<f4').reshape(n, v)
    res = {'tag': a.tag, 'ref': a.ref, 'test': a.test, 'n_tokens': n, 'logits': logits_stats(r, t)}
    s = res['logits']
    print(f"[{a.tag}] logits: max_abs {s['max_abs']:.4g}  nmse {s['nmse']:.3e}  "
          f"kld mean {s['kld_mean']:.3e} max {s['kld_max']:.3e}@{s['kld_argmax']}  "
          f"top1 {s['top1'] * 100:.1f}%  first_flip {s['first_flip']}  nmse_var {s['nmse_var']:.3e}  pos0 {s['pos0']:.4g}")
    if a.layers:
        nl, e = int(mr['n_layer']), int(mr['n_embd'])
        res['layers'] = layer_stats(a.ref, a.test, n, e, nl)
        for x in res['layers']:
            print(f"  l_out-{x['layer']:<2d} nmse {x['nmse']:.3e}  cos_min {x['cos_min']:.6f}@{x['cos_min_pos']}  max_abs {x['max_abs']:.4g}")
    if a.json:
        with open(a.json, 'a') as f:
            f.write(json.dumps(res) + '\n')


if __name__ == '__main__':
    main()
