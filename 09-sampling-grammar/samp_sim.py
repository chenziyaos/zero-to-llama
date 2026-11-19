"""用 numpy 按 llama.cpp 0ec4027 的规则重算各个截断采样器，读 samp_bench 存下的 logits。

看三件事：
1. 每个采样器留下几个候选、保住多少概率质量、截断后的熵；
2. 温度放在截断前还是截断后，候选数差多少（llama.cpp 缺省把 temp 放最后）；
3. 各种温度下 top_p 和 min_p 的候选数怎么变。

用法：python3 samp_sim.py logits1.f32 [logits2.f32 ...]
不给文件时用一组合成的 logits 演示。
"""
import sys

import numpy as np


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    m = x[np.isfinite(x)].max()
    e = np.where(np.isfinite(x), np.exp(x - m), 0.0)
    return e / e.sum()


def top_k(lg, k):
    if k <= 0 or k >= len(lg):
        return np.ones(len(lg), bool)
    idx = np.argpartition(-lg, k - 1)[:k]
    keep = np.zeros(len(lg), bool)
    keep[idx] = True
    return keep


def top_p(lg, p, min_keep=1):
    if p >= 1:
        return np.ones(len(lg), bool)
    pr = softmax(lg)
    order = np.argsort(-pr, kind='stable')
    cum = np.cumsum(pr[order])
    n = int(np.searchsorted(cum, p) + 1)   # 累积到刚好不小于 p 的那个也留下
    n = max(n, min_keep)
    keep = np.zeros(len(lg), bool)
    keep[order[:n]] = True
    return keep


def min_p(lg, p):
    # 源码在 logit 空间比：logit >= max + ln(p)，等价于 prob >= p * prob_max
    if p <= 0:
        return np.ones(len(lg), bool)
    return lg >= lg.max() + np.log(p)


def top_n_sigma(lg, n):
    if n <= 0:
        return np.ones(len(lg), bool)
    f = lg[np.isfinite(lg)]
    return lg >= f.max() - n * f.std()


def typical(lg, p, min_keep=1):
    if p >= 1:
        return np.ones(len(lg), bool)
    pr = softmax(lg)
    with np.errstate(divide='ignore'):
        logp = np.where(pr > 0, np.log(pr), -np.inf)
    ent = -np.sum(np.where(pr > 0, pr * logp, 0))
    shift = np.abs(-logp - ent)
    order = np.argsort(shift, kind='stable')
    cum = np.cumsum(pr[order])
    n = int(np.searchsorted(cum, p) + 1)
    n = max(n, min_keep)
    keep = np.zeros(len(lg), bool)
    keep[order[:n]] = True
    return keep


def penalties(lg, counts, repeat=1.0, freq=0.0, present=0.0):
    # llama-sampler.cpp penalties_apply：logit 小于等于 0 乘 repeat，大于 0 除以 repeat，再减频率项和出现项
    out = np.asarray(lg, dtype=np.float64).copy()
    for tok, c in counts.items():
        if c <= 0:
            continue
        out[tok] = out[tok] * repeat if out[tok] <= 0 else out[tok] / repeat
        out[tok] -= c * freq + present
    return out


def stats(lg, keep):
    pr = softmax(lg)
    mass = pr[keep].sum()
    q = pr[keep] / mass
    ent = -np.sum(q * np.log(q))
    return int(keep.sum()), float(mass), float(ent)


def chain(lg, k=40, p=0.95, mp=0.05, temp=0.8, temp_first=False):
    """common 的缺省顺序 top_k -> top_p -> min_p -> temp；temp_first 时把温度挪到最前。"""
    x = lg / temp if temp_first else lg.copy()
    m = top_k(x, k)
    x = np.where(m, x, -np.inf)
    m = top_p(x, p) & m
    x = np.where(m, x, -np.inf)
    m = min_p(x, mp) & m
    return m


def report(name, lg):
    pr = softmax(lg)
    ent = -np.sum(pr[pr > 0] * np.log(pr[pr > 0]))
    print(f'## {name}  n_vocab {len(lg)}  p_max {pr.max():.4f}  entropy {ent:.3f}')
    print(f'{"sampler":<22}{"kept":>8}{"mass":>9}{"entropy":>9}')
    for nm, keep in (('top_k 40', top_k(lg, 40)), ('top_k 20', top_k(lg, 20)), ('top_p 0.95', top_p(lg, 0.95)),
                     ('top_p 0.9', top_p(lg, 0.9)), ('min_p 0.05', min_p(lg, 0.05)), ('min_p 0.1', min_p(lg, 0.1)),
                     ('top_n_sigma 1', top_n_sigma(lg, 1.0)), ('typical 0.9', typical(lg, 0.9)),
                     ('chain default', chain(lg))):
        n, mass, e = stats(lg, keep)
        print(f'{nm:<22}{n:>8}{mass:>9.4f}{e:>9.3f}')
    print('# 温度的位置：top_p 0.95 + min_p 0.05，不加 top_k')
    print(f'{"temp":<8}{"temp last":>12}{"temp first":>12}')
    for t in (0.6, 1.0, 1.5, 2.0):
        a = chain(lg, k=0, temp=t).sum()
        b = chain(lg, k=0, temp=t, temp_first=True).sum()
        print(f'{t:<8}{a:>12}{b:>12}')
    print('# 温度放最前时各采样器单独的候选数')
    print(f'{"temp":<8}{"top_p0.95":>12}{"min_p0.05":>12}{"nsigma1":>12}')
    for t in (0.6, 1.0, 1.5, 2.0):
        x = lg / t
        print(f'{t:<8}{top_p(x, 0.95).sum():>12}{min_p(x, 0.05).sum():>12}{top_n_sigma(x, 1.0).sum():>12}')
    print()


def main():
    files = sys.argv[1:]
    if not files:
        rng = np.random.default_rng(0)
        lg = rng.normal(0, 2, 32000).astype(np.float32)
        lg[:5] += np.array([12, 11, 10.5, 9, 8])
        report('synthetic', lg)
        return
    for f in files:
        report(f.rsplit('/', 1)[-1], np.fromfile(f, dtype=np.float32).astype(np.float64))


if __name__ == '__main__':
    main()
