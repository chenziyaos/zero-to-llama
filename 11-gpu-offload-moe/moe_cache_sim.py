#!/usr/bin/env python3
"""moe_cache_sim.py：拿 moe_trace 抓的路由回放 MoE 专家缓存，比较几种放法在 GPU 上命中多少专家读取。

缓存语义照 llama.cpp src/llama-moe-cache.cpp：
- 同一设备、同样专家张量布局的层共用一组槽位和一个 SLRU，键是 il*n_expert + expert；
- plan：本次请求选中的去重专家，先把命中的 touch 进 protected，再把未命中的按专家号排序，
  逐个取 victim 填进去，放到 probation 的尾部；
- touch：移到 protected 尾部，protected 超过 n_slots/2 时把它的头部退回 probation；
- victim：先扫 probation 再扫 protected，从头部（最久未用）找第一个空槽、别的层的槽，
  或者本次请求没选中的专家；
- 只有 n_tokens 不超过 32、且 n_tokens*n_used 不超过槽位数时才走缓存，大批只读不换。
这里按 decode 回放，一次一个 token，所以每次请求就是一层的 n_used 个专家。

对照组：
- layers：-ncmoe 式的静态放法，前若干层专家全在主存，后面整层上卡，命中率等于上卡专家的比例；
- hot：按暖缓存那段（或 --profile 给的另一份 trace）统计频次，把最热的（层，专家）钉在卡上，
  llama.cpp 的 -ot 只能按整个张量放，一个张量含一层全部专家，做不到这种逐专家的钉法，这里当参照；
- oracle：同上，但频次用整段路由统计，偷看了未来，是静态放法的上限；
- lru：一条链表，命中移到尾部，victim 规则同上；
- slru：上面描述的现行实现。

用法：
  python3 moe_cache_sim.py TRACE.bin [--mult 0.5,1,2 | --frac 0.1,0.5] [--warm 512] [--stats] [--profile OTHER.bin]
"""
import argparse, struct, sys
from collections import OrderedDict


def load(fn):
    with open(fn, 'rb') as f:
        n_layer, n_expert, n_used, n_tok = struct.unpack('<4i', f.read(16))
        data = f.read()
    assert len(data) == n_tok * n_layer * n_used, 'trace 长度不对'
    return n_layer, n_expert, n_used, n_tok, data


def sel(data, n_layer, n_used, t, il):
    o = (t * n_layer + il) * n_used
    return data[o:o + n_used]


class SLRU:
    """两段 OrderedDict 模拟 probation(0) 和 protected(1)，头部是最久未用"""

    def __init__(self, n_slots, segmented=True):
        self.n = n_slots
        self.seg = [OrderedDict(), OrderedDict()]
        self.where = {}          # key -> segment
        self.free = n_slots      # 空槽在 probation 头部，扫描时最先被选中
        self.segmented = segmented

    def _move(self, key, protect):
        src = self.where.get(key)
        if src is not None:
            del self.seg[src][key]
        self.seg[protect][key] = None
        self.where[key] = protect

    def touch(self, key):
        if not self.segmented:
            self._move(key, 0)
            return
        self._move(key, 1)
        if len(self.seg[1]) > self.n // 2:
            k = next(iter(self.seg[1]))
            self._move(k, 0)

    def victim(self, il, n_expert, seen):
        if self.free:
            return None
        for s in (0, 1):
            for key in self.seg[s]:
                if key // n_expert != il or (key % n_expert) not in seen:
                    return key
        raise RuntimeError('no SLRU victim')

    def plan(self, il, ids, n_expert):
        uniq = set(ids)
        if len(uniq) > self.n:
            return None
        hit = 0
        for e in uniq:
            k = il * n_expert + e
            if k in self.where:
                self.touch(k)
                hit += 1
        miss = 0
        for e in sorted(uniq):
            k = il * n_expert + e
            if k in self.where:
                continue
            v = self.victim(il, n_expert, uniq)
            if v is None:
                self.free -= 1
            else:
                del self.seg[self.where.pop(v)][v]
            self._move(k, 0)
            miss += 1
        return hit, miss


def replay(cache, data, n_layer, n_expert, n_used, n_tok, warm):
    h = m = 0
    for t in range(n_tok):
        for il in range(n_layer):
            r = cache.plan(il, sel(data, n_layer, n_used, t, il), n_expert)
            if t >= warm:
                h += r[0]
                m += r[1]
    return h / (h + m)


def counts(data, n_layer, n_expert, n_used, t0, t1):
    cnt = [0] * (n_layer * n_expert)
    for t in range(t0, t1):
        for il in range(n_layer):
            for e in sel(data, n_layer, n_used, t, il):
                cnt[il * n_expert + e] += 1
    return cnt


def pinned(cnt, data, n_layer, n_expert, n_used, n_tok, n_slots, warm):
    hot = set(sorted(range(len(cnt)), key=lambda k: -cnt[k])[:n_slots])
    h = n = 0
    for t in range(warm, n_tok):
        for il in range(n_layer):
            for e in sel(data, n_layer, n_used, t, il):
                h += (il * n_expert + e) in hot
                n += 1
    return h / n


def oracle(data, n_layer, n_expert, n_used, n_tok, n_slots, warm):
    cnt = [0] * (n_layer * n_expert)
    for t in range(n_tok):
        for il in range(n_layer):
            for e in sel(data, n_layer, n_used, t, il):
                cnt[il * n_expert + e] += 1
    hot = set(sorted(range(len(cnt)), key=lambda k: -cnt[k])[:n_slots])
    h = n = 0
    for t in range(warm, n_tok):
        for il in range(n_layer):
            for e in sel(data, n_layer, n_used, t, il):
                h += (il * n_expert + e) in hot
                n += 1
    return h / n


def stats(data, n_layer, n_expert, n_used, n_tok):
    """每层的频次偏斜、相邻 token 的专家重合、连续 n 个 token 用到的专家比例"""
    top_q = []
    reuse = []
    for il in range(n_layer):
        c = [0] * n_expert
        for t in range(n_tok):
            for e in sel(data, n_layer, n_used, t, il):
                c[e] += 1
        c.sort(reverse=True)
        top_q.append(sum(c[:max(1, n_expert // 4)]) / sum(c))
        same = 0
        for t in range(1, n_tok):
            same += len(set(sel(data, n_layer, n_used, t, il)) & set(sel(data, n_layer, n_used, t - 1, il)))
        reuse.append(same / ((n_tok - 1) * n_used))
    uniq = {}
    for b in (1, 4, 8, 16, 32, 64, 128, 512):
        if b > n_tok:
            break
        tot = cnt = 0
        for t0 in range(0, n_tok - b + 1, b):
            for il in range(n_layer):
                s = set()
                for t in range(t0, t0 + b):
                    s.update(sel(data, n_layer, n_used, t, il))
                tot += len(s) / n_expert
                cnt += 1
        uniq[b] = tot / cnt
    return top_q, reuse, uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trace')
    ap.add_argument('--frac', default='', help='槽位占全部专家的比例，逗号分隔')
    ap.add_argument('--mult', default='0.5,1,1.25,1.5,2,2.5,3', help='槽位是单个 token 用量的几倍，--frac 给了就不用它')
    ap.add_argument('--warm', type=int, default=512, help='前多少个 token 只暖缓存不计数')
    ap.add_argument('--tokens', type=int, default=0, help='只用前多少个 token，0 是全部')
    ap.add_argument('--stats', action='store_true')
    ap.add_argument('--profile', default='', help='hot 列用另一份 trace 的频次，比如拿 wiki 的频次去钉生成代码时的专家')
    a = ap.parse_args()
    n_layer, n_expert, n_used, n_tok, data = load(a.trace)
    if a.tokens:
        n_tok = min(n_tok, a.tokens)
    warm = min(a.warm, n_tok // 4)
    print(f'{a.trace}: {n_layer} 层 × {n_expert} 专家，选 {n_used}，{n_tok} 个 token，前 {warm} 个暖缓存')
    if a.stats:
        top_q, reuse, uniq = stats(data, n_layer, n_expert, n_used, n_tok)
        uni = (n_expert // 4) / n_expert
        print(f'最热 1/4 专家占的选择：均匀时 {uni:.3f}，各层 {min(top_q):.3f} 到 {max(top_q):.3f}，平均 {sum(top_q) / n_layer:.3f}')
        print(f'和上一个 token 重合的专家比例：随机时 {n_used / n_expert:.3f}，各层 {min(reuse):.3f} 到 {max(reuse):.3f}，平均 {sum(reuse) / n_layer:.3f}')
        print('连续 n 个 token 用到的专家比例：' + '  '.join(f'n={b} {v:.3f}' for b, v in uniq.items()))
    total = n_layer * n_expert
    if a.profile:
        pl, pe, pu, pt, pd = load(a.profile)
        assert (pl, pe, pu) == (n_layer, n_expert, n_used), 'profile 和 trace 不是同一个模型'
        prof = counts(pd, n_layer, n_expert, n_used, 0, pt)
    else:
        prof = counts(data, n_layer, n_expert, n_used, 0, warm)
    foot = n_layer * n_used
    print(f'一个 token 要 {foot} 个（层，专家），占全部专家的 {foot / total:.3f}')
    print('槽位比例  槽位数  槽位/单token  layers  hot     oracle  lru     slru')
    sizes = [round(total * float(f)) for f in a.frac.split(',')] if a.frac else \
            [round(foot * float(x)) for x in a.mult.split(',')]
    for n_slots in sizes:
        n_slots = min(total, max(n_used, n_slots))
        f = n_slots / total
        lru = replay(SLRU(n_slots, False), data, n_layer, n_expert, n_used, n_tok, warm)
        slru = replay(SLRU(n_slots, True), data, n_layer, n_expert, n_used, n_tok, warm)
        orc = oracle(data, n_layer, n_expert, n_used, n_tok, n_slots, warm)
        hot = pinned(prof, data, n_layer, n_expert, n_used, n_tok, n_slots, warm)
        lay = (n_slots // n_expert) / n_layer
        print(f'{f:8.3f}  {n_slots:6d}  {n_slots / foot:12.2f}  {lay:.3f}   {hot:.3f}   {orc:.3f}   {lru:.3f}   {slru:.3f}')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
