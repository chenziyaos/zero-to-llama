"""比较两个 GGUF：KV 的增删改、张量类型的变化、各段大小差多少。只用标准库。

    python3 gguf_diff.py a.gguf b.gguf            # 人读
    python3 gguf_diff.py a.gguf b.gguf --types    # 再列出每个类型变了的张量

常见用法：量化前后看哪些张量换了类型；改元数据之后确认只动了想动的键；
合并分卷之后确认张量数据没变，只多了 split.* 之类的键。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from gguf_lite import GGUF  # noqa: E402
from gguf_check import check  # noqa: E402


def short(v, n=60):
    s = repr(v)
    if isinstance(v, list):
        s = f'list[{len(v)}]'
    return s if len(s) <= n else s[:n] + '...'


def diff(pa, pb):
    a, b = GGUF(pa, keep_arrays=True), GGUF(pb, keep_arrays=True)
    ka, kb = set(a.kv), set(b.kv)
    out = {'only_a': sorted(ka - kb), 'only_b': sorted(kb - ka), 'changed': []}
    for k in sorted(ka & kb):
        if a.kv[k] != b.kv[k]:
            out['changed'].append((k, short(a.kv[k]), short(b.kv[k])))
    ta = {t.name: t for t in a.tensors}
    tb = {t.name: t for t in b.tensors}
    out['tensors_only_a'] = sorted(set(ta) - set(tb))
    out['tensors_only_b'] = sorted(set(tb) - set(ta))
    out['retyped'] = [(n, ta[n].type_name, tb[n].type_name, ta[n].nbytes, tb[n].nbytes)
                      for n in sorted(set(ta) & set(tb)) if ta[n].type_name != tb[n].type_name]
    out['reshaped'] = [(n, ta[n].shape, tb[n].shape)
                       for n in sorted(set(ta) & set(tb)) if list(ta[n].shape) != list(tb[n].shape)]
    sa, sb = check(pa), check(pb)
    out['sections'] = {k: (sa['sections'][k], sb['sections'][k]) for k in sa.get('sections', {})}
    return out


def main(argv):
    paths = [x for x in argv if not x.startswith('--')]
    if len(paths) != 2:
        print(__doc__)
        return 2
    o = diff(*paths)
    print(f'A = {paths[0]}\nB = {paths[1]}')
    print('分段（A → B，字节）：')
    for k, (x, y) in o['sections'].items():
        if x or y:
            print(f'  {k:16s} {x:>14,} → {y:>14,}  {y - x:+,}')
    for k in o['only_a']:
        print('  KV 只在 A：', k)
    for k in o['only_b']:
        print('  KV 只在 B：', k)
    for k, x, y in o['changed']:
        print(f'  KV 变了：{k}  {x} → {y}')
    for n in o['tensors_only_a']:
        print('  张量只在 A：', n)
    for n in o['tensors_only_b']:
        print('  张量只在 B：', n)
    for n, x, y in o['reshaped']:
        print(f'  形状变了：{n} {x} → {y}')
    pairs = {}
    for n, x, y, nx, ny in o['retyped']:
        p = pairs.setdefault((x, y), [0, 0, 0])
        p[0] += 1
        p[1] += nx
        p[2] += ny
    for (x, y), (c, nx, ny) in sorted(pairs.items(), key=lambda z: -z[1][0]):
        print(f'  类型 {x} → {y}：{c} 个张量，{nx / 2**20:.1f} → {ny / 2**20:.1f} MiB')
    if '--types' in argv:
        for n, x, y, _, _ in o['retyped']:
            print(f'    {n:32s} {x} → {y}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
