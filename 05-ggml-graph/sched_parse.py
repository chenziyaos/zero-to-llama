"""统计 GGML_SCHED_DEBUG=2 日志里每张图的节点。

用法：python3 sched_parse.py sched.log[.gz]
调度器只打印需要计算的节点，视图节点（VIEW、RESHAPE、PERMUTE、TRANSPOSE）的序号会被跳过，
所以 max idx + 1 是图的总节点数，listed 是真正干活的节点数。
日志末尾那张图可能被截断，这里取打印行数最多的一张。
"""
import collections
import gzip
import re
import sys


def load(path):
    op = gzip.open if path.endswith('.gz') else open
    with op(path, 'rt', errors='replace') as f:
        return f.read()


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    parts = load(sys.argv[1]).split('## SPLIT #0')[1:]
    print('graphs printed', len(parts))
    if not parts:
        return
    pat = re.compile(r'node #\s*(\d+) \(\s*([A-Z_]+)\)')
    ops = max((pat.findall(p) for p in parts), key=len)
    cnt = collections.Counter(o for _, o in ops)
    mx = max(int(i) for i, _ in ops)
    print(f'largest graph: listed {len(ops)}, max idx {mx}, total nodes {mx + 1}, views {mx + 1 - len(ops)}')
    print('  '.join(f'{k} {v}' for k, v in cnt.most_common()))


if __name__ == '__main__':
    main()
