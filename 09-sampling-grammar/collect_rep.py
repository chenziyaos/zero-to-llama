"""统计重复程度：重复的 4-gram 占比、重复的行数、最长的重复片段。

用法：python3 collect_rep.py rep-*.txt
按空白切词，只看生成部分（exp09.sh 用 --no-display-prompt）。
"""
import os
import sys
from collections import Counter


def ngram_rep(words, n=4):
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return 0.0
    c = Counter(grams)
    dup = sum(v for v in c.values() if v > 1)
    return dup / len(grams)


def longest_repeat(words, max_n=200):
    best = 0
    for n in range(2, max_n + 1):
        seen = set()
        hit = False
        for i in range(len(words) - n + 1):
            g = tuple(words[i:i + n])
            if g in seen:
                hit = True
                break
            seen.add(g)
        if hit:
            best = n
        else:
            break
    return best


def main():
    print(f'{"config":<12}{"words":>7}{"rep4":>8}{"dup_lines":>11}{"longest":>9}')
    for fn in sys.argv[1:]:
        t = open(fn, errors='ignore').read()
        w = t.split()
        lines = [l.strip() for l in t.splitlines() if l.strip()]
        dl = len(lines) - len(set(lines))
        name = os.path.basename(fn)[4:-4]
        print(f'{name:<12}{len(w):>7}{ngram_rep(w):>8.1%}{dl:>11}{longest_repeat(w):>9}')


if __name__ == '__main__':
    main()
