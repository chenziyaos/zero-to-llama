"""把 llama-bench 或 test-backend-ops 的 --output sql 输出灌进 SQLite。

机器上没有 sqlite3 命令行，Python 自带的 sqlite3 模块就够用。
--label 把这批记录的 build_commit 改成一个固定标签，
这样同一个 commit 的两种构建（或两种环境变量）也能交给 compare-llama-bench.py 用 -b、-c 比较。
标签长度要一致，compare-llama-bench.py 遇到长度不同的 build_commit 会按最短的截断。

用法：
    ./bin/test-backend-ops perf -b CPU -o MUL_MAT --output sql > a.sql
    python3 load_sql.py out.sqlite a.sql --label fuse1
"""
import argparse
import sqlite3
import sys


def load(db, sql_text, label=None):
    con = sqlite3.connect(db)
    cur = con.cursor()
    tables_before = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    counts_before = {t: cur.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0] for t in tables_before}
    cur.executescript(sql_text)
    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    added = {}
    for t in tables:
        n0 = counts_before.get(t, 0)
        n1 = cur.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
        added[t] = n1 - n0
        if label is not None and n1 > n0:
            # 新插入的行 rowid 最大，只改这一批
            cur.execute(f'UPDATE {t} SET build_commit = ? WHERE rowid > (SELECT MAX(rowid) FROM {t}) - ?', (label, n1 - n0))
    con.commit()
    con.close()
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('db')
    ap.add_argument('sql', nargs='?', default='-')
    ap.add_argument('--label', default=None)
    a = ap.parse_args()
    text = sys.stdin.read() if a.sql == '-' else open(a.sql).read()
    added = load(a.db, text, a.label)
    for t, n in added.items():
        print(f'{t}: +{n} rows' + (f' as {a.label}' if a.label else ''))


if __name__ == '__main__':
    main()
