"""解析 test-backend-ops 和 ctest 的日志，给 test_test_contrib.py 和正文的数字用。

用法：python3 parse_tbo.py logs/bug-RMS_NORM_MUL_ADD.log
"""
import re
import sys

ANSI = re.compile(r'\x1b\[[0-9;]*m')
CASE = re.compile(r'^\s*(?:\[(\w+)\] ERR = ([\d.eE+-]+) > ([\d.eE+-]+)\s+)?(\w+)\((.*)\): (OK|FAIL|not supported.*)$')


def clean(text):
    return ANSI.sub('', text)


def cases(text):
    """每个用例一条：op、参数串、ne、结果；失败的再带出错节点、误差和阈值。"""
    out = []
    for line in clean(text).splitlines():
        m = CASE.match(line)
        if not m:
            continue
        node, err, thr, op, params, res = m.groups()
        ne = re.search(r'ne=\[([\d,]+)\]', params)
        out.append({
            'op': op, 'params': params, 'result': res.split()[0] if res.startswith('not') is False else 'unsupported',
            'ne': [int(x) for x in ne.group(1).split(',')] if ne else None,
            'node': node, 'err': float(err) if err else None, 'thr': float(thr) if thr else None,
        })
    return out


def summary(text):
    """“N/M tests passed”和 runlog.py 的尾行。"""
    t = clean(text)
    m = re.search(r'(\d+)/(\d+) tests passed', t)
    r = re.search(r'== rc=(\d+) wall_s=([\d.]+) maxrss_mib=(\d+)', t)
    return {
        'passed': int(m.group(1)) if m else None, 'total': int(m.group(2)) if m else None,
        'rc': int(r.group(1)) if r else None, 'wall_s': float(r.group(2)) if r else None,
        'maxrss_mib': int(r.group(3)) if r else None,
    }


def ctest(text):
    """ctest 的逐个测试结果和总耗时。"""
    t = clean(text)
    tests = [(m.group(1), m.group(2), float(m.group(3)))
             for m in re.finditer(r'Test +#\d+: (\S+) \.+\s*(\*{0,3}\w+[\w ]*?)\s+([\d.]+) sec', t)]
    tot = re.search(r'Total Test time \(real\) =\s+([\d.]+) sec', t)
    pct = re.search(r'(\d+)% tests passed, (\d+) tests failed out of (\d+)', t)
    return {
        'tests': tests, 'real_s': float(tot.group(1)) if tot else None,
        'failed': int(pct.group(2)) if pct else None, 'n': int(pct.group(3)) if pct else None,
    }


def params_dict(params):
    """'ne=[1,2],eps=0.1,mul=1' 拆成字典，值保留字符串。"""
    return dict(x.split('=', 1) for x in re.findall(r'(\w+=(?:\[[^\]]*\]|[^,]+))', params))


def fusable(case):
    """CPU 后端会不会把 RMS_NORM 和后面的 MUL 融合。

    ggml_cpu_try_fuse_ops 要求权重的 ne[0] 等于 rms_norm 输出的 ne[0]。
    RMS_NORM_MUL_ADD 的 broadcast=1 把输入 a 放大成 [2*ne0, ...]，权重 b 还是 ne0 宽，条件不成立，两个算子分开算。
    RMS_NORM_MUL_ROPE 只要 mul=1 就有 RMS_NORM 加 MUL，broadcast=1 是一维权重，正好是模型里的写法。
    """
    p = params_dict(case['params'])
    if case['op'] == 'RMS_NORM_MUL_ADD':
        return p.get('broadcast') == '0'
    if case['op'] == 'RMS_NORM_MUL_ROPE':
        return p.get('mul') == '1'
    return False


def tail_bug_predicts(ne0, block=16, fused=True):
    """fused-tail.patch 只在融合路径里算到 ne0 向下取 16 的倍数，余下的列没写。
    融合了、且 ne0 不是 16 的倍数才会错。"""
    return fused and ne0 % block != 0


if __name__ == '__main__':
    txt = open(sys.argv[1], errors='replace').read()
    print(summary(txt))
    for c in cases(txt):
        if c['result'] == 'FAIL':
            print(c['node'], c['err'], c['ne'])
