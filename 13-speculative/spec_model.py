"""投机解码的期望收益模型，和 llama.cpp 的统计口径对齐。

用法：
  python3 spec_model.py table                 # 打印 alpha × gamma 的期望接受长度表
  python3 spec_model.py fit LOGDIR            # 从 exp13.sh 的日志读出 alpha，按实测代价曲线预测加速比，和实测对比
  python3 spec_model.py synth L K             # 复算 --spec-synth-len：给定平均接受长度 L 和 n_max K，反解每位置接受率 p
  python3 spec_model.py temp                  # 合成分布上比较两种验证：草稿取 argmax 再比对，和草稿采样再做拒绝采样

公式出处：Leviathan 等人 2023 的式 (1)：每轮期望产出 E = (1 - a^(g+1)) / (1 - a)，
其中 a 是每个草稿 token 被接受的概率（假设各位置独立同分布），g 是草稿长度。
加速比 = E / (g * c + v(g))，c 是草稿一次前向和目标一次前向的耗时比，
v(g) 是目标一次喂 g+1 个 token 和喂 1 个 token 的耗时比。论文假设 v(g) = 1，CPU 上不成立，这里用实测。
"""
import sys, os, re, glob, json, csv, math, random


def expected_len(a, g):
    """一轮验证的期望产出 token 数，含验证时目标自己多采的那 1 个。"""
    if a >= 1.0:
        return g + 1.0
    return (1.0 - a ** (g + 1)) / (1.0 - a)


def speedup(a, g, c, v=1.0):
    return expected_len(a, g) / (g * c + v)


def synth_p(L, K):
    """和 common_speculative_synth_rates_resolve 同样的 32 次二分：p + p^2 + ... + p^K = L - 1。"""
    if L >= K + 1:
        return 1.0
    if L <= 1:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        s = sum(mid ** j for j in range(1, K + 1))
        if s < L - 1:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def alpha_from_mean_len(L, K):
    """反解 Leviathan 模型里的 a：E(a, K) = L。E 对 a 单调，二分即可。"""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if expected_len(mid, K) < L:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------- 读日志 ----------

def read_cost(logdir, model):
    """llama-bench -p 1,2,3,5,9,17：一次喂 n 个 token 的耗时（ms）。"""
    f = os.path.join(logdir, f'cost-{model}.csv')
    out = {}
    with open(f) as fh:
        for r in csv.DictReader(fh):
            if r['n_gen'] == '0':
                out[int(r['n_prompt'])] = float(r['avg_ns']) / 1e6
            else:
                out['tg'] = float(r['avg_ns']) / 1e6 / int(r['n_gen'])
    return out


def interp(cost, n):
    ks = sorted(k for k in cost if isinstance(k, int))
    if n in cost:
        return cost[n]
    for a, b in zip(ks, ks[1:]):
        if a < n < b:
            return cost[a] + (cost[b] - cost[a]) * (n - a) / (b - a)
    return cost[ks[-1]] * n / ks[-1]


def parse_spec(path):
    t = open(path, errors='replace').read()
    d = {}
    m = re.search(r'decoded\s+(\d+) tokens in\s+([\d.]+) seconds, speed:\s+([\d.]+)', t)
    if m:
        d['n_dec'], d['sec'], d['tps'] = int(m.group(1)), float(m.group(2)), float(m.group(3))
    for k in ('n_drafted', 'n_accept', 'n_draft'):
        m = re.search(rf'{k}\s+= (\d+)', t)
        if m:
            d[k] = int(m.group(1))
    m = re.search(r'#gen drafts =\s+(\d+), #acc drafts =\s+(\d+), #gen tokens =\s+(\d+), #acc tokens =\s+(\d+), #mean acc len = ([\d.]+), #acc rate/pos = \(([^)]*)\)', t)
    if m:
        d['gen_drafts'], d['acc_drafts'], d['gen_tok'], d['acc_tok'] = map(int, m.group(1, 2, 3, 4))
        d['mean_len'] = float(m.group(5))
        d['rate_pos'] = [float(x) for x in m.group(6).split(',')]
    return d


def parse_base(path):
    t = open(path, errors='replace').read()
    m = re.search(r'\beval time =\s+([\d.]+) ms /\s+(\d+) runs\s+\(\s*([\d.]+) ms per token,\s+([\d.]+) tokens per second', t)
    return {'ms': float(m.group(1)), 'runs': int(m.group(2)), 'ms_tok': float(m.group(3)), 'tps': float(m.group(4))} if m else {}


def fit(logdir):
    rows = []
    for f in sorted(glob.glob(os.path.join(logdir, 'draft-*.err'))):
        name = os.path.basename(f)[6:-4]
        m = re.match(r'(.+?)-(q-Q4_0|q-Q2_K|qi-IQ2_M)-k(\d+)-(\w+)$', name)
        if not m:
            continue
        tgt, dft, k, p = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        s = parse_spec(f)
        b = parse_base(os.path.join(logdir, f'base-{tgt}-{p}.err'))
        if 'mean_len' not in s or not b:
            continue
        ct, cd = read_cost(logdir, tgt), read_cost(logdir, dft)
        c = cd[1] / ct[1]
        v = interp(ct, k + 1) / ct[1]
        a = alpha_from_mean_len(s['mean_len'], k)
        rows.append(dict(tgt=tgt, dft=dft, k=k, prompt=p, mean_len=s['mean_len'], alpha=round(a, 3),
                         c=round(c, 3), v=round(v, 3),
                         pred_ideal=round(speedup(a, k, c, 1.0), 3),
                         pred=round(speedup(a, k, c, v), 3),
                         meas=round(s['tps'] / b['tps'], 3), tps=s['tps'], base_tps=b['tps']))
    return rows


# ---------- 两种验证在温度下的接受率（合成分布） ----------

def softmax(z, T):
    m = max(z)
    e = [math.exp((x - m) / T) for x in z]
    s = sum(e)
    return [x / s for x in e]


def temp_curve(n_vocab=200, noise=1.0, trials=400, seed=7):
    """目标 logits z 取标准正态乘 3，草稿 logits = z + 噪声。
    greedy：草稿给 argmax q，目标采样，接受概率 = p[argmax q]（逐个比对时的期望）。
    rejection：草稿按 q 采样，接受概率 = sum min(p, q) = 1 - TV(p, q)。"""
    rnd = random.Random(seed)
    out = []
    for T in (0.05, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2):
        g = r = 0.0
        for _ in range(trials):
            z = [rnd.gauss(0, 3) for _ in range(n_vocab)]
            zq = [x + rnd.gauss(0, noise) for x in z]
            p, q = softmax(z, T), softmax(zq, T)
            am = max(range(n_vocab), key=lambda i: q[i])
            g += p[am]
            r += sum(min(a, b) for a, b in zip(p, q))
        out.append((T, round(g / trials, 3), round(r / trials, 3)))
    return out


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'table'
    if cmd == 'table':
        gs = [1, 2, 4, 8, 16]
        print('alpha ' + ' '.join(f'g={g:<5d}' for g in gs))
        for a in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            print(f'{a:<5} ' + ' '.join(f'{expected_len(a, g):7.2f}' for g in gs))
    elif cmd == 'fit':
        rows = fit(sys.argv[2])
        print(json.dumps(rows, ensure_ascii=False, indent=0))
    elif cmd == 'synth':
        L, K = float(sys.argv[2]), int(sys.argv[3])
        p = synth_p(L, K)
        print(f'p = {p:.4f}, rates = {[round(p ** (i + 1), 4) for i in range(K)]}, alpha(Leviathan) = {alpha_from_mean_len(L, K):.4f}')
    elif cmd == 'temp':
        for T, g, r in temp_curve():
            print(f'T={T:<4} greedy-match={g:.3f} rejection={r:.3f}')


if __name__ == '__main__':
    main()
