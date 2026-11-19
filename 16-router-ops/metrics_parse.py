"""解析 llama-server /metrics 的 Prometheus 文本，按 llama.cpp 0ec4027 的口径算派生指标。

对应源码：
  tools/server/server-task.cpp   server_task_result_metrics::to_metrics：计数器、仪表和带 position 标签的计数器
  tools/server/server-common.h   server_metrics::bucket：速率用 decode 步数（首 token 不算），bucket 每次抓取后清零
  tools/server/server-context.cpp metrics_on_prediction：一条请求的生成统计在槽位 reset 时才并进 bucket

这里不起服务。Emu 按上面三条规则重放一串请求，渲染出和服务端同格式的文本，再用 parse/derive 读回来。
速度取 14-server/logs/bb-kvu.jsonl 的实测（同一上下文里 B 路并发时每步的总吞吐），没有就用缺省值。
用法：python3 metrics_parse.py [--sample 输出文件] [14-server 的 logs 目录]
"""
import json
import os
import re
import sys

LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)$')

COUNTERS = [
    ('prompt_tokens_total', 'Number of prompt tokens processed, excluding cached tokens'),
    ('prompt_tokens_cached_total', 'Number of prompt tokens reused from the cache'),
    ('prompt_seconds_total', 'Total time spent processing prompts'),
    ('tokens_predicted_total', 'Number of generation tokens processed'),
    ('tokens_predicted_seconds_total', 'Total time spent generating tokens'),
    ('n_decode_total', 'Total number of llama_decode() calls, excluding speculative decoding and multimodal decoding'),
    ('n_tokens_max', 'Largest observed sequence length (prompt + generation)'),
    ('spec_decode_num_draft_tokens_total', 'Speculative: Total draft tokens generated'),
    ('spec_decode_num_accepted_tokens_total', 'Speculative: Total draft tokens accepted by the target model'),
    ('spec_decode_num_drafts_total', 'Speculative: Total speculative decoding verification steps'),
]
GAUGES = [
    ('prompt_tokens_seconds', 'Average prompt throughput in tokens/s'),
    ('predicted_tokens_seconds', 'Average generation throughput in tokens/s'),
    ('requests_processing', 'Number of requests processing'),
    ('requests_deferred', 'Number of requests deferred'),
    ('n_busy_slots_per_decode', 'Average number of busy slots per llama_decode() call'),
]


def parse(text):
    """返回 {名字（去掉 llamacpp: 前缀）: 值}，带标签的放成 {名字: {标签串: 值}}；另返回类型表。"""
    vals, types = {}, {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('# TYPE '):
            _, _, n, t = line.split(None, 3)
            types[n.split(':', 1)[-1]] = t
            continue
        if line.startswith('#'):
            continue
        m = LINE.match(line)
        if not m:
            raise ValueError(f'bad metrics line: {line}')
        name = m.group(1).split(':', 1)[-1]
        v = float(m.group(3))
        if m.group(2):
            vals.setdefault(name, {})[m.group(2)] = v
        else:
            vals[name] = v
    return vals, types


def derive(a, b, dt):
    """两次抓取之间的派生量。a、b 是 parse 的结果，dt 是两次抓取的墙钟间隔（秒）。"""
    d = lambda k: b.get(k, 0.0) - a.get(k, 0.0)
    gen, gen_s = d('tokens_predicted_total'), d('tokens_predicted_seconds_total')
    pp, pp_s, cached = d('prompt_tokens_total'), d('prompt_seconds_total'), d('prompt_tokens_cached_total')
    nd = d('n_decode_total')
    drafts, acc = d('spec_decode_num_draft_tokens_total'), d('spec_decode_num_accepted_tokens_total')
    return {
        'gen_tps_wall': gen / dt if dt > 0 else 0.0,            # 整机生成吞吐：这才是容量
        'gen_tps_per_req': gen / gen_s if gen_s > 0 else 0.0,   # 每条请求自己看到的速度（含首 token）
        'gen_busy': gen_s / dt if dt > 0 else 0.0,              # 生成时间之和除以墙钟，约等于平均并发
        'prompt_tps_wall': pp / dt if dt > 0 else 0.0,
        'cache_hit': cached / (cached + pp) if cached + pp > 0 else 0.0,
        'decode_per_s': nd / dt if dt > 0 else 0.0,
        'draft_accept': acc / drafts if drafts > 0 else 0.0,
    }


def render(counters, gauges, per_pos=None):
    """按 to_metrics 的格式出文本。"""
    out = []
    for kind, items, vals in (('counter', COUNTERS, counters), ('gauge', GAUGES, gauges)):
        for n, desc in items:
            v = vals.get(n, 0)
            out.append(f'# HELP llamacpp:{n} {desc}')
            out.append(f'# TYPE llamacpp:{n} {kind}')
            out.append(f'llamacpp:{n} {fmt(v)}')
    if per_pos:
        out.append('# HELP llamacpp:spec_decode_num_accepted_tokens_per_pos_total Accepted tokens per draft position')
        out.append('# TYPE llamacpp:spec_decode_num_accepted_tokens_per_pos_total counter')
        for i, v in enumerate(per_pos):
            out.append(f'llamacpp:spec_decode_num_accepted_tokens_per_pos_total{{position="{i}"}} {v}')
    return '\n'.join(out) + '\n'


def fmt(v):
    # std::stringstream 的缺省格式：6 位有效数字
    return f'{v:.6g}'


class Emu:
    """按 server_metrics 的规则累计：cumulative 一直加，bucket 在每次抓取后清零；
    生成统计在请求结束（槽位 reset）时一次并入，prompt 统计在预填充结束时并入。"""

    def __init__(self):
        self.c = {n: 0.0 for n, _ in COUNTERS}
        self.bp = [0, 0.0]     # prompt bucket：步数（=token 数）、秒
        self.bg = [0, 0.0]     # predict bucket：步数（n_gen-1）、秒
        self.busy = 0
        self.ev = []           # (t, kind, data)

    def request(self, t0, n_prompt, n_cached, t_prompt, n_gen, t_step):
        """一条请求：t0 开始，预填充 t_prompt 秒，然后每步 t_step 秒出一个 token（首 token 免费）。"""
        t1 = t0 + t_prompt
        t2 = t1 + (n_gen - 1) * t_step
        self.ev.append((t1, 'prompt', (n_prompt, n_cached, t_prompt)))
        self.ev.append((t2, 'done', (n_gen, t2 - t1, n_prompt + n_gen)))
        return t2

    def scrape_at(self, times):
        out = []
        evs = sorted(self.ev)
        i = 0
        for ts in times:
            while i < len(evs) and evs[i][0] <= ts:
                _, k, d = evs[i]
                if k == 'prompt':
                    n, nc, tp = d
                    self.c['prompt_tokens_total'] += n
                    self.c['prompt_tokens_cached_total'] += nc
                    self.c['prompt_seconds_total'] += tp
                    self.bp[0] += n
                    self.bp[1] += tp
                else:
                    n, tg, ntok = d
                    self.c['tokens_predicted_total'] += n
                    self.c['tokens_predicted_seconds_total'] += tg
                    self.c['n_tokens_max'] = max(self.c['n_tokens_max'], ntok)
                    self.bg[0] += n - 1
                    self.bg[1] += tg
                i += 1
            g = {
                'prompt_tokens_seconds': self.bp[0] / self.bp[1] if self.bp[1] > 0 else 0.0,
                'predicted_tokens_seconds': self.bg[0] / self.bg[1] if self.bg[1] > 0 else 0.0,
            }
            out.append((ts, render(dict(self.c), g)))
            self.bp, self.bg = [0, 0.0], [0, 0.0]     # GET /metrics 默认带 reset_bucket
        return out


def bb_speeds(d):
    """pl 路并发时的总生成速度 t/s 和单路预填充速度。"""
    tg = {1: 53.7, 2: 60.8, 4: 76.5}
    pp = 319.2
    p = os.path.join(d or '', 'bb-kvu.jsonl')
    if d and os.path.exists(p):
        for line in open(p):
            if line.startswith('{'):
                j = json.loads(line)
                if j['pp'] == 128:
                    tg[j['pl']] = j['speed_tg']
                    if j['pl'] == 1:
                        pp = j['speed_pp']
    return tg, pp


def demo(d):
    tg, pp = bb_speeds(d)
    rows = []
    # 同样 15 s 的窗口，分别 1 路和 4 路并发，每路连续发 128 token 提示、生成 128 token 的请求
    for pl in (1, 4):
        e = Emu()
        step = pl / tg[pl]                     # 每路每步的耗时
        for s in range(pl):
            t = 0.0
            while t < 60:
                t = e.request(t, 128, 0, 128 / pp * pl, 128, step)
        sc = e.scrape_at([15.0, 30.0, 45.0])
        a, _ = parse(sc[1][1])
        b, _ = parse(sc[2][1])
        dv = derive(a, b, 15.0)
        rows.append(('conc', pl, round(b['predicted_tokens_seconds'], 1), round(dv['gen_tps_wall'], 1),
                     round(dv['gen_tps_per_req'], 1), round(dv['gen_busy'], 2)))
    # 一条长请求跨过抓取点：生成 2000 token，抓取间隔 5 s
    e = Emu()
    e.request(0.0, 128, 0, 128 / pp, 2000, 1 / tg[1])
    sc = e.scrape_at([5.0 * k for k in range(1, 10)])
    long_rows = []
    for ts, txt in sc:
        v, _ = parse(txt)
        r = round(v['predicted_tokens_seconds'], 1)
        long_rows.append((ts, r, int(v['tokens_predicted_total'])))
    return rows, long_rows, sc[-1][1]


def main():
    args = sys.argv[1:]
    sample = None
    if '--sample' in args:
        i = args.index('--sample')
        sample = args[i + 1]
        del args[i:i + 2]
    d = args[0] if args else None
    rows, long_rows, txt = demo(d)
    print('# conc  pl gauge_predicted_tps wall_tps per_req_tps busy')
    for r in rows:
        print(*r)
    print('\n# long  t_scrape gauge_predicted_tps tokens_predicted_total')
    for r in long_rows:
        print('long', *r)
    if sample:
        open(sample, 'w').write(txt)
        v, t = parse(txt)
        print('\nsample', sample, len(v), 'series', sum(1 for x in t.values() if x == 'counter'), 'counters')


if __name__ == '__main__':
    main()
