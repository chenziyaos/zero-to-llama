"""按 llama.cpp 0ec4027 的服务端源码复刻槽位选择、提示缓存和 KV 账本。

对应源码：
  tools/server/server-context.cpp  get_available_slot、get_free_slots、prompt_save、prompt_load
  tools/server/server-task.cpp     server_prompt_cache::alloc、load、update
  src/llama-context.cpp            n_ctx 和 n_ctx_seq 的取整

不启动服务，只在内存里跑一份多轮对话的请求流，数每个请求要重新算多少提示 token。
"""
import random

KIB = 1024
MIB = 1024 * 1024
# Qwen3-0.6B：28 层，8 个 KV 头，头维 128，K 和 V 各一份，f16
KV_BYTES_PER_TOKEN = 28 * 8 * 128 * 2 * 2


def pad(x, n):
    return (x + n - 1) // n * n


def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


# ---------------- KV 账本 ----------------

def ctx_layout(n_ctx, n_seq, unified):
    """llama-context.cpp 的取整：先把 n_ctx 补到 256 的倍数；不统一时每条序列 n_ctx/n_seq 再补到 256，总量随之改写。"""
    n_ctx = pad(n_ctx, 256)
    if unified:
        return n_ctx, n_ctx
    seq = pad(n_ctx // n_seq, 256)
    if seq == 0:
        raise ValueError('n_ctx_seq == 0')
    return seq * n_seq, seq


def kv_mib(n_cells, bpt=KV_BYTES_PER_TOKEN):
    return n_cells * bpt / MIB


# ---------------- 提示缓存 ----------------

class PromptCache:
    """server_prompt_cache：一条 list，旧的在前。limit_size 单位字节，0 表示不限；limit_tokens 取 n_ctx。"""

    def __init__(self, limit_mib, limit_tokens, bpt=KV_BYTES_PER_TOKEN):
        self.limit_size = int(limit_mib * MIB) if limit_mib > 0 else 0
        self.limit_tokens = limit_tokens
        self.bpt = bpt
        self.states = []          # 每项是 token 元组
        self.evicted = 0

    def size(self):
        return sum(len(s) for s in self.states) * self.bpt

    def n_tokens(self):
        return sum(len(s) for s in self.states)

    def alloc(self, tokens):
        for s in self.states:
            if lcp(s, tokens) == len(tokens):
                return False                      # 已经整段在缓存里
        new = len(tokens) * self.bpt
        if self.limit_size > 0 and new > self.limit_size:
            return False                          # 单条超限，不进缓存
        self.states = [s for s in self.states if lcp(s, tokens) != len(s)]   # 被新条目完全包含的旧条目删掉
        if self.limit_size > 0:
            while self.states and self.size() + new > self.limit_size:
                self.states.pop(0)
                self.evicted += 1
        self.states.append(tuple(tokens))
        return True

    def load(self, cur, new):
        """返回换进槽位的 token；没有更好的就原样返回 cur。"""
        l0 = lcp(cur, new)
        keep_best = l0 / len(cur) if cur else -1.0
        sim_best = l0 / len(new)
        best = None
        for i, s in enumerate(self.states):
            l = lcp(s, new)
            keep, sim = l / len(s), l / len(new)
            if keep < 0.25:
                continue                          # 不拿大提示去换一小段前缀
            if keep_best < keep and sim_best < sim:
                keep_best, sim_best, best = keep, sim, i
        if best is None:
            return cur
        return self.states.pop(best)

    def update(self):
        if self.limit_size > 0:
            while self.states and self.size() > self.limit_size:
                self.states.pop(0)
                self.evicted += 1
        spt = max(1.0, self.size() / max(1, self.n_tokens()))
        cur = max(self.limit_tokens, int(self.limit_size / spt)) if self.limit_size > 0 else self.limit_tokens
        if self.limit_tokens > 0:
            while self.states and self.n_tokens() > cur:
                self.states.pop(0)
                self.evicted += 1


# ---------------- 槽位 ----------------

class Slot:
    def __init__(self, i):
        self.id = i
        self.tokens = ()
        self.t_last = 0
        self.busy = False


def get_available_slot(slots, task, sps, cache):
    """返回 (槽位, 选法)。选法是 lcp、lru 或 None。"""
    ret, how, update = None, None, False
    if sps != 0.0:
        best = 0.0
        for s in slots:
            if s.busy or not s.tokens:
                continue
            f = lcp(s.tokens, task) / len(task)
            if f > best and f > sps:
                best, ret = f, s
        if ret is not None:
            how = 'lcp'
            f_keep = best * len(task) / len(ret.tokens)
            if f_keep < 0.5:
                update = True
    if ret is None:
        free = [s for s in slots if not s.busy]
        free.sort(key=lambda s: (len(s.tokens) != 0, s.t_last))   # 空槽优先，再按最久未用
        if free:
            ret, how, update = free[0], 'lru', True
    if ret is not None and update and cache is not None:
        if ret.tokens:
            cache.alloc(ret.tokens)
        ret.tokens = cache.load(ret.tokens, task)
        cache.update()
    return ret, how


# ---------------- 请求流 ----------------

def make_sessions(n_users, n_turns, sys_len, seed):
    """每个用户一段多轮对话：共享系统提示 + 逐轮追加的用户消息和回复。token 用整数编号，互不重复。"""
    rng = random.Random(seed)
    nxt = [10_000_000]

    def fresh(n):
        out = tuple(range(nxt[0], nxt[0] + n))
        nxt[0] += n
        return out

    system = tuple(range(sys_len))
    sessions = []
    for _ in range(n_users):
        turns = []
        for _ in range(n_turns):
            turns.append((fresh(rng.randint(40, 80)), fresh(rng.randint(100, 200))))
        sessions.append(turns)
    return system, sessions


def simulate(n_slots, n_users, n_turns, sys_len, sps, cache_mib, n_ctx, seed=1):
    """请求按波次到达，每波最多 n_slots 个不同用户；同一波里的请求同时占用槽位，波次结束后全部空闲。"""
    system, sessions = make_sessions(n_users, n_turns, sys_len, seed)
    rng = random.Random(seed + 1)
    slots = [Slot(i) for i in range(n_slots)]
    cache = PromptCache(cache_mib, n_ctx) if cache_mib != 0 else None
    hist = [system for _ in range(n_users)]
    pos = [0] * n_users
    t = 0
    tot = rec = 0
    hows = {'lcp': 0, 'lru': 0}
    while True:
        pend = [u for u in range(n_users) if pos[u] < n_turns]
        if not pend:
            break
        wave = rng.sample(pend, min(n_slots, len(pend)))
        for u in wave:
            msg, reply = sessions[u][pos[u]]
            task = hist[u] + msg
            s, how = get_available_slot(slots, task, sps, cache)
            hows[how] += 1
            keep = lcp(s.tokens, task)
            tot += len(task)
            rec += len(task) - keep
            s.tokens = task + reply      # 生成结束后槽位里留着提示加回复
            s.busy = True
            t += 1
            s.t_last = t
            hist[u] = task + reply
            pos[u] += 1
        for s in slots:
            s.busy = False
    return {
        'prompt_tokens': tot,
        'recomputed': rec,
        'reuse': 1 - rec / tot,
        'lcp': hows['lcp'],
        'lru': hows['lru'],
        'cached_prompts': len(cache.states) if cache else 0,
        'cache_mib': cache.size() / MIB if cache else 0.0,
        'evicted': cache.evicted if cache else 0,
    }


def main():
    print('# KV 账本：Qwen3-0.6B f16，每 token %d 字节 = %.0f KiB' % (KV_BYTES_PER_TOKEN, KV_BYTES_PER_TOKEN / KIB))
    print('ledger n_ctx np unified n_ctx_eff n_ctx_seq kv_mib')
    for n_ctx in (4096, 16384, 32768):
        for np_ in (1, 4, 8, 16):
            for uni in (0, 1):
                tot, seq = ctx_layout(n_ctx, np_, uni)
                print('ledger %d %d %d %d %d %.1f' % (n_ctx, np_, uni, tot, seq, kv_mib(tot)))
    for n_ctx, np_ in ((10000, 3), (5000, 16)):
        tot, seq = ctx_layout(n_ctx, np_, 0)
        print('ledger %d %d 0 %d %d %.1f' % (n_ctx, np_, tot, seq, kv_mib(tot)))
    print()
    print('# 提示缓存能装多少 token')
    for mib in (256, 1024, 8192):
        print('cachecap %d %d' % (mib, mib * MIB // KV_BYTES_PER_TOKEN))
    print()
    print('# 多轮对话：16 用户 × 6 轮，系统提示 300 token，4 个槽位，n_ctx 16384（缓存 token 上限取它）')
    print('sim sps cache_mib prompt_tokens recomputed reuse lcp lru cached_prompts cache_mib_used evicted')
    for sps in (0.0, 0.1, 0.5):
        for mib in (0, 256, 1024, 8192):
            r = simulate(4, 16, 6, 300, sps, mib, 16384)
            print('sim %.1f %d %d %d %.3f %d %d %d %.1f %d' % (
                sps, mib, r['prompt_tokens'], r['recomputed'], r['reuse'], r['lcp'], r['lru'],
                r['cached_prompts'], r['cache_mib'], r['evicted']))
    print()
    print('# 用户数对复用率的影响：sps 0.1，cache 8192 MiB 对比 0')
    print('users n_users cache_mib reuse evicted')
    for nu in (2, 4, 8, 16, 32):
        for mib in (0, 8192):
            r = simulate(4, nu, 6, 300, 0.1, mib, 16384)
            print('users %d %d %.3f %d' % (nu, mib, r['reuse'], r['evicted']))


if __name__ == '__main__':
    main()
