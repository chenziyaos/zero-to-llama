"""按 llama.cpp 0ec4027 的 tools/server/server-models-sched.cpp 复刻 router 的排队、LRU 卸载和内存余量。

对应源码：
  has_capacity / pick_victim / join / leave / try_claim / tick   请求排队和按 models_max 卸载
  limits_exceeded / make_room                                   加载前腾地方
  calibrate / reserve / release / available / deficit           设备内存账，单位这里一律用 MiB
  server-models.cpp ensure_model_ready / load / proxy_request   排队、加载、req_count 和 last_used

不起服务，在一条离散时间线上跑请求。时间单位是秒。加载耗时取 logs/load.txt 的实测（没有就用缺省值），
模型的内存取 logs/fit_print.txt 的实测（-c 4096 的 min）。GPU 场景里的设备是假设的，只用来走通余量逻辑。
用法：python3 router_sim.py [logs 目录]
"""
import heapq
import os
import re
import sys

STOP_S = 0.5          # 子进程收到退出信号到真正退出；源码里 stop-timeout 是强杀前的上限，缺省 10 s


class Model:
    def __init__(self, name, load_s, min_mem, actual=None, warm_s=None):
        self.name = name
        self.status = 'unloaded'      # unloaded / loading / loaded / sleeping
        self.last_used = 0.0
        self.req = 0
        self.load_s = load_s          # 冷加载
        self.warm_s = warm_s or load_s
        self.loaded_once = False
        self.min = dict(min_mem)      # 测量子进程报的 min：{设备: MiB}
        self.actual = dict(actual or min_mem)   # 实例真实占用，含测量看不到的部分（mmproj）
        self.eff = None               # on_ready 报上来的 effective
        self.n_loads = 0
        self.unseen = 0               # 实例占了但 llama_get_memory_breakdown 不报的部分（mmproj）
        self.fit_max = None           # 不钉 -c 时 --fit 最多吃到多少；None 表示上下文已钉死

    def running(self):
        return self.status in ('loaded', 'loading', 'sleeping')

    def ready_or_sleep(self):
        return self.status in ('loaded', 'sleeping')


class Router:
    def __init__(self, models, models_max=4, margin=0, devices=None, external=None):
        self.m = {x.name: x for x in models}
        self.models_max = models_max
        self.margin = margin                  # --models-memory-margin，MiB，0 表示关
        self.devices = devices or {}          # 设备名 -> 总量 MiB；空字典表示纯 CPU，只有 host
        self.ext_true = dict(external or {})  # 真实的外部占用（别的程序）
        self.dm = {}                          # router 记的 {设备: [total, external]}
        self.held = {}                        # {模型: {设备: MiB}}
        self.queue = []                       # [[model, n_waiters, loading]]
        self.stopping = set()
        self.t = 0.0
        self.ev = []
        self.seq = 0
        self.log = []
        self.waiting = []                     # 排队中的请求 (rid, model, dur)
        self.room = {}                        # 正在 make_room 的模型 -> 在等哪个模型卸载
        self.lat = {}                         # rid -> (到达, 开始)
        self.load_total = 0.0

    # ---------- 事件 ----------
    def at(self, t, kind, *data):
        self.seq += 1
        heapq.heappush(self.ev, (t, self.seq, kind, data))

    def say(self, s):
        self.log.append(f'{self.t:7.2f}  {s}')

    # ---------- 设备内存（server-models-sched.cpp 508-593） ----------
    def mem_enabled(self):
        return self.margin > 0

    def free_now(self, dev):
        used = sum(x.actual.get(dev, 0) for x in self.m.values() if x.running())
        return self.devices[dev] - self.ext_true.get(dev, 0) - used

    def calibrate(self, mem, reporter):
        for dev, self_mib in mem.items():
            if dev not in self.devices:
                continue                       # host 行：is_device() 为假，跳过
            tracked = sum(h.get(dev, 0) for n, h in self.held.items() if n != reporter)
            s = 0 if reporter == '' else self_mib
            total = self.devices[dev]
            self.dm[dev] = [total, max(0, total - self.free_now(dev) - s - tracked)]

    def reserve(self, name):
        x = self.m[name]
        if name in self.held:
            return
        mem = x.eff if x.eff else x.min
        self.held[name] = {d: v for d, v in mem.items() if d in self.devices}

    def release(self, name):
        self.held.pop(name, None)

    def available(self, dev):
        if dev not in self.dm:
            return float('inf')
        total, ext = self.dm[dev]
        return total - ext - sum(h.get(dev, 0) for h in self.held.values())

    def deficit(self, name):
        res = {}
        for dev, need in self.m[name].min.items():
            if dev not in self.devices or dev not in self.dm:
                continue                       # host 内存当作无限，和 --fit 一致
            avail = self.available(dev) - self.margin
            if need > avail:
                res[dev] = need - avail
        return res

    # ---------- 排队（85-230） ----------
    def count_running(self):
        return sum(1 for x in self.m.values() if x.running())

    def has_capacity(self):
        return self.models_max <= 0 or self.count_running() < self.models_max

    def find(self, name):
        for e in self.queue:
            if e[0] == name:
                return e
        return None

    def pick_victim(self):
        v = None
        for n, x in self.m.items():
            if x.req != 0 or not x.ready_or_sleep():
                continue
            if n in self.stopping or self.find(n):
                continue
            if v is None or x.last_used < self.m[v].last_used:
                v = n
        return v

    def join(self, name):
        e = self.find(name)
        if e:
            e[1] += 1
        else:
            self.queue.append([name, 1, False])

    def leave(self, name):
        e = self.find(name)
        if e:
            e[1] -= 1
            if e[1] <= 0:
                self.queue.remove(e)

    def limits_exceeded(self, name, verbose=False):
        if not self.has_capacity():
            return True
        if not self.mem_enabled():
            return False
        miss = self.deficit(name)
        if not miss:
            return False
        any_running = any(x.running() for x in self.m.values())
        if verbose and not any_running:
            for dev, b in miss.items():
                self.say(f'WARN {name} 预计在 {dev} 上差 {b:.0f} MiB，但没有别的模型可卸，照样加载')
        return any_running

    def try_claim(self, name):
        if not self.queue or self.queue[0][0] != name or self.queue[0][2]:
            return False
        if not self.has_capacity():
            return False
        if self.mem_enabled() and self.limits_exceeded(name) and not self.pick_victim():
            return False
        self.queue[0][2] = True
        return True

    def claim_done(self, name, ok):
        if ok:
            return
        e = self.find(name)
        if e:
            e[2] = False

    def tick(self):
        if self.models_max <= 0 or not self.queue:
            return
        n_running = self.count_running()
        n_stopping = sum(1 for n in self.stopping if self.m[n].running())
        n_needed = sum(1 for e in self.queue if not e[2])
        n_claimed = sum(1 for e in self.queue if e[2] and not self.m[e[0]].running())
        n_free = self.models_max - n_running + n_stopping - n_claimed
        while n_free < n_needed:
            v = self.pick_victim()
            if not v:
                return
            self.say(f'tick：为排队的请求卸载最久没用的 {v}')
            self.stop(v)
            n_free += 1

    # ---------- 实例生命周期 ----------
    def stop(self, name):
        self.stopping.add(name)
        self.at(self.t + STOP_S, 'unloaded', name)

    def make_room(self, name):
        """返回 None 表示可以 spawn，返回模型名表示要等它卸载完再来。"""
        while True:
            if not self.limits_exceeded(name, verbose=True):
                return None
            v = self.pick_victim()
            if not v:
                return None                    # 都忙，交给 load() 再查一次
            self.say(f'make_room：超限，卸载最久没用的 {v}')
            self.stop(v)
            return v

    def load(self, name):
        if self.limits_exceeded(name):
            raise RuntimeError('model limit reached, try again later')
        x = self.m[name]
        x.status = 'loading'
        x.last_used = self.t
        x.eff = None
        if x.fit_max is not None:              # --fit 按此刻的空闲挑上下文，留 1024 MiB
            dev = next(iter(x.min))
            free = self.devices[dev] - self.ext_true.get(dev, 0) - sum(
                y.actual.get(dev, 0) for y in self.m.values() if y.running() and y is not x)
            x.actual = {dev: max(x.min[dev], min(x.fit_max, free - 1024)) + x.unseen}
        self.reserve(name)                     # on_spawned
        dt = x.warm_s if x.loaded_once else x.load_s
        x.n_loads += 1
        self.load_total += dt
        self.say(f'spawn {name}（加载 {dt:.1f} s）')
        self.at(self.t + dt, 'ready', name)

    def begin_load(self, name):
        wait = self.make_room(name)
        if wait:
            self.room[name] = wait
            return
        try:
            self.load(name)
            self.claim_done(name, True)
        except RuntimeError as e:
            self.say(f'{name} 这次没加载成：{e}')
            self.claim_done(name, False)
        self.tick()

    def pump(self):
        progress = True
        while progress:
            progress = False
            for r in list(self.waiting):
                rid, name, dur = r
                x = self.m[name]
                if x.ready_or_sleep() and name not in self.stopping:
                    self.waiting.remove(r)
                    self.leave(name)
                    self.start(rid, name, dur)
                    progress = True
                elif x.status == 'unloaded' and name not in self.room and self.try_claim(name):
                    self.begin_load(name)
                    progress = True

    def start(self, rid, name, dur):
        x = self.m[name]
        x.last_used = self.t
        x.req += 1
        a = self.lat[rid][0]
        self.lat[rid] = (a, self.t)
        self.say(f'请求 {rid} 交给 {name}（等了 {self.t - a:.1f} s）')
        self.at(self.t + dur, 'done', rid, name)

    def run(self, reqs):
        for rid, (t, name, dur) in enumerate(reqs):
            self.at(t, 'arrive', rid, name, dur)
        while self.ev:
            self.t, _, kind, data = heapq.heappop(self.ev)
            if kind == 'arrive':
                rid, name, dur = data
                self.lat[rid] = (self.t, None)
                x = self.m[name]
                if x.ready_or_sleep() and name not in self.stopping:
                    self.start(rid, name, dur)
                else:
                    self.say(f'请求 {rid} 要 {name}，排队')
                    if x.status == 'unloaded' or name in self.stopping:
                        self.join(name)
                        self.tick()
                    self.waiting.append((rid, name, dur))
            elif kind == 'done':
                rid, name = data
                x = self.m[name]
                x.req -= 1
                if x.req == 0:
                    self.tick()
            elif kind == 'ready':
                name, = data
                x = self.m[name]
                x.status = 'loaded'
                x.loaded_once = True
                if self.mem_enabled():                    # on_ready：用实例报上来的真实占用
                    x.eff = {d: v - x.unseen for d, v in x.actual.items()}
                    self.release(name)
                    self.reserve(name)
                    self.calibrate(x.eff, name)
                self.say(f'{name} 就绪，占 {sum(x.actual.values()):.0f} MiB')
            elif kind == 'unloaded':
                name, = data
                x = self.m[name]
                x.status = 'unloaded'
                self.stopping.discard(name)
                self.release(name)
                x.eff = None
                self.say(f'{name} 已卸载')
                for n, w in list(self.room.items()):
                    if w == name:
                        del self.room[n]
                        self.begin_load(n)
                self.tick()
            self.pump()
        return self

    def resident(self):
        return sorted(n for n, x in self.m.items() if x.running())

    def ledger(self, dev):
        total, ext = self.dm.get(dev, [self.devices.get(dev, 0), 0])
        held = sum(h.get(dev, 0) for h in self.held.values())
        return dict(total=total, external=ext, held=held, avail=self.available(dev), real_free=self.free_now(dev))


# ---------------- 从 logs 读实测 ----------------

def read_logs(d):
    load, mem = {}, {}
    p = os.path.join(d or '', 'load.txt')
    if d and os.path.exists(p):
        for line in open(p):
            m = re.match(r'(\S+) run=(\d) load_ms=([\d.]+)', line)
            if m:
                load.setdefault(m.group(1), []).append(float(m.group(3)) / 1000)
    p = os.path.join(d or '', 'fit_print.txt')
    if d and os.path.exists(p):
        for line in open(p):
            m = re.match(r'(\S+) np=1 c=4096 Host (\d+) (\d+) (\d+)', line)
            if m:
                mem[m.group(1)] = sum(int(m.group(i)) for i in (2, 3, 4))
    return load, mem


def maxmem(d):
    """-c 40960（训练长度，fit 在大内存上会挑到的上限）的三项之和。"""
    out = {'q-Q4_0': 5031, 'q-Q8_0': 5587, 'qwen3-0.6b-bf16': 6259}
    p = os.path.join(d or '', 'fit_print.txt')
    if d and os.path.exists(p):
        for line in open(p):
            m = re.match(r'(\S+) np=1 c=40960 Host (\d+) (\d+) (\d+)', line)
            if m:
                out[m.group(1)] = sum(int(m.group(i)) for i in (2, 3, 4))
    return out


DEFAULT_LOAD = {'q-Q4_0': [3.7, 1.5, 1.5], 'q-Q8_0': [2.1, 2.0, 2.0], 'qwen3-0.6b-bf16': [11.2, 3.7, 3.7]}
DEFAULT_MEM = {'q-Q4_0': 965, 'q-Q8_0': 1521, 'qwen3-0.6b-bf16': 2193}
NAMES = ['q-Q4_0', 'q-Q8_0', 'qwen3-0.6b-bf16']
SHORT = {'q-Q4_0': 'A', 'q-Q8_0': 'B', 'qwen3-0.6b-bf16': 'C'}


def mk(load, mem, dev='host', extra=None, fit=None):
    out = []
    for n in NAMES:
        ls = load.get(n) or DEFAULT_LOAD[n]
        mm = mem.get(n) or DEFAULT_MEM[n]
        ex = (extra or {}).get(n, 0)
        x = Model(SHORT[n], ls[0], {dev: mm}, {dev: mm + ex}, warm_s=min(ls[1:]) if len(ls) > 1 else ls[0])
        x.unseen = ex
        if fit:
            x.fit_max = fit.get(n)
        out.append(x)
    return out


def trace(pattern, gap=10.0, dur=2.0):
    return [(i * gap, c, dur) for i, c in enumerate(pattern)]


def summary(r):
    waits = [s - a for a, s in r.lat.values() if s is not None]
    return dict(loads=sum(x.n_loads for x in r.m.values()), load_s=round(r.load_total, 2),
                wait_mean=round(sum(waits) / len(waits), 2), wait_max=round(max(waits), 2))


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else None
    load, mem = read_logs(d)
    print('# 输入：A=Q4_0 B=Q8_0 C=bf16；加载秒数(冷, 热) 和 -c 4096 下的 min MiB')
    for n in NAMES:
        ls = load.get(n) or DEFAULT_LOAD[n]
        print('input', SHORT[n], round(ls[0], 2), round(min(ls[1:]), 2), mem.get(n) or DEFAULT_MEM[n])

    # 1 models_max 和访问模式：轮转三个模型 vs 有局部性
    print('\n# lru  models_max pattern loads load_s wait_mean wait_max')
    pats = {'rr': 'ABCABCABCABC', 'local': 'AAAABBBBAAAACCCC', 'hot2': 'ABABABABABAC'}
    for mx in (1, 2, 3):
        for pn, p in pats.items():
            r = Router(mk(load, mem), models_max=mx).run(trace(p))
            s = summary(r)
            print('lru', mx, pn, s['loads'], s['load_s'], s['wait_mean'], s['wait_max'])

    # 2 忙的模型不卸：models_max=1，A 正在跑 30 s 的长请求，B 的请求到了
    print('\n# busy')
    r = Router(mk(load, mem), models_max=1).run([(0, 'A', 30.0), (5, 'B', 2.0)])
    for line in r.log:
        print('busy', line)

    # 3 纯 CPU：余量开到 99999 MiB 也不卸，因为 host 行被跳过
    print('\n# cpu  margin resident_after loads')
    for mg in (0, 1024, 99999):
        r = Router(mk(load, mem), models_max=4, margin=mg).run(trace('ABC', gap=20))
        print('cpu', mg, ''.join(r.resident()), summary(r)['loads'])

    # 4 假设一块 6144 MiB 的 GPU，模型全放在上面，外部程序占 512 MiB
    print('\n# gpu  margin resident_after loads evictions')
    for mg in (0, 512, 1024, 2048):
        ms = mk(load, mem, dev='GPU0')
        r = Router(ms, models_max=4, margin=mg, devices={'GPU0': 6144}, external={'GPU0': 512})
        r.dm['GPU0'] = [6144, 512]             # 启动时测量子进程报过一次
        r.run(trace('ABC', gap=20))
        ev = sum(1 for line in r.log if '卸载最久' in line)
        print('gpu', mg, ''.join(r.resident()), summary(r)['loads'], ev)
    ms = mk(load, mem, dev='GPU0')
    r = Router(ms, models_max=4, margin=1024, devices={'GPU0': 6144}, external={'GPU0': 512})
    r.dm['GPU0'] = [6144, 512]
    r.run(trace('ABC', gap=20))
    for line in r.log:
        print('gpulog', line)
    print('gpuledger', r.ledger('GPU0'))

    # 5 什么都没加载时，min 比可用还大：只报警，照样加载
    print('\n# anyway')
    ms = mk(load, mem, dev='GPU0')
    r = Router(ms, models_max=4, margin=1024, devices={'GPU0': 2048}, external={'GPU0': 0})
    r.dm['GPU0'] = [2048, 0]
    r.run([(0, 'C', 1.0)])
    for line in r.log:
        print('anyway', line)

    # 6 mmproj 不进测量：B 实际多占 extra MiB。3072 MiB 的卡上 A 已在，B 按 min 判断放得下，不卸 A
    print('\n# mmproj  extra_mib resident_after evictions real_free_after router_avail_after')
    for extra in (0, 220, 600):
        ms = mk(load, mem, dev='GPU0', extra={'q-Q8_0': extra})
        r = Router(ms, models_max=4, margin=512, devices={'GPU0': 3072}, external={'GPU0': 0})
        r.dm['GPU0'] = [3072, 0]
        r.run(trace('AB', gap=20))
        ev = sum(1 for line in r.log if '卸载最久' in line)
        lg = r.ledger('GPU0')
        print('mmproj', extra, ''.join(r.resident()), ev, lg['real_free'], lg['avail'])

    # 7 不钉 -c：--fit 让每个实例吃到“此刻空闲减 1024”为止，后来的模型就得把它卸掉
    print('\n# fitgreedy  pinned resident_after loads evictions')
    fmax = maxmem(d)
    for pinned in (True, False):
        ms = mk(load, mem, dev='GPU0', fit=None if pinned else fmax)
        r = Router(ms, models_max=4, margin=512, devices={'GPU0': 8192}, external={'GPU0': 0})
        r.dm['GPU0'] = [8192, 0]
        r.run(trace('ABCABC', gap=20))
        ev = sum(1 for line in r.log if '卸载最久' in line)
        print('fitgreedy', int(pinned), ''.join(r.resident()), summary(r)['loads'], ev)
        if not pinned:
            for line in r.log:
                print('fitlog', line)

if __name__ == '__main__':
    main()
