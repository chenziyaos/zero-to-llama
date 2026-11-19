"""按 llama.cpp 0ec4027 的 common/preset.cpp 和 tools/server/server-models.cpp 复刻 router 的预设合并。

能做的事：
  解析 --models-preset 的 INI（[*] 全局段、version 键跳过、取反布尔键、标签规范化）
  扫描 --models-dir（子目录一个模型，文件名含 mmproj 的是投影器，mtp-/dspark-/dflash- 开头的是草稿模型）
  按源码顺序合并：global 分别盖到 cache/models-dir/preset 三个来源，models-dir 覆盖 cache，
  preset 再 merge 进去，最后 router 自己的命令行盖到每个模型上
  删掉 router 保留的键，补上 host/port/alias，渲染成子进程命令行

选项表只收了常用的二十来个，够演示规则；真实程序的选项表来自 common/arg.cpp。
用法：python3 preset_tool.py presets.ini [--models-dir DIR] [--router-args "..."]
"""
import os
import re
import sys
import shlex
import argparse
from collections import OrderedDict

# (args, args_neg, env, 取值, 仅预设) ；args 的第一个决定 std::map 里的顺序，最后一个是渲染时用的长名
OPTS = [
    (['-a', '--alias'], [], 'LLAMA_ARG_ALIAS', True, False),
    (['-c', '--ctx-size'], [], 'LLAMA_ARG_CTX_SIZE', True, False),
    (['-cram', '--cache-ram'], [], 'LLAMA_ARG_CACHE_RAM', True, False),
    (['-hf', '-hfr', '--hf-repo'], [], 'LLAMA_ARG_HF_REPO', True, False),
    (['-kvu', '--kv-unified'], ['-no-kvu', '--no-kv-unified'], 'LLAMA_ARG_KV_UNIFIED', False, False),
    (['-m', '--model'], [], 'LLAMA_ARG_MODEL', True, False),
    (['-md', '--spec-draft-model'], [], 'LLAMA_ARG_SPEC_DRAFT_MODEL', True, False),
    (['-mm', '--mmproj'], [], 'LLAMA_ARG_MMPROJ', True, False),
    (['-ngl', '--gpu-layers', '--n-gpu-layers'], [], 'LLAMA_ARG_N_GPU_LAYERS', True, False),
    (['-np', '--parallel'], [], 'LLAMA_ARG_N_PARALLEL', True, False),
    (['--api-key'], [], 'LLAMA_API_KEY', True, False),
    (['--host'], [], 'LLAMA_ARG_HOST', True, False),
    (['--jinja'], ['--no-jinja'], 'LLAMA_ARG_JINJA', False, False),
    (['--log-colors'], [], 'LLAMA_ARG_LOG_COLORS', True, False),
    (['--log-file'], [], 'LLAMA_ARG_LOG_FILE', True, False),
    (['--metrics'], [], 'LLAMA_ARG_ENDPOINT_METRICS', False, False),
    (['--models-autoload'], ['--no-models-autoload'], 'LLAMA_ARG_MODELS_AUTOLOAD', False, False),
    (['--models-dir'], [], 'LLAMA_ARG_MODELS_DIR', True, False),
    (['--models-max'], [], 'LLAMA_ARG_MODELS_MAX', True, False),
    (['--models-memory-margin'], [], 'LLAMA_ARG_MODELS_MEMORY_MARGIN', True, False),
    (['--models-preset'], [], 'LLAMA_ARG_MODELS_PRESET', True, False),
    (['--port'], [], 'LLAMA_ARG_PORT', True, False),
    (['--sleep-idle-seconds'], [], None, True, False),
    (['--temp', '--temperature'], [], 'LLAMA_ARG_TEMPERATURE', True, False),
    (['load-on-startup'], [], 'LLAMA_ARG_PRESET_LOAD_ON_STARTUP', True, True),
    (['stop-timeout'], [], 'LLAMA_ARG_PRESET_STOP_TIMEOUT', True, True),
    (['dedup-cache-models'], [], 'LLAMA_ARG_PRESET_DEDUP_CACHE_MODELS', True, True),
]


class Opt:
    def __init__(self, args, neg, env, val, ponly):
        self.args, self.neg, self.env, self.val, self.ponly = args, neg, env, val, ponly

    def __repr__(self):
        return self.args[-1]


TABLE = [Opt(*o) for o in OPTS]
DEFAULT_NAME = 'default'
CHILD_ADDR = '127.0.0.1'
DEFAULT_STOP_TIMEOUT = 10
RESERVED = ['LLAMA_ARG_SSL_KEY_FILE', 'LLAMA_ARG_SSL_CERT_FILE', 'LLAMA_API_KEY', 'LLAMA_ARG_API_KEY_FILE',
            'LLAMA_ARG_MODELS_DIR', 'LLAMA_ARG_MODELS_MAX', 'LLAMA_ARG_MODELS_MEMORY_MARGIN',
            'LLAMA_ARG_MODELS_PRESET', 'LLAMA_ARG_MODELS_AUTOLOAD', 'LLAMA_ARG_CONNECT', 'LLAMA_ARG_CONNECT_CODE']
RESERVED_MODEL = ['LLAMA_ARG_MODEL', 'LLAMA_ARG_MMPROJ', 'LLAMA_ARG_ALIAS', 'LLAMA_ARG_HF_REPO']
DRAFT_PREFIXES = ('mtp-', 'dspark-', 'dflash-')
FALSEY = {'false', '0', 'off', 'no', 'disabled'}
TRUTHY = {'true', '1', 'on', 'yes', 'enabled'}


def rm_dashes(s):
    return s.lstrip('-')


def key_map():
    """get_map_key_opt：env 名和去掉横杠的每个参数名都能当键。"""
    m = {}
    for o in TABLE:
        if o.env:
            m[o.env] = o
        for a in o.args + o.neg:
            m[rm_dashes(a)] = o
    return m


KEYS = key_map()


def canonical_tag(tag):
    """量化标签规范化：取最后一个 - 或 . 之后的部分转大写，否则整体大写。"""
    m = re.search(r'[-.]([A-Z0-9_]+)$', tag, re.I)
    return (m.group(1) if m else tag).upper()


class Preset:
    def __init__(self, name):
        self.name = name
        self.options = {}            # Opt -> str

    def ordered(self):
        # std::map<common_arg, ...> 按 args[0] 的 strcmp 排序
        return sorted(self.options.items(), key=lambda kv: kv[0].args[0].encode())

    def merge(self, other):
        for o, v in other.options.items():
            self.options[o] = v

    def set(self, env, value):
        self.options[KEYS[env]] = value

    def get(self, env):
        for o, v in self.options.items():
            if o.env == env:
                return v
        return None

    def unset(self, env):
        for o in list(self.options):
            if o.env == env:
                del self.options[o]
                return

    def copy(self, name=None):
        p = Preset(self.name if name is None else name)
        p.options = dict(self.options)
        return p

    def to_args(self, bin_path='llama-server'):
        args = [bin_path] if bin_path else []
        for o, v in self.ordered():
            if o.ponly:
                continue
            args.append(o.args[-1])
            if not o.val and v.lower() in FALSEY:
                if o.neg:
                    args[-1] = o.neg[-1]
                else:
                    args.pop()
            if o.val:
                args.append(v)
        return args


INI_LINE_HDR = re.compile(r'^[ \t]*\[[ \t]*([^\]]*?)[ \t]*\][ \t]*(?:[;#].*)?$')
INI_LINE_KV = re.compile(r'^[ \t]*([A-Za-z_][A-Za-z0-9_.-]*)[ \t]*=[ \t]*(.*?)[ \t]*(?:[;#].*)?$')


def parse_ini(text):
    """parse_ini_from_file：没有段名的键落在 default 段；同段同键后写的覆盖先写的。"""
    parsed = OrderedDict()
    cur = DEFAULT_NAME
    for ln, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.strip()[0] in ';#':
            continue
        m = INI_LINE_HDR.match(line)
        if m:
            cur = m.group(1)
            parsed[cur] = OrderedDict()
            continue
        m = INI_LINE_KV.match(line)
        if not m:
            raise ValueError(f'failed to parse server config file at line {ln}')
        parsed.setdefault(cur, OrderedDict())[m.group(1)] = m.group(2)
    return parsed


def load_from_ini(text, ignore_unknown=False, allowed=None):
    """返回 (presets, global)。"""
    out, glob = OrderedDict(), Preset('*')
    warns = []
    for sec, kv in parse_ini(text).items():
        name = sec or DEFAULT_NAME
        if name not in ('*', DEFAULT_NAME) and ':' in name:
            i = name.rfind(':')
            tag = name[i + 1:]
            ct = canonical_tag(tag)
            if ct != tag:
                name = name[:i + 1] + ct
        p = Preset(name)
        for k, v in kv.items():
            if k == 'version':
                continue
            if allowed is not None and k not in allowed:
                raise ValueError(f"option '{k}' is not allowed in remote presets")
            if k in KEYS:
                o = KEYS[k]
                if o.neg:     # 布尔参数：写的是取反键就把值反过来
                    if any(rm_dashes(n) == k for n in o.neg):
                        v = 'false' if v.lower() in TRUTHY else 'true'
                p.options[o] = v
            elif ignore_unknown:
                warns.append(f"ignoring option '{k}': not supported by this program")
            else:
                raise ValueError(f"option '{k}' not recognized in preset '{name}'")
        if name == DEFAULT_NAME and not p.options:
            continue
        if name == '*':
            glob = p
        else:
            out[name] = p
    return out, glob, warns


def is_mmproj(f):
    return 'mmproj' in f


def is_draft(f):
    return f.startswith(DRAFT_PREFIXES)


def load_from_models_dir(d):
    models = []
    for e in sorted(os.listdir(d)):
        p = os.path.join(d, e)
        if os.path.isdir(p):
            model = shard = mm = draft = None
            for f in sorted(os.listdir(p)):
                if not f.endswith('.gguf') or not os.path.isfile(os.path.join(p, f)):
                    continue
                fp = os.path.join(p, f)
                if is_mmproj(f):
                    mm = fp
                elif is_draft(f):
                    draft = draft or fp
                elif '-00001-of-' in f:
                    shard = fp
                else:
                    model = fp
            path = shard or model
            if path:
                models.append((e, path, mm, draft))
        elif e.endswith('.gguf') and not is_mmproj(e) and not is_draft(e):
            models.append((e.replace('.gguf', ''), p, None, None))
    out = OrderedDict()
    for name, path, mm, draft in models:
        pr = Preset(name)
        pr.set('LLAMA_ARG_MODEL', path)
        if mm:
            pr.set('LLAMA_ARG_MMPROJ', mm)
        if draft:
            pr.set('LLAMA_ARG_SPEC_DRAFT_MODEL', draft)
        out[name] = pr
    return out


def parse_cli(argv):
    """common_params_to_map 的简化版：把命令行解析成一个 default 预设。"""
    p = Preset(DEFAULT_NAME)
    i = 0
    flat = {}
    for o in TABLE:
        if o.ponly:
            continue
        for a in o.args:
            flat[a] = (o, True)
        for a in o.neg:
            flat[a] = (o, False)
    while i < len(argv):
        a = argv[i]
        if a not in flat:
            raise ValueError(f'unknown argument: {a}')
        o, pos = flat[a]
        if o.val:
            p.options[o] = argv[i + 1]
            i += 2
        else:
            p.options[o] = 'true' if pos else 'false'
            i += 1
    return p


def unset_reserved(p, model_args):
    for e in RESERVED:
        p.unset(e)
    if model_args:
        for e in RESERVED_MODEL:
            p.unset(e)


def cascade_base(base, presets):
    out = OrderedDict()
    for n, pr in presets.items():
        t = base.copy(n)
        t.merge(pr)
        out[n] = t
    return out


def build(ini_text='', models_dir=None, cached=None, router_argv=()):
    """server_models::load_models 的合并顺序。返回 {name: (preset, source)}。"""
    base = parse_cli(list(router_argv))
    unset_reserved(base, True)
    base.unset('LLAMA_ARG_LOG_FILE')
    cached = cached or OrderedDict()
    local = load_from_models_dir(models_dir) if models_dir else OrderedDict()
    custom, glob, _ = load_from_ini(ini_text) if ini_text else (OrderedDict(), Preset('*'), [])
    cached, local, custom = cascade_base(glob, cached), cascade_base(glob, local), cascade_base(glob, custom)
    final, src = OrderedDict(), {}
    for n, p in cached.items():
        final[n], src[n] = p, 'cache'
    for n, p in local.items():
        final[n], src[n] = p, 'models-dir'
    for n, p in custom.items():
        if n in final:
            final[n].merge(p)
        else:
            final[n] = p
        src[n] = 'preset'
    for p in final.values():
        p.merge(base)
    return OrderedDict((n, (p, src[n])) for n, p in sorted(final.items()))


def child_args(name, preset, port):
    """server_model_meta::update_args。"""
    p = preset.copy()
    unset_reserved(p, False)
    p.set('LLAMA_ARG_HOST', CHILD_ADDR)
    p.set('LLAMA_ARG_PORT', str(port))
    p.set('LLAMA_ARG_ALIAS', name)
    p.set('LLAMA_ARG_LOG_COLORS', 'off')
    return p.to_args()


def stop_timeout(preset):
    v = preset.get('LLAMA_ARG_PRESET_STOP_TIMEOUT')
    if v is None:
        return DEFAULT_STOP_TIMEOUT
    try:
        return int(v)
    except ValueError:
        return DEFAULT_STOP_TIMEOUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ini')
    ap.add_argument('--models-dir')
    ap.add_argument('--router-args', default='')
    a = ap.parse_args()
    text = open(a.ini).read()
    print('router argv:', a.router_args)
    _, glob, _ = load_from_ini(text)
    print('global [*]:', {repr(o): v for o, v in glob.ordered()})
    port = 39001
    for n, (p, src) in build(text, a.models_dir, router_argv=shlex.split(a.router_args)).items():
        print(f'\n[{n}] source={src} load-on-startup={p.get("LLAMA_ARG_PRESET_LOAD_ON_STARTUP") or "-"} stop-timeout={stop_timeout(p)}')
        print('  ' + ' '.join(shlex.quote(x) for x in child_args(n, p, port)))
        port += 1


if __name__ == '__main__':
    main()
