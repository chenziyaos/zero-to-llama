#!/usr/bin/env python3
"""编出来的 llama.cpp 和这台机器对得上吗：对比 system_info、CMakeCache 和 /proc/cpuinfo。

用法：
  python3 check_build.py BUILD_DIR [MODEL]
  python3 check_build.py --sysinfo "CPU : SSE3 = 1 | AVX = 1 | ..."   # 只给一行 system_info

给了 MODEL 时会跑一次 llama-completion -n 1 取 system_info；没给时只读 CMakeCache。
只依赖标准库。输出三类问题：
  MISSING  编进去了但这台 CPU 没有的指令集，换机器跑会非法指令或加载失败
  UNUSED   CPU 有但编译时没打开的指令集，白白丢性能
  NOTE     其他值得看一眼的开关，比如 NATIVE 和 BACKEND_DL 同开、BLAS 开着
"""
import os, re, subprocess, sys

# system_info 里的名字 -> /proc/cpuinfo 里的 flag；有些要多个 flag 同时在
FLAGS = {
    'SSE3': ['pni'], 'SSSE3': ['ssse3'], 'AVX': ['avx'], 'AVX2': ['avx2'], 'F16C': ['f16c'], 'FMA': ['fma'],
    'BMI2': ['bmi2'], 'AVX_VNNI': ['avx_vnni'],
    'AVX512': ['avx512f', 'avx512cd', 'avx512vl', 'avx512dq', 'avx512bw'],
    'AVX512_VBMI': ['avx512vbmi'], 'AVX512_VNNI': ['avx512_vnni'], 'AVX512_BF16': ['avx512_bf16'],
    'AMX_INT8': ['amx_int8'],
}
# 跟指令集无关、只是编译开关的条目
OPTS = {'LLAMAFILE', 'OPENMP', 'REPACK', 'KLEIDIAI', 'ACCELERATE'}


def cpu_flags():
    try:
        for line in open('/proc/cpuinfo'):
            if line.startswith('flags'):
                return set(line.split(':', 1)[1].split())
    except OSError:
        pass
    return set()


def parse_sysinfo(s):
    """CPU : SSE3 = 1 | AVX = 1 | ... -> {'SSE3': 1, ...}；有多个后端时只取 CPU 段。"""
    m = re.search(r'CPU : (.*?)(?:\| [A-Z][A-Za-z]+ : |$)', s)
    seg = m.group(1) if m else s
    return {k: int(v) for k, v in re.findall(r'([A-Z0-9_]+) = (\d+)', seg)}


def cache(build):
    p = os.path.join(build, 'CMakeCache.txt')
    out = {}
    if os.path.exists(p):
        for line in open(p):
            m = re.match(r'(GGML_[A-Z0-9_]+|BUILD_SHARED_LIBS|CMAKE_BUILD_TYPE):\w+=(.*)', line.strip())
            if m:
                out[m.group(1)] = m.group(2)
    return out


def run_sysinfo(build, model):
    exe = os.path.join(os.path.abspath(build), 'bin', 'llama-completion')
    model = os.path.abspath(model)
    r = subprocess.run([exe, '-m', model, '-n', '1', '-p', 'hi', '-no-cnv', '-t', '1'],
                       capture_output=True, text=True, cwd='/tmp')
    txt = r.stdout + r.stderr
    loads = re.findall(r'loaded (\S+) backend from (\S+)', txt)
    line = next((l for l in txt.splitlines() if 'system_info' in l), '')
    return line, loads


def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__); return 2
    have = cpu_flags()
    loads, cc = [], {}
    if a[0] == '--sysinfo':
        line = ' '.join(a[1:])
    else:
        cc = cache(a[0])
        line = ''
        if len(a) > 1:
            line, loads = run_sysinfo(a[0], a[1])
    info = parse_sysinfo(line) if line else {}
    probs = []
    if info:
        for k, fl in FLAGS.items():
            on = info.get(k, 0) == 1
            cpu = all(f in have for f in fl) if have else None
            if on and cpu is False:
                probs.append(('MISSING', f'{k} 编进去了，这台 CPU 没有 {"、".join(f for f in fl if f not in have)}'))
            if not on and cpu and k in info:
                probs.append(('UNUSED', f'{k} 这台 CPU 有，但编译时没打开'))
            if not on and cpu and k not in info and k not in ('AVX_VNNI', 'AMX_INT8'):
                probs.append(('UNUSED', f'{k} 这台 CPU 有，system_info 里没有这一项，多半没编进去'))
        for k in sorted(OPTS & set(info)):
            print(f'  opt {k} = {info[k]}')
    if cc:
        g = lambda k: cc.get(k, '').upper() in ('ON', 'TRUE', '1')
        if cc.get('CMAKE_BUILD_TYPE', '') not in ('Release', 'RelWithDebInfo'):
            probs.append(('NOTE', f'CMAKE_BUILD_TYPE={cc.get("CMAKE_BUILD_TYPE") or "空"}，不是 Release，速度会差很多'))
        if g('GGML_NATIVE') and not g('GGML_BACKEND_DL'):
            probs.append(('NOTE', 'GGML_NATIVE=ON：二进制只保证在本机这代 CPU 上能跑，拷到老机器可能非法指令'))
        if g('GGML_BLAS'):
            probs.append(('NOTE', 'GGML_BLAS=ON：只接管三个维度都 ≥32 的 F32 矩阵乘，解码不受影响'))
        if g('GGML_BACKEND_DL'):
            probs.append(('NOTE', 'GGML_BACKEND_DL=ON：CPU 后端运行时按 score 挑，看 loaded CPU backend from 那一行'))
        gpus = [k[5:] for k in ('GGML_CUDA', 'GGML_HIP', 'GGML_VULKAN', 'GGML_METAL', 'GGML_SYCL', 'GGML_OPENCL') if g(k)]
        print('  build', cc.get('CMAKE_BUILD_TYPE'), 'shared', cc.get('BUILD_SHARED_LIBS'), 'native', cc.get('GGML_NATIVE'),
              'dl', cc.get('GGML_BACKEND_DL', 'OFF'), 'gpu', ','.join(gpus) or 'none')
    for b, p in loads:
        print(f'  loaded {b} <- {os.path.basename(p)}')
    if line:
        print('  ' + line.split('system_info: ', 1)[-1][:300])
    for k, s in probs:
        print(f'{k:8s} {s}')
    bad = sum(1 for k, _ in probs if k == 'MISSING')
    print(f'missing {bad} unused {sum(1 for k, _ in probs if k == "UNUSED")}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
