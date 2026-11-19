"""跑一条命令，报子进程的峰值常驻内存（ru_maxrss）和 load time。

router 的每个模型实例是一个独立进程，内存账最后落在这个进程的常驻内存上。
llama.cpp 分配 KV 缓存后会立刻清零一遍（llama-memory.cpp 的 ggml_backend_buffer_clear），
所以 -c 开多大，页就真的占多大，不会因为没用到而不占。
用法：python3 rss.py -- llama-completion -m x.gguf -c 4096 ...
"""
import re
import resource
import subprocess
import sys


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, errors='replace')
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    m = re.search(r'load time =\s*([\d.]+) ms', p.stderr)
    return p.returncode, ru.ru_maxrss // 1024, float(m.group(1)) if m else -1.0


if __name__ == '__main__':
    i = sys.argv.index('--')
    rc, mib, load = run(sys.argv[i + 1:])
    print(f'rc={rc} maxrss_mib={mib} load_ms={load}')
