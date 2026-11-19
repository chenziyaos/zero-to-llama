"""跑一条命令，打印子进程的峰值 RSS、墙钟、缺页次数和 llama.cpp 报的 load time。

    python3 rss.py llama-completion -m model.gguf -lm mmap -c 512 -n 1 -p hi -no-cnv
"""
import resource
import subprocess
import sys
import time

t = time.time()
p = subprocess.run(sys.argv[1:], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
load = [ln for ln in p.stderr.splitlines() if 'load time' in ln]
print(f'maxrss={ru.ru_maxrss / 1024:.0f}MiB wall={time.time() - t:.2f}s '
      f'minflt={ru.ru_minflt} majflt={ru.ru_majflt}',
      load[-1].split('=')[-1].strip() if load else '')
