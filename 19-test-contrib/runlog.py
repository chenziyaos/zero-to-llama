"""跑一条命令，stdout 和 stderr 一起写进日志，末尾追加墙钟和子进程峰值常驻内存。

用法：python3 runlog.py logs/convert.log -- python3 convert_hf_to_gguf.py ...
"""
import resource
import subprocess
import sys
import time


def main():
    i = sys.argv.index('--')
    log, cmd = sys.argv[1], sys.argv[i + 1:]
    t0 = time.time()
    with open(log, 'w') as f:
        f.write('$ ' + ' '.join(cmd) + '\n')
        f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    mib = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024
    tail = f'== rc={p.returncode} wall_s={wall:.1f} maxrss_mib={mib}\n'
    with open(log, 'a') as f:
        f.write(tail)
    print(tail, end='')
    return p.returncode


if __name__ == '__main__':
    sys.exit(main())
