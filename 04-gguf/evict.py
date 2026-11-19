"""把文件从页缓存里踢出去（posix_fadvise DONTNEED），用来测冷启动。不需要 root。"""
import os
import sys

for p in sys.argv[1:]:
    fd = os.open(p, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
