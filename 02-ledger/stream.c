/* 最小 STREAM：测本机可持续内存带宽，口径同 McCalpin STREAM（copy、scale、add、triad），另加一个只读求和 read，每种取最快一次。
   gcc -O3 -march=native -ffast-math -fopenmp stream.c -o stream && OMP_NUM_THREADS=4 ./stream [MiB_per_array] */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
int main(int argc, char **argv) {
    size_t mib = argc > 1 ? atol(argv[1]) : 64, n = mib * 1024 * 1024 / sizeof(double);
    double *a = malloc(n * 8), *b = malloc(n * 8), *c = malloc(n * 8), s = 3.0;
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) { a[i] = 1; b[i] = 2; c[i] = 0; }
    const char *nm[5] = {"copy", "scale", "add", "triad", "read"};
    double mult[5] = {2, 2, 3, 3, 1}, best[5] = {1e9, 1e9, 1e9, 1e9, 1e9}, sink = 0;
    for (int r = 0; r < 10; r++) {
        for (int k = 0; k < 5; k++) {
            double t = omp_get_wtime();
            if (k == 0) {
                #pragma omp parallel for
                for (size_t i = 0; i < n; i++) c[i] = a[i];
            } else if (k == 1) {
                #pragma omp parallel for
                for (size_t i = 0; i < n; i++) b[i] = s * c[i];
            } else if (k == 2) {
                #pragma omp parallel for
                for (size_t i = 0; i < n; i++) c[i] = a[i] + b[i];
            } else if (k == 3) {
                #pragma omp parallel for
                for (size_t i = 0; i < n; i++) a[i] = b[i] + s * c[i];
            } else {  /* 只读：decode 读权重就是这种访问 */
                double acc = 0;
                #pragma omp parallel for reduction(+:acc)
                for (size_t i = 0; i < n; i++) acc += a[i];
                sink += acc;
            }
            t = omp_get_wtime() - t;
            if (r && t < best[k]) best[k] = t;
        }
    }
    printf("{\"threads\": %d, \"mib_per_array\": %zu", omp_get_max_threads(), mib);
    for (int k = 0; k < 5; k++) printf(", \"%s_GBps\": %.2f", nm[k], mult[k] * n * 8 / best[k] / 1e9);
    printf("}\n");
    return sink < 0;
}
