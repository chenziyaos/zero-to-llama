/* 测单核和多核的峰值算力：AVX512 FP32 FMA、AVX512_BF16 点积（vdpbf16ps）和 AVX512_VNNI int8 点积（vpdpbusd）。
   每个线程 12 条独立依赖链，必须 -O3：-O2 不展开内层循环，累加器落回内存，测出来只有一半，掩盖指令延迟。gcc -O3 -march=native -fopenmp peak.c -o peak */
#include <immintrin.h>
#include <stdio.h>
#include <omp.h>
#define ITERS 100000000L
int main(void) {
    double t0, t1, t2, t3;
    float sinkf = 0; int sinki = 0;
    t0 = omp_get_wtime();
    #pragma omp parallel reduction(+:sinkf)
    {
        __m512 a[12], b = _mm512_set1_ps(1e-7f), c = _mm512_set1_ps(0.999999f);
        for (int j = 0; j < 12; j++) a[j] = _mm512_set1_ps((float)j);
        for (long i = 0; i < ITERS / 12; i++)
            for (int j = 0; j < 12; j++) a[j] = _mm512_fmadd_ps(a[j], c, b);
        for (int j = 0; j < 12; j++) sinkf += _mm512_reduce_add_ps(a[j]);
    }
    t1 = omp_get_wtime();
    #pragma omp parallel reduction(+:sinki)
    {
        __m512i a[12], b = _mm512_set1_epi8(1), c = _mm512_set1_epi8(2);
        for (int j = 0; j < 12; j++) a[j] = _mm512_set1_epi32(j);
        for (long i = 0; i < ITERS / 12; i++)
            for (int j = 0; j < 12; j++) a[j] = _mm512_dpbusd_epi32(a[j], b, c);
        for (int j = 0; j < 12; j++) sinki += _mm512_reduce_add_epi32(a[j]);
    }
    t2 = omp_get_wtime();
    #pragma omp parallel reduction(+:sinkf)
    {
        __m512 a[12]; __m512bh b = (__m512bh)_mm512_set1_epi16(0x3f80), c = (__m512bh)_mm512_set1_epi16(0x3380);
        for (int j = 0; j < 12; j++) a[j] = _mm512_set1_ps((float)j);
        for (long i = 0; i < ITERS / 12; i++)
            for (int j = 0; j < 12; j++) a[j] = _mm512_dpbf16_ps(a[j], b, c);
        for (int j = 0; j < 12; j++) sinkf += _mm512_reduce_add_ps(a[j]);
    }
    t3 = omp_get_wtime();
    int n = omp_get_max_threads();
    /* 一条 FMA = 16 路 x 2 flop；一条 vpdpbusd = 64 个 int8 乘加 = 128 op */
    double fma = (double)ITERS / 12 * 12 * n * 32 / (t1 - t0) / 1e9;
    double vnni = (double)ITERS / 12 * 12 * n * 128 / (t2 - t1) / 1e9;
    /* 一条 vdpbf16ps = 32 个 bf16 乘加 = 64 flop */
    double bf16 = (double)ITERS / 12 * 12 * n * 64 / (t3 - t2) / 1e9;
    printf("{\"threads\": %d, \"fp32_fma_GFLOPS\": %.1f, \"bf16_dp_GFLOPS\": %.1f, \"int8_vnni_GOPS\": %.1f, \"sink\": %d}\n", n, fma, bf16, vnni, (int)sinkf + sinki);
    return 0;
}
