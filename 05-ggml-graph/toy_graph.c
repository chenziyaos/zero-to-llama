// 05 实验：用 ggml 搭一个 L 层 MLP 计算图，对比 gallocr 规划出的缓冲和所有中间张量之和
// 用法：toy_graph [L] [D] [T]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char ** argv) {
    int L = argc > 1 ? atoi(argv[1]) : 8;      // 层数
    int D = argc > 2 ? atoi(argv[2]) : 1024;   // 隐藏维
    int T = argc > 3 ? atoi(argv[3]) : 512;    // token 数

    struct ggml_init_params ip = { ggml_tensor_overhead() * 4096 + ggml_graph_overhead(), NULL, true };
    struct ggml_context * ctx = ggml_init(ip);   // no_alloc：只建元数据
    struct ggml_context * wctx = ggml_init(ip);  // 权重单独放，算图缓冲里只剩激活

    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, T);
    ggml_set_name(x, "x"); ggml_set_input(x);
    struct ggml_tensor * cur = x;
    size_t naive = 0;
    for (int l = 0; l < L; l++) {
        struct ggml_tensor * w1 = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, D, 4 * D);
        struct ggml_tensor * w2 = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, 4 * D, D);
        struct ggml_tensor * h = ggml_rms_norm(ctx, cur, 1e-6f);
        struct ggml_tensor * u = ggml_mul_mat(ctx, w1, h);     // [4D, T]
        u = ggml_silu(ctx, u);
        struct ggml_tensor * o = ggml_mul_mat(ctx, w2, u);     // [D, T]
        cur = ggml_add(ctx, cur, o);
        naive += ggml_nbytes(h) + 2 * ggml_nbytes(u) + ggml_nbytes(o) + ggml_nbytes(cur);
    }
    ggml_set_name(cur, "out"); ggml_set_output(cur);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cur);

    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, be);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_reserve(ga, gf);
    size_t planned = ggml_gallocr_get_buffer_size(ga, 0);

    int n_nodes = ggml_graph_n_nodes(gf);
    printf("L=%d D=%d T=%d nodes=%d weights=%.0f MiB\n", L, D, T, n_nodes, 2.0 * L * 4 * D * D * 4 / 1048576);
    printf("sum_of_intermediates=%.2f MiB planned=%.2f MiB ratio=%.1fx\n",
           naive / 1048576.0, planned / 1048576.0, (double) naive / planned);
    printf("largest_single=%.2f MiB (u: 4D x T x 4B)\n", 4.0 * D * T * 4 / 1048576.0);

    ggml_gallocr_free(ga);
    ggml_backend_buffer_free(wbuf);
    ggml_backend_free(be);
    ggml_free(wctx);
    ggml_free(ctx);
    return 0;
}
