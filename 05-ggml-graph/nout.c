// 05 实验：n_outputs_max 对 CPU compute buffer 的影响
// 用法：nout MODEL N_UBATCH N_OUTPUTS_MAX [FA:0/1] [N_CTX]
#include "llama.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model ub nout [fa] [ctx]\n", argv[0]); return 1; }
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    struct llama_model * m = llama_model_load_from_file(argv[1], mp);
    if (!m) return 1;
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = argc > 5 ? atoi(argv[5]) : 4096;
    cp.n_batch = 2048;
    cp.n_ubatch = atoi(argv[2]);
    cp.n_outputs_max = atoi(argv[3]);
    cp.flash_attn_type = (argc > 4 && atoi(argv[4]) == 0) ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_threads = 4;
    struct llama_context * c = llama_init_from_model(m, cp);
    if (!c) return 1;
    llama_free(c);
    llama_model_free(m);
    return 0;
}
