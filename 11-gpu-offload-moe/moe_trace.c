// 11 实验：用 cb_eval 抓 MoE 路由，记下每个 token 在每层选了哪几个专家
// 用法：
//   moe_trace MODEL text TEXT_FILE OUT.bin [N_TOKENS] [CHUNK]   按 CHUNK 个 token 一段做 prefill，相当于逐个 decode 这段文字
//   moe_trace MODEL gen  "提示词"  OUT.bin [N_GEN]              贪心生成 N_GEN 个 token，记下生成时的路由
// 输出：int32 头 {n_layer, n_expert, n_used, n_tok}，后面是 n_tok × n_layer × n_used 个 uint8 专家号
// 最后一层的 FFN 只对要输出 logits 的 token 计算，text 模式给每个位置都要 logits
// 因果注意力下，一段文字做 prefill 时第 i 个 token 的路由，和逐个 decode 到第 i 个 token 时一样
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static void quiet(enum ggml_log_level l, const char * t, void * u) { (void) l; (void) t; (void) u; }

static int g_layer, g_expert, g_used;
static uint8_t * g_ub;     // 当前 ubatch：[n_tokens][n_layer][n_used]
static int g_ub_tokens;
static int g_seen;         // 本 ubatch 里抓到了几层
static int g_record = 1;   // gen 模式的提示词 prefill 不记

static bool cb(struct ggml_tensor * t, bool ask, void * ud) {
    (void) ud;
    if (strncmp(t->name, "ffn_moe_topk-", 13) != 0) return ask ? false : true;
    if (ask) return g_record;
    if (!g_record) return true;
    int il = atoi(t->name + 13);
    int k = (int) t->ne[0], nt = (int) t->ne[1];
    if (il < 0 || il >= g_layer || k != g_used || nt != g_ub_tokens) {
        fprintf(stderr, "unexpected %s ne=%lld,%lld\n", t->name, (long long) t->ne[0], (long long) t->ne[1]);
        exit(1);
    }
    // topk 是 argsort 结果的视图，行距沿用 argsort 的 nb[1]；本机只有 CPU，数据在主存里
    if (!t->buffer || !ggml_backend_buffer_is_host(t->buffer)) { fprintf(stderr, "non-host tensor\n"); exit(1); }
    for (int j = 0; j < nt; j++) {
        const int32_t * row = (const int32_t *) ((const char *) t->data + j * t->nb[1]);
        for (int e = 0; e < k; e++) g_ub[((size_t) j * g_layer + il) * g_used + e] = (uint8_t) row[e];
    }
    g_seen++;
    return true;
}

static char * read_file(const char * fn, long * n) {
    FILE * f = fopen(fn, "rb");
    if (!f) { perror(fn); exit(1); }
    fseek(f, 0, SEEK_END);
    *n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char * s = malloc(*n + 1);
    if (fread(s, 1, *n, f) != (size_t) *n) exit(1);
    s[*n] = 0;
    fclose(f);
    return s;
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s MODEL text|gen INPUT OUT [N] [CHUNK]\n", argv[0]); return 1; }
    const int is_text = !strcmp(argv[2], "text");
    const int n_want = argc > 5 ? atoi(argv[5]) : (is_text ? 16384 : 256);
    const int chunk = argc > 6 ? atoi(argv[6]) : 512;

    llama_log_set(quiet, NULL);
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    struct llama_model * m = llama_model_load_from_file(argv[1], mp);
    if (!m) return 1;
    const struct llama_vocab * vocab = llama_model_get_vocab(m);

    char arch[64] = {0}, key[128], val[64];
    llama_model_meta_val_str(m, "general.architecture", arch, sizeof arch);
    snprintf(key, sizeof key, "%s.expert_count", arch);
    if (llama_model_meta_val_str(m, key, val, sizeof val) < 0) { fprintf(stderr, "not a MoE model\n"); return 1; }
    g_expert = atoi(val);
    snprintf(key, sizeof key, "%s.expert_used_count", arch);
    llama_model_meta_val_str(m, key, val, sizeof val);
    g_used = atoi(val);
    g_layer = llama_model_n_layer(m);
    if (g_expert > 256) { fprintf(stderr, "uint8 only holds 256 experts\n"); return 1; }

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = is_text ? chunk : 2048;
    cp.n_batch = cp.n_ubatch = is_text ? chunk : 512;
    cp.n_threads = cp.n_threads_batch = 4;
    cp.cb_eval = cb;
    cp.cb_eval_user_data = NULL;
    struct llama_context * ctx = llama_init_from_model(m, cp);
    if (!ctx) return 1;

    const size_t row = (size_t) g_layer * g_used;
    uint8_t * all = malloc(row * (size_t) n_want);
    g_ub = malloc(row * (size_t) (chunk > 512 ? chunk : 512));
    int n_tok = 0;

    if (is_text) {
        long nbytes;
        char * text = read_file(argv[3], &nbytes);
        int cap = (int) (nbytes / 2 + 16);
        llama_token * toks = malloc(sizeof(llama_token) * cap);
        int nt = llama_tokenize(vocab, text, (int) nbytes, toks, cap, false, false);
        if (nt < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
        for (int off = 0; off + chunk <= nt && n_tok + chunk <= n_want; off += chunk) {
            llama_memory_clear(llama_get_memory(ctx), true);
            g_ub_tokens = chunk; g_seen = 0;
            // 最后一层只算要出 logits 的 token，所以每个位置都要 logits
            llama_batch b = llama_batch_init(chunk, 0, 1);
            for (int j = 0; j < chunk; j++) {
                b.token[j] = toks[off + j]; b.pos[j] = j; b.n_seq_id[j] = 1; b.seq_id[j][0] = 0; b.logits[j] = 1;
            }
            b.n_tokens = chunk;
            if (llama_decode(ctx, b)) return 1;
            llama_batch_free(b);
            if (g_seen != g_layer) { fprintf(stderr, "saw %d layers\n", g_seen); return 1; }
            memcpy(all + row * n_tok, g_ub, row * chunk);
            n_tok += chunk;
        }
        free(toks); free(text);
    } else {
        llama_token toks[1024];
        int nt = llama_tokenize(vocab, argv[3], strlen(argv[3]), toks, 1024, true, true);
        g_record = 0;
        if (llama_decode(ctx, llama_batch_get_one(toks, nt))) return 1;
        g_record = 1;
        llama_token id = 0;
        for (int i = 0; i < n_want; i++) {
            const float * lg = llama_get_logits_ith(ctx, -1);
            int nv = llama_vocab_n_tokens(vocab), best = 0;
            for (int v = 1; v < nv; v++) if (lg[v] > lg[best]) best = v;
            id = best;
            if (llama_vocab_is_eog(vocab, id)) break;
            g_ub_tokens = 1; g_seen = 0;
            if (llama_decode(ctx, llama_batch_get_one(&id, 1))) break;
            memcpy(all + row * n_tok, g_ub, row);
            n_tok++;
        }
    }

    FILE * f = fopen(argv[4], "wb");
    int32_t hdr[4] = { g_layer, g_expert, g_used, n_tok };
    fwrite(hdr, sizeof hdr, 1, f);
    fwrite(all, row, n_tok, f);
    fclose(f);
    printf("%s: arch=%s layers=%d experts=%d used=%d tokens=%d -> %s\n", argv[2], arch, g_layer, g_expert, g_used, n_tok, argv[4]);
    free(all); free(g_ub);
    llama_free(ctx);
    llama_model_free(m);
    return 0;
}
