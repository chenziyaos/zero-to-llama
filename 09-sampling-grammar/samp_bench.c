// 09 实验：在真实 logits 上测每个采样器的耗时和留下的候选数，以及语法约束的拒绝率
// 用法：
//   samp_bench MODEL micro "提示词" [迭代次数] [logits 输出文件]
//   samp_bench MODEL gen   "提示词" GRAMMAR_FILE N_TOKENS TEMP [first|reject]   环境变量 SEED 换随机种子
// micro：对最后一个位置的 logits，每个配置各跑若干次，每次从完整词表的副本开始
// gen  ：带 GBNF 生成，统计 rejection 路径下有多少步要重采样
#include "llama.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

static void quiet(enum ggml_log_level l, const char * t, void * u) { (void) l; (void) t; (void) u; }

static size_t n_alive(const llama_token_data_array * a) {
    size_t n = 0;
    for (size_t i = 0; i < a->size; i++) n += a->data[i].logit != -INFINITY;
    return n;
}

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static llama_token_data * g_base;
static llama_token_data * g_work;
static int g_nv;

static void fill(llama_token_data_array * a) {
    memcpy(g_work, g_base, sizeof(llama_token_data) * g_nv);
    a->data = g_work;
    a->size = g_nv;
    a->selected = -1;
    a->sorted = false;
}

static char * read_file(const char * fn) {
    FILE * f = fopen(fn, "rb");
    if (!f) { perror(fn); exit(1); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char * s = malloc(n + 1);
    if (fread(s, 1, n, f) != (size_t) n) { exit(1); }
    s[n] = 0;
    fclose(f);
    return s;
}

// 一条链：名字 + 若干采样器，最后是否接 dist
static struct llama_sampler * mk(const char * what, const struct llama_vocab * vocab) {
    struct llama_sampler_chain_params cp = llama_sampler_chain_default_params();
    cp.no_perf = true;
    struct llama_sampler * c = llama_sampler_chain_init(cp);
    int nv = llama_vocab_n_tokens(vocab);
    if (!strcmp(what, "greedy")) {
        llama_sampler_chain_add(c, llama_sampler_init_greedy());
    } else if (!strcmp(what, "dist")) {
        llama_sampler_chain_add(c, llama_sampler_init_dist(1));
    } else if (!strcmp(what, "top_k40")) {
        llama_sampler_chain_add(c, llama_sampler_init_top_k(40));
    } else if (!strcmp(what, "top_k20")) {
        llama_sampler_chain_add(c, llama_sampler_init_top_k(20));
    } else if (!strcmp(what, "top_p0.95")) {
        llama_sampler_chain_add(c, llama_sampler_init_top_p(0.95f, 1));
    } else if (!strcmp(what, "min_p0.05")) {
        llama_sampler_chain_add(c, llama_sampler_init_min_p(0.05f, 1));
    } else if (!strcmp(what, "top_nsigma1")) {
        llama_sampler_chain_add(c, llama_sampler_init_top_n_sigma(1.0f));
    } else if (!strcmp(what, "typical0.9")) {
        llama_sampler_chain_add(c, llama_sampler_init_typical(0.9f, 1));
    } else if (!strcmp(what, "penalties")) {
        llama_sampler_chain_add(c, llama_sampler_init_penalties(nv, 64, 1.1f, 0.0f, 0.5f));
    } else if (!strcmp(what, "temp0.8")) {
        llama_sampler_chain_add(c, llama_sampler_init_temp_ext(0.8f, 0.0f, 1.0f));
    } else if (!strcmp(what, "chain_default")) {
        // common 的缺省顺序；penalties、dry、top_n_sigma、typical、xtc 缺省都是空操作
        llama_sampler_chain_add(c, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(c, llama_sampler_init_top_p(0.95f, 0));
        llama_sampler_chain_add(c, llama_sampler_init_min_p(0.05f, 0));
        llama_sampler_chain_add(c, llama_sampler_init_temp_ext(0.8f, 0.0f, 1.0f));
        llama_sampler_chain_add(c, llama_sampler_init_dist(1));
    } else if (!strcmp(what, "chain_qwen3")) {
        // GGUF 里 general.sampling.* 覆盖后的值：top_k 20，top_p 0.95，temp 0.6
        llama_sampler_chain_add(c, llama_sampler_init_top_k(20));
        llama_sampler_chain_add(c, llama_sampler_init_top_p(0.95f, 0));
        llama_sampler_chain_add(c, llama_sampler_init_min_p(0.05f, 0));
        llama_sampler_chain_add(c, llama_sampler_init_temp_ext(0.6f, 0.0f, 1.0f));
        llama_sampler_chain_add(c, llama_sampler_init_dist(1));
    } else if (!strcmp(what, "chain_topk0")) {
        // --top-k 0：top_p 要面对整个词表
        llama_sampler_chain_add(c, llama_sampler_init_top_p(0.95f, 0));
        llama_sampler_chain_add(c, llama_sampler_init_min_p(0.05f, 0));
        llama_sampler_chain_add(c, llama_sampler_init_temp_ext(0.8f, 0.0f, 1.0f));
        llama_sampler_chain_add(c, llama_sampler_init_dist(1));
    } else if (!strcmp(what, "chain_minp_only")) {
        // --top-k 0 --top-p 1：只靠 min_p 截断
        llama_sampler_chain_add(c, llama_sampler_init_min_p(0.05f, 0));
        llama_sampler_chain_add(c, llama_sampler_init_temp_ext(0.8f, 0.0f, 1.0f));
        llama_sampler_chain_add(c, llama_sampler_init_dist(1));
    } else {
        fprintf(stderr, "unknown %s\n", what);
        exit(1);
    }
    return c;
}

static void micro(struct llama_context * ctx, const struct llama_vocab * vocab, int iters, const char * dump,
                  const char * gbnf) {
    float * logits = llama_get_logits_ith(ctx, -1);
    for (int i = 0; i < g_nv; i++) {
        g_base[i].id = i;
        g_base[i].logit = logits[i];
        g_base[i].p = 0.0f;
    }
    if (dump && *dump) {
        FILE * f = fopen(dump, "wb");
        fwrite(logits, sizeof(float), g_nv, f);
        fclose(f);
    }
    // 熵和最大概率，帮助解读候选数
    double mx = -1e30, s = 0, h = 0;
    for (int i = 0; i < g_nv; i++) mx = fmax(mx, logits[i]);
    for (int i = 0; i < g_nv; i++) s += exp(logits[i] - mx);
    for (int i = 0; i < g_nv; i++) {
        double p = exp(logits[i] - mx) / s;
        if (p > 0) h -= p * log(p);
    }
    printf("n_vocab %d p_max %.4f entropy_nats %.3f\n", g_nv, 1.0 / s, h);

    const char * names[] = {"greedy", "dist", "top_k40", "top_k20", "top_p0.95", "min_p0.05", "top_nsigma1", "typical0.9",
                            "penalties", "temp0.8", "chain_default", "chain_qwen3", "chain_topk0", "chain_minp_only"};
    int nn = sizeof(names) / sizeof(names[0]);
    llama_token_data_array a;
    for (int k = 0; k < nn; k++) {
        struct llama_sampler * c = mk(names[k], vocab);
        // penalties 要有历史才会改 logits：先喂 64 个 token
        if (!strcmp(names[k], "penalties")) {
            for (int t = 0; t < 64; t++) llama_sampler_accept(c, (t * 7919) % g_nv);
        }
        fill(&a);
        llama_sampler_apply(c, &a);  // 预热
        size_t kept = n_alive(&a);  // top_n_sigma 把截掉的置成 -inf，不缩数组
        double t0 = now_us();
        for (int it = 0; it < iters; it++) {
            fill(&a);
            llama_sampler_apply(c, &a);
        }
        double per = (now_us() - t0) / iters;
        printf("%-16s kept %6zu  us %9.1f\n", names[k], kept, per);
        llama_sampler_free(c);
    }
    double t0 = now_us();
    for (int it = 0; it < iters; it++) fill(&a);
    printf("%-16s kept %6d  us %9.1f\n", "copy_only", g_nv, (now_us() - t0) / iters);

    if (gbnf) {
        // 语法采样器作用在整个词表上：每个候选都要按 UTF-8 解码再和语法栈比
        char * g = read_file(gbnf);
        struct llama_sampler * gr = llama_sampler_init_grammar(vocab, g, "root");
        int n_it = iters / 10 > 0 ? iters / 10 : 1;
        fill(&a);
        llama_sampler_apply(gr, &a);
        size_t ok = 0;
        for (size_t i = 0; i < a.size; i++) ok += a.data[i].logit != -INFINITY;
        double t1 = now_us();
        for (int it = 0; it < n_it; it++) {
            fill(&a);
            llama_sampler_apply(gr, &a);
        }
        printf("%-16s kept %6zu  us %9.1f\n", "grammar_full", ok, (now_us() - t1) / n_it);
        // rejection 路径：只检查一个 token
        llama_token_data one = {g_base[0].id, 1.0f, 0.0f};
        llama_token_data_array oa = {&one, 1, -1, false};
        t1 = now_us();
        for (int it = 0; it < iters; it++) {
            one.logit = 1.0f;
            llama_sampler_apply(gr, &oa);
        }
        printf("%-16s kept %6d  us %9.1f\n", "grammar_one", 1, (now_us() - t1) / iters);
        llama_sampler_free(gr);
        free(g);
    }
}

static void gen(struct llama_context * ctx, const struct llama_vocab * vocab, const char * gbnf, int n, float temp, int first) {
    char * g = read_file(gbnf);
    struct llama_sampler * gr = llama_sampler_init_grammar(vocab, g, "root");
    struct llama_sampler_chain_params cp = llama_sampler_chain_default_params();
    struct llama_sampler * ch = llama_sampler_chain_init(cp);
    llama_sampler_chain_add(ch, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(ch, llama_sampler_init_top_p(0.95f, 0));
    llama_sampler_chain_add(ch, llama_sampler_init_min_p(0.05f, 0));
    llama_sampler_chain_add(ch, llama_sampler_init_temp_ext(temp, 0.0f, 1.0f));
    const char * se = getenv("SEED");  // 换种子看拒绝率的波动
    uint32_t seed = se ? (uint32_t) atoi(se) : 1;
    llama_sampler_chain_add(ch, temp > 0 ? llama_sampler_init_dist(seed) : llama_sampler_init_greedy());

    int rejected = 0, steps = 0;
    int depth = 0, opened = 0, in_str = 0, esc = 0;
    double t_samp = 0;
    llama_token_data_array a;
    for (int i = 0; i < n; i++) {
        float * logits = llama_get_logits_ith(ctx, -1);
        double t0 = now_us();
        for (int j = 0; j < g_nv; j++) {
            g_base[j].id = j;
            g_base[j].logit = logits[j];
            g_base[j].p = 0.0f;
        }
        fill(&a);
        if (first) llama_sampler_apply(gr, &a);
        llama_sampler_apply(ch, &a);
        llama_token id = a.data[a.selected].id;
        if (!first) {
            llama_token_data one = {id, 1.0f, 0.0f};
            llama_token_data_array oa = {&one, 1, -1, false};
            llama_sampler_apply(gr, &oa);
            if (one.logit == -INFINITY) {
                rejected++;
                fill(&a);
                llama_sampler_apply(gr, &a);
                llama_sampler_apply(ch, &a);
                id = a.data[a.selected].id;
            }
        }
        llama_sampler_accept(gr, id);
        llama_sampler_accept(ch, id);
        t_samp += now_us() - t0;
        steps++;
        char buf[256];
        int m = llama_token_to_piece(vocab, id, buf, sizeof(buf) - 1, 0, false);
        if (m > 0) { buf[m] = 0; fputs(buf, stdout); }
        if (llama_vocab_is_eog(vocab, id)) break;
        // JSON 收尾以后语法只允许空白，继续生成没有意义：按字符串外的括号深度判断是否已闭合
        int closed = 0;
        for (int c = 0; c < m; c++) {
            char ch = buf[c];
            if (in_str) {
                if (esc) esc = 0;
                else if (ch == '\\') esc = 1;
                else if (ch == '"') in_str = 0;
            } else if (ch == '"') in_str = 1;
            else if (ch == '{' || ch == '[') { depth++; opened = 1; }
            else if (ch == '}' || ch == ']') { depth--; if (opened && depth == 0) closed = 1; }
        }
        if (closed) break;
        llama_batch b = llama_batch_get_one(&id, 1);
        if (llama_decode(ctx, b)) break;
    }
    printf("\n[gen] mode %s temp %.2f steps %d rejected %d (%.1f%%) sample_us_per_token %.1f\n",
           first ? "first" : "reject", temp, steps, rejected, 100.0 * rejected / steps, t_samp / steps);
    llama_sampler_free(gr);
    llama_sampler_free(ch);
    free(g);
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model micro|gen prompt ...\n", argv[0]);
        return 1;
    }
    llama_log_set(quiet, NULL);
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    struct llama_model * m = llama_model_load_from_file(argv[1], mp);
    if (!m) return 1;
    const struct llama_vocab * vocab = llama_model_get_vocab(m);
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    struct llama_context * ctx = llama_init_from_model(m, cp);
    if (!ctx) return 1;
    g_nv = llama_vocab_n_tokens(vocab);
    g_base = malloc(sizeof(llama_token_data) * g_nv);
    g_work = malloc(sizeof(llama_token_data) * g_nv);

    const char * prompt = argv[3];
    llama_token toks[1024];
    int nt = llama_tokenize(vocab, prompt, strlen(prompt), toks, 1024, true, true);
    if (nt <= 0) return 1;
    if (llama_decode(ctx, llama_batch_get_one(toks, nt))) return 1;

    if (!strcmp(argv[2], "micro")) {
        int iters = argc > 4 ? atoi(argv[4]) : 200;
        micro(ctx, vocab, iters, argc > 5 ? argv[5] : NULL, argc > 6 ? argv[6] : NULL);
    } else {
        if (argc < 7) return 1;
        gen(ctx, vocab, argv[4], atoi(argv[5]), atof(argv[6]), argc > 7 && !strcmp(argv[7], "first"));
    }
    free(g_base);
    free(g_work);
    llama_free(ctx);
    llama_model_free(m);
    return 0;
}
