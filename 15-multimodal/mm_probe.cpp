// mm_probe：直接调 libmtmd，不开端口
//
//   mm_probe count  -m M --mmproj P [--max-tok N] WxH ...
//       占位位图（data = nullptr）只走预处理和分块，不跑编码器；打印每个分辨率切成几块、多少 token
//   mm_probe time   -m M --mmproj P [-t 4] [--max-tok N] [--reps 3] (file | WxH) ...
//       真跑：tokenize、逐块编码、把嵌入喂进语言模型；再用同样长度的文本 token 做一遍预填充对照
//   mm_probe batch  -m M --mmproj P [-t 4] [--bmax N] file
//       同一张图切出的块，一块一块编码 vs 交给 mtmd_batch 一次编码
//
// 合成图是确定性的彩色渐变加噪点，不依赖外部图片。
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

static void quiet(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
}

struct opts {
    std::string mode, model, mmproj;
    int threads = 4, max_tok = -1, min_tok = -1, reps = 3, bmax = 1024, ub = 2048;
    std::vector<std::string> items;
};

static bool parse_wh(const std::string & s, int & w, int & h) {
    return sscanf(s.c_str(), "%dx%d", &w, &h) == 2 && w > 0 && h > 0;
}

static std::vector<unsigned char> synth(int w, int h) {
    std::vector<unsigned char> d((size_t) w * h * 3);
    uint32_t r = 12345;
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
        r = r * 1103515245u + 12345u;
        unsigned char n = (r >> 16) & 31;
        size_t i = ((size_t) y * w + x) * 3;
        d[i + 0] = (unsigned char) (x * 255 / w) ^ n;
        d[i + 1] = (unsigned char) (y * 255 / h) ^ n;
        d[i + 2] = (unsigned char) ((x + y) * 255 / (w + h));
    }
    return d;
}

struct env {
    llama_model * model = nullptr;
    llama_context * lctx = nullptr;
    mtmd_context * mctx = nullptr;
};

static env load(const opts & o, bool need_ctx) {
    env e;
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    e.model = llama_model_load_from_file(o.model.c_str(), mp);
    if (!e.model) { fprintf(stderr, "load model failed\n"); exit(1); }
    if (need_ctx) {
        auto cp = llama_context_default_params();
        cp.n_ctx = 16384; cp.n_batch = o.ub; cp.n_ubatch = o.ub;
        cp.n_threads = cp.n_threads_batch = o.threads;
        cp.no_perf = false;
        e.lctx = llama_init_from_model(e.model, cp);
        if (!e.lctx) { fprintf(stderr, "init ctx failed\n"); exit(1); }
    }
    auto mpar = mtmd_context_params_default();
    mpar.use_gpu = false;
    mpar.print_timings = false;
    mpar.n_threads = o.threads;
    mpar.warmup = need_ctx;
    mpar.image_max_tokens = o.max_tok;
    mpar.image_min_tokens = o.min_tok;
    mpar.batch_max_tokens = o.bmax;
    e.mctx = mtmd_init_from_file(o.mmproj.c_str(), e.model, mpar);
    if (!e.mctx) { fprintf(stderr, "load mmproj failed\n"); exit(1); }
    return e;
}

static mtmd_bitmap * make_bitmap(const env & e, const std::string & it, bool placeholder, int & w, int & h) {
    if (parse_wh(it, w, h)) {
        if (placeholder) return mtmd_bitmap_init(w, h, nullptr);
        auto d = synth(w, h);
        auto * b = mtmd_bitmap_init(w, h, d.data());
        mtmd_bitmap_set_id(b, it.c_str());
        return b;
    }
    auto r = mtmd_helper_bitmap_init_from_file(e.mctx, it.c_str(), placeholder, mtmd_helper_init_opt_default());
    if (!r.bitmap) { fprintf(stderr, "cannot load %s\n", it.c_str()); exit(1); }
    w = mtmd_bitmap_get_nx(r.bitmap); h = mtmd_bitmap_get_ny(r.bitmap);
    return r.bitmap;
}

static mtmd_input_chunks * tokenize(const env & e, mtmd_bitmap * b) {
    std::string p = std::string("Describe: ") + mtmd_default_marker() + "\n";
    mtmd_input_text t{p.c_str(), p.size(), true, true};
    auto * ch = mtmd_input_chunks_init();
    const mtmd_bitmap * bs[1] = {b};
    int r = mtmd_tokenize(e.mctx, ch, &t, bs, 1);
    if (r != 0) { fprintf(stderr, "tokenize failed %d\n", r); exit(1); }
    return ch;
}

struct tally { int n_media = 0, n_media_tok = 0, n_text_tok = 0, n_pos = 0; std::vector<int> per; };

static tally count_chunks(const mtmd_input_chunks * ch) {
    tally t;
    for (size_t i = 0; i < mtmd_input_chunks_size(ch); i++) {
        auto * c = mtmd_input_chunks_get(ch, i);
        int n = (int) mtmd_input_chunk_get_n_tokens(c);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_TEXT) t.n_text_tok += n;
        else { t.n_media++; t.n_media_tok += n; t.per.push_back(n); }
    }
    t.n_pos = (int) mtmd_helper_get_n_pos(ch);
    return t;
}

static int do_count(const opts & o) {
    env e = load(o, false);
    printf("# model=%s mmproj=%s max_tok=%d\n", o.model.c_str(), o.mmproj.c_str(), o.max_tok);
    printf("%-11s %6s %9s %9s %7s  %s\n", "size", "chunks", "media_tok", "text_tok", "n_pos", "per_chunk");
    for (auto & it : o.items) {
        int w, h;
        auto * b = make_bitmap(e, it, true, w, h);
        auto * ch = tokenize(e, b);
        auto t = count_chunks(ch);
        std::string per;
        for (size_t i = 0; i < t.per.size(); i++) { if (i) per += ","; per += std::to_string(t.per[i]); }
        printf("%-11s %6d %9d %9d %7d  %s\n", (std::to_string(w) + "x" + std::to_string(h)).c_str(),
               t.n_media, t.n_media_tok, t.n_text_tok, t.n_pos, per.c_str());
        mtmd_input_chunks_free(ch);
        mtmd_bitmap_free(b);
    }
    return 0;
}

static double ms_since(int64_t t0) { return (ggml_time_us() - t0) / 1000.0; }

// 文本对照：n 个 token 一次性预填充，返回毫秒
static double text_prefill(const env & e, int n) {
    llama_memory_clear(llama_get_memory(e.lctx), true);
    const llama_vocab * v = llama_model_get_vocab(e.model);
    int nv = llama_vocab_n_tokens(v);
    std::vector<llama_token> toks(n);
    for (int i = 0; i < n; i++) toks[i] = 1000 + (i * 7919) % (nv - 2000);
    int64_t t0 = ggml_time_us();
    int nb = (int) llama_n_batch(e.lctx);
    for (int i = 0; i < n; i += nb) {
        int k = std::min(nb, n - i);
        auto b = llama_batch_get_one(toks.data() + i, k);
        if (llama_decode(e.lctx, b) != 0) { fprintf(stderr, "decode failed\n"); exit(1); }
    }
    llama_synchronize(e.lctx);
    return ms_since(t0);
}

static int do_time(const opts & o) {
    env e = load(o, true);
    const int n_embd = llama_model_n_embd_inp(e.model);
    printf("# model=%s mmproj=%s threads=%d max_tok=%d reps=%d\n", o.model.c_str(), o.mmproj.c_str(), o.threads, o.max_tok, o.reps);
    printf("%-11s %6s %9s %8s %9s %9s %9s %9s %8s\n", "size", "chunks", "media_tok", "tok_ms", "enc_ms", "dec_ms", "text_ms", "enc/tok", "img/txt");
    for (auto & it : o.items) {
        std::vector<double> vt, ve, vd, vx;
        tally t;
        int w = 0, h = 0;
        for (int r = 0; r < o.reps; r++) {
            auto * b = make_bitmap(e, it, false, w, h);
            int64_t t0 = ggml_time_us();
            auto * ch = tokenize(e, b);
            double tok = ms_since(t0);
            t = count_chunks(ch);
            llama_memory_clear(llama_get_memory(e.lctx), true);
            double enc = 0, dec = 0;
            llama_pos n_past = 0;
            for (size_t i = 0; i < mtmd_input_chunks_size(ch); i++) {
                auto * c = mtmd_input_chunks_get(ch, i);
                llama_pos np = n_past;
                if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                    int64_t t1 = ggml_time_us();
                    mtmd_helper_eval_chunk_single(e.mctx, e.lctx, c, n_past, 0, llama_n_batch(e.lctx), false, &np);
                    llama_synchronize(e.lctx);
                    dec += ms_since(t1);
                } else {
                    int64_t t1 = ggml_time_us();
                    if (mtmd_encode_chunk(e.mctx, c) != 0) { fprintf(stderr, "encode failed\n"); return 1; }
                    enc += ms_since(t1);
                    float * embd = mtmd_get_output_embd(e.mctx);
                    size_t n = mtmd_input_chunk_get_n_tokens(c);
                    std::vector<float> copy(embd, embd + n * n_embd);
                    int64_t t2 = ggml_time_us();
                    mtmd_helper_decode_image_chunk(e.mctx, e.lctx, c, copy.data(), n_past, 0, llama_n_batch(e.lctx), &np, nullptr, nullptr);
                    llama_synchronize(e.lctx);
                    dec += ms_since(t2);
                }
                n_past = np;
            }
            double txt = text_prefill(e, t.n_media_tok + t.n_text_tok);
            vt.push_back(tok); ve.push_back(enc); vd.push_back(dec); vx.push_back(txt);
            mtmd_input_chunks_free(ch);
            mtmd_bitmap_free(b);
        }
        auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
        double enc = med(ve), dec = med(vd), txt = med(vx);
        printf("%-11s %6d %9d %8.1f %9.1f %9.1f %9.1f %9.3f %8.2f\n",
               (std::to_string(w) + "x" + std::to_string(h)).c_str(), t.n_media, t.n_media_tok,
               med(vt), enc, dec, txt, enc / std::max(1, t.n_media_tok), (enc + dec) / txt);
        fflush(stdout);
    }
    return 0;
}

static int do_batch(const opts & o) {
    env e = load(o, true);
    printf("# model=%s mmproj=%s threads=%d bmax=%d\n", o.model.c_str(), o.mmproj.c_str(), o.threads, o.bmax);
    for (auto & it : o.items) {
        int w, h;
        for (int r = 0; r < o.reps; r++) {
            auto * b = make_bitmap(e, it, false, w, h);
            auto * ch = tokenize(e, b);
            std::vector<const mtmd_input_chunk *> media;
            for (size_t i = 0; i < mtmd_input_chunks_size(ch); i++) {
                auto * c = mtmd_input_chunks_get(ch, i);
                if (mtmd_input_chunk_get_type(c) != MTMD_INPUT_CHUNK_TYPE_TEXT) media.push_back(c);
            }
            int64_t t0 = ggml_time_us();
            for (auto * c : media) mtmd_encode_chunk(e.mctx, c);
            double one = ms_since(t0);
            t0 = ggml_time_us();
            int n_batches = 0;
            size_t i = 0;
            while (i < media.size()) {
                auto * mb = mtmd_batch_init(e.mctx);
                mtmd_batch_add_chunk(mb, media[i++]);
                while (i < media.size() && mtmd_batch_add_chunk(mb, media[i]) == 0) i++;
                if (mtmd_batch_encode(mb) != 0) { fprintf(stderr, "batch encode failed\n"); return 1; }
                mtmd_batch_free(mb);
                n_batches++;
            }
            double bat = ms_since(t0);
            printf("%dx%d rep=%d chunks=%zu one_by_one_ms=%.1f batched_ms=%.1f n_batches=%d\n", w, h, r, media.size(), one, bat, n_batches);
            fflush(stdout);
            mtmd_input_chunks_free(ch);
            mtmd_bitmap_free(b);
        }
    }
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: mm_probe count|time|batch -m M --mmproj P ...\n"); return 1; }
    opts o;
    o.mode = argv[1];
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1); } return std::string(argv[++i]); };
        if (a == "-m") o.model = next();
        else if (a == "--mmproj") o.mmproj = next();
        else if (a == "-t") o.threads = atoi(next().c_str());
        else if (a == "--max-tok") o.max_tok = atoi(next().c_str());
        else if (a == "--min-tok") o.min_tok = atoi(next().c_str());
        else if (a == "--reps") o.reps = atoi(next().c_str());
        else if (a == "--bmax") o.bmax = atoi(next().c_str());
        else if (a == "-ub") o.ub = atoi(next().c_str());
        else o.items.push_back(a);
    }
    llama_log_set(quiet, nullptr);
    mtmd_helper_log_set(quiet, nullptr);
    llama_backend_init();
    int r = 1;
    if (o.mode == "count") r = do_count(o);
    else if (o.mode == "time") r = do_time(o);
    else if (o.mode == "batch") r = do_batch(o);
    else fprintf(stderr, "unknown mode %s\n", o.mode.c_str());
    llama_backend_free();
    return r;
}
