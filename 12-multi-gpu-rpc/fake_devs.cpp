// fake_devs.cpp：在一台只有 CPU 的机器上注册 N 个"假 GPU"，让 llama.cpp 走一遍多卡的切层、
// 调度和显存账本。权重不真分配（no_alloc），不启动任何服务，不监听端口。
//
// 每个假设备：
//   - 类型报 GPU，空闲显存由 --free 指定，device_id 互不相同（否则会被去重）
//   - buffer type 复制 CPU 的分配函数，但 device 指向自己、is_host=false，像一块独立显存
//   - supports_op 交给 CPU 设备判断；backend 用 ggml_backend_cpu_init() 再改掉 device 指针
//   - --rpc K 另注册一个名叫 "RPC" 的 reg（在 FAKE 之后注册），llama.cpp 会把它排到设备表最前面
//   - --free 依次给 RPC 设备、FAKE 设备
//   - --events 1 时 caps.async/events 报 1，满足流水线并行的开启条件（event_new 仍为 NULL）
//
// 编译：g++ -O2 -std=c++17 fake_devs.cpp -I$LD/include -I$LD/ggml/include -I$LD/src -I$LD/ggml/src
//        -L$B -lllama -lggml -lggml-base -lggml-cpu -Wl,-rpath,$B
// 运行：./fake_devs MODEL [--n 2] [--rpc K] [--free 8000,8000] [--ts 1,1] [--events 0|1] [--sm layer|tensor]
//                         [--ngl -1] [--ctx 4096] [--ub 512]

#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct fake_dev {
    std::string name, desc, id;
    struct ggml_backend_reg * reg;
    size_t free_b = 0;
    struct ggml_backend_buffer_type buft;
    struct ggml_backend_device dev;
};

// 两个 reg：FAKE 先注册，RPC 后注册，用来验证 llama.cpp 会把 RPC 设备挪到最前
static std::vector<fake_dev *> g_fake, g_rpc;
static bool g_events = false;
static struct ggml_backend_reg g_reg_fake, g_reg_rpc;

static fake_dev * ctx_of(ggml_backend_dev_t d) { return (fake_dev *) d->context; }

// ---- buffer type ----
static const char * buft_name(ggml_backend_buffer_type_t b) { return ((fake_dev *) b->context)->name.c_str(); }
static ggml_backend_buffer_t buft_alloc(ggml_backend_buffer_type_t b, size_t size) {
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    if (buf) buf->buft = b;   // 账本按 buft 记，挂回自己名下
    return buf;
}
static size_t buft_align(ggml_backend_buffer_type_t) { return ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type()); }
static bool   buft_is_host(ggml_backend_buffer_type_t) { return false; }

// ---- device ----
static const char * dev_name(ggml_backend_dev_t d) { return ctx_of(d)->name.c_str(); }
static const char * dev_desc(ggml_backend_dev_t d) { return ctx_of(d)->desc.c_str(); }
static void dev_mem(ggml_backend_dev_t d, size_t * f, size_t * t) { *f = ctx_of(d)->free_b; *t = ctx_of(d)->free_b; }
static enum ggml_backend_dev_type dev_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; }
static void dev_props(ggml_backend_dev_t d, struct ggml_backend_dev_props * p) {
    p->name = dev_name(d); p->description = dev_desc(d);
    dev_mem(d, &p->memory_free, &p->memory_total);
    p->type = GGML_BACKEND_DEVICE_TYPE_GPU;
    p->device_id = ctx_of(d)->id.c_str();
    p->caps.async = g_events; p->caps.host_buffer = false; p->caps.buffer_from_host_ptr = false;
    p->caps.events = g_events; p->caps.mmap_support = false;
}
static const char * be_name(ggml_backend_t be) { return dev_name(be->device); }
static ggml_backend_t dev_init(ggml_backend_dev_t d, const char *) {
    ggml_backend_t be = ggml_backend_cpu_init();
    if (be) { be->device = d; be->iface.get_name = be_name; }   // 计算仍是 CPU 内核，名字换成设备名
    return be;
}
static ggml_backend_buffer_type_t dev_buft(ggml_backend_dev_t d) { return &ctx_of(d)->buft; }
static bool dev_supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) {
    return ggml_backend_dev_supports_op(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), op);
}
static bool dev_supports_buft(ggml_backend_dev_t d, ggml_backend_buffer_type_t b) { return b == &ctx_of(d)->buft; }

// ---- reg ----
static std::vector<fake_dev *> & devs_of(ggml_backend_reg_t r) { return r == &g_reg_rpc ? g_rpc : g_fake; }
static const char * reg_name(ggml_backend_reg_t r) { return r == &g_reg_rpc ? "RPC" : "FAKE"; }
static size_t reg_count(ggml_backend_reg_t r) { return devs_of(r).size(); }
static ggml_backend_dev_t reg_get(ggml_backend_reg_t r, size_t i) { return &devs_of(r)[i]->dev; }
static void * reg_proc(ggml_backend_reg_t, const char *) { return NULL; }

// 前 n_rpc 个空闲显存值给 RPC 设备，其余给 FAKE 设备
static void make_devs(int n_fake, int n_rpc, const std::vector<double> & free_mib) {
    int k = 0;
    for (int r = 0; r < 2; r++) {
        struct ggml_backend_reg * reg = r == 0 ? &g_reg_fake : &g_reg_rpc;
        reg->api_version = GGML_BACKEND_API_VERSION;
        reg->iface = { reg_name, reg_count, reg_get, reg_proc };
        reg->context = NULL;
    }
    for (int i = 0; i < n_rpc + n_fake; i++) {
        bool is_rpc = i < n_rpc;
        int idx = is_rpc ? i : i - n_rpc;
        fake_dev * f = new fake_dev();
        f->reg  = is_rpc ? &g_reg_rpc : &g_reg_fake;
        f->name = std::string(is_rpc ? "RPC" : "FAKE") + std::to_string(idx);
        f->desc = std::string(is_rpc ? "fake rpc " : "fake gpu ") + std::to_string(idx);
        f->id   = std::string(is_rpc ? "rpc:" : "fake:") + std::to_string(idx);
        double m = free_mib.empty() ? 8000.0 : free_mib[k < (int) free_mib.size() ? k : free_mib.size() - 1];
        k++;
        f->free_b = (size_t) (m * 1024 * 1024);
        f->buft.iface = { buft_name, buft_alloc, NULL, buft_align, NULL, NULL, NULL, buft_is_host };
        f->buft.device = &f->dev;
        f->buft.context = f;
        f->dev.iface = { dev_name, dev_desc, dev_mem, dev_type, dev_props, dev_init, dev_buft, NULL, NULL,
                         dev_supports_op, dev_supports_buft, NULL, NULL, NULL, NULL };
        f->dev.reg = f->reg;
        f->dev.context = f;
        (is_rpc ? g_rpc : g_fake).push_back(f);
    }
    if (n_fake) ggml_backend_register(&g_reg_fake);
    if (n_rpc)  ggml_backend_register(&g_reg_rpc);
}

// ---- 日志：只留和多卡相关的行，逐层分配压成区间 ----
static const char * KEEP[] = { "using device", "pipeline parallelism", "nodes = ", "sched copies",
    "compute buffer size", "offloaded", "model buffer size", "KV buffer size",
    "retrying without", "skipping device", "creating a Meta", ": - device", "flash_attn since", NULL };
static std::string g_run_dev; static int g_run_a = -1, g_run_b = -1;
static void flush_run() {
    if (g_run_a >= 0) printf("layers %3d-%3d -> %s\n", g_run_a, g_run_b, g_run_dev.c_str());
    g_run_a = g_run_b = -1;
}
// GGML_SCHED_DEBUG=1 时调度器分几段打印 "## SPLIT #k: 后端 # n inputs: [张量 (大小)] ..."，拼起来原样输出
static bool g_in_split = false;
static void log_cb(enum ggml_log_level, const char * text, void *) {
    if (strstr(text, "## SPLIT #")) { flush_run(); g_in_split = true; fputs(text + (text[0] == '\n'), stdout); return; }
    if (g_in_split) { fputs(text, stdout); if (strcmp(text, "\n") == 0) g_in_split = false; return; }
    const char * p = strstr(text, "assigned to device ");
    if (p) {
        int il = atoi(strstr(text, "layer") + 5);
        std::string dev(p + strlen("assigned to device "));
        dev = dev.substr(0, dev.find(','));
        if (dev == g_run_dev && il == g_run_b + 1) { g_run_b = il; return; }
        flush_run(); g_run_dev = dev; g_run_a = g_run_b = il; return;
    }
    for (int i = 0; KEEP[i]; i++) if (strstr(text, KEEP[i])) { flush_run(); fputs(text, stdout); return; }
}

static std::vector<double> parse_list(const char * s) {
    std::vector<double> v; std::string t(s); size_t p = 0;
    while (p <= t.size()) { size_t q = t.find(',', p); if (q == std::string::npos) q = t.size();
        if (q > p) v.push_back(atof(t.substr(p, q - p).c_str()));
        p = q + 1; }
    return v;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "用法：%s MODEL [--n N] [--rpc K] [--free a,b] [--ts a,b] [--events 0|1] [--ngl K] [--ctx C] [--ub U]\n", argv[0]); return 1; }
    const char * path = argv[1];
    int n = 2, n_rpc = 0, sm = 1, ngl = -1, ctx = 4096, ub = 512;
    std::vector<double> free_mib, ts;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--n" && i + 1 < argc) n = atoi(argv[++i]);
        else if (a == "--rpc" && i + 1 < argc) n_rpc = atoi(argv[++i]);
        else if (a == "--free" && i + 1 < argc) free_mib = parse_list(argv[++i]);
        else if (a == "--ts" && i + 1 < argc) ts = parse_list(argv[++i]);
        else if (a == "--events" && i + 1 < argc) g_events = atoi(argv[++i]) != 0;
        else if (a == "--sm" && i + 1 < argc) sm = std::string(argv[++i]) == "tensor" ? 3 : 1;
        else if (a == "--ngl" && i + 1 < argc) ngl = atoi(argv[++i]);
        else if (a == "--ctx" && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (a == "--ub" && i + 1 < argc) ub = atoi(argv[++i]);
    }
    llama_log_set(log_cb, NULL);
    ggml_backend_load_all();
    make_devs(n, n_rpc, free_mib);
    llama_backend_init();

    printf("== FAKE × %d + RPC × %d  events=%d  ngl=%d  ctx=%d  ub=%d\n", n, n_rpc, (int) g_events, ngl, ctx, ub);

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    mp.no_alloc = true;
    mp.split_mode = (enum llama_split_mode) sm;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    std::vector<float> tsf(llama_max_devices(), 0.0f);
    if (!ts.empty()) { for (size_t i = 0; i < ts.size() && i < tsf.size(); i++) tsf[i] = (float) ts[i]; mp.tensor_split = tsf.data(); }

    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) { fprintf(stderr, "模型加载失败\n"); return 1; }
    printf("model devices = %d\n", llama_model_n_devices(model));

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = ctx; cp.n_batch = ub; cp.n_ubatch = ub; cp.n_seq_max = 1;
    if (sm == 3) cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context * lctx = llama_init_from_model(model, cp);
    if (!lctx) { fprintf(stderr, "上下文创建失败\n"); llama_model_free(model); return 1; }

    llama_memory_breakdown mb = llama_get_memory_breakdown(lctx);
    printf("%-12s %10s %10s %10s %10s\n", "buffer", "model", "context", "compute", "total");
    double sum = 0;
    for (auto & kv : mb) {
        const double M = 1024.0 * 1024.0;
        printf("%-12s %10.1f %10.1f %10.1f %10.1f\n", ggml_backend_buft_name(kv.first),
               kv.second.model / M, kv.second.context / M, kv.second.compute / M, kv.second.total() / M);
        sum += kv.second.total() / M;
    }
    printf("%-12s %43.1f MiB\n", "sum", sum);

    llama_free(lctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
