// 把一个 GGUF 在给定 token 序列上的全部位置 logits 写成 float32 二进制，
// 可选再把每层的中间张量（按名字前缀挑）一起写出来，给 compare.py 对拍 HF。
//
// 用法：
//   dump_logits -m model.gguf -i tokens.bin -o out_dir [-t 4] [--fa 0|1]
//               [--keep l_out,attn_norm] [--kv key=str:val] [--kv key=float:val]
// tokens.bin 是 int32 小端序列（hf_ref.py 写的）。输出：
//   out_dir/logits.bin      n_tokens x n_vocab，float32
//   out_dir/<name>.bin      每个被 --keep 选中的张量，按 ggml 内存顺序原样写出
//   out_dir/meta.txt        n_tokens、n_vocab、各张量的形状
// 不监听任何端口。
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

struct keep_ctx {
    std::vector<std::string> prefixes;
    std::string dir;
    std::map<std::string, std::string> shapes;
};

static bool want(const keep_ctx * k, const char * name) {
    for (const auto & p : k->prefixes) {
        if (strncmp(name, p.c_str(), p.size()) == 0 && (name[p.size()] == '-' || name[p.size()] == 0)) {
            return true;
        }
    }
    return false;
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * ud) {
    auto * k = (keep_ctx *) ud;
    if (ask) {
        return want(k, t->name);
    }
    if (!want(k, t->name) || t->type != GGML_TYPE_F32) {
        return true;
    }
    std::vector<float> buf(ggml_nelements(t));
    if (ggml_is_contiguous(t)) {
        ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
    } else {
        // 非连续张量按行取
        size_t i = 0;
        std::vector<char> row(t->nb[1]);
        for (int64_t i3 = 0; i3 < t->ne[3]; i3++)
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++)
        for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
            size_t off = i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
            for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
                float v;
                ggml_backend_tensor_get(t, &v, off + i0 * t->nb[0], sizeof(float));
                buf[i++] = v;
            }
        }
    }
    std::ofstream f(k->dir + "/" + t->name + ".bin", std::ios::binary);
    f.write((const char *) buf.data(), buf.size() * sizeof(float));
    char shp[128];
    snprintf(shp, sizeof(shp), "%lld %lld %lld %lld", (long long) t->ne[0], (long long) t->ne[1],
             (long long) t->ne[2], (long long) t->ne[3]);
    k->shapes[t->name] = shp;
    return true;
}

static void split(const std::string & s, char c, std::vector<std::string> & out) {
    size_t a = 0;
    while (a <= s.size()) {
        size_t b = s.find(c, a);
        if (b == std::string::npos) b = s.size();
        if (b > a) out.push_back(s.substr(a, b - a));
        a = b + 1;
    }
}

int main(int argc, char ** argv) {
    std::string model, tok_path, dir;
    int n_threads = 4;
    int fa = -1;
    keep_ctx keep;
    std::vector<llama_model_kv_override> kvo;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "-m") model = next();
        else if (a == "-i") tok_path = next();
        else if (a == "-o") dir = next();
        else if (a == "-t") n_threads = atoi(next().c_str());
        else if (a == "--fa") fa = atoi(next().c_str());
        else if (a == "--keep") split(next(), ',', keep.prefixes);
        else if (a == "--kv") {
            std::string s = next();
            size_t eq = s.find('='), col = s.find(':', eq);
            llama_model_kv_override o{};
            strncpy(o.key, s.substr(0, eq).c_str(), sizeof(o.key) - 1);
            std::string ty = s.substr(eq + 1, col - eq - 1), v = s.substr(col + 1);
            if (ty == "str") { o.tag = LLAMA_KV_OVERRIDE_TYPE_STR; strncpy(o.val_str, v.c_str(), sizeof(o.val_str) - 1); }
            else if (ty == "float") { o.tag = LLAMA_KV_OVERRIDE_TYPE_FLOAT; o.val_f64 = atof(v.c_str()); }
            else if (ty == "int") { o.tag = LLAMA_KV_OVERRIDE_TYPE_INT; o.val_i64 = atoll(v.c_str()); }
            else if (ty == "bool") { o.tag = LLAMA_KV_OVERRIDE_TYPE_BOOL; o.val_bool = v == "true"; }
            kvo.push_back(o);
        } else {
            fprintf(stderr, "unknown arg %s\n", a.c_str());
            return 2;
        }
    }
    if (model.empty() || tok_path.empty() || dir.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf -i tokens.bin -o out_dir\n", argv[0]);
        return 2;
    }
    keep.dir = dir;
    std::filesystem::create_directories(dir);

    std::ifstream tf(tok_path, std::ios::binary | std::ios::ate);
    std::vector<llama_token> toks(tf.tellg() / sizeof(llama_token));
    tf.seekg(0);
    tf.read((char *) toks.data(), toks.size() * sizeof(llama_token));
    const int n = (int) toks.size();

    llama_backend_init();
    auto mp = llama_model_default_params();
    if (!kvo.empty()) {
        kvo.push_back(llama_model_kv_override{});  // key[0] == 0 结束
        mp.kv_overrides = kvo.data();
    }
    llama_model * m = llama_model_load_from_file(model.c_str(), mp);
    if (!m) { fprintf(stderr, "load failed\n"); return 1; }

    auto cp = llama_context_default_params();
    cp.n_ctx = ((n + 255) / 256) * 256;
    cp.n_batch = cp.n_ctx;
    cp.n_ubatch = cp.n_ctx;  // 一个物理批算完，中间张量只有一份
    cp.n_threads = cp.n_threads_batch = n_threads;
    if (fa >= 0) cp.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (!keep.prefixes.empty()) {
        cp.cb_eval = cb_eval;
        cp.cb_eval_user_data = &keep;
    }
    llama_context * c = llama_init_from_model(m, cp);
    if (!c) { fprintf(stderr, "ctx failed\n"); return 1; }

    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        b.token[i] = toks[i];
        b.pos[i] = i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = 1;
    }
    b.n_tokens = n;
    if (llama_decode(c, b) != 0) { fprintf(stderr, "decode failed\n"); return 1; }

    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(m));
    const float * lg = llama_get_logits(c);
    std::ofstream lf(dir + "/logits.bin", std::ios::binary);
    lf.write((const char *) lg, (size_t) n * nv * sizeof(float));
    if (!lf) { fprintf(stderr, "write %s/logits.bin failed\n", dir.c_str()); return 1; }

    std::ofstream mf(dir + "/meta.txt");
    mf << "n_tokens " << n << "\nn_vocab " << nv << "\n";
    for (auto & kv : keep.shapes) mf << kv.first << " " << kv.second << "\n";
    fprintf(stderr, "dump_logits: %d tokens, %d vocab, %zu tensors -> %s\n", n, nv, keep.shapes.size(), dir.c_str());

    llama_batch_free(b);
    llama_free(c);
    llama_model_free(m);
    llama_backend_free();
    return 0;
}
