"""启动前估算 llama.cpp 在 CPU 上要占多少内存，口径对齐日志里的 memory breakdown 表。

    python3 estimate_memory.py MODEL.gguf [--ctx 4096] [--ubatch 512] [--ctk f16] [--ctv f16]
                               [--fa on|off] [--cpu avx512|avx2|none]

三项和日志一一对应（基线 llama.cpp 0ec4027）：
  model    张量字节，直接从 GGUF 张量目录累加；能 repack 的那部分另记到 CPU_REPACK 一行
  context  KV cache = 层数 x 2(K,V) x n_embd_kv_gqa x 槽位数 x 每元素字节，槽位数是 n_ctx 向上取整到 256
           （src/llama-context.cpp:387 的 GGML_PAD(n_ctx, 256)）
  compute  sched_reserve 按最坏情况（n_ubatch 个 token、n_outputs = n_ubatch）预留的计算图缓冲。
           第一项是 logits，n_vocab x n_ubatch x 4 B，精确；后两项是用 Qwen3-0.6B 在 6 组
           (ubatch, ctx) 上实测拟合的系数，换架构时只能当量级参考，02-ledger 会推广到一般情况

CPU repack 的条件抄自 ggml/src/ggml-cpu/repack.cpp:4930 ggml_repack_get_optimal_repack_type：
  x86 上 Q4_0、Q4_K、IQ4_NL、MXFP4 要 AVX2，Q2_K 要 AVX512，且行数 ne[1] 是 8 的倍数；
  token_embd 走 GET_ROWS 不做 repack（日志里那句 "cannot be used with preferred buffer type CPU_REPACK"）。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from gguf_lite import GGUF, TYPE_ID, type_bytes  # noqa: E402

MiB = 2 ** 20
REPACK_X86 = {'Q4_0': 'avx2', 'Q4_K': 'avx2', 'IQ4_NL': 'avx2', 'MXFP4': 'avx2', 'Q2_K': 'avx512'}
LEVEL = {'none': 0, 'avx2': 1, 'avx512': 2}
# Qwen3-0.6B，FA on，实测拟合：compute = 4*V*ub + A*ub + B*ub*n_ctx（字节）
FIT_A, FIT_B = 25982.0, 5.0 / 3.0
FIT_B_FA_OFF_EXTRA = 3.0   # FA off 时每 ub*ctx 多出的字节（ub512 c4096 实测 318.77 对 312.77 MiB）


def can_repack(t, cpu):
    need = REPACK_X86.get(t.type_name)
    if need is None or t.name == 'token_embd.weight' or len(t.shape) < 2:
        return False
    return LEVEL[cpu] >= LEVEL[need] and t.shape[1] % 8 == 0


def estimate(g, ctx, ubatch, ctk, ctv, fa, cpu):
    n_layer = g.hp('block_count')
    n_head_kv = g.hp('attention.head_count_kv')
    hd_k = g.hp('attention.key_length') or g.hp('embedding_length') // g.hp('attention.head_count')
    hd_v = g.hp('attention.value_length') or hd_k
    n_ctx_train = g.hp('context_length')
    vocab = g.kv.get(f'{g.arch}.vocab_size')
    if vocab is None:
        tok = g.kv.get('tokenizer.ggml.tokens')
        vocab = tok[2] if isinstance(tok, tuple) else len(tok)
    if ctx == 0:
        ctx = n_ctx_train                       # llama-context.cpp:225，n_ctx=0 取训练长度
    cells = (ctx + 255) // 256 * 256
    ub = min(ubatch, cells)

    repack = sum(t.nbytes for t in g.tensors if can_repack(t, cpu))
    model = g.tensor_bytes - repack
    k = n_layer * type_bytes(TYPE_ID[ctk.upper()], n_head_kv * hd_k * cells)
    v = n_layer * type_bytes(TYPE_ID[ctv.upper()], n_head_kv * hd_v * cells)
    b = FIT_B + (FIT_B_FA_OFF_EXTRA if fa == 'off' else 0.0)
    compute = 4 * vocab * ub + FIT_A * ub + b * ub * cells
    return dict(n_ctx=cells, n_ctx_train=n_ctx_train, n_ubatch=ub, vocab=vocab, n_layer=n_layer,
                n_head_kv=n_head_kv, head_dim=hd_k, model=model, repack=repack, k=k, v=v,
                compute=compute, host=model + k + v + compute)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--ctx', type=int, default=4096)
    ap.add_argument('--ubatch', type=int, default=512)
    ap.add_argument('--ctk', default='f16')
    ap.add_argument('--ctv', default='f16')
    ap.add_argument('--fa', choices=['on', 'off'], default='on')
    ap.add_argument('--cpu', choices=list(LEVEL), default='avx512')
    a = ap.parse_args()
    if a.fa == 'off' and a.ctv.lower() != 'f16':
        sys.exit('量化 V cache 要求 Flash Attention，-fa off 时 --ctv 只能是 f16')
    g = GGUF(a.model)
    e = estimate(g, a.ctx, a.ubatch, a.ctk, a.ctv, a.fa, a.cpu)
    print(f'{os.path.basename(a.model)}：{g.arch}，{e["n_layer"]} 层，KV 头 {e["n_head_kv"]} x {e["head_dim"]}，'
          f'词表 {e["vocab"]}')
    print(f'n_ctx {e["n_ctx"]}（训练长度 {e["n_ctx_train"]}），n_ubatch {e["n_ubatch"]}，'
          f'K {a.ctk} / V {a.ctv}，FA {a.fa}')
    print('| memory breakdown [MiB] | self   model   context   compute |')
    print(f'|   - Host               | {e["host"] / MiB:5.0f} = {e["model"] / MiB:5.0f} + '
          f'{(e["k"] + e["v"]) / MiB:7.0f} + {e["compute"] / MiB:7.0f} |')
    if e['repack']:
        print(f'|   - CPU_REPACK         | {e["repack"] / MiB:5.0f} = {e["repack"] / MiB:5.0f} + '
              f'      0 +       0 |')
    print(f'KV：K {e["k"] / MiB:.2f} MiB，V {e["v"] / MiB:.2f} MiB；'
          f'compute 里 logits 占 {4 * e["vocab"] * e["n_ubatch"] / MiB:.2f} MiB')
    print(f'合计 {(e["host"] + e["repack"]) / MiB:.0f} MiB，-fit 默认还要再留 1024 MiB 余量')


if __name__ == '__main__':
    main()
