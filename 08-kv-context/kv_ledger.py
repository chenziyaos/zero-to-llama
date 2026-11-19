#!/usr/bin/env python3
"""KV cache 账本：按 llama.cpp 0ec4027 的分配规则算 K、V 各占多少字节。

对照：
  src/llama-context.cpp      n_ctx 向上取整到 256；非 unified 时 n_ctx_seq = n_ctx / n_seq_max 再取整到 256
  src/llama-kv-cache.cpp     每层 K 是 n_embd_k_gqa x kv_size x n_stream，MLA 没有 V
  src/llama-kv-cache-iswa.cpp  SWA 层 size_swa = PAD(min(size_base, n_swa*(unified?n_seq:1) + n_ubatch), 256)
  src/llama-hparams.cpp      set_swa_pattern：il % n_pattern < n_pattern-1 的层是 SWA
  common/arg.cpp             -ctk/-ctv 允许的类型

    python3 kv_ledger.py --preset qwen3-0.6b -c 4096 40960 -t f16 q8_0 q4_0
    python3 kv_ledger.py --gguf models/q-Q4_K_M.gguf -c 32768 --np 4
    python3 kv_ledger.py --list
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))

import hybrid_ledger as hl  # noqa: E402

MiB = 2 ** 20
# -ctk/-ctv 允许的类型 -> (每块元素数, 每块字节数)，照 ggml-common.h
KV_TYPES = {
    'f32': (1, 4), 'f16': (1, 2), 'bf16': (1, 2),
    'q8_0': (32, 34), 'q4_0': (32, 18), 'q4_1': (32, 20),
    'iq4_nl': (32, 18), 'q5_0': (32, 22), 'q5_1': (32, 24),
}
# load_swa_pattern 的 n_pattern，按 general.architecture
SWA_PATTERN = {'gemma3': 6, 'cohere2': 4, 'llama4': 4, 'gpt-oss': 2, 'openai-moe': 2}

# 公开 config.json 里的数；mla 为 (kv_lora_rank, qk_rope_head_dim)
PRESETS = {
    'qwen3-0.6b': dict(n_layer=28, n_head_kv=8, head_k=128, head_v=128, n_ctx_train=40960),
    'qwen3-8b': dict(n_layer=36, n_head_kv=8, head_k=128, head_v=128, n_ctx_train=40960),
    'qwen3-32b': dict(n_layer=64, n_head_kv=8, head_k=128, head_v=128, n_ctx_train=40960),
    'qwen3-30b-a3b': dict(n_layer=48, n_head_kv=4, head_k=128, head_v=128, n_ctx_train=40960),
    'llama3.1-8b': dict(n_layer=32, n_head_kv=8, head_k=128, head_v=128, n_ctx_train=131072),
    'mistral-small-24b': dict(n_layer=40, n_head_kv=8, head_k=128, head_v=128, n_ctx_train=131072),
    'gemma3-27b': dict(n_layer=62, n_head_kv=16, head_k=128, head_v=128, n_ctx_train=131072,
                       n_swa=1024, swa_pattern=6),
    'gpt-oss-20b': dict(n_layer=24, n_head_kv=8, head_k=64, head_v=64, n_ctx_train=131072,
                        n_swa=128, swa_pattern=2),
    'deepseek-v3': dict(n_layer=61, n_head_kv=1, head_k=512 + 64, head_v=0, n_ctx_train=163840, mla=True),
}

# 双模型混合架构基线（Gated DeltaNet + 少量 full attention）。这两个模型只有 full attention 层存
# softmax KV，DeltaNet 层存固定大小的循环状态，不能套上面“每层都存 KV”的 ledger()。几何口径在
# common/dual_models.json，这里只放 manifest 的 key，账本统一走 common/hybrid_ledger.py。
HYBRID_MODELS = {
    'qwen3.8-27b': 'qwen3.8-27b',
    'qwen3.5-35b-a3b': 'qwen3.5-35b-a3b',
}
_HYBRID_MANIFEST = hl.load_manifest()


def hybrid_geometry(preset):
    """混合架构预设 -> hybrid_ledger 的几何 dict。"""
    return hl.geometry(_HYBRID_MANIFEST, HYBRID_MODELS[preset])


def hybrid_ledger(preset, n_ctx, ctk='f16', n_seq=1, unified=True, n_rs_seq=0):
    """混合架构一条序列的账：full-attention KV（随上下文）+ DeltaNet 循环状态（常数）。

    n_ctx 语义和 hybrid_ledger.kv_bytes 对齐，表示“整池总上下文”，由共享账本统一做
    pad / 多槽平分，这里不再预先 pad，避免非 unified 下把已 pad 的整池再平分一次。
    返回字段对齐传统 ledger()，额外给 rec_bytes 和 split_ctx 算出的整池/每槽长度。
    K、V 同类型（混合架构的 -ctk/-ctv 同源）。
    """
    g = hybrid_geometry(preset)
    if not n_ctx:
        n_ctx = g['n_ctx_train']
    total_ctx, per_ctx = hl.split_ctx(n_ctx, n_seq, unified)
    kv = hl.kv_bytes(g, n_ctx, ctk, n_seq, unified)
    rec = hl.rec_total(g, n_seq, n_rs_seq)
    return dict(n_ctx=total_ctx, ctx_per_seq=per_ctx, kv_bytes=kv, rec_bytes=rec,
                total=kv + rec, per_token=hl.kv_per_token(g, ctk),
                rec_per_seq=hl.rec_per_seq(g))



def pad(x, n=256):
    return (x + n - 1) // n * n


def nbytes(t, n_elems):
    blk, nb = KV_TYPES[t]
    if n_elems % blk:
        raise ValueError(f'{t} 的块大小 {blk} 不整除 {n_elems}，llama.cpp 会拒绝这个组合')
    return n_elems // blk * nb


def is_swa(il, n_pattern):
    return n_pattern > 0 and il % n_pattern < n_pattern - 1


def cells(m, n_ctx, n_seq=1, unified=False, n_ubatch=512, swa_full=False):
    """返回 (n_ctx_padded, size_base, size_swa, n_stream)。"""
    n_ctx = pad(n_ctx or m['n_ctx_train'])
    if unified:
        size_base, n_stream = n_ctx, 1
    else:
        size_base, n_stream = pad(n_ctx // n_seq), n_seq
    size_swa = size_base
    if m.get('n_swa') and not swa_full:
        size_swa = pad(min(size_base, m['n_swa'] * (n_seq if unified else 1) + n_ubatch))
    return n_ctx, size_base, size_swa, n_stream


def ledger(m, n_ctx=0, ctk='f16', ctv='f16', n_seq=1, unified=False, n_ubatch=512, swa_full=False):
    # llama_init_from_model：块大小必须整除每个头的维度
    for t, hd in ((ctk, m['head_k']), (ctv, m['head_v'])):
        if hd and hd % KV_TYPES[t][0]:
            raise ValueError(f'{t} 的块大小 {KV_TYPES[t][0]} 不整除 head_dim {hd}，llama.cpp 会拒绝这个组合')
    n_ctx, size_base, size_swa, n_stream = cells(m, n_ctx, n_seq, unified, n_ubatch, swa_full)
    pat = m.get('swa_pattern', 0) if m.get('n_swa') else 0
    k = v = 0
    n_swa_layers = 0
    for il in range(m['n_layer']):
        sw = is_swa(il, pat)
        n_swa_layers += sw
        kv = (size_swa if sw else size_base) * n_stream
        k += nbytes(ctk, m['n_head_kv'] * m['head_k'] * kv)
        if not m.get('mla'):
            v += nbytes(ctv, m['n_head_kv'] * m['head_v'] * kv)
    per_tok = sum(nbytes(ctk, m['n_head_kv'] * m['head_k'] * 32) for _ in range(m['n_layer'])) / 32
    if not m.get('mla'):
        per_tok += m['n_layer'] * nbytes(ctv, m['n_head_kv'] * m['head_v'] * 32) / 32
    return dict(n_ctx=n_ctx, size_base=size_base, size_swa=size_swa, n_stream=n_stream,
                n_swa_layers=n_swa_layers, k=k, v=v, total=k + v, per_token_full=per_tok)


def from_gguf(path):
    from gguf_lite import GGUF
    g = GGUF(path)
    hp = g.hp
    n_head = hp('attention.head_count')
    n_embd = hp('embedding_length')
    head_k = hp('attention.key_length', n_embd // n_head)
    head_v = hp('attention.value_length', head_k)
    n_kv = hp('attention.head_count_kv', n_head)
    if isinstance(n_kv, list):  # 少数架构按层给，取最大值做保守估计
        n_kv = max(n_kv)
    m = dict(n_layer=hp('block_count'), n_head_kv=n_kv, head_k=head_k, head_v=head_v,
             n_ctx_train=hp('context_length'))
    lora = hp('attention.kv_lora_rank')
    if lora:
        rope = hp('rope.dimension_count', 64)
        m.update(n_head_kv=1, head_k=lora + rope, head_v=0, mla=True)
    swa = hp('attention.sliding_window')
    if swa and g.arch in SWA_PATTERN:
        m.update(n_swa=swa, swa_pattern=SWA_PATTERN[g.arch])
    return g.arch, m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--preset', choices=sorted(set(PRESETS) | set(HYBRID_MODELS)))
    ap.add_argument('--gguf')
    ap.add_argument('--list', action='store_true', help='每个预设在训练上下文长度下的 f16 账本')
    ap.add_argument('-c', type=int, nargs='+', default=[0], help='整池总 n_ctx，0 表示训练长度')
    ap.add_argument('-t', nargs='+', default=['f16'], help='K、V 同类型；写 q8_0/q4_0 表示 K q8_0 V q4_0')
    ap.add_argument('--np', type=int, default=1, help='n_seq_max，即 -np')
    ap.add_argument('--kvu', action='store_true', help='unified KV（llama-server 缺省就是 unified）')
    ap.add_argument('--n-rs-seq', type=int, default=0, help='混合架构循环态回滚快照数，等于草稿最大 token 数')
    ap.add_argument('-ub', type=int, default=512)
    ap.add_argument('--swa-full', action='store_true')
    a = ap.parse_args()

    if a.list:
        print(f'{"预设":18} {"层":>3} {"KV头":>4} {"head":>5} {"训练长度":>8} {"KiB/token":>10} {"满长 f16 MiB":>12}')
        for name, m in PRESETS.items():
            r = ledger(m)
            print(f'{name:18} {m["n_layer"]:3} {m["n_head_kv"]:4} {m["head_k"]:5} {m["n_ctx_train"]:8} '
                  f'{r["per_token_full"] / 1024:10.1f} {r["total"] / MiB:12.0f}')
        print('-- 混合架构（full-attn KV 随上下文，DeltaNet 循环状态为常数，单列）--')
        for name in sorted(HYBRID_MODELS):
            g = hybrid_geometry(name)
            r = hybrid_ledger(name, g['n_ctx_train'], 'f16')
            print(f'{name:18} {g["n_layer"]:3} {g["n_kv"]:4} {g["head_dim"]:5} {g["n_ctx_train"]:8} '
                  f'{r["per_token"] / 1024:10.1f} {r["kv_bytes"] / MiB:12.0f}'
                  f'  循环态 {r["rec_per_seq"] / MiB:8.1f} MiB/seq')
        return
    if a.gguf:
        name, m = from_gguf(a.gguf)
        name = f'{os.path.basename(a.gguf)} ({name})'
    elif a.preset:
        if a.preset in HYBRID_MODELS:
            name = a.preset
            g = hybrid_geometry(name)
            print(f'{name}: 混合架构，full-attn {g["kv_layers"]} 层 / DeltaNet {g["rec_layers"]} 层，'
                  f'KV 头 {g["n_kv"]}，head_dim {g["head_dim"]}，ffn={g["ffn"]}')
            for c in a.c:
                for t in a.t:
                    ctk, _, ctv = t.partition('/')
                    ctk = ctk or ctv
                    r = hybrid_ledger(name, c, ctk, a.np, a.kvu, a.n_rs_seq)
                    print(f'  整池 n_ctx {r["n_ctx"]:6}（每槽 {r["ctx_per_seq"]:6}）{ctk:6} '
                          f'{"unified" if a.kvu else "non-unified"}: '
                          f'full-attn KV {r["kv_bytes"] / MiB:9.2f} MiB  '
                          f'循环态 {r["rec_bytes"] / MiB:9.2f} MiB  合计 {r["total"] / MiB:9.2f} MiB')
            return
        name, m = a.preset, PRESETS[a.preset]
    else:
        ap.error('给 --preset 或 --gguf')
    print(f'{name}: {m["n_layer"]} 层，KV 头 {m["n_head_kv"]}，head_dim K {m["head_k"]} V {m["head_v"]}'
          + ('，MLA 只存 K' if m.get('mla') else '')
          + (f'，SWA {m["n_swa"]} 每 {m["swa_pattern"]} 层 1 层全局' if m.get('n_swa') else ''))
    for c in a.c:
        for t in a.t:
            ctk, _, ctv = t.partition('/')
            ctv = ctv or ctk
            r = ledger(m, c, ctk, ctv, a.np, a.kvu, a.ub, a.swa_full)
            swa = f' swa_cells {r["size_swa"]}' if r['n_swa_layers'] else ''
            print(f'  n_ctx {r["n_ctx"]:6} K {ctk:6} V {ctv:6} cells {r["size_base"]}x{r["n_stream"]}{swa}: '
                  f'K {r["k"] / MiB:9.2f} MiB  V {r["v"] / MiB:9.2f} MiB  合计 {r["total"] / MiB:9.2f} MiB')


if __name__ == '__main__':
    main()
