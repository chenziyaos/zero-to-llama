"""HF transformers 参考前向：分词、跑一遍、把每个位置的 logits 和每层输出写成 float32 二进制。

用法：python3 hf_ref.py --model models/Qwen3-0.6B --text prompt.txt --out ref/f32 [--dtype f32|bf16]
输出：
  tokens.bin   int32 小端，交给 dump_logits -i
  logits.bin   n_tokens x n_vocab
  layers.bin   n_layer x n_tokens x n_embd，第 i 层 decoder 的输出，对应 llama.cpp 的 l_out-i
  norm.bin     n_tokens x n_embd，最后的 RMSNorm 输出，对应 result_norm
  meta.txt
层输出用 forward hook 取，不用 output_hidden_states：后者最后一项已经过了 norm，和 l_out 对不上。
"""
import argparse
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--text', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dtype', default='f32', choices=['f32', 'bf16'])
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    text = open(a.text, encoding='utf-8').read()
    ids = tok(text, add_special_tokens=False)['input_ids']
    np.asarray(ids, dtype='<i4').tofile(os.path.join(a.out, 'tokens.bin'))

    dt = torch.float32 if a.dtype == 'f32' else torch.bfloat16
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt)
    m.eval()
    t_load = time.time() - t0

    layers = m.model.layers
    outs = [None] * len(layers)
    norm_out = {}

    def hook(i):
        def f(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            outs[i] = h[0].float().numpy().copy()
        return f

    hs = [l.register_forward_hook(hook(i)) for i, l in enumerate(layers)]
    hs.append(m.model.norm.register_forward_hook(
        lambda _m, _i, o: norm_out.__setitem__('x', o[0].float().numpy().copy())))

    t0 = time.time()
    with torch.no_grad():
        lg = m(torch.tensor([ids]), use_cache=False).logits[0].float().numpy()
    t_fwd = time.time() - t0
    for h in hs:
        h.remove()

    lg.astype('<f4').tofile(os.path.join(a.out, 'logits.bin'))
    np.stack(outs).astype('<f4').tofile(os.path.join(a.out, 'layers.bin'))
    norm_out['x'].astype('<f4').tofile(os.path.join(a.out, 'norm.bin'))
    cfg = m.config
    with open(os.path.join(a.out, 'meta.txt'), 'w') as f:
        f.write(f'n_tokens {len(ids)}\nn_vocab {lg.shape[1]}\nn_layer {len(layers)}\n'
                f'n_embd {cfg.hidden_size}\ndtype {a.dtype}\nload_s {t_load:.1f}\nforward_s {t_fwd:.1f}\n')
    print(f'hf_ref: {len(ids)} tokens, dtype {a.dtype}, load {t_load:.1f}s, forward {t_fwd:.1f}s -> {a.out}')


if __name__ == '__main__':
    main()
