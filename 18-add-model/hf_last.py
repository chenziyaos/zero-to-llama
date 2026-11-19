"""官方 model-conversion 流程的口径：短提示、只比最后一个位置。拿来和全位置对拍对照。

用法：python3 hf_last.py --model models/Qwen3-0.6B --text "Hello, my name is"
        [--tokens-out short.bin] [--llama file.bin ...] [--dump dir ...]
--llama 是 llama-debug --save-logits 写的 n_vocab 个 float32；--dump 是 dump_logits 的输出目录，取最后一行。
每一份都报 nmse_var（check-nmse.py 的口径）、top-1 是否一致、top-10 重合数。
"""
import argparse
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def grade(nmse):
    # check-nmse.py 的分档
    for th, s in [(1e-6, 'essentially identical'), (1e-4, 'excellent'), (1e-3, 'very good'),
                  (1e-2, 'good'), (0.1, 'acceptable'), (1.0, 'poor')]:
        if nmse < th:
            return s
    return 'very poor'


def report(name, r, t):
    d = t.astype(np.float64) - r
    nmse = float((d * d).mean() / r.astype(np.float64).var())
    t10r, t10t = set(np.argsort(-r)[:10]), set(np.argsort(-t)[:10])
    print(f'{name}: nmse_var {nmse:.3e} ({grade(nmse)})  top1 {"same" if r.argmax() == t.argmax() else "DIFF"}  '
          f'top10 overlap {len(t10r & t10t)}/10  max_abs {np.abs(d).max():.4g}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--text', required=True)
    ap.add_argument('--tokens-out')
    ap.add_argument('--llama', nargs='*', default=[])
    ap.add_argument('--dump', nargs='*', default=[])
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = tok(a.text, add_special_tokens=False)['input_ids']
    print(f'tokens {ids}')
    if a.tokens_out:
        np.asarray(ids, dtype='<i4').tofile(a.tokens_out)
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    with torch.no_grad():
        r = m(torch.tensor([ids]), use_cache=False).logits[0, -1].numpy()
    for p in a.llama:
        report(os.path.basename(p), r, np.fromfile(p, dtype='<f4'))
    for d in a.dump:
        lg = np.fromfile(os.path.join(d, 'logits.bin'), dtype='<f4').reshape(-1, r.shape[0])
        report(os.path.basename(d.rstrip('/')), r, lg[-1])


if __name__ == '__main__':
    main()
