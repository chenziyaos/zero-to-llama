"""分词对拍：同一批文本分别过 HF tokenizer 和 llama-tokenize，逐条比 token id。

用法：python3 tok_compare.py --hf models/Qwen3-0.6B --gguf model.gguf --tokenize llama-tokenize
        [--tag 名字] [--json out.json]
--gguf 可以是只含词表的 GGUF（convert_hf_to_gguf.py --vocab-only），预分词器选错的对照组用
gguf_new_metadata.py --pre-tokenizer 改 tokenizer.ggml.pre 生成。llama-tokenize 只加载词表，
不吃 --override-kv，所以对照组必须是单独的文件。
样本覆盖英文、中文、代码、数字、空白、emoji、特殊 token 文本和组合字符。
"""
import argparse
import json
import subprocess

from transformers import AutoTokenizer

SAMPLES = [
    'Hello world',
    ' Hello world',
    'The quick brown fox jumps over the lazy dog.',
    "I'm sure it's fine, isn't it? We'll see.",
    '把一个新模型接进推理引擎，最怕的是程序照常运行。',
    '混合 mixed 文本 text 123 和 English。',
    'def f(x):\n    return x * 2\n',
    'for (int i = 0; i < n; i++) { s += a[i]; }',
    '3.14159 2026 1234567890 1,000,000',
    '   leading spaces and\ttabs\t\tand\n\n\nnewlines',
    'trailing spaces   ',
    'emoji 🙂👍🏽 and flags 🇨🇳🇺🇸',
    '<|im_start|>user\nhi<|im_end|>',
    'café naïve résumé',
    'é combining acute',
    'Ünïcödé ΑΒΓ Ελληνικά русский язык',
    'ｆｕｌｌｗｉｄｔｈ　全角空格',
    "\"quotes\" 'single' `back`",
    'https://example.com/path?q=1&r=2',
    'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
    '日本語のテキストとカタカナ',
    '한국어 텍스트',
    'x' * 3 + ' ' * 7 + 'y',
    '\n',
    '1+1=2, 10-3=7, 6/2=3',
]


def llama_ids(bin_, gguf, text):
    cmd = [bin_, '-m', gguf, '--ids', '--log-disable', '--stdin']
    p = subprocess.run(cmd, input=text.encode('utf-8'), capture_output=True)
    out = p.stdout.decode().strip().splitlines()
    return json.loads(out[-1]) if out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hf', required=True)
    ap.add_argument('--gguf', required=True)
    ap.add_argument('--tokenize', required=True)
    ap.add_argument('--tag', default='gguf')
    ap.add_argument('--json')
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.hf)
    rows, bad = [], 0
    for s in SAMPLES:
        h = tok(s, add_special_tokens=False)['input_ids']
        g = llama_ids(a.tokenize, a.gguf, s)
        ok = h == g
        bad += not ok
        rows.append({'text': s, 'hf': h, 'llama': g, 'ok': ok})
        if not ok:
            k = next((i for i, (x, y) in enumerate(zip(h, g or [])) if x != y), min(len(h), len(g or [])))
            print(f'MISMATCH {s!r}\n  hf    {h}\n  llama {g}\n  first diff at {k}')
    tag = a.tag
    print(f'tok_compare[{tag}]: {len(SAMPLES) - bad}/{len(SAMPLES)} identical')
    if a.json:
        with open(a.json, 'w') as f:
            json.dump({'tag': tag, 'n': len(SAMPLES), 'bad': bad, 'rows': rows}, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
