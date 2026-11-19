"""从 llama-perplexity --kl-divergence 日志里抽 PPL、Mean KLD、99% KLD、Same top p。
用法：python3 collect_kld.py logs/kld-*.log"""
import re, sys, os

PAT = {
    'ppl': r'Mean PPL\(Q\)\s*:\s*([\d.]+)\s*±\s*([\d.]+)',
    'kld': r'Mean\s+KLD:\s*([\d.\-]+)\s*±\s*([\d.]+)',
    'kld99': r'99\.0%\s+KLD:\s*([\d.\-]+)',
    'top': r'Same top p:\s*([\d.]+)\s*±\s*([\d.]+)',
    'rot': r'n_rot_k\s*=\s*(\d+)',
}


def parse(text):
    out = {}
    for k, p in PAT.items():
        m = re.search(p, text)
        out[k] = m.groups() if m else None
    return out


if __name__ == '__main__':
    print('%-14s %16s %18s %10s %16s' % ('name', 'PPL(Q)', 'Mean KLD', '99% KLD', 'same top %'))
    for fn in sys.argv[1:]:
        r = parse(open(fn, errors='replace').read())
        name = os.path.basename(fn)[4:-4]
        f = lambda v, n=2: ('%s ± %s' % v[:2]) if v else '-'
        print('%-14s %16s %18s %10s %16s' % (name, f(r['ppl']), f(r['kld']), r['kld99'][0] if r['kld99'] else '-', f(r['top'])))
