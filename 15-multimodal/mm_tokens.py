"""按 llama.cpp 0ec4027 的 libmtmd 源码复刻：一张图、一段音频会变成多少个嵌入 token。

对应源码：
  tools/mtmd/mtmd-image.cpp  calc_size_preserved_ratio、mtmd_image_preprocessor_idefics3、mtmd_image_preprocessor_lfm2
  tools/mtmd/clip.cpp        load_hparams 里 IDEFICS3、LFM2 的参数；clip_n_output_tokens 里 QWEN3A 的 13 token 每 100 帧
  tools/mtmd/mtmd-audio.cpp  mtmd_audio_preprocessor_qwen3a::preprocess 的 800 帧窗口

不加载模型，结果和 mm_probe count 的日志逐条对账（见 test_mm.py）。
"""
import math
import struct


def f32(x):
    """按 C++ float 截断精度。"""
    return struct.unpack('f', struct.pack('f', x))[0]

# ---------------- 通用：保持长宽比缩放并对齐 ----------------

def calc_size_preserved_ratio(w, h, align, min_pixels=0, max_pixels=0, longest_edge=0):
    """img_tool::calc_size_preserved_ratio：先按最长边放缩并向上对齐，再按像素上下限收缩或放大。"""
    def ceil_f(x):
        return int(math.ceil(x / align)) * align

    def floor_f(x):
        return int(math.floor(x / align)) * align

    if longest_edge > 0:
        # 源码用 float 算缩放比：2048/4000 存成 float 略大于 0.512，3000 乘它得 1536.0001，向上对齐成 2048
        s = f32(min(f32(longest_edge / w), f32(longest_edge / h)))
        wb, hb = ceil_f(f32(w * s)), ceil_f(f32(h * s))
    else:
        wb = max(align, int(round(w / align)) * align)
        hb = max(align, int(round(h / align)) * align)
    if max_pixels > 0 and wb * hb > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        hb, wb = max(align, floor_f(h / beta)), max(align, floor_f(w / beta))
    elif min_pixels > 0 and wb * hb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb, wb = ceil_f(h * beta), ceil_f(w * beta)
    return wb, hb


# ---------------- SmolVLM2：IDEFICS3 ----------------

SMOL = dict(image_size=512, patch=16, n_merge=4, longest_edge=2048)
SMOL_OLD = dict(SMOL, longest_edge=0)      # 早期转换的 f16 mmproj，缺 clip.vision.preproc_image_size


def idefics3(w, h, max_tok=-1, p=SMOL):
    """返回每个块的 token 数列表：第一个是 512 的总览图，后面是网格切块。"""
    t = p['image_size']
    per = (t // p['patch'] // p['n_merge']) ** 2          # 512/16/4 = 8，每块 64 token
    if max_tok <= 0 and p['longest_edge'] <= 0:
        # 旧 GGUF 没有 preproc_image_size：边长按 512 四舍五入，不先放大到 2048
        rw, rh = calc_size_preserved_ratio(w, h, t)
        gx, gy = math.ceil(rw / t), math.ceil(rh / t)
    elif max_tok <= 0:
        L = p['longest_edge']
        ar = w / h
        if w >= h:
            rw, rh = L, int(L / ar)
            rh += rh % 2
        else:
            rh, rw = L, int(L * ar)
            rw += rw % 2
        gx, gy = (rw + t - 1) // t, (rh + t - 1) // t
    else:
        area = (p['patch'] * p['n_merge']) ** 2
        rw, rh = calc_size_preserved_ratio(w, h, t, 0, max_tok * area, p['longest_edge'])
        gx, gy = math.ceil(rw / t), math.ceil(rh / t)
    return [per] + [per] * (gx * gy)


# ---------------- LFM2-VL ----------------

LFM2 = dict(patch=16, n_merge=2, min_tok=64, max_tok=256, tile=512, min_tiles=2, max_tiles=10, tol=2.0)


def _lfm2_limits(p, max_tok, min_tok):
    area = (p['patch'] * p['n_merge']) ** 2
    lo = (min_tok if min_tok > 0 else p['min_tok']) * area
    hi = (max_tok if max_tok > 0 else p['max_tok']) * area
    return lo, hi


def lfm2_smart_resize(w, h, lo, hi, a=32):
    """Lfm2VlImageProcessor.smart_resize：Python 的四舍六入五成双。"""
    hb = max(a, round(h / a) * a)
    wb = max(a, round(w / a) * a)
    if hb * wb > hi:
        beta = math.sqrt(h * w / hi)
        hb = max(a, math.floor(h / beta / a) * a)
        wb = max(a, math.floor(w / beta / a) * a)
    elif hb * wb < lo:
        beta = math.sqrt(lo / (h * w))
        hb = math.ceil(h * beta / a) * a
        wb = math.ceil(w * beta / a) * a
    return wb, hb


def lfm2_ratios(p):
    out = []
    for n in range(p['min_tiles'], p['max_tiles'] + 1):
        for gw in range(1, n + 1):
            for gh in range(1, n + 1):
                if p['min_tiles'] <= gw * gh <= p['max_tiles'] and (gw, gh) not in out:
                    out.append((gw, gh))
    out.sort(key=lambda r: r[0] * r[1])                  # 稳定排序，和 std::sort 在这组数据上一致
    return out


def lfm2_grid(w, h, p=LFM2):
    ar = w / h
    best, bd = (1, 1), float('inf')
    for gw, gh in lfm2_ratios(p):
        d = abs(ar - gw / gh)
        if d < bd:
            bd, best = d, (gw, gh)
        elif d == bd and w * h > 0.5 * p['tile'] ** 2 * gw * gh:
            best = (gw, gh)
    return best


def lfm2(w, h, max_tok=-1, min_tok=-1, p=LFM2):
    """返回每个块的 token 数：不切块时只有一块；切块时是 N 个 512 块（各 256）加一张缩略图。"""
    lo, hi = _lfm2_limits(p, max_tok, min_tok)
    a = p['patch'] * p['n_merge']
    ow, oh = lfm2_smart_resize(w, h, lo, hi, a)
    thumb = (ow // a) * (oh // a)
    hb = max(p['patch'], round(h / a) * a)
    wb = max(p['patch'], round(w / a) * a)
    tile = p['max_tiles'] > 1 and hb * wb > hi * p['tol']
    if not tile:
        return [thumb]
    gw, gh = lfm2_grid(w, h, p)
    return [(p['tile'] // a) ** 2] * (gw * gh) + [thumb]


# ---------------- Qwen3-ASR：QWEN3A ----------------

def qwen3a_tokens(n_samples, sr=16000, hop=160):
    """有效帧数 n/hop+1，按 800 帧一窗；每窗补到 100 的倍数，每 100 帧出 13 个 token。"""
    n_eff = n_samples // hop + 1
    tot, wins = 0, []
    for off in range(0, n_eff, 800):
        win = min(800, n_eff - off)
        k = (win + 99) // 100 * 13
        wins.append(k)
        tot += k
    return tot, wins


def main():
    sizes = [(256, 256), (512, 512), (640, 488), (1024, 768), (1536, 1536), (2048, 1024), (4000, 3000)]
    print('# SmolVLM2 IDEFICS3：每块 64 token')
    print('model size max_tok chunks media_tok')
    for mt in (-1, 64, 256, 1024):
        for w, h in sizes:
            c = idefics3(w, h, mt)
            print('smol %dx%d %d %d %d' % (w, h, mt, len(c), sum(c)))
    for w, h in sizes:
        c = idefics3(w, h, -1, SMOL_OLD)
        print('smol_old %dx%d -1 %d %d' % (w, h, len(c), sum(c)))
    print()
    print('# LFM2-VL：单图 64 到 256 token，超过 2 倍上限就切 512 块加缩略图')
    for mt in (-1, 64, 128):
        for w, h in sizes:
            c = lfm2(w, h, mt)
            print('lfm2 %dx%d %d %d %d thumb=%d' % (w, h, mt, len(c), sum(c), c[-1]))
    print()
    print('# Qwen3-ASR：16 kHz，每秒 100 帧')
    for sec in (1, 5, 8, 10, 30, 60):
        tot, wins = qwen3a_tokens(sec * 16000)
        print('audio %ds tokens=%d windows=%d' % (sec, tot, len(wins)))


if __name__ == '__main__':
    main()
