"""mm_tokens 的回归用例：每条都对着 0ec4027 的源码和 mm_probe count 的实测日志写。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mm_tokens as M  # noqa: E402

# mm_probe count 的实测（logs/count-*.txt），格式：尺寸 -> (块数, 媒体 token)
SMOL_DEFAULT = {(256, 256): (17, 1088), (512, 512): (17, 1088), (640, 488): (17, 1088),
                (1024, 768): (13, 832), (1536, 1536): (17, 1088), (2048, 1024): (9, 576),
                (4000, 3000): (13, 832)}
SMOL_F16_OLD = {(256, 256): (2, 128), (640, 488): (2, 128), (1024, 768): (5, 320),
                (2048, 1024): (9, 576), (4000, 3000): (49, 3136)}
LFM2_DEFAULT = {(256, 256): (1, 64), (512, 512): (1, 256), (640, 488): (1, 234),
                (1024, 768): (7, 1770), (1536, 1536): (10, 2560), (2048, 1024): (9, 2290),
                (4000, 3000): (7, 1770)}


class T(unittest.TestCase):

    def test_smol_default_matches_log(self):
        for (w, h), (n, tok) in SMOL_DEFAULT.items():
            c = M.idefics3(w, h)
            assert (len(c), sum(c)) == (n, tok), (w, h, c)

    def test_smol_small_image_upscaled(self):
        # 256x256 先放大到 2048 边长，切 4x4 加总览，比 1024x768 还多
        assert sum(M.idefics3(256, 256)) > sum(M.idefics3(1024, 768))

    def test_smol_max_tok(self):
        for w, h in SMOL_DEFAULT:
            assert sum(M.idefics3(w, h, 64)) == 128
        assert sum(M.idefics3(256, 256, 256)) == 320
        assert sum(M.idefics3(640, 488, 256)) == 192
        assert sum(M.idefics3(640, 488, 1024)) == 1088
        # 默认路径 4000x3000 是 13 块，max-tok 1024 却是 17 块：float 精度让 1536 变成 1536.0001 再向上对齐
        assert (len(M.idefics3(4000, 3000)), len(M.idefics3(4000, 3000, 1024))) == (13, 17)

    def test_smol_old_gguf(self):
        # 缺 preproc_image_size 的 f16 mmproj：同一张图块数完全不同
        for (w, h), (n, tok) in SMOL_F16_OLD.items():
            c = M.idefics3(w, h, -1, M.SMOL_OLD)
            assert (len(c), sum(c)) == (n, tok), (w, h, c)

    def test_lfm2_default_matches_log(self):
        for (w, h), (n, tok) in LFM2_DEFAULT.items():
            c = M.lfm2(w, h)
            assert (len(c), sum(c)) == (n, tok), (w, h, c)

    def test_lfm2_thumbnail(self):
        c = M.lfm2(1024, 768)
        assert c[:-1] == [256] * 6 and c[-1] == 234

    def test_lfm2_lower_max_triggers_tiling(self):
        # 上限压到 64，640x488 超过 2 倍上限就切块，总 token 反而从 234 涨到 1590
        assert sum(M.lfm2(640, 488)) == 234
        c = M.lfm2(640, 488, 64)
        assert len(c) == 7 and sum(c) == 1590 and c[-1] == 54
        assert sum(M.lfm2(640, 488, 128)) == 1644
        assert sum(M.lfm2(1024, 768, 128)) == 1653
        assert sum(M.lfm2(2048, 1024, 128)) == 2176

    def test_lfm2_token_formula(self):
        # 不切块时 token = (w/32)*(h/32)，对齐后
        w, h = M.lfm2_smart_resize(640, 488, 64 * 1024, 256 * 1024)
        assert (w // 32) * (h // 32) == 234

    def test_calc_size_max_pixels(self):
        assert M.calc_size_preserved_ratio(640, 488, 512, 0, 64 * 4096, 2048) == (512, 512)

    def test_qwen3a_tokens(self):
        assert M.qwen3a_tokens(16000)[0] == 26
        tot, wins = M.qwen3a_tokens(30 * 16000)
        assert wins == [104, 104, 104, 91] and tot == 403

    def test_qwen3a_matches_cli(self):
        # mtmd-cli 对 test-2.mp3 打印的三批是 104、104、26：前两窗满 800 帧，第三窗不足 200 帧。
        # 反推音频长度落在 17.0 到 18.0 秒之间，两端都检查
        for sec in (17.0, 17.99):
            tot, wins = M.qwen3a_tokens(int(sec * 16000))
            assert wins == [104, 104, 26] and tot == 234
        assert M.qwen3a_tokens(18 * 16000)[1] != [104, 104, 26]


if __name__ == '__main__':
    unittest.main(verbosity=1)
