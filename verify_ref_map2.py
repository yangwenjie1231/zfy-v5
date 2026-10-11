# -*- coding: utf-8 -*-
"""用**包络互相关**验证映射 —— 比时长指纹可靠。

## 上一版的错误
`verify_ref_map.py` 用时长差判断"是否同一段"，结论是 10/12 "不同段"。
**但这是错的** —— 我早先记录过：

    同名文件在不同目录时长不同：
        5-wav32k  5.650s   <- **裁剪过**（去掉首尾静音）
        剧情语音   6.948s   <- 原始

所以**源文件比训练集长是正常的**（裁掉了静音），不能据此判定"不同段"。

## 正确方法：包络互相关
    1. 把两条音频都降为 **20ms 帧的 RMS 包络**
    2. 用较短的包络在较长的上滑动，找最大归一化互相关
    3. 峰值 > 0.8 -> **同一段录音**（训练集是源的裁剪版）
       峰值 < 0.5 -> 确实不同

**为什么可靠**：
  · 裁剪只改变长度，不改变包络形状
  · 不同语音的包络形状差异极大
  · 对重采样/编码差异不敏感

## 输出
    dist/ref_best2/map_verify2.txt
    dist/ref_best2/cands_final.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")
GSV = V5 / "GPT-SoVITS"
OUT = V5 / "dist" / "ref_best2"
TXT = GSV / "logs" / "zfy_v5" / "2-name2text.txt"
TRAIN = GSV / "logs" / "zfy_v5" / "5-wav32k"
FRAME = 0.02          # 20ms


def envelope(p: Path, sr_t: int = 16000):
    """返回 20ms 帧的 RMS 包络（归一化）。"""
    try:
        x, sr = sf.read(str(p), always_2d=True, dtype="float64")
    except Exception:  # noqa: BLE001
        return None
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if sr != sr_t:
        try:
            from math import gcd

            from scipy.signal import resample_poly
            g = gcd(int(sr), sr_t)
            x = resample_poly(x, sr_t // g, int(sr) // g).astype(np.float64)
        except Exception:  # noqa: BLE001
            n = int(round(len(x) * sr_t / sr))
            x = np.interp(np.linspace(0, len(x) - 1, n),
                          np.arange(len(x)), x)
    n = int(FRAME * sr_t)
    m = (len(x) // n) * n
    if m < n * 4:
        return None
    env = np.sqrt((x[:m].reshape(-1, n) ** 2).mean(axis=1))
    env = env - env.mean()
    nn = np.linalg.norm(env)
    return env / nn if nn > 1e-9 else None


def xcorr_peak(a: np.ndarray, b: np.ndarray) -> float:
    """较短包络在较长上滑动，返回**归一化**互相关峰值。

    ⚠️ 我第一版用 FFT 做互相关，但只把 `a` 归一化了，
       **没有对 `b` 的滑动窗口做局部归一化** —— 导致峰值被稀释：
       实测连"同 key"的同一段音频也只有 0.116（应该接近 1.0）。
       因为 `b` 比 `a` 长，窗口只覆盖 `b` 的一部分，
       `dot(a, b_win)` 被 `||b||`（全局）除，必然偏小。

    正确做法：每个偏移处用**该窗口自身的范数**归一化：
        corr(k) = dot(a, b[k:k+len(a)]) / (||a|| * ||b[k:k+len(a)]||)
    这样同一段音频（即使长度不同）也会得到接近 1.0 的峰值。
    """
    if len(a) > len(b):
        a, b = b, a
    if len(a) < 4 or len(b) < len(a):
        return 0.0
    na = np.linalg.norm(a)
    if na < 1e-9:
        return 0.0
    a = a / na
    L = len(a)
    # 滑动窗口的范数（用累积平方和，O(n)）
    csum = np.concatenate([[0.0], np.cumsum(b ** 2)])
    win_sq = csum[L:] - csum[:-L]          # 每个窗口的平方和
    win_norm = np.sqrt(np.maximum(win_sq, 1e-18))
    # 每个窗口的点积：用相关（卷积）
    nfft = 1 << (len(a) + len(b) - 1).bit_length()
    c = np.fft.irfft(np.fft.rfft(b, nfft) * np.conj(np.fft.rfft(a, nfft)),
                     nfft)[:len(b) - L + 1]
    corr = c / win_norm
    return float(np.abs(corr).max())


print("=" * 104)
print("映射验证（包络互相关）")
print("=" * 104)

# 训练集包络缓存
train_env: dict[str, np.ndarray] = {}
for w in sorted(TRAIN.glob("*.wav")):
    e = envelope(w)
    if e is not None:
        train_env[w.stem] = e
print(f"  训练集包络: {len(train_env)} 条")

tm: dict[str, str] = {}
for line in TXT.read_text(encoding="utf-8").splitlines():
    p = line.split("\t")
    if len(p) >= 4:
        tm[p[0].replace(".wav", "")] = p[3].strip()

rows = json.loads((OUT / "texts_mapped.json").read_text(encoding="utf-8"))

print()
print(f"  {'#':>3s} {'候选':<22s} {'训练集对应':<38s} {'相关':>7s} "
      f"{'源时长':>7s} {'训练':>7s} {'Δ':>7s}  判定")
print("  " + "-" * 104)

results = []
for r in rows:
    p = Path(r["path"])
    e_c = envelope(p)
    how = r["how"]

    if how in ("精确", "前缀"):
        # 文本直接来自训练集同 key -> 可靠
        st = r["stem"]
        full = next((k for k in train_env if k.startswith(st[:16])), None)
        corr = xcorr_peak(e_c, train_env[full]) if (e_c is not None and full) else float("nan")
        results.append({**r, "corr": corr, "train_stem": full, "ok": True,
                        "note": "同 key"})
        print(f"  {r['rank']:3d} {r['stem'][:22]:<22s} "
              f"{(full or '(同key)')[:38]:<38s} {corr:7.3f} "
              f"{r['dur']:6.2f}s {'':>7s} {'':>7s}  ✓ {how}")
        continue

    # 映射来的：从 how 提取训练集 stem 前缀
    src = how[how.index("(") + 1:how.index(")")] if "(" in how else ""
    full = next((k for k in train_env if k.startswith(src)), None)
    if e_c is None or full is None:
        results.append({**r, "corr": float("nan"), "ok": False,
                        "note": "无法计算"})
        print(f"  {r['rank']:3d} {r['stem'][:22]:<22s} {src[:38]:<38s} "
              f"{'?':>7s} {'':>7s} {'':>7s} {'':>7s}  ✗ 无法验证")
        continue

    corr = xcorr_peak(e_c, train_env[full])
    ok = corr > 0.80
    d_c = r["dur"]
    try:
        i = sf.info(str(TRAIN / f"{full}.wav"))
        d_t = i.frames / float(i.samplerate)
    except Exception:  # noqa: BLE001
        d_t = float("nan")
    results.append({**r, "corr": corr, "train_stem": full, "ok": ok,
                    "note": f"corr={corr:.3f}"})
    print(f"  {r['rank']:3d} {r['stem'][:22]:<22s} {full[:38]:<38s} "
          f"{corr:7.3f} {d_c:6.2f}s {d_t:6.2f}s {d_c-d_t:+6.2f}s  "
          f"{'✓ 同一段' if ok else '✗ 不同段'}")

good = [x for x in results if x.get("ok")]
print()
print("=" * 104)
print(f"验证通过: **{len(good)} / {len(results)}**")
print("=" * 104)
print()
print(f"  {'#':>3s} {'文件':<24s} {'相似度':>8s} {'尾平坦':>8s} "
      f"{'尾/体':>8s} {'时长':>7s} {'相关':>7s}  文本")
print("  " + "-" * 104)
for v in sorted(good, key=lambda x: x["flat"]):
    t = v["text"][:32] + ("…" if len(v["text"]) > 32 else "")
    print(f"  {v['rank']:3d} {v['stem'][:24]:<24s} {v['sim']:8.4f} "
          f"{v['flat']:8.4f} {v['tb']:+8.2f} {v['dur']:6.2f}s "
          f"{v['corr']:7.3f}  {t}")

# 选取：orig + 尾最干净的 2 个 + 相似度最高 1 个
by_flat = sorted(good, key=lambda x: x["flat"])
sel, seen = [], set()
for v in by_flat[:2]:
    sel.append(v)
    seen.add(v["stem"])
for v in sorted(good, key=lambda x: -x["sim"]):
    if v["stem"] not in seen:
        sel.append(v)
        seen.add(v["stem"])
        break
for v in sorted(good, key=lambda x: -x["sim"]):
    if v["stem"] not in seen and len(sel) < 4:
        sel.append(v)
        seen.add(v["stem"])

lines = ["CANDS = [",
         '    ("orig", GSV / "ref" / "default.wav",',
         '     (GSV / "ref" / "ref.txt").read_text(encoding="utf-8").strip()),']
for v in sel:
    lines.append(f'    ("r{v["rank"]:02d}", Path(r"{v["path"]}"),')
    lines.append(f'     {v["text"]!r}),   # flat={v["flat"]:.4f} '
                 f'sim={v["sim"]:.4f} corr={v["corr"]:.3f}')
lines.append("]")
(OUT / "cands_final.py").write_text("\n".join(lines) + "\n", encoding="utf-8")

print()
print("=" * 104)
print("最终 CANDS")
print("=" * 104)
print("\n".join(lines))
print()
print(f"  -> {OUT / 'cands_final.py'}")

rep = ["=" * 80, "映射验证（包络互相关）", "=" * 80, ""]
rep.append(f"通过 {len(good)}/{len(results)}   阈值 corr > 0.80")
rep.append("")
rep.append(f"{'#':>3s} {'候选':<24s} {'corr':>7s} {'ok':>4s}  训练集对应")
for v in results:
    rep.append(f"{v['rank']:3d} {v['stem'][:24]:<24s} {v.get('corr',float('nan')):7.3f} "
               f"{'OK' if v.get('ok') else 'FAIL':>4s}  {v.get('train_stem','')[:44]}")
rep.append("")
rep.append("通过者（按尾平坦排序）:")
for i, v in enumerate(by_flat, 1):
    rep.append(f"{i:2d}. {v['stem'][:30]:<32s} flat={v['flat']:.4f} "
               f"sim={v['sim']:.4f} tb={v['tb']:+.2f} corr={v['corr']:.3f}")
(OUT / "map_verify2.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
print(f"  -> {OUT / 'map_verify2.txt'}")
print("=" * 104)
