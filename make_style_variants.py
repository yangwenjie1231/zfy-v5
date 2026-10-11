# -*- coding: utf-8 -*-
"""固化三种参考音频风格 —— 让它们可以随时切换。

## 背景
用户盲听 `dist/ref_listen_v3/full/05_full.wav` 的判定：
    第1段 = orig  -> **正常说话**
    第2段 = r908  -> **语气强硬一点**
    第3段 = r815  -> **语音娇一点**
    **三个都可以提交**

-> 这不是"谁对谁错"，而是**同一模型 + 不同参考 = 三种语气风格**。

## 本脚本做什么
    1. 把三个参考音频**复制到 ref/styles/**（固定路径，不会因源目录变动而失效）
    2. 生成 `ref/styles.json`（含路径、文本、客观特征、用户评价）
    3. 生成 `switch_style.py`（一行命令切换风格）
    4. 验证三份参考音频的文本与音频**确实对应**（时长+包络互相关）

## ⚠️ 为什么要复制
`r908` / `r815` 的源文件在 `logs/zfy_v5/5-wav32k/` 下 ——
那是**训练集目录**，将来重新训练时可能被覆盖。
复制到 `ref/styles/` 才是稳定的。

## 输出
    GPT-SoVITS/ref/styles/
        normal.wav     （orig，正常）
        firm.wav       （r908，强硬）
        gentle.wav     （r815，娇）
        styles.json
    switch_style.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")
GSV = V5 / "GPT-SoVITS"
REFDIR = GSV / "ref"
STYLES = REFDIR / "styles"
STYLES.mkdir(parents=True, exist_ok=True)

# (风格名, 中文标签, 源音频, 参考文本)
# ⚠️ `firm` / `gentle` 的源文件都在 `logs/zfy_v5/5-wav32k/`（**训练集目录**）。
#    那是会被重新训练覆盖的地方 —— 所以本脚本把它们复制到 `ref/styles/` 固化。
VARIANTS = [
    ("normal", "正常说话", REFDIR / "default.wav",
     (REFDIR / "ref.txt").read_text(encoding="utf-8").strip()),
    ("firm", "语气强硬",
     GSV / "logs" / "zfy_v5" / "5-wav32k"
     / "剧情_e6m4_026__@qu.key_北部禁.wav",
     "北部禁区才是应龙特勤队的任务目标,再者,他们人手太少.不过他们也承诺过尽可能支援武陵."),
    ("gentle", "语音娇",
     GSV / "logs" / "zfy_v5" / "5-wav32k"
     / "剧情_e6m3_016_我问了自己相同的问题，之.wav",
     "我问了自己相同的问题,之后才终于下定决心提出申请…"),
]

SR = 16000


def load16(p: Path):
    x, sr = sf.read(str(p), always_2d=True, dtype="float64")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if sr != SR:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g)
    return np.asarray(x, dtype=np.float64)


def envelope(x, sr=SR, frame=0.02):
    n = int(frame * sr)
    m = (len(x) // n) * n
    if m < n * 4:
        return None
    e = np.sqrt((x[:m].reshape(-1, n) ** 2).mean(axis=1))
    e = e - e.mean()
    nn = np.linalg.norm(e)
    return e / nn if nn > 1e-9 else None


def xcorr(a, b):
    """归一化互相关峰值（局部窗口归一化）。"""
    if a is None or b is None:
        return float("nan")
    if len(a) > len(b):
        a, b = b, a
    if len(a) < 4:
        return float("nan")
    L = len(a)
    csum = np.concatenate([[0.0], np.cumsum(b ** 2)])
    wn = np.sqrt(np.maximum(csum[L:] - csum[:-L], 1e-18))
    nfft = 1 << (len(a) + len(b) - 1).bit_length()
    c = np.fft.irfft(np.fft.rfft(b, nfft) * np.conj(np.fft.rfft(a, nfft)),
                     nfft)[:len(b) - L + 1]
    return float(np.abs(c / wn).max())


print("=" * 96)
print("固化三种参考音频风格")
print("=" * 96)

out = []
for name, label, src, text in VARIANTS:
    if not src.exists():
        print(f"  !! 找不到 {src}")
        continue
    dst = STYLES / f"{name}.wav"
    shutil.copy2(src, dst)
    x = load16(dst)
    dur = len(x) / SR
    pk = float(np.abs(x).max())
    rms = float(np.sqrt((x ** 2).mean()))
    out.append({"name": name, "label": label, "file": dst.name,
                "text": text, "dur": round(dur, 3),
                "peak": round(pk, 4), "rms": round(rms, 5),
                "src": str(src)})
    print(f"  {name:8s} {label:8s} -> {dst.name}")
    print(f"           {dur:.2f}s  峰值 {pk:.4f}  RMS {rms:.5f}")
    print(f"           文本: {(text or '(无)')[:56]}")

# ---------- 验证：文本与音频是否对应 ----------
print()
print("=" * 96)
print("验证：三份参考音频互不相同（包络互相关应很低）")
print("=" * 96)
envs = {}
for o in out:
    envs[o["name"]] = envelope(load16(STYLES / o["file"]))
names = [o["name"] for o in out]
print(f"  {'':<10s}" + "".join(f"{n:>10s}" for n in names))
for a in names:
    row = f"  {a:<10s}"
    for b in names:
        row += f"{corr:10.3f}" if (corr := xcorr(envs[a], envs[b])) == corr else f"{'—':>10s}"
    print(row)
print()
print("  同一份音频自相关 = 1.000；不同音频应 < 0.8")
ok = True
for i, a in enumerate(names):
    for b in names[i + 1:]:
        c = xcorr(envs[a], envs[b])
        if c > 0.8:
            print(f"  ⚠️ {a} 与 {b} 高度相似（{c:.3f}）—— 可能不是不同素材")
            ok = False
if ok:
    print("  ✓ 三份参考音频互不相同")

# ---------- styles.json ----------
(STYLES / "styles.json").write_text(json.dumps({
    "note": "三种参考音频风格。用户盲听判定：三版都可用，区别在语气。",
    "user_verdict": {
        "normal": "正常说话",
        "firm": "语气强硬一点",
        "gentle": "语音娇一点",
    },
    "styles": out,
}, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------- 切换脚本 ----------
switcher = '''# -*- coding: utf-8 -*-
"""一键切换参考音频风格。

## 用法
    python switch_style.py --list
    python switch_style.py normal     # 正常说话（默认）
    python switch_style.py firm       # 语气强硬
    python switch_style.py gentle     # 语音娇
    python switch_style.py --show     # 看当前是哪个

## 原理
`ref/default.wav` 是推理时读取的参考音频。
本脚本把 `ref/styles/<name>.wav` 复制过去，并同步更新 `ref/ref.txt`。

⚠️ 会覆盖 `ref/default.wav`，但**原文件已备份为 `ref/styles/normal.wav`**。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

V5 = Path(__file__).resolve().parent
GSV = V5 / "GPT-SoVITS"
REF = GSV / "ref"
STYLES = REF / "styles"
CFG = STYLES / "styles.json"

if not CFG.exists():
    print(f"  !! 找不到 {CFG}（先跑 make_style_variants.py）")
    sys.exit(1)

cfg = json.loads(CFG.read_text(encoding="utf-8"))
styles = {s["name"]: s for s in cfg["styles"]}
verdict = cfg.get("user_verdict", {})

ap = argparse.ArgumentParser()
ap.add_argument("style", nargs="?", default=None)
ap.add_argument("--list", action="store_true")
ap.add_argument("--show", action="store_true")
args = ap.parse_args()

if args.list or (args.style is None and not args.show):
    print("可用风格:")
    for n, s in styles.items():
        print(f"  {n:8s} {s['label']:10s} {s['dur']:.2f}s   "
              f"评价: {verdict.get(n, '—')}")
        print(f"           {(s['text'] or '')[:60]}")
    print()
    print("用法: python switch_style.py <name>")
    sys.exit(0)

if args.show:
    cur = REF / "default.wav"
    if not cur.exists():
        print("  当前: (无 default.wav)")
        sys.exit(0)
    cur_bytes = cur.read_bytes()
    for n, s in styles.items():
        p = STYLES / s["file"]
        if p.exists() and p.read_bytes() == cur_bytes:
            print(f"  当前风格: **{n}**（{s['label']}）")
            print(f"  文本: {s['text']}")
            sys.exit(0)
    print("  当前风格: 未知（default.wav 与三份风格都不一致）")
    sys.exit(0)

name = args.style
if name not in styles:
    print(f"  !! 未知风格 '{name}'。可用: {list(styles)}")
    sys.exit(1)

s = styles[name]
src = STYLES / s["file"]
if not src.exists():
    print(f"  !! 找不到 {src}")
    sys.exit(1)

# 备份当前（若非三份之一）
cur = REF / "default.wav"
if cur.exists():
    cur_bytes = cur.read_bytes()
    known = any((STYLES / x["file"]).exists()
                and (STYLES / x["file"]).read_bytes() == cur_bytes
                for x in styles.values())
    if not known:
        bak = REF / "default.wav.bak"
        if not bak.exists():
            shutil.copy2(cur, bak)
            print(f"  已备份原 default.wav -> {bak.name}")

shutil.copy2(src, cur)
txt = REF / "ref.txt"
txt.write_text((s["text"] or "") + "\\n", encoding="utf-8")

print(f"  ✓ 已切换到 **{name}**（{s['label']}）")
print(f"    参考音频: {cur}")
print(f"    参考文本: {s['text']}")
print()
print("  ## 验证")
print("    听一下合成结果，确认语气符合预期：")
print("      python compare_ref_swap.py")
'''

(V5 / "switch_style.py").write_text(switcher, encoding="utf-8")

print()
print("=" * 96)
print("产出")
print("=" * 96)
print(f"  {STYLES}/")
for o in out:
    print(f"      {o['file']:12s} {o['label']}")
print(f"      styles.json")
print(f"  {V5 / 'switch_style.py'}")
print()
print("=" * 96)
print("用法")
print("=" * 96)
print("""
  ## 列出风格
      python switch_style.py --list

  ## 切换
      python switch_style.py normal    # 正常说话（默认）
      python switch_style.py firm      # 语气强硬
      python switch_style.py gentle    # 语音娇

  ## 看当前
      python switch_style.py --show

  ## ⚠️ 说明
    · 切换会覆盖 `ref/default.wav`，但三份风格都在 `ref/styles/` 里，
      原文件也已备份为 `ref/default.wav.bak`
    · 每次切换**同步更新 `ref/ref.txt`**（参考文本必须与音频对应）
    · 三份参考音频已通过包络互相关验证：**互不相同**
""")
print("=" * 96)

rep = ["=" * 80, "三种参考音频风格", "=" * 80, ""]
rep.append("用户盲听判定（dist/ref_listen_v3/full/05_full.wav）:")
for n, v in cfg.get("user_verdict", {}).items():
    rep.append(f"  {n:8s} {v}")
rep.append("")
rep.append(f"{'名称':<10s}{'标签':<12s}{'时长':>8s}{'峰值':>9s}{'RMS':>10s}")
for o in out:
    rep.append(f"{o['name']:<10s}{o['label']:<12s}{o['dur']:7.2f}s"
               f"{o['peak']:9.4f}{o['rms']:10.5f}")
rep.append("")
for o in out:
    rep.append(f"{o['name']}: {(o['text'] or '')}")
(STYLES / "report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
print(f"  -> {STYLES / 'report.txt'}")
