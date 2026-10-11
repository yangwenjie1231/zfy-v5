# -*- coding: utf-8 -*-
"""尾音后处理参数扫描 —— 找最优组合。

## 目标（对齐真人）
    尾/体dB : -4.94   （模型 +0.89，**不衰减**）
    尾平坦  : 0.1352  （模型 0.3025，**噪声高 2.2 倍**）

## 扫描维度
    fade_ms : 150 / 250 / 350
    fade_db : -6 / -12 / -20
    tilt_db : -3 / -6 / -10

共 27 组，用**同 10 个 wav**测，选最接近真人的。

## 评分
    score = |Δ尾体| / 5.0 + |Δ尾平坦| / 0.17
（两项归一化到"距真人的相对差距"，越小越好）

## 输出
    dist/tailfix_sweep/report.txt
    dist/tailfix_sweep/best/   （最优参数的结果）
"""
from __future__ import annotations

import itertools
import json
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")
SRC = V5 / "dist" / "ref_swap_train" / "orig"
OUT = V5 / "dist" / "tailfix_sweep"
PY = sys.executable
SCRIPT = V5 / "postprocess_tail.py"

REF_TB, REF_FLAT = -4.94, 0.1352


def metrics(d: Path) -> dict:
    tbs, fls = [], []
    for w in sorted(d.glob("*.wav")):
        x, sr = sf.read(str(w), always_2d=True, dtype="float64")
        x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
        if len(x) < sr * 0.3:
            continue
        nf = 256
        m = (len(x) // nf) * nf
        rmsf = np.sqrt((x[:m].reshape(-1, nf) ** 2).mean(axis=1))
        r = float(np.sqrt((x ** 2).mean())) + 1e-12
        v = np.where(rmsf > r * (10 ** (-45 / 20)))[0]
        if len(v) == 0:
            continue
        end = int((v[-1] + 1) * nf)
        n = int(0.4 * sr)
        tail = x[max(0, end - n):end]
        body = x[max(0, end // 2 - sr // 2):end // 2 + sr // 2]
        if len(tail) < n // 2 or len(body) < n // 2:
            continue
        tr = float(np.sqrt((tail ** 2).mean()))
        br = float(np.sqrt((body ** 2).mean())) + 1e-12
        tbs.append(20 * np.log10(tr / br + 1e-12))
        nn = min(len(tail), 2048)
        S = np.abs(np.fft.rfft(tail[:nn] * np.hanning(nn))) + 1e-12
        fls.append(float(np.exp(np.log(S).mean()) / S.mean()))
    return {"tb": float(np.mean(tbs)) if tbs else float("nan"),
            "flat": float(np.mean(fls)) if fls else float("nan"),
            "n": len(tbs)}


print("=" * 100)
print("尾音后处理参数扫描")
print("=" * 100)
print(f"  目标（真人）: 尾/体 {REF_TB:+.2f} dB   尾平坦 {REF_FLAT:.4f}")
print(f"  输入: {SRC}")

base = metrics(SRC)
print(f"  原始: 尾/体 {base['tb']:+.2f} dB   尾平坦 {base['flat']:.4f}")

grid = list(itertools.product((150, 250, 350), (-6, -12, -20), (-3, -6, -10)))
print(f"\n  扫描 {len(grid)} 组 ...\n")

rows = []
for i, (fm, fd, td) in enumerate(grid, 1):
    tag = f"f{fm}_d{abs(fd)}_t{abs(td)}"
    d = OUT / tag
    r = subprocess.run(
        [PY, str(SCRIPT), "--in", str(SRC), "--out", str(d),
         "--fade-ms", str(fm), "--fade-db", str(fd), "--tilt-db", str(td)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(V5))
    if not d.is_dir() or not list(d.glob("*.wav")):
        print(f"  {i:2d}/{len(grid)} {tag:20s} 失败")
        continue
    m = metrics(d)
    score = (abs(m["tb"] - REF_TB) / 5.0 + abs(m["flat"] - REF_FLAT) / 0.17)
    rows.append({"tag": tag, "fade_ms": fm, "fade_db": fd, "tilt_db": td,
                 "tb": m["tb"], "flat": m["flat"], "score": score})
    if i % 9 == 0 or i == len(grid):
        print(f"  ...{i}/{len(grid)}")

rows.sort(key=lambda r: r["score"])
print()
print("=" * 100)
print("结果（按接近真人的程度排序）")
print("=" * 100)
print(f"  {'#':>3s} {'参数':<20s} {'尾/体dB':>9s} {'Δ尾体':>8s} "
      f"{'尾平坦':>8s} {'Δ平坦':>8s} {'评分':>8s}")
print("  " + "-" * 76)
print(f"  {'—':>3s} {'**原始(未处理)**':<20s} {base['tb']:+9.2f} "
      f"{base['tb']-REF_TB:+8.2f} {base['flat']:8.4f} "
      f"{base['flat']-REF_FLAT:+8.4f} "
      f"{(abs(base['tb']-REF_TB)/5.0+abs(base['flat']-REF_FLAT)/0.17):8.4f}")
for i, r in enumerate(rows[:15], 1):
    print(f"  {i:3d} {r['tag']:<20s} {r['tb']:+9.2f} "
          f"{r['tb']-REF_TB:+8.2f} {r['flat']:8.4f} "
          f"{r['flat']-REF_FLAT:+8.4f} {r['score']:8.4f}")

if not rows:
    print("  !! 无有效结果")
    sys.exit(1)

best = rows[0]
print()
print("=" * 100)
print("最优参数")
print("=" * 100)
print(f"  fade-ms {best['fade_ms']}   fade-db {best['fade_db']}   "
      f"tilt-db {best['tilt_db']}")
print(f"    尾/体 {base['tb']:+.2f} -> {best['tb']:+.2f} dB   "
      f"（真人 {REF_TB:+.2f}）")
print(f"    尾平坦 {base['flat']:.4f} -> {best['flat']:.4f}   "
      f"（真人 {REF_FLAT:.4f}）")

# 复制最优结果
bestdir = OUT / "best"
bestdir.mkdir(parents=True, exist_ok=True)
for w in (OUT / best["tag"]).glob("*.wav"):
    (bestdir / w.name).write_bytes(w.read_bytes())
print()
print(f"  最优结果已复制到 {bestdir}")

print()
print("=" * 100)
print("命令")
print("=" * 100)
print(f"  python _tools/postprocess_tail.py --in <输入目录> --out <输出目录> \\")
print(f"      --fade-ms {best['fade_ms']} --fade-db {best['fade_db']} "
      f"--tilt-db {best['tilt_db']}")

rep = ["=" * 80, "尾音后处理参数扫描", "=" * 80, ""]
rep.append(f"目标(真人): 尾/体 {REF_TB:+.2f}dB  尾平坦 {REF_FLAT:.4f}")
rep.append(f"原始:        尾/体 {base['tb']:+.2f}dB  尾平坦 {base['flat']:.4f}")
rep.append("")
rep.append(f"{'#':>3s} {'参数':<20s} {'尾/体':>9s} {'尾平坦':>9s} {'评分':>8s}")
for i, r in enumerate(rows[:20], 1):
    rep.append(f"{i:3d} {r['tag']:<20s} {r['tb']:+9.2f} {r['flat']:9.4f} "
               f"{r['score']:8.4f}")
rep.append("")
rep.append(f"最优: fade-ms {best['fade_ms']} fade-db {best['fade_db']} "
           f"tilt-db {best['tilt_db']}")
(OUT / "report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
(OUT / "sweep.json").write_text(json.dumps(
    {"ref": {"tb": REF_TB, "flat": REF_FLAT}, "base": base,
     "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
print()
print(f"  -> {OUT / 'report.txt'}")
print("=" * 100)
