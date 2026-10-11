# -*- coding: utf-8 -*-
"""挑「尾部干净」的参考音频 —— 解决"尾音气声强"。

## 已知问题（实测）
    v2Pro 输出 : 尾/体 **+0.83 dB**, 尾平坦 **0.3338**
    真人录音   : 尾/体 **-4.94 dB**, 尾平坦 **0.1352**
    -> 模型句末**不衰减**且**噪声性高 2.5 倍** = 听感"气声强"

    参考音频 ref/default : 尾/体 **+29.83 dB**, 尾平坦 **0.5619**
    -> **参考音频本身尾部极脏**，模型在模仿它

## 思路
参考音频是**推理时参数**（不需要重训）。换一个：
    · 尾部**衰减**（尾/体 < 0）
    · 尾部**噪声低**（平坦 < 0.15）
    · 时长 3~10s（技能要求）
    · 音色仍是庄方宜（从训练集里挑，天然满足）

## 筛选条件（可调）
    dur        3.0 ~ 10.0s
    尾/体dB    < +1.0      （尾部不异常抬升）
    尾平坦     < 0.18      （噪声性低）
    尾部低频   < 60%       （不以低频为主）
    无长静音   静音尾 < 0.6s
    总 RMS     > 0.01      （不是静音文件）

## 输出
    dist/ref_candidates/report.txt
    dist/ref_candidates/top_N.wav   （复制前几名，便于试听）
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
OUT = V5 / "dist" / "ref_candidates"
BANDS = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000),
         (4000, 8000), (8000, 16000)]

SRC = [
    GSV / "logs" / "zfy_v5" / "5-wav32k",
    Path(r"D:\Users\A\Downloads\仿通行证\新素材\音频\终末地解包\剧情语音"),
    Path(r"D:\Users\A\Downloads\仿通行证\新素材\音频\终末地解包\干员语音"),
]

# 阈值
MIN_DUR, MAX_DUR = 3.0, 10.0
MAX_TB = 1.0
MAX_FLAT = 0.18
MAX_LO = 0.60
MAX_SIL = 0.60
MIN_RMS = 0.01


def load(p: Path):
    x, sr = sf.read(str(p), always_2d=True, dtype="float64")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    return x, int(sr)


def metrics(p: Path) -> dict | None:
    try:
        x, sr = load(p)
    except Exception:  # noqa: BLE001
        return None
    dur = len(x) / sr
    if not (MIN_DUR <= dur <= MAX_DUR):
        return None
    rms = float(np.sqrt((x ** 2).mean()))
    if rms < MIN_RMS:
        return None
    # 裁尾部静音
    nf = 256
    m = (len(x) // nf) * nf
    if m < nf * 8:
        return None
    rmsf = np.sqrt((x[:m].reshape(-1, nf) ** 2).mean(axis=1))
    ref = rms + 1e-12
    voiced = np.where(rmsf > ref * (10 ** (-45 / 20)))[0]
    if len(voiced) == 0:
        return None
    end = int((voiced[-1] + 1) * nf)
    sil_tail = (len(x) - end) / sr
    if sil_tail > MAX_SIL:
        return None
    n = int(0.4 * sr)
    tail = x[max(0, end - n):end]
    body = x[max(0, end // 2 - sr // 2):end // 2 + sr // 2]
    if len(tail) < n // 2 or len(body) < n // 2:
        return None
    tr = float(np.sqrt((tail ** 2).mean()))
    br = float(np.sqrt((body ** 2).mean())) + 1e-12
    tb = 20 * np.log10(tr / br + 1e-12)
    # 谱平坦度
    nn = min(len(tail), 2048)
    S = np.abs(np.fft.rfft(tail[:nn] * np.hanning(nn))) + 1e-12
    flat = float(np.exp(np.log(S).mean()) / S.mean())
    # 频段
    S2 = np.abs(np.fft.rfft(tail[:nn] * np.hanning(nn)))
    f = np.fft.rfftfreq(nn, 1 / sr)
    S2 = S2 / (S2.sum() + 1e-12)
    lo = float(S2[(f >= 0) & (f < min(2000, sr / 2))].sum())
    hi = float(S2[(f >= 4000) & (f < min(16000, sr / 2))].sum())
    # 峰值（避免削波）
    pk = float(np.abs(x).max())
    return {"dur": dur, "rms": rms, "tb": tb, "flat": flat,
            "lo": lo, "hi": hi, "sil_tail": sil_tail, "peak": pk,
            "sr": sr}


print("=" * 100)
print("挑「尾部干净」的参考音频")
print("=" * 100)
print(f"  条件: 时长 {MIN_DUR}~{MAX_DUR}s  尾/体<{MAX_TB:+.1f}dB  "
      f"平坦<{MAX_FLAT}  低频<{MAX_LO*100:.0f}%  静音尾<{MAX_SIL}s")
print()

cands = []
seen = set()
for d in SRC:
    if not d.is_dir():
        continue
    ws = sorted(d.rglob("*.wav"))
    print(f"  扫描 {d.name}: {len(ws)} 个")
    for w in ws:
        # 去重（同名只算一次）
        key = w.stem[:60]
        if key in seen:
            continue
        seen.add(key)
        mt = metrics(w)
        if mt is None:
            continue
        if (mt["tb"] < MAX_TB and mt["flat"] < MAX_FLAT
                and mt["lo"] < MAX_LO and mt["peak"] < 0.99):
            cands.append((w, mt))

# 评分：尾部越"干净"越好
#   平坦度权重最高（直接对应"气声"）
#   尾/体 越负越好（衰减）
#   低频越低越好
def score(mt: dict) -> float:
    return (mt["flat"] * 3.0
            + max(0.0, mt["tb"] + 5.0) / 10.0
            + mt["lo"] * 0.5)


cands.sort(key=lambda t: score(t[1]))

print()
print("=" * 100)
print(f"合格候选: {len(cands)} 个（按尾部干净度排序）")
print("=" * 100)
print(f"  {'#':>3s} {'文件':<44s} {'时长':>6s} {'尾/体dB':>8s} "
      f"{'尾平坦':>8s} {'尾低频':>7s} {'尾高频':>7s} {'评分':>7s}")
print("  " + "-" * 96)
for i, (w, mt) in enumerate(cands[:25], 1):
    nm = w.stem[:44]
    print(f"  {i:3d} {nm:<44s} {mt['dur']:5.2f}s {mt['tb']:+8.2f} "
          f"{mt['flat']:8.4f} {mt['lo']*100:6.1f}% {mt['hi']*100:6.1f}% "
          f"{score(mt):7.4f}")

# 对照组
print()
print("=" * 100)
print("对照")
print("=" * 100)
REF = GSV / "ref" / "default.wav"
if REF.exists():
    mt = metrics(REF)
    if mt:
        print(f"  当前 ref/default.wav : 尾/体 {mt['tb']:+.2f}dB  "
              f"平坦 {mt['flat']:.4f}  低频 {mt['lo']*100:.1f}%  "
              f"评分 {score(mt):.4f}")
    else:
        x, sr = load(REF)
        print(f"  当前 ref/default.wav : 时长 {len(x)/sr:.2f}s（超出 "
              f"{MIN_DUR}~{MAX_DUR}s 范围，未参与评分）")
print(f"  真人录音典型值       : 尾/体 -4.94dB  平坦 0.1352  低频 51.7%")
print(f"  v2Pro 输出典型值     : 尾/体 +0.83dB  平坦 0.3338  低频 57.5%")

# 导出前几名
OUT.mkdir(parents=True, exist_ok=True)
for i, (w, mt) in enumerate(cands[:10], 1):
    dst = OUT / f"top{i:02d}_{w.stem[:40]}.wav"
    shutil.copy2(w, dst)

print()
print("=" * 100)
print("判读")
print("=" * 100)
if cands:
    best_w, best_m = cands[0]
    print(f"  最优候选: {best_w.name}")
    print(f"    时长 {best_m['dur']:.2f}s  尾/体 {best_m['tb']:+.2f}dB  "
          f"平坦 {best_m['flat']:.4f}  低频 {best_m['lo']*100:.1f}%")
    print()
    print(f"  与当前 ref/default 对比：")
    if REF.exists() and metrics(REF):
        rm = metrics(REF)
        print(f"    平坦度 {rm['flat']:.4f} -> {best_m['flat']:.4f}  "
              f"（降 {(1-best_m['flat']/rm['flat'])*100:.0f}%）")
        print(f"    尾/体  {rm['tb']:+.2f} -> {best_m['tb']:+.2f} dB")
    print()
    print("  ## 下一步")
    print("    用最优候选重新合成，对比尾音：")
    print(f"      python _tools/compare_ref_candidates.py")
    print()
    print("  ## ⚠️ 但要注意")
    print("    · 换参考音频会**改变音色相似度**（技能记录：21 个候选里")
    print("      `006_更换武器` 相似度最优 0.6459）")
    print("    · 所以需要**同时看**：尾音干净度 + 相似度")
    print("    · 本脚本只筛了尾音，**没筛相似度**")
else:
    print("  !! 没有合格候选 —— 说明训练集里普遍尾部偏脏")
    print("     那就只能靠**后处理**（尾部淡出）")

print()
print("=" * 100)
rep = ["=" * 78, "参考音频候选（按尾部干净度）", "=" * 78, ""]
rep.append(f"条件: {MIN_DUR}~{MAX_DUR}s  尾/体<{MAX_TB}  平坦<{MAX_FLAT}  "
           f"低频<{MAX_LO}  静音尾<{MAX_SIL}")
rep.append(f"合格: {len(cands)}")
rep.append("")
rep.append(f"{'#':>3s} {'文件':<46s} {'时长':>7s} {'尾/体':>8s} {'平坦':>8s} "
           f"{'低频':>7s}")
for i, (w, mt) in enumerate(cands[:30], 1):
    rep.append(f"{i:3d} {w.stem[:46]:<46s} {mt['dur']:6.2f}s "
               f"{mt['tb']:+8.2f} {mt['flat']:8.4f} {mt['lo']*100:6.1f}%")
(OUT / "report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
(OUT / "candidates.json").write_text(json.dumps(
    [{"path": str(w), **mt} for w, mt in cands[:50]],
    ensure_ascii=False, indent=2), encoding="utf-8")
print(f"  -> {OUT / 'report.txt'}")
print(f"  -> 前 10 名已复制到 {OUT}")
print("=" * 100)
