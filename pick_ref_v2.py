# -*- coding: utf-8 -*-
"""参考音频筛选 v2 —— 按**用户听感对应的声学特征**筛。

## 为什么重做
前面 4 个后处理版本全部失败（v1 顾此失彼 / v2 过头 / v3 过头）。
**根源：在"输出端"打补丁，而不是换"输入端"。**
用户提示得对：数据量扩大了，应该能找到**更好的参考音频**。

## 我现在的筛选依据（4 轮失败换来的认知）
用户的三个听感 -> 参考音频该有的特征：

    「往上高」 -> 参考**结尾音高走向**必须 <= 0（降调/平调）
                 ⚠️ 当前 ref/default.wav 结尾是「…好吗？」= **问句升调**
                    实测末 0.5s 斜率 **+4.82 semi/s**
    「虚」     -> 参考 **body(150-800Hz)** 不能偏低
    「浊」     -> 参考 **500-800Hz** 要接近真人（当前 10.05% vs 真人 4.64%）

## 数据源（只用文本零风险的）
`5-wav32k` —— 文件名即 `2-name2text.txt` 的 key，**不可能错位**。
（其他源需推断映射，实测 12 个只验过 5 个，风险高）

## 筛选条件（全部要满足）
    时长        3.0 ~ 10.0s        （技能要求）
    文本        必须在 2-name2text.txt 里
    **末段F0斜率 <= +0.5 semi/s**   （不是升调结尾）  <- 关键新增
    尾平坦      < 0.20              （尾部干净）
    500-800Hz   < 8.0%              （不浊）
    body        > 70%               （不虚）
    相似度      越高越好

## 评分
    score = 相似度
          - |500-800 - 4.64| / 40      （浊度惩罚）
          - max(0, tail_slope) / 10    （升调惩罚）
          - 尾平坦 * 0.3               （脏尾惩罚）

## 输出
    dist/ref_v2/
        report.txt
        top01~top10.wav
        CANDS.py      可直接用于合成对比
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")
GSV = V5 / "GPT-SoVITS"
TRAIN = GSV / "logs" / "zfy_v5" / "5-wav32k"
TXT = GSV / "logs" / "zfy_v5" / "2-name2text.txt"
HELD = V5 / "dist" / "heldout" / "heldout_clean.json"
CUR_REF = GSV / "ref" / "default.wav"
OUT = V5 / "dist" / "ref_v2"
OUT.mkdir(parents=True, exist_ok=True)
SR = 16000

# 真人目标（实测，来自 dist/zhuo_audit）
TGT = {"b500_800": 4.64, "body": 78.35, "cent": 898.0, "air": 2.33}

MIN_DUR, MAX_DUR = 3.0, 10.0
MAX_SLOPE = 0.5          # semi/s，末段 F0 斜率上限
MAX_FLAT = 0.20
MAX_MUD = 8.0            # 500-800Hz %
MIN_BODY = 70.0          # 150-800Hz %


def load16(p: Path):
    x, sr = sf.read(str(p), always_2d=True, dtype="float64")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if sr != SR:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g)
    return np.asarray(x, dtype=np.float64)


def voiced_span(x, sr=SR, t=-45.0):
    n = int(0.01 * sr)
    m = (len(x) // n) * n
    if m < n * 4:
        return 0, len(x)
    r = np.sqrt((x[:m].reshape(-1, n) ** 2).mean(axis=1))
    ref = r.max() + 1e-12
    v = np.where(r > ref * (10 ** (t / 20)))[0]
    return (int(v[0]) * n, int(v[-1] + 1) * n) if len(v) else (0, len(x))


def f0_slope(x, sr=SR, tail=0.5):
    """末段 F0 斜率（semi/s）。正=升调，负=降调。"""
    n, h = int(0.04 * sr), int(0.01 * sr)
    if len(x) < n * 4:
        return float("nan")
    ts, f0s = [], []
    for i in range(0, len(x) - n, h):
        seg = x[i:i + n]
        if np.sqrt((seg ** 2).mean()) < 1e-3:
            continue
        z = (seg - seg.mean()) * np.hanning(n)
        ac = np.correlate(z, z, "full")[len(z) - 1:]
        if ac[0] <= 0:
            continue
        ac /= ac[0]
        lo, hi = int(sr / 350), min(int(sr / 70), len(ac) - 1)
        if hi <= lo:
            continue
        k = int(np.argmax(ac[lo:hi]))
        if ac[lo + k] < 0.4:
            continue
        ts.append(i / sr)
        f0s.append(sr / (lo + k))
    if len(ts) < 6:
        return float("nan")
    ts, f0s = np.array(ts), np.array(f0s)
    m = ts >= ts[-1] - tail
    if m.sum() < 4:
        return float("nan")
    s = 12 * np.log2(f0s[m] / np.median(f0s))
    return float(np.polyfit(ts[m] - ts[m][0], s, 1)[0])


def spec_feats(x, sr=SR):
    nfft = 2048
    acc = np.zeros(nfft // 2 + 1)
    cnt = 0
    for i in range(0, max(1, len(x) - nfft), nfft // 4):
        seg = x[i:i + nfft]
        if len(seg) < nfft or np.sqrt((seg ** 2).mean()) < 1e-4:
            continue
        acc += np.abs(np.fft.rfft(seg * np.hanning(nfft))) ** 2
        cnt += 1
    if cnt == 0:
        return {}
    f = np.fft.rfftfreq(nfft, 1 / sr)
    P = acc / (acc.sum() + 1e-20)
    return {"cent": float((f * P).sum()),
            "body": float(P[(f >= 150) & (f < 800)].sum()) * 100,
            "b500_800": float(P[(f >= 500) & (f < 800)].sum()) * 100,
            "air": float(P[(f >= 6000) & (f < min(9000, SR / 2))].sum()) * 100}


def tail_flat(x, sr=SR, sec=0.4):
    n = int(sec * sr)
    t = x[-n:] if len(x) >= n else x
    nn = min(len(t), 2048)
    if nn < 64:
        return float("nan")
    S = np.abs(np.fft.rfft(t[:nn] * np.hanning(nn))) + 1e-12
    return float(np.exp(np.log(S).mean()) / S.mean())


print("=" * 104)
print("参考音频筛选 v2（按用户听感对应的声学特征）")
print("=" * 104)
print(f"  数据源: 5-wav32k（{len(list(TRAIN.glob('*.wav')))} 条，文本零风险）")
print()
print("  筛选条件:")
print(f"    时长       {MIN_DUR}~{MAX_DUR}s")
print(f"    末段F0斜率 <= {MAX_SLOPE:+.1f} semi/s   <- **不是升调结尾**（修'往上高'）")
print(f"    尾平坦     < {MAX_FLAT}              <- 尾部干净")
print(f"    500-800Hz  < {MAX_MUD:.1f}%              <- 不浊（真人 {TGT['b500_800']:.2f}%）")
print(f"    body       > {MIN_BODY:.0f}%               <- 不虚（真人 {TGT['body']:.2f}%）")

# 文本表
tm: dict[str, str] = {}
for line in TXT.read_text(encoding="utf-8").splitlines():
    p = line.split("\t")
    if len(p) >= 4:
        tm[p[0].replace(".wav", "")] = p[3].strip()
print(f"\n  文本表: {len(tm)} 条")

# 当前参考做基准
if CUR_REF.exists():
    cx = load16(CUR_REF)
    ca, cb = voiced_span(cx)
    cv = cx[ca:cb]
    cf = spec_feats(cv)
    print()
    print("  当前 ref/default.wav 基准:")
    print(f"    时长 {len(cx)/SR:.2f}s  末段F0斜率 {f0_slope(cv):+.2f} semi/s  "
          f"尾平坦 {tail_flat(cv):.4f}")
    print(f"    500-800Hz {cf.get('b500_800',0):.2f}%  body {cf.get('body',0):.2f}%  "
          f"质心 {cf.get('cent',0):.0f}")
    print(f"    -> 结尾是问句「好吗？」= **升调** {f0_slope(cv):+.2f}")

# ---------- 阶段 1：声学筛选 ----------
print()
print("=" * 104)
print("阶段 1：声学筛选")
print("=" * 104)
cands = []
n_dur = n_slope = n_flat = n_mud = n_body = 0
for w in sorted(TRAIN.glob("*.wav")):
    st = w.stem
    if st not in tm:
        continue
    try:
        x = load16(w)
    except Exception:  # noqa: BLE001
        continue
    dur = len(x) / SR
    if not (MIN_DUR <= dur <= MAX_DUR):
        n_dur += 1
        continue
    a, b = voiced_span(x)
    xv = x[a:b]
    if len(xv) / SR < MIN_DUR * 0.7:
        n_dur += 1
        continue
    sl = f0_slope(xv)
    if np.isnan(sl) or sl > MAX_SLOPE:
        n_slope += 1
        continue
    fl = tail_flat(xv)
    if np.isnan(fl) or fl >= MAX_FLAT:
        n_flat += 1
        continue
    ft = spec_feats(xv)
    if ft.get("b500_800", 99) >= MAX_MUD:
        n_mud += 1
        continue
    if ft.get("body", 0) <= MIN_BODY:
        n_body += 1
        continue
    cands.append({"path": w, "stem": st, "text": tm[st], "dur": dur,
                  "slope": sl, "flat": fl, **ft})

print(f"  逐条淘汰: 时长 {n_dur}  升调 {n_slope}  尾脏 {n_flat}  "
      f"浊 {n_mud}  虚 {n_body}")
print(f"  **通过: {len(cands)} 条**")
if not cands:
    print("  !! 无候选")
    sys.exit(1)

# ---------- 阶段 2：相似度 ----------
print()
print("=" * 104)
print("阶段 2：相似度")
print("=" * 104)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
sys.path[:0] = [str(GSV), str(GSV / "GPT_SoVITS"), str(GSV / "tools")]
os.chdir(str(GSV))
from GPT_SoVITS.sv import SV  # noqa: E402

sv = SV(DEV, is_half=False)
hd = json.loads(HELD.read_text(encoding="utf-8"))
targets = [Path(t["path"]) for t in hd.get("targets", [])
           if Path(t["path"]).exists()]


def emb(p: Path):
    try:
        import torchaudio
        w, sr = torchaudio.load(str(p))
        if w.shape[0] > 1:
            w = w.mean(0, keepdim=True)
        if sr != 16000:
            w = torchaudio.functional.resample(w, sr, 16000)
        w = w.squeeze(0)
        if w.shape[0] > 160000:
            w = w[:160000]
        e = sv.compute_embedding3(w.to(DEV).unsqueeze(0).float())
        return F.normalize(e.float(), dim=-1)
    except Exception:  # noqa: BLE001
        return None


T = [e for e in (emb(p) for p in targets) if e is not None]
T = torch.stack([t.reshape(-1) for t in T])
print(f"  目标 {len(T)} 条   候选 {len(cands)} 条")

for i, c in enumerate(cands):
    e = emb(c["path"])
    c["sim"] = float((T @ e.reshape(-1)).max().item()) if e is not None else 0.0
    if (i + 1) % 20 == 0:
        print(f"    ...{i+1}/{len(cands)}")

# 评分
for c in cands:
    c["score"] = (c["sim"]
                  - abs(c["b500_800"] - TGT["b500_800"]) / 40.0
                  - max(0.0, c["slope"]) / 10.0
                  - c["flat"] * 0.3)
cands.sort(key=lambda c: -c["score"])

# 当前参考的相似度
cur_sim = 0.0
if CUR_REF.exists():
    e = emb(CUR_REF)
    if e is not None:
        cur_sim = float((T @ e.reshape(-1)).max().item())

print()
print("=" * 104)
print("TOP 25")
print("=" * 104)
print(f"  {'#':>3s} {'文件':<38s} {'相似度':>8s} {'F0斜率':>8s} "
      f"{'尾平坦':>8s} {'500-800':>8s} {'body':>8s} {'质心':>7s} {'时长':>6s}")
print("  " + "-" * 100)
if CUR_REF.exists():
    print(f"  {'—':>3s} {'**当前 ref/default.wav**':<38s} {cur_sim:8.4f} "
          f"{f0_slope(cv):+8.2f} {tail_flat(cv):8.4f} "
          f"{cf.get('b500_800',0):7.2f}% {cf.get('body',0):7.2f}% "
          f"{cf.get('cent',0):7.0f} {len(cx)/SR:5.2f}s")
for i, c in enumerate(cands[:25], 1):
    print(f"  {i:3d} {c['stem'][:38]:<38s} {c['sim']:8.4f} {c['slope']:+8.2f} "
          f"{c['flat']:8.4f} {c['b500_800']:7.2f}% {c['body']:7.2f}% "
          f"{c['cent']:7.0f} {c['dur']:5.2f}s")

print()
print("=" * 104)
print("真人目标对照")
print("=" * 104)
print(f"  500-800Hz {TGT['b500_800']:.2f}%   body {TGT['body']:.2f}%   "
      f"质心 {TGT['cent']:.0f}   air {TGT['air']:.2f}%")

# ---------- 导出 ----------
for i, c in enumerate(cands[:10], 1):
    shutil.copy2(c["path"], OUT / f"top{i:02d}_{c['stem'][:44]}.wav")

# CANDS：当前 + 相似度最高 + 综合最优 + 最干净
sel, seen = [], set()


def add(c):
    if c["stem"] not in seen:
        sel.append(c)
        seen.add(c["stem"])


if cands:
    add(cands[0])                                        # 综合最优
    for c in sorted(cands, key=lambda x: -x["sim"])[:1]:  # 相似度最高
        add(c)
    for c in sorted(cands, key=lambda x: x["flat"])[:1]:  # 尾最干净
        add(c)
    for c in sorted(cands, key=lambda x: x["b500_800"])[:1]:  # 最不浊
        add(c)

lines = ["CANDS = [",
         '    ("orig", GSV / "ref" / "default.wav",',
         '     (GSV / "ref" / "ref.txt").read_text(encoding="utf-8").strip()),']
for c in sel:
    lines.append(f'    ("r{c["sim"]*1000:.0f}", Path(r"{c["path"]}"),')
    lines.append(f'     {c["text"]!r}),   # sim={c["sim"]:.4f} '
                 f'slope={c["slope"]:+.2f} flat={c["flat"]:.4f} '
                 f'mud={c["b500_800"]:.2f}%')
lines.append("]")
(OUT / "CANDS.py").write_text("\n".join(lines) + "\n", encoding="utf-8")

print()
print("=" * 104)
print("推荐 CANDS")
print("=" * 104)
print("\n".join(lines))
print()
print("  选取维度:")
print("    · 综合最优（相似度 + 不浊 + 不升调 + 尾干净）")
print("    · 相似度最高")
print("    · 尾部最干净")
print("    · 500-800Hz 最少（最不浊）")

rep = ["=" * 80, "参考音频筛选 v2", "=" * 80, ""]
rep.append(f"数据源 5-wav32k（{len(tm)} 条有文本）")
rep.append(f"淘汰: 时长 {n_dur} 升调 {n_slope} 尾脏 {n_flat} 浊 {n_mud} 虚 {n_body}")
rep.append(f"通过 {len(cands)}")
rep.append("")
rep.append(f"当前 ref/default: sim={cur_sim:.4f} slope={f0_slope(cv):+.2f} "
           f"flat={tail_flat(cv):.4f} mud={cf.get('b500_800',0):.2f}% "
           f"body={cf.get('body',0):.2f}%")
rep.append("")
rep.append(f"{'#':>3s} {'文件':<40s}{'sim':>8s}{'slope':>8s}{'flat':>8s}"
           f"{'mud':>8s}{'body':>8s}")
for i, c in enumerate(cands[:40], 1):
    rep.append(f"{i:3d} {c['stem'][:40]:<40s}{c['sim']:8.4f}{c['slope']:+8.2f}"
               f"{c['flat']:8.4f}{c['b500_800']:7.2f}%{c['body']:7.2f}%")
(OUT / "report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
(OUT / "candidates.json").write_text(json.dumps(
    [{**{k: v for k, v in c.items() if k != "path"}, "path": str(c["path"])}
     for c in cands[:80]], ensure_ascii=False, indent=2), encoding="utf-8")
print()
print(f"  -> {OUT / 'CANDS.py'}")
print(f"  -> {OUT / 'report.txt'}")
print(f"  -> 前 10 名 wav")
print("=" * 104)
