# -*- coding: utf-8 -*-
"""排查「尾音气声强」的来源 —— 是模型、参考音频、还是我的听包处理？

## 用户观察
「我听不出明显的区别，但是**所有句子**的尾音的气声都会强很多」

**关键**：三方对比包里 base / A_e4 / B_e4 **都有** ——
而 GPT 是唯一变量，SoVITS 固定 v2Pro。
所以**不是 GPT 的问题**。可能来源：

    A. v2Pro 模型本身的输出特性
    B. 参考音频 ref/default.wav 的"气声"被模仿
    C. **我的听包处理引入的**（RMS 对齐 / 重采样 / 拼接）
    D. 用户之前没注意，拼接后更明显（3 段连续 -> 听到 3 个尾音）

## 判据
对**同一条文本**的多个来源，测量尾部的声学特征：

    · 尾部 RMS（裁掉静音后）
    · **谱平坦度**（spectral flatness）—— 噪声性越高越"气声"
    · **高频比**（4k+/4k-）—— 气声能量集中在 2~8kHz
    · **谐噪比代理**（HNR proxy = 峰值自相关）—— 越低越"气"

对比：
    1. `dist/gpt_ab/base/NN.wav`        原始合成（32kHz，未处理）
    2. `dist/gpt_listen/NN_*_3way.wav`  我的处理（48kHz，RMS 对齐 + 拼接）
    3. `ref/default.wav`                参考音频（模型模仿的对象）
    4. 留出集真实录音                    真人对照

**如果 1 和 2 的尾部特征一致** -> 不是我引入的，是模型
**如果 3 的尾部也很"气"**       -> 参考音频的影响
**如果 4 的尾部不气**           -> 模型/参考的问题，不是数据本身

## 输出
    dist/tail_breath_report.txt
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")
RAW = V5 / "dist" / "gpt_ab" / "base"
LISTEN = V5 / "dist" / "gpt_listen"
REF = V5 / "GPT-SoVITS" / "ref" / "default.wav"
HELD = V5 / "dist" / "heldout" / "heldout_clean.json"

TEXTS = {
    "01": "让我想想。", "02": "好。",
    "03": "迎敌或是迎客，都得整装后再出发。",
    "04": "出门前，要不要先喝杯茶提提神？", "05": "嗯……我理一理。",
    "06": "再好的武器，也得朝夕相处磨合上几天。等用惯了，我再和你说说感受，好吗？",
    "07": "武陵城外有一片梨树林，每年春天都会开满白色的花。",
    "08": "她就在那阵花香里，只是他自己不知道。",
    "09": "作画者把心画在纸上，纸会烂。庄方宜把心砌在城里，城也会塌。",
    "10": "这件事我三点五十分再去处理。",
    "11": "管理员，今天的实验数据我已经整理好了。",
    "12": "清波城那边的情况，我一直在关注。",
    "13": "若是要谈，就该拿出诚意来。",
    "14": "我答应你，会好好照顾自己。",
    "15": "这些话，我本想留到以后再说的。",
    "16": "你来了。坐吧，茶还热着。",
    "17": "有些事，不是努力就能改变的。",
    "18": "但不去试，就永远不知道结果。",
    "19": "汤汤说，我是个念旧的人。",
    "20": "或许吧。毕竟这座城，承载了太多。",
}


def load(p: Path):
    x, sr = sf.read(str(p), always_2d=True, dtype="float64")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    return x, int(sr)


def trim_tail(x: np.ndarray, sr: int, thresh_db: float = -45.0):
    """裁掉尾部静音，返回有声区间的结束位置。"""
    nf = 256
    if len(x) < nf * 4:
        return len(x)
    m = (len(x) // nf) * nf
    rmsf = np.sqrt((x[:m].reshape(-1, nf) ** 2).mean(axis=1))
    ref = float(np.sqrt((x ** 2).mean())) + 1e-12
    voiced = np.where(rmsf > ref * (10 ** (thresh_db / 20)))[0]
    return int((voiced[-1] + 1) * nf) if len(voiced) else len(x)


def seg_stats(x: np.ndarray, sr: int, a: int, b: int) -> dict:
    """区间 [a,b) 的声学特征。"""
    s = x[a:b]
    if len(s) < 64:
        return {}
    rms = float(np.sqrt((s ** 2).mean()))
    # 谱平坦度：几何均值/算术均值（1=白噪声，0=纯音）
    n = min(len(s), 2048)
    S = np.abs(np.fft.rfft(s[:n] * np.hanning(n))) + 1e-12
    flat = float(np.exp(np.log(S).mean()) / S.mean())
    f = np.fft.rfftfreq(n, 1 / sr)
    hi = float(S[f >= 4000].sum() / (S[f < 4000].sum() + 1e-12))
    # 谐噪比代理：归一化自相关的峰值（1=很谐，0=很噪）
    z = s - s.mean()
    if len(z) > 32:
        ac = np.correlate(z, z, "full")[len(z) - 1:]
        ac = ac / (ac[0] + 1e-12)
        lo = int(sr / 500)          # 500Hz 对应周期
        hi_i = min(int(sr / 60), len(ac) - 1)
        hnr = float(ac[lo:hi_i].max()) if hi_i > lo else float("nan")
    else:
        hnr = float("nan")
    return {"rms": rms, "flat": flat, "hf": hi, "hnr": hnr}


def analyze(path: Path, label: str, tail_sec: float = 0.4) -> dict:
    try:
        x, sr = load(path)
    except Exception as e:  # noqa: BLE001
        return {"label": label, "err": f"{type(e).__name__}: {e}"}
    if len(x) < sr * 0.2:
        return {"label": label, "err": "太短"}
    end = trim_tail(x, sr)
    n_tail = int(tail_sec * sr)
    tail = seg_stats(x, sr, max(0, end - n_tail), end)
    # 主体（中间 1 秒）
    mid = (end // 2)
    body = seg_stats(x, sr, max(0, mid - sr // 2), mid + sr // 2)
    d = {"label": label, "sr": sr, "dur": len(x) / sr,
         "voiced_end": end / sr, "sil_tail": (len(x) - end) / sr}
    for k, v in tail.items():
        d[f"tail_{k}"] = v
    for k, v in body.items():
        d[f"body_{k}"] = v
    if tail.get("rms") and body.get("rms"):
        d["tail_body_db"] = 20 * np.log10(tail["rms"] / (body["rms"] + 1e-12))
    if tail.get("flat") and body.get("flat"):
        d["flat_ratio"] = tail["flat"] / (body["flat"] + 1e-12)
    if tail.get("hnr") is not None and body.get("hnr"):
        d["hnr_drop"] = body["hnr"] - tail["hnr"]
    return d


print("=" * 100)
print("尾音气声排查")
print("=" * 100)

# ---------- 1) 同句多来源对比（取 3 条长句）----------
print()
print("1) 同句多来源（长句 06/09/11）")
print("-" * 100)
print(f"  {'来源':<34s} {'时长':>6s} {'尾部RMS':>9s} {'尾/体dB':>8s} "
      f"{'尾平坦':>8s} {'平坦比':>7s} {'尾HNR':>7s} {'HNR降':>7s}")
print("  " + "-" * 96)

rows = []
for idx in ("06", "09", "11"):
    for label, p in (
        ("原始合成 base", RAW / f"{idx}.wav"),
        ("我的听包(3way)", next(iter(sorted(LISTEN.glob(f"{idx}_*_3way.wav"))), None)),
    ):
        if p is None or not Path(p).exists():
            continue
        d = analyze(Path(p), f"{idx} {label}")
        rows.append(d)
        if "err" in d:
            print(f"  {d['label']:<34s} {d['err']}")
            continue
        print(f"  {d['label']:<34s} {d['dur']:5.2f}s {d['tail_rms']:9.5f} "
              f"{d.get('tail_body_db', float('nan')):+8.2f} "
              f"{d['tail_flat']:8.4f} {d.get('flat_ratio', float('nan')):7.3f} "
              f"{d['tail_hnr']:7.3f} {d.get('hnr_drop', float('nan')):+7.3f}")
    print()

# ---------- 2) 参考音频 vs 真实录音 ----------
print("2) 参考音频 / 真实录音 对照")
print("-" * 100)
print(f"  {'来源':<34s} {'时长':>6s} {'尾部RMS':>9s} {'尾/体dB':>8s} "
      f"{'尾平坦':>8s} {'平坦比':>7s} {'尾HNR':>7s} {'HNR降':>7s}")
print("  " + "-" * 96)

if REF.exists():
    d = analyze(REF, "ref/default.wav（参考音频）")
    rows.append(d)
    if "err" not in d:
        print(f"  {d['label']:<34s} {d['dur']:5.2f}s {d['tail_rms']:9.5f} "
              f"{d.get('tail_body_db', float('nan')):+8.2f} "
              f"{d['tail_flat']:8.4f} {d.get('flat_ratio', float('nan')):7.3f} "
              f"{d['tail_hnr']:7.3f} {d.get('hnr_drop', float('nan')):+7.3f}")

if HELD.exists():
    hd = json.loads(HELD.read_text(encoding="utf-8"))
    tg = hd.get("targets", [])[:3]
    for t in tg:
        p = Path(t["path"])
        if not p.exists():
            continue
        d = analyze(p, f"真实录音 {p.stem[:20]}")
        rows.append(d)
        if "err" not in d:
            print(f"  {d['label']:<34s} {d['dur']:5.2f}s {d['tail_rms']:9.5f} "
                  f"{d.get('tail_body_db', float('nan')):+8.2f} "
                  f"{d['tail_flat']:8.4f} {d.get('flat_ratio', float('nan')):7.3f} "
                  f"{d['tail_hnr']:7.3f} {d.get('hnr_drop', float('nan')):+7.3f}")

# ---------- 3) 汇总 ----------
print()
print("=" * 100)
print("3) 汇总（按来源分组）")
print("=" * 100)
grp: dict[str, list] = {}
for d in rows:
    if "err" in d:
        continue
    key = d["label"].split(" ", 1)[-1]
    if "原始合成" in key:
        key = "原始合成 base"
    elif "听包" in key:
        key = "我的听包(3way)"
    elif "ref/" in key:
        key = "参考音频 ref/default.wav"
    elif "真实录音" in key:
        key = "真实录音（留出集）"
    grp.setdefault(key, []).append(d)

print(f"  {'来源':<28s} {'n':>3s} {'尾/体dB':>9s} {'尾平坦':>8s} "
      f"{'平坦比':>7s} {'尾HNR':>7s} {'HNR降':>7s}  判读")
print("  " + "-" * 92)
for k, v in grp.items():
    def m(f):
        xs = [x.get(f) for x in v if x.get(f) is not None]
        return float(np.nanmean(xs)) if xs else float("nan")
    tb, fl, fr, hn, hd_ = (m("tail_body_db"), m("tail_flat"),
                           m("flat_ratio"), m("tail_hnr"), m("hnr_drop"))
    # 判读：平坦比 >1.5 且 HNR 降 >0.2 -> 尾部明显更"气"
    verdict = ""
    if fr > 1.5 or hd_ > 0.2:
        verdict = "**尾部明显更气**"
    elif fr < 1.2 and hd_ < 0.1:
        verdict = "尾部与主体接近（不气）"
    print(f"  {k:<28s} {len(v):3d} {tb:+9.2f} {fl:8.4f} {fr:7.3f} "
          f"{hn:7.3f} {hd_:+7.3f}  {verdict}")

print()
print("=" * 100)
print("4) 判读方法")
print("=" * 100)
print("""
  ## 指标含义
    · **尾/体dB**   : 尾部 RMS 相对主体。负得多 = 尾部很轻
    · **尾平坦**    : 谱平坦度。**越高越像噪声（气声）**，越低越像纯音
    · **平坦比**    : 尾平坦 / 体平坦。**>1.5 说明尾部比主体更"气"**
    · **尾HNR**     : 谐噪比代理（自相关峰值）。**越低越气**
    · **HNR降**     : 主体HNR - 尾部HNR。**>0.2 说明尾部谐波性明显下降**

  ## 三种可能的结论
    ① 原始合成 ≈ 我的听包  -> **不是我引入的**，是模型特性
    ② 参考音频也很"气"      -> 模型**模仿了参考音频**的尾音
    ③ 真实录音不气          -> 模型在**生成时加了气声**（可能是训练数据的特性）

  ## 如果确认是模型特性，能怎么办
    · 换参考音频（挑一个尾音干净的）
    · 用**去气声**的后处理（轻量高通搁架的反向：2~8kHz 轻微衰减）
    · 或在参考音频的尾音处做淡出
""")
print("=" * 100)

out = V5 / "dist" / "tail_breath_report.txt"
lines = ["=" * 78, "尾音气声排查", "=" * 78, ""]
for d in rows:
    if "err" in d:
        continue
    lines.append(f"{d['label']}")
    lines.append(f"  时长 {d['dur']:.2f}s  有声结束 {d['voiced_end']:.2f}s  "
                 f"静音尾 {d['sil_tail']:.2f}s")
    lines.append(f"  尾部: RMS={d['tail_rms']:.5f} 平坦={d['tail_flat']:.4f} "
                 f"HF={d['tail_hf']:.4f} HNR={d['tail_hnr']:.3f}")
    lines.append(f"  主体: RMS={d['body_rms']:.5f} 平坦={d['body_flat']:.4f} "
                 f"HF={d['body_hf']:.4f} HNR={d['body_hnr']:.3f}")
    lines.append(f"  尾/体 {d.get('tail_body_db', float('nan')):+.2f} dB   "
                 f"平坦比 {d.get('flat_ratio', float('nan')):.3f}   "
                 f"HNR降 {d.get('hnr_drop', float('nan')):+.3f}")
    lines.append("")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"  -> {out}")
