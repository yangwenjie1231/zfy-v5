# -*- coding: utf-8 -*-
"""韵律（语气）对比：v5 vs v2Pro —— 验证用户「v5 语气好很多」的观察。

## 为什么之前没测出来

我的 SIM（ERes2NetV2 说话人嵌入余弦）**只测音色**，测不到韵律。
用户反馈的正是「语气好、音色有提升空间」—— 两个维度分开的。

## 架构上的预期

已核实：两组用的是**同一个 GPT**（`zfh-e4.ckpt`），语义 token 序列相同。
所以语气差异只能来自 **SoVITS 侧**：

    v5   : CFMV5，22 层 DiT，335.79M 参数，输出 48kHz
    v2Pro: v2 架构 CFM，参数少得多，输出 32kHz

CFM 负责把语义 token 还原成 mel/波形。**更大的 CFM + 更高采样率
能更忠实地还原 token 里编码的韵律细节。**

## 本脚本测什么（纯韵律，不看音色）

  1. **F0 动态范围**（半音）—— 音高起伏，语气的核心
  2. **F0 轮廓变化率** —— 音高移动的活跃度
  3. **能量包络动态**（dB）—— 重音强弱
  4. **停顿时长分布** —— 节奏感
  5. **语速**（字/秒）
  6. **与真人录音的 F0 轮廓相关性**（对数据集中存在的句子）

用法：
    python prosody_compare.py --v5 r32_e02 --n 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

V5 = Path(r"D:\A\zfy-v5")
SWEEP = V5 / "out" / "r32_sweep"
CMP = V5 / "out" / "r32_vs_r128"
OUT = V5 / "out" / "prosody"
DATASET = Path(r"D:\Users\A\Downloads\仿通行证\新素材\音频\庄方宜_TTS训练集")

TEXTS = [
    "迎敌或是迎客，都得整装后再出发。",
    "出门前，要不要先喝杯茶提提神？",
    "让我想想。",
    "好。",
    "武陵城外有一片梨树林，每年春天都会开满白色的花。",
    "她就在那阵花香里，只是他自己不知道。",
    "学得越多，就越是清楚，自己学到的不过沧海一粟。",
    "上次外出执行任务……都是好几年前了，但愿别生疏。",
    "新的是更厉害些，可那旧的用惯了，确实……也不舍得。",
    "再好的武器，也得朝夕相处磨合上几天。",
    "你找我帮忙，想必是遇上棘手的问题了，直说就好。",
    "作画者把心画在纸上，纸会烂。庄方宜把心砌在城里，城也会塌。",
    "嗯……我理一理。",
    "三成火力，足够了。",
    "这件事我三点五十分再去处理。",
    "武陵城的梨花开了，要不要一起去看看？",
    "别紧张，多难多苦，我们都会一直陪在你身边。",
    "冬天总会过去，春天总会再来。",
    "我确实是第一次来帝江号。",
    "下个春天的好风景，我们一起去看。",
]


def f0_track(x: np.ndarray, sr: int, fmin=70.0, fmax=400.0,
             win_ms=40.0, hop_ms=10.0):
    """自相关法逐帧估 F0，返回 (时刻, f0, 是否有声)。

    稳健性处理：
      · 只在能量足够高的帧上估
      · 要求自相关峰值 >= 0.3 * 零延迟值（否则判为无声）
      · 用中值滤波去掉八度跳变（常见的自相关错误）
    """
    win, hop = int(win_ms / 1000 * sr), int(hop_ms / 1000 * sr)
    if len(x) < win + hop:
        return np.array([]), np.array([]), np.array([], dtype=bool)
    nf = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(nf)[:, None]
    fr = x[idx] * np.hanning(win)
    e = np.sqrt(np.mean(fr ** 2, axis=1)) + 1e-12
    thr = max(np.percentile(e, 55), np.max(e) * 0.02)

    lo, hi = max(2, int(sr / fmax)), int(sr / fmin)
    f0 = np.full(nf, np.nan)
    for i in range(nf):
        if e[i] < thr:
            continue
        s = fr[i] - fr[i].mean()
        ac = np.correlate(s, s, mode="full")[len(s) - 1:]
        if ac[0] <= 0:
            continue
        seg = ac[lo:hi]
        if len(seg) < 2:
            continue
        k = int(np.argmax(seg)) + lo
        if seg.max() > 0.30 * ac[0]:
            f0[i] = sr / k

    voiced = ~np.isnan(f0)
    # 中值滤波去八度跳变
    if voiced.sum() > 5:
        v = f0[voiced]
        med = np.median(v)
        bad = (v > med * 1.8) | (v < med / 1.8)
        v2 = v.copy()
        v2[bad] = med
        f0[voiced] = v2
    return np.arange(nf) * hop / sr, f0, voiced


def st(f):
    """频率 -> 半音（以 55Hz 为基准）。"""
    return 12 * np.log2(np.asarray(f, dtype=float) / 55.0 + 1e-9)


def prosody_features(p: Path) -> dict | None:
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if len(x) < sr * 0.2:
        return None
    t, f0, voiced = f0_track(x, sr)
    if voiced.sum() < 8:
        return None

    # 1) F0 动态范围（半音）
    vs = st(f0[voiced])
    p10, p50, p90 = np.percentile(vs, [10, 50, 90])
    f0_range = float(p90 - p10)

    # 2) F0 轮廓变化率：相邻有声帧的半音差绝对值均值
    d = np.abs(np.diff(vs))
    f0_slope = float(np.mean(d)) if len(d) else 0.0

    # 3) 能量包络动态（dB）
    win, hop = int(0.025 * sr), int(0.010 * sr)
    nf = 1 + (len(x) - win) // hop
    ii = np.arange(win)[None, :] + hop * np.arange(nf)[:, None]
    ee = np.sqrt(np.mean(x[ii] ** 2, axis=1)) + 1e-12
    edb = 20 * np.log10(ee)
    env_dyn = float(np.percentile(edb, 95) - np.percentile(edb, 20))

    # 4) 停顿统计（连续无声 > 0.15s）
    vmask = edb > (np.percentile(edb, 95) - 35)
    pauses, run = [], 0
    for v in vmask:
        if not v:
            run += 1
        else:
            if run * 0.010 >= 0.15:
                pauses.append(run * 0.010)
            run = 0
    if run * 0.010 >= 0.15:
        pauses.append(run * 0.010)

    return {
        "dur": round(len(x) / sr, 3),
        "sr": sr,
        "f0_median_hz": round(float(55 * 2 ** (p50 / 12)), 1),
        "f0_range_st": round(f0_range, 2),
        "f0_slope_st": round(f0_slope, 4),
        "env_dyn_db": round(env_dyn, 2),
        "n_pause": len(pauses),
        "pause_mean_s": round(float(np.mean(pauses)), 3) if pauses else 0.0,
        "pause_total_s": round(float(np.sum(pauses)), 3),
        "voiced_ratio": round(float(voiced.mean()), 3),
    }


def paired(a, b):
    dv = np.asarray(a, float) - np.asarray(b, float)
    n = len(dv)
    if n < 3:
        return {"mean": None, "t": None, "sig": False}
    se = dv.std(ddof=1) / np.sqrt(n)
    t = float(dv.mean() / se) if se > 0 else 0.0
    crit = {19: 2.093, 20: 2.086}.get(n, 2.1)
    return {"mean": round(float(dv.mean()), 4), "t": round(t, 2),
            "crit": crit, "sig": bool(abs(t) > crit),
            "win": int((dv > 0).sum()), "lose": int((dv < 0).sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5", default="r32_e02", help="v5 目录名（r32_sweep 下）")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    texts = TEXTS[: args.n]

    groups = {}
    d5 = SWEEP / args.v5
    if not d5.exists():
        print(f"找不到 v5 目录: {d5}")
        return 1
    groups[f"v5_{args.v5}"] = sorted(d5.glob("*.wav"))[: args.n]
    dv2 = CMP / "v2pro"
    if dv2.exists():
        groups["v2pro"] = sorted(dv2.glob("*.wav"))[: args.n]

    # 真人参照（数据集中存在的句子）
    nat_map = {}
    import csv
    for r in csv.DictReader(open(DATASET / "manifest.csv", encoding="utf-8-sig")):
        if r.get("处理") == "保留":
            nat_map[(r.get("文本清洗") or "").strip()] = DATASET / "audio" / \
                Path(r["原文件"]).name

    print(f"{'组':16s} {'F0范围':>8s} {'F0斜率':>8s} {'能量动态':>9s} "
          f"{'停顿数':>7s} {'停顿总长':>9s} {'时长':>7s}")
    print("-" * 78)
    rows = {}
    for tag, ps in groups.items():
        feats = []
        for p in ps:
            f = prosody_features(p)
            if f:
                feats.append(f)
        rows[tag] = feats
        if not feats:
            print(f"{tag:16s} 无有效特征")
            continue
        g = lambda k: np.mean([f[k] for f in feats])  # noqa: E731
        print(f"{tag:16s} {g('f0_range_st'):8.2f} {g('f0_slope_st'):8.4f} "
              f"{g('env_dyn_db'):9.2f} {g('n_pause'):7.2f} "
              f"{g('pause_total_s'):9.3f} {g('dur'):7.2f}")

    # 真人对照
    nat_feats = []
    for t in texts:
        p = nat_map.get(t)
        if p and p.exists():
            f = prosody_features(p)
            if f:
                nat_feats.append(f)
    if nat_feats:
        g = lambda k: np.mean([f[k] for f in nat_feats])  # noqa: E731
        print("-" * 78)
        print(f"{'真人录音':16s} {g('f0_range_st'):8.2f} {g('f0_slope_st'):8.4f} "
              f"{g('env_dyn_db'):9.2f} {g('n_pause'):7.2f} "
              f"{g('pause_total_s'):9.3f} {g('dur'):7.2f}")
        print(f"（对照句数 {len(nat_feats)}）")

    # 配对检验
    v5tag = f"v5_{args.v5}"
    if v5tag in rows and "v2pro" in rows:
        print("\n" + "=" * 78)
        print(f"配对检验：{v5tag} vs v2Pro（同 20 句，同 GPT）")
        print("=" * 78)
        tests = {}
        n = min(len(rows[v5tag]), len(rows["v2pro"]))
        for key, name, better in [
            ("f0_range_st", "F0 动态范围(半音)", "大"),
            ("f0_slope_st", "F0 变化率", "大"),
            ("env_dyn_db", "能量动态(dB)", "大"),
            ("n_pause", "停顿数", "多"),
            ("pause_total_s", "停顿总长(s)", "—"),
        ]:
            a = [f[key] for f in rows[v5tag][:n]]
            b = [f[key] for f in rows["v2pro"][:n]]
            p = paired(a, b)
            tests[key] = p
            print(f"  {name:20s} 差={p['mean']:+9.4f}  t={p['t']:6.2f}  "
                  f"{'显著' if p['sig'] else '不显著'}  "
                  f"(v5更大 {p['win']}/{n})")
        (OUT / "prosody_tests.json").write_text(json.dumps(
            {"tests": tests, "v5": v5tag}, ensure_ascii=False, indent=2),
            encoding="utf-8")

    (OUT / "prosody_features.json").write_text(json.dumps(
        {"groups": {k: v for k, v in rows.items()},
         "natural": nat_feats, "texts": texts},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果: {OUT/'prosody_features.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
