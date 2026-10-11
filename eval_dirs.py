# -*- coding: utf-8 -*-
"""通用相似度对比 —— 对任意目录测「留出集相似度」。

## 为什么需要
`eval_from_wav.py` 的目录是**硬编码**的（`baseline_v2pro` / `e16_merged`），
无法测新的目录（`ref_swap_ab/*`、`gpt_ab/*` 等）。

## 指标
    相似度 = 合成音频的说话人嵌入 · 留出集真实录音的说话人嵌入
    天然上限 = 留出集内部互比（真实 vs 真实）= **0.8874**

**达成率 = 相似度 / 0.8874** —— 这是跨批次可比的归一化指标。

## 留出集
`dist/heldout/heldout_clean.json`（131 条干净留出，训练集外）
其中 `targets` 35 条作为目标音色。

## 用法
    python eval_dirs.py --dirs dist/ref_swap_ab/orig dist/ref_swap_ab/cand3
    python eval_dirs.py --dirs dist/gpt_ab/base dist/gpt_ab/A_e4 --tag gpt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

V5 = Path(r"D:\A\zfy-v5")
OUT = V5 / "dist" / "eval_dirs"
HELD = V5 / "dist" / "heldout" / "heldout_clean.json"

ap = argparse.ArgumentParser()
ap.add_argument("--dirs", nargs="+", required=True)
ap.add_argument("--tag", default="cmp")
ap.add_argument("--ceiling", type=float, default=0.8874,
                help="天然上限（默认 0.8874，来自 eval_from_wav.py）")
args = ap.parse_args()

import torch  # noqa: E402
import torchaudio  # noqa: E402
import soundfile as sf  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SR = 16000
# 用与 eval_from_wav.py 相同的模型（保证可比）
MODEL_DIR = V5 / "GPT-SoVITS" / "GPT_SoVITS" / "pretrained_models" / \
    "sv" / "pretrained_eres2netv2w24s4ep4.ckpt"

print("=" * 100)
print("通用相似度对比")
print("=" * 100)
print(f"  device : {DEV}")
print(f"  模型   : {MODEL_DIR.name}")
print(f"  上限   : {args.ceiling}")

if not HELD.exists():
    print(f"  !! 找不到留出集 {HELD}")
    sys.exit(1)
hd = json.loads(HELD.read_text(encoding="utf-8"))
targets = [Path(t["path"]) for t in hd.get("targets", []) if Path(t["path"]).exists()]
print(f"  留出集目标: {len(targets)} 条")
if not targets:
    print("  !! 留出集目标为空")
    sys.exit(1)


def load16(p: Path) -> np.ndarray:
    x, sr = sf.read(str(p), always_2d=True, dtype="float32")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if sr != SR:
        try:
            from math import gcd

            from scipy.signal import resample_poly
            g = gcd(int(sr), SR)
            x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
        except Exception:  # noqa: BLE001
            n = int(round(len(x) * SR / sr))
            x = np.interp(np.linspace(0, len(x) - 1, n),
                          np.arange(len(x)), x).astype(np.float32)
    return x


# 加载说话人模型
# ⚠️ `eval_from_wav.py:191-193` 用的是 `SV(device, is_half=False)` ——
#    **不是** `SV(ckpt_path, device=...)`。我第一版写错了签名。
sys.path[:0] = [str(V5 / "GPT-SoVITS"), str(V5 / "GPT-SoVITS" / "GPT_SoVITS")]
os.chdir(str(V5 / "GPT-SoVITS"))
try:
    from GPT_SoVITS.sv import SV  # noqa: E402
    sv = SV(DEV, is_half=False)
    print("  ✓ 说话人模型加载成功")
except Exception as e1:  # noqa: BLE001
    print(f"  !! 无法加载说话人模型: {type(e1).__name__}: {str(e1)[:160]}")
    sys.exit(1)


def embed(p: Path) -> np.ndarray | None:
    """**逐字复刻** `eval_from_wav.py:195-206` 的 `emb()`，保证可比。

    ⚠️ 我前两版都写错了：
        v1: `SV(ckpt_path, device=...)`  —— 签名错（是 `SV(device, is_half)`）
        v2: `sv.compute_embedding(t, SR)` —— 方法名错
        实际: `sv.compute_embedding3(w_16k_1d.unsqueeze(0).float())`
    参数细节：
        · torchaudio 加载 -> 单声道 -> 重采样 16k -> 截断 160000 样本(10s)
        · 返回后 F.normalize
    """
    import torch.nn.functional as F
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


# 目标嵌入（torch tensor, [1, D]，已 L2 归一化）
tg = []
for p in targets:
    e = embed(p)
    if e is not None:
        tg.append(e.reshape(-1))
print(f"  有效目标嵌入: {len(tg)}")
if not tg:
    print("  !! 目标嵌入全失败")
    sys.exit(1)
T = torch.stack(tg)          # [N, D]


def sim_to_targets(e) -> float:
    v = e.reshape(-1)
    return float((T @ v).max().item())


# 上限（留出集内部）
ceiling_vals = []
for i in range(len(T)):
    mask = torch.ones(len(T), dtype=torch.bool)
    mask[i] = False
    others = T[mask]
    ceiling_vals.append(float((others @ T[i]).max().item()))
ceiling = float(np.mean(ceiling_vals))
print(f"  **实测天然上限 = {ceiling:.4f}**（留出集内部互比）")

print()
print("=" * 100)
print("逐目录评估")
print("=" * 100)
print(f"  {'目录':<34s} {'句数':>4s} {'相似度':>9s} {'达成率':>8s} "
      f"{'时长':>7s} {'语速':>6s}")
print("  " + "-" * 78)

results = {}
for ds in args.dirs:
    d = (V5 / ds) if not Path(ds).is_absolute() else Path(ds)
    if not d.is_dir():
        print(f"  {ds:<34s} 目录不存在")
        continue
    wavs = sorted(d.glob("*.wav"))
    vals, durs, sps = [], [], []
    for w in wavs:
        e = embed(w)
        if e is None:
            continue
        vals.append(sim_to_targets(e))
        try:
            x, sr = sf.read(str(w), always_2d=True, dtype="float64")
            x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
            durs.append(len(x) / sr)
        except Exception:  # noqa: BLE001
            pass
    if not vals:
        print(f"  {ds:<34s} 无有效样本")
        continue
    sim = float(np.mean(vals))
    results[ds] = {"n": len(vals), "sim": sim, "std": float(np.std(vals)),
                   "ratio": sim / ceiling,
                   "dur": float(np.mean(durs)) if durs else float("nan")}
    r = results[ds]
    print(f"  {ds:<34s} {r['n']:4d} {sim:9.4f} {r['ratio']*100:7.1f}% "
          f"{r['dur']:6.2f}s")

if len(results) > 1:
    print()
    print("=" * 100)
    print("相对第一个目录")
    print("=" * 100)
    keys = list(results)
    base = keys[0]
    b = results[base]
    print(f"  基准: {base}  ({b['sim']:.4f}, 达成率 {b['ratio']*100:.1f}%)")
    print()
    for k in keys[1:]:
        r = results[k]
        d = r["sim"] - b["sim"]
        dr = (r["ratio"] - b["ratio"]) * 100
        verdict = ("✓ 更好" if d > 0.01 else ("✗ 更差" if d < -0.01 else "~ 噪声内"))
        print(f"  {k:<34s} Δ相似度 {d:+.4f}  Δ达成率 {dr:+.1f}pp   {verdict}")
    print()
    print("  ⚠️ 阈值说明：")
    print("    · |Δ| < 0.01  -> 在噪声内（之前 v2Pro vs v5merged 差 0.0048 就属此类）")
    print("    · 只有 |Δ| > 0.02 才算真实差异")

print()
print("=" * 100)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / f"{args.tag}_report.json").write_text(json.dumps(
    {"ceiling_measured": ceiling, "ceiling_ref": args.ceiling,
     "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"  -> {OUT / f'{args.tag}_report.json'}")
print("=" * 100)
