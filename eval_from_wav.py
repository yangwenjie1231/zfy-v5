# -*- coding: utf-8 -*-
"""客观评估（离线版）—— 直接评估**已有的 wav**，不需要跑 TTS。

## 为什么需要离线版
`eval_objective.py` 要加载权重跑 TTS，但 v5 全量权重（23.5 GB）在云端，
本地没有。而**已经有两批合成好的 wav**：

    dist/baseline_v2pro/NN_*_v2pro.wav            10 个  (v2Pro 生产版)
    dist/sweep2/out_sweep/e16_merged/NN_*_e16_merged.wav  10 个 (v5 全量+ssl_proj修复)

这两批是**同一批文本、同一参考音频、同一套推理参数**生成的，
所以可以直接对比 —— 而且**省掉合成时间**。

## 落实技能里的三条铁律

### 铁律 1：相似度目标必须用**训练集外**素材
`pick_heldout_clean.py` 筛出 131 条干净留出集，其中 24 条 3~8s 的做目标。
（排查中已排除两个"假留出集"：`wav24k`、`语音合集_分条` 都是训练素材的别名）

### 铁律 2：必须先测「天然上限」
真实录音 vs 另一段真实录音（同角色不同录音）。
若模型已达上限，该维度**没有提升空间**，不必再折腾。

### 铁律 3：频谱指标必须**固定带宽**
v2Pro 是 32kHz（Nyquist 16k），v5 是 48kHz（Nyquist 24k）。
直接比全频带会得到虚假差异 —— 技能里的真事故：
「合成输出比原声频谱平坦度高 12 倍」→ 实际同频带内只有 1.17 倍。
**本脚本把所有音频统一到 16kHz 重采样后再算谱指标。**

## 输出
    D:\\A\\zfy-v5\\dist\\eval\\offline_report.json
    D:\\A\\zfy-v5\\dist\\eval\\offline_report.txt
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

V5 = Path(r"D:\A\zfy-v5")
GSV = V5 / "GPT-SoVITS"
OUT = V5 / "dist" / "eval"
HELD = V5 / "dist" / "heldout" / "heldout_clean.json"

# 待评估的两批 wav
BATCHES = {
    "v2Pro(生产)": V5 / "dist" / "baseline_v2pro",
    "v5merged(全量+ssl_proj)": V5 / "dist" / "sweep2" / "out_sweep" / "e16_merged",
}
# 文本（用于算语速）
TEXTS = {
    "01": "让我想想。", "02": "好。",
    "03": "迎敌或是迎客，都得整装后再出发。",
    "04": "出门前，要不要先喝杯茶提提神？",
    "05": "嗯……我理一理。",
    "06": "再好的武器，也得朝夕相处磨合上几天。等用惯了，我再和你说说感受，好吗？",
    "07": "武陵城外有一片梨树林，每年春天都会开满白色的花。",
    "08": "她就在那阵花香里，只是他自己不知道。",
    "09": "作画者把心画在纸上，纸会烂。庄方宜把心砌在城里，城也会塌。",
    "10": "这件事我三点五十分再去处理。",
}
# 共同带宽（v2Pro Nyquist = 16kHz）
COMMON_SR = 16000
COMMON_HZ = 7600.0


def to_common(x: np.ndarray, sr: int) -> np.ndarray:
    """统一重采样到 16kHz（**多相 FIR**，不是线性插值）。

    ⚠️ 线性插值的频响是 sinc²(f/sr_out)，16kHz 处 -3.3 dB。
       上一轮我用它做盲听包，导致 v2Pro 被削高频、v5 没有 ->
       用户听出"前段糊"，差点误判成模型差异。
    """
    if sr == COMMON_SR:
        return x
    try:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(int(sr), COMMON_SR)
        return resample_poly(x, COMMON_SR // g, int(sr) // g).astype(np.float64)
    except Exception:  # noqa: BLE001
        n = int(round(len(x) * COMMON_SR / sr))
        return np.interp(np.linspace(0, len(x) - 1, n),
                         np.arange(len(x)), x).astype(np.float64)


def load_mono(p: Path):
    x, sr = sf.read(str(p), always_2d=True, dtype="float64")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    return x, int(sr)


def spectral16(x: np.ndarray):
    """在 16kHz 域算谱指标（**固定带宽**）。"""
    n, hop = 2048, 512
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    fr = np.lib.stride_tricks.sliding_window_view(x, n)[::hop]
    mag = np.abs(np.fft.rfft(fr * np.hanning(n), axis=1)).mean(axis=0)
    f = np.fft.rfftfreq(n, 1 / COMMON_SR)
    sel = f < COMMON_HZ
    m, ff = mag[sel], f[sel]
    tot = m.sum() + 1e-12
    centroid = float((ff * m).sum() / tot)
    # 4~7.6k / 0~4k —— "清亮度/气音"代理
    e_hi = m[ff >= 4000].sum()
    e_lo = m[ff < 4000].sum()
    return centroid, float(e_hi / (e_lo + 1e-12))


def tail_rms(x: np.ndarray, sr: int, sec: float = 0.4) -> float:
    """尾音 RMS —— **先裁掉尾部静音再量**。

    ⚠️ 为什么必须裁（实测踩到的坑）：
      直接用 `x[-0.4s:]` 的 RMS 会被**末尾静音填充长度**污染。
      实测两批输出的静音尾长度不同：
          v2Pro   0.34s 静音尾
          v5merged 0.23s 静音尾
      于是同一指标在不同窗长下给出完全不同的比值：
          窗长 0.1s -> 两批都是 0.00000（纯静音），比值 0.0x
          窗长 0.4s -> 0.00124 vs 0.02888，比值 **23.4x**  <- 假象
          窗长 0.8s -> 0.05414 vs 0.06298，比值 1.2x
          窗长 1.5s -> 0.06265 vs 0.06845，比值 1.1x
      **比值随窗长剧烈变化 = 测的是"末尾有多少静音"，不是"尾音有没有力气"。**

    正确做法：从尾部往前找到最后一个"有声"帧，从那里往前取 sec 秒。
    """
    n_frame = 256
    if len(x) < n_frame * 4:
        return float("nan")
    m = (len(x) // n_frame) * n_frame
    fr = x[:m].reshape(-1, n_frame)
    rms_fr = np.sqrt((fr ** 2).mean(axis=1))
    ref = float(np.sqrt((x ** 2).mean())) + 1e-12
    thr = ref * (10 ** (-45.0 / 20))          # -45 dB 相对整体 RMS
    voiced = np.where(rms_fr > thr)[0]
    if len(voiced) == 0:
        return float("nan")
    end = int((voiced[-1] + 1) * n_frame)     # 最后一个有声帧的结束位置
    n = int(sec * sr)
    seg = x[max(0, end - n):end]
    if len(seg) < n // 2:
        return float("nan")
    return float(np.sqrt((seg ** 2).mean()))


def silence_tail(x: np.ndarray, sr: int, thresh_db: float = -45.0) -> float:
    """从尾部往前数，连续低于阈值的时长（秒）。"""
    n_frame = 256
    if len(x) < n_frame * 2:
        return float("nan")
    m = (len(x) // n_frame) * n_frame
    fr = x[:m].reshape(-1, n_frame)
    rms_fr = np.sqrt((fr ** 2).mean(axis=1))
    ref = float(np.sqrt((x ** 2).mean())) + 1e-12
    thr = ref * (10 ** (thresh_db / 20))
    cnt = 0
    for v in rms_fr[::-1]:
        if v < thr:
            cnt += 1
        else:
            break
    return cnt * n_frame / sr


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 96)
    print("客观评估（离线 wav 版）")
    print("=" * 96)
    print(f"  device: {device}")

    data = json.loads(HELD.read_text(encoding="utf-8"))
    targets = data.get("targets", [])
    print(f"  留出集目标: {len(targets)} 条（训练集外，同角色）")

    # ---------- 载入 SV ----------
    sys.path[:0] = [str(GSV), str(GSV / "GPT_SoVITS"), str(GSV / "tools")]
    os.chdir(str(GSV))
    from GPT_SoVITS.sv import SV  # noqa: E402

    sv = SV(device, is_half=False)

    def emb(p: Path):
        import torchaudio
        w, sr = torchaudio.load(str(p))
        if w.shape[0] > 1:
            w = w.mean(0, keepdim=True)
        if sr != 16000:
            w = torchaudio.functional.resample(w, sr, 16000)
        w = w.squeeze(0)
        if w.shape[0] > 160000:
            w = w[:160000]
        e = sv.compute_embedding3(w.to(device).unsqueeze(0).float())
        return F.normalize(e.float(), dim=-1)

    # ---------- 目标音色 + 天然上限 ----------
    print()
    print("--- 留出集目标音色 ---")
    tgt_list = []
    for t in targets[:24]:
        try:
            tgt_list.append(emb(Path(t["path"])))
        except Exception as e:  # noqa: BLE001
            print(f"    跳过 {Path(t['path']).name}: {type(e).__name__}")
    tgt = torch.cat(tgt_list, dim=0)
    print(f"  目标段: {tgt.shape[0]}")

    # 天然上限：留出集自己 vs 目标（排除对角线）
    sim = torch.einsum("id,jd->ij", tgt, tgt)
    n = tgt.shape[0]
    off = sim.clone()
    for i in range(n):
        off[i, i] = -1
    ceiling = off.max(dim=1).values.mean().item()
    print(f"  **天然上限 = {ceiling:.4f}**  （真实录音 vs 另一段真实录音）")

    # ---------- 逐批评估 ----------
    results = []
    for label, d in BATCHES.items():
        print()
        print("=" * 96)
        print(f"评估: {label}")
        print(f"  {d}")
        print("=" * 96)
        if not d.is_dir():
            print("  目录不存在，跳过")
            continue
        wavs = sorted(d.glob("*.wav"))
        if not wavs:
            print("  无 wav，跳过")
            continue

        sims, cents, hfs, tails, sils, speeds, durs, rows = [], [], [], [], [], [], [], []
        for w in wavs:
            m = re.match(r"(\d+)_", w.stem)
            if not m:
                continue
            key = m.group(1)
            text = TEXTS.get(key)
            if text is None:
                continue
            try:
                x, sr = load_mono(w)
                if len(x) < sr * 0.2:
                    continue
                dur = len(x) / sr
                xc = to_common(x, sr)
                c, h = spectral16(xc)
                # ⚠️ tail_rms 必须在**原始采样率**上算（避免重采样边界效应），
                #    且函数内部会先裁掉尾部静音再量
                tr = tail_rms(x, sr)
                sil = silence_tail(x, sr)
                n_ch = len(re.sub(r"[，。？！、…,.?!\s]", "", text))
                sp = n_ch / dur

                # 相似度（用 16k 版本）
                tmp = OUT / f"_tmp_{abs(hash(label))%9999}_{w.stem}.wav"
                tmp.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(tmp), xc.astype(np.float32), COMMON_SR)
                e = emb(tmp)
                s = torch.einsum("id,jd->ij", e, tgt).max().item()
                tmp.unlink(missing_ok=True)

                sims.append(s)
                cents.append(c)
                hfs.append(h)
                tails.append(tr)
                sils.append(sil)
                speeds.append(sp)
                durs.append(dur)
                rows.append((key, s, dur, sp, c, h, tr))
            except Exception as ex:  # noqa: BLE001
                print(f"  {w.name}: {type(ex).__name__}: {str(ex)[:60]}")

        if not sims:
            print("  无有效样本")
            continue

        print(f"  {'#':4s} {'相似度':>8s} {'时长':>7s} {'语速':>6s} "
              f"{'质心':>7s} {'高频':>7s} {'尾音':>7s}")
        for k, s, du, sp, c, h, tr in rows:
            print(f"  {k:4s} {s:8.4f} {du:6.2f}s {sp:6.2f} "
                  f"{c:7.0f} {h:7.4f} {tr:7.5f}")

        row = {
            "label": label,
            "n": len(sims),
            "sim_mean": float(np.mean(sims)),
            "sim_std": float(np.std(sims)),
            "sim_min": float(np.min(sims)),
            "sim_max": float(np.max(sims)),
            "reach": float(np.mean(sims) / ceiling * 100),
            "centroid": float(np.mean(cents)),
            "hf_ratio": float(np.mean(hfs)),
            "tail_rms": float(np.nanmean(tails)),
            "silence_tail": float(np.nanmean(sils)),
            "speed": float(np.mean(speeds)),
            "dur_total": float(np.sum(durs)),
        }
        results.append(row)
        print()
        print(f"  **{label} 汇总**")
        print(f"    泛化相似度 : {row['sim_mean']:.4f} ± {row['sim_std']:.4f}  "
              f"(min {row['sim_min']:.4f} / max {row['sim_max']:.4f})")
        print(f"    达成率     : {row['reach']:.1f}%  (相对天然上限 {ceiling:.4f})")
        print(f"    谱质心     : {row['centroid']:.0f} Hz")
        print(f"    高频比     : {row['hf_ratio']:.4f}")
        print(f"    尾音 RMS   : {row['tail_rms']:.5f}")
        print(f"    平均语速   : {row['speed']:.2f} 字/秒  (正常 3~5)")

    # ---------- 对比 ----------
    print()
    print("=" * 96)
    print("A/B 对比")
    print("=" * 96)
    if len(results) >= 2:
        a, b = results[0], results[1]
        print(f"  {'指标':<14}{a['label']:>22}{b['label']:>26}{'差异':>12}")
        print("  " + "-" * 74)
        for k, name, fmt in (
            ("sim_mean", "泛化相似度", "{:.4f}"),
            ("reach", "达成率%", "{:.1f}"),
            ("centroid", "谱质心Hz", "{:.0f}"),
            ("hf_ratio", "高频比", "{:.4f}"),
            ("tail_rms", "尾音RMS", "{:.5f}"),
            ("speed", "语速字/秒", "{:.2f}"),
            ("dur_total", "总时长s", "{:.2f}"),
        ):
            va, vb = a[k], b[k]
            diff = vb - va
            mark = ""
            if k == "sim_mean":
                mark = "  <- 越高越好"
            elif k == "speed":
                mark = "  <- 应在 3~5"
            print(f"  {name:<14}{fmt.format(va):>22}{fmt.format(vb):>26}"
                  f"{diff:>+12.4f}{mark}")
        print()
        d_sim = b["sim_mean"] - a["sim_mean"]
        print(f"  相似度差异: {d_sim:+.4f}")
        if abs(d_sim) < 0.01:
            print("    -> 差异 <0.01，**在噪声范围内**，两者听感等价")
            print("       （这与你说的『人耳分辨不出来』一致）")
        elif d_sim > 0:
            print(f"    -> v5merged 更好 {d_sim/ceiling*100:+.1f}pp 达成率")
        else:
            print(f"    -> v2Pro 更好 {abs(d_sim)/ceiling*100:.1f}pp 达成率")
        print()
        print(f"  天然上限 {ceiling:.4f}：")
        for r in results:
            gap = ceiling - r["sim_mean"]
            print(f"    {r['label']:<26} 距上限 {gap:+.4f}  "
                  f"({r['reach']:.1f}%)")
            if gap < 0.02:
                print(f"      -> **已触顶**，该维度没有提升空间了")

    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"ceiling": ceiling, "n_targets": len(targets), "results": results}
    (OUT / "offline_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["=" * 78, "客观评估（离线 wav 版）", "=" * 78, ""]
    lines.append(f"留出集目标 : {len(targets)} 条（训练集外，同角色）")
    lines.append(f"天然上限   : {ceiling:.4f}  （真实录音 vs 真实录音）")
    lines.append(f"共同带宽   : 0~{COMMON_HZ/1000:.1f} kHz @ {COMMON_SR} Hz")
    lines.append("")
    lines.append(f"{'模型':<26}{'相似度':>10}{'±':>8}{'达成率':>9}"
                 f"{'质心':>8}{'高频':>9}{'语速':>7}")
    lines.append("-" * 78)
    for r in results:
        lines.append(
            f"{r['label']:<26}{r['sim_mean']:>10.4f}{r['sim_std']:>8.4f}"
            f"{r['reach']:>8.1f}%{r['centroid']:>8.0f}"
            f"{r['hf_ratio']:>9.4f}{r['speed']:>7.2f}")
    lines.append("")
    lines.append("判读：")
    lines.append("  · 相似度接近天然上限 -> 该维度已触顶，不必再折腾")
    lines.append("  · 达成率 <80%       -> 还有提升空间")
    lines.append("  · 语速应在 3~5 字/秒")
    (OUT / "offline_report.txt").write_text("\n".join(lines) + "\n",
                                            encoding="utf-8")
    print()
    print("\n".join(lines))
    print(f"\n  -> {OUT / 'offline_report.json'}")
    print(f"  -> {OUT / 'offline_report.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
