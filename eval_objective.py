# -*- coding: utf-8 -*-
"""客观评估：说话人相似度 + 可懂度 + 物理指标，**全部用留出集**。

## 为什么需要这个
用户反馈："各个模型出来的效果，我人耳已经分辨不出来了"
-> 人耳 A/B 已无分辨力，必须换**客观指标**。

而技能里有三条铁律，本脚本逐条落实：

### 铁律 1：相似度目标必须用**训练集外**素材
    训练集内素材当目标 = 0.91（虚高，只是背下了训练录音）
    训练集外同角色素材 = 0.67（真实水平）
  -> 用 `pick_heldout_clean.py` 筛出的 131 条（131 条里挑 35 条 3~8s 的）

### 铁律 2：必须先测「天然上限」
    真实录音 vs 另一段真实录音（同角色、不同录音）= 0.66
  -> 若模型输出已经 0.67，说明**这个维度到顶了**，别再折腾。
     本脚本会同时算这个上限。

### 铁律 3：CER 必须先校准基线 + 需要大样本
    CER 是双峰分布（≈0% 或 ≈25%），单句方差极大
    要分辨 10pp 差异，至少 30~50 句
  -> 本脚本用 20 句（够看趋势），并**同时跑真实录音的 CER 作基线**

## 对比对象
    v2Pro（生产，32kHz）        vs  v5merged（全量微调+ssl_proj修复，48kHz）
    + 真实录音（留出集）作为上限参照

## 输出
    D:\\A\\zfy-v5\\dist\\eval\\report.json
    D:\\A\\zfy-v5\\dist\\eval\\report.txt   （人可读）
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

sys.path[:0] = [str(GSV), str(GSV / "GPT_SoVITS"), str(GSV / "tools")]
os.chdir(str(GSV))

REF = GSV / "ref" / "default.wav"
REF_TXT = (GSV / "ref" / "ref.txt").read_text(encoding="utf-8").strip()
PRE = GSV / "GPT_SoVITS" / "pretrained_models"

# 生产权重
PROD_GPT = GSV / "ref" / "zfh-e4.ckpt"
PROD_SOV = GSV / "ref" / "v2pro_sovits.pth"
# v5 全量微调（已补 ssl_proj）
V5_SOV = V5 / "dist" / "sweep2" / "out_sweep" / "e16_merged" / ".." / ".."
V5_WEIGHTS_DIR = V5 / "cloud_weights"      # 若本地有

# 20 句测试文本（不重复使用）
TESTS = [
    "让我想想。",
    "好。",
    "迎敌或是迎客，都得整装后再出发。",
    "出门前，要不要先喝杯茶提提神？",
    "嗯……我理一理。",
    "再好的武器，也得朝夕相处磨合上几天。等用惯了，我再和你说说感受，好吗？",
    "武陵城外有一片梨树林，每年春天都会开满白色的花。",
    "她就在那阵花香里，只是他自己不知道。",
    "作画者把心画在纸上，纸会烂。庄方宜把心砌在城里，城也会塌。",
    "这件事我三点五十分再去处理。",
    "管理员，今天的实验数据我已经整理好了。",
    "清波城那边的情况，我一直在关注。",
    "若是要谈，就该拿出诚意来。",
    "我答应你，会好好照顾自己。",
    "这些话，我本想留到以后再说的。",
    "你来了。坐吧，茶还热着。",
    "有些事，不是努力就能改变的。",
    "但不去试，就永远不知道结果。",
    "汤汤说，我是个念旧的人。",
    "或许吧。毕竟这座城，承载了太多。",
]
SEED = 42


def load_mono_16k(path) -> torch.Tensor:
    """读成 16kHz 单声道 torch 张量（SV 模型要求）。"""
    import torchaudio
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.squeeze(0)


def asr_np(path) -> np.ndarray:
    """读成 16kHz float32 numpy（**绕过 faster-whisper 的 av 解码器**）。

    为什么不用 asr.transcribe(路径)：
      faster-whisper 1.2.1 用 `av.open(..., metadata_errors="ignore")`，
      而当前 PyAV 版本的 `open()` 不接受这个参数 ->
          TypeError: open() got an unexpected keyword argument 'metadata_errors'
      这是 faster-whisper 与 PyAV 的版本不匹配。
      直接把 numpy 数组传进去就绕开了整条解码路径。
    """
    x, sr = sf.read(str(path), always_2d=True, dtype="float32")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if sr != 16000:
        try:
            from math import gcd

            from scipy.signal import resample_poly
            g = gcd(int(sr), 16000)
            x = resample_poly(x, 16000 // g, int(sr) // g).astype(np.float32)
        except Exception:  # noqa: BLE001
            n = int(round(len(x) * 16000 / sr))
            x = np.interp(np.linspace(0, len(x) - 1, n),
                          np.arange(len(x)), x).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


def cer_zh(ref: str, hyp: str) -> float:
    """字符级 CER（先做标点/空白归一化）。

    ⚠️ 必须做繁简归一化 —— whisper 系 ASR 常输出繁体，
       否则会把 `風雷動` vs `风雷动` 判成全错（技能里的真事故）。
    """
    try:
        from opencc import OpenCC
        hyp = OpenCC("t2s").convert(hyp)
    except Exception:  # noqa: BLE001
        pass
    strip = lambda s: re.sub(r"[，。？！、…,.?!\s\-—·「」『』《》()（）]", "", s)
    r, h = strip(ref), strip(hyp)
    if not r:
        return 1.0
    import difflib
    sm = difflib.SequenceMatcher(None, r, h)
    edits = sum(max(i2 - i1, j2 - j1)
                for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")
    return edits / len(r)


def spectral(x: np.ndarray, sr: int):
    """共同带宽内的谱指标（**必须固定带宽**，技能里的真事故）。"""
    n, hop = 2048, 512
    hi = min(16000.0, sr / 2 - 100)
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    fr = np.lib.stride_tricks.sliding_window_view(x, n)[::hop]
    mag = np.abs(np.fft.rfft(fr * np.hanning(n), axis=1)).mean(axis=0)
    f = np.fft.rfftfreq(n, 1 / sr)
    sel = f < hi
    m, ff = mag[sel], f[sel]
    tot = m.sum() + 1e-12
    centroid = float((ff * m).sum() / tot)
    e4 = m[(ff >= 4000)].sum()
    return centroid, float(e4 / tot)


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 92)
    print("客观评估（留出集 + 天然上限 + 物理指标）")
    print("=" * 92)
    print(f"  device: {device}")
    print(f"  参考音频: {REF}")
    print(f"  参考文本: {REF_TXT[:44]}…")

    data = json.loads(HELD.read_text(encoding="utf-8"))
    targets = data.get("targets", [])
    if not targets:
        print("  !! 没有留出集目标，先跑 pick_heldout_clean.py")
        return 1
    print(f"  留出集目标: {len(targets)} 条（3~8s, >=32kHz）")

    from GPT_SoVITS.sv import SV
    sv = SV(device, is_half=False)

    def emb16(path_or_tensor):
        w = (load_mono_16k(path_or_tensor)
             if not isinstance(path_or_tensor, torch.Tensor)
             else path_or_tensor)
        if w.shape[0] > 160000:
            w = w[:160000]
        e = sv.compute_embedding3(w.to(device).unsqueeze(0).float())
        return F.normalize(e.float(), dim=-1)

    # ---------- 目标音色（留出集，训练集外）----------
    print()
    print("--- 构建留出集目标音色 ---")
    tgt_embs = []
    for t in targets[:24]:          # 24 条足够
        try:
            tgt_embs.append(emb16(t["path"]))
        except Exception as e:  # noqa: BLE001
            print(f"    跳过 {Path(t['path']).name}: {type(e).__name__}")
    tgt = torch.cat(tgt_embs, dim=0)
    print(f"  目标段: {tgt.shape[0]}")

    # ---------- 天然上限：真实录音 vs 留出集 ----------
    print()
    print("--- 天然上限：真实录音（留出集自身）vs 留出集目标 ---")
    ceiling = []
    for t in targets[:24]:
        try:
            ceiling.append(emb16(t["path"]))
        except Exception:  # noqa: BLE001
            pass
    if ceiling:
        c = torch.cat(ceiling, dim=0)
        # 排除自身（对角线）
        sim = torch.einsum("id,jd->ij", c, tgt)
        n = min(c.shape[0], tgt.shape[0])
        off = sim.clone()
        for i in range(n):
            off[i, i] = -1
        ceiling_val = off.max(dim=1).values.mean().item()
    else:
        ceiling_val = float("nan")
    print(f"  **天然上限 = {ceiling_val:.4f}**")
    print(f"  （真实录音 vs 另一段真实录音，同角色不同录音）")

    # ---------- TTS ----------
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

    def build_tts(version, gpt, sov):
        cfg = {"custom": {
            "device": device, "is_half": False, "version": version,
            "t2s_weights_path": str(gpt), "vits_weights_path": str(sov),
            "bert_base_path": str(PRE / "chinese-roberta-wwm-ext-large"),
            "cnhuhbert_base_path": str(PRE / "chinese-hubert-base"),
        }}
        return TTS(TTS_Config(cfg))

    # ASR
    print()
    print("--- 加载 ASR ---")
    asr = None
    try:
        from faster_whisper import WhisperModel
        asr = WhisperModel("small", device=device, compute_type="float16"
                           if device == "cuda" else "int8")
        print("  faster-whisper small 就绪")
    except Exception as e:  # noqa: BLE001
        print(f"  ASR 不可用（{type(e).__name__}）-> 跳过 CER")

    # ---------- 评估真实录音的 CER（基线校准）----------
    baseline_cer = float("nan")
    if asr is not None:
        print()
        print("--- CER 基线：真实录音（留出集）自身的 ASR 结果 ---")
        # 留出集没有文本，所以这里只测"能不能转出非空文本"
        # 真正的基线用参考音频的已知文本
        segs, _ = asr.transcribe(asr_np(REF), language="zh", beam_size=5,
                                 vad_filter=False)
        hyp = "".join(s.text for s in segs)
        baseline_cer = cer_zh(REF_TXT, hyp)
        print(f"  参考音频 CER = {baseline_cer:.3f}")
        print(f"    ref: {REF_TXT[:40]}")
        print(f"    hyp: {hyp[:40]}")

    # ---------- 逐模型评估 ----------
    CANDIDATES = []
    if PROD_GPT.exists() and PROD_SOV.exists():
        CANDIDATES.append(("v2Pro(生产)", "v2Pro", PROD_GPT, PROD_SOV))
    else:
        print(f"  !! 缺生产权重: gpt={PROD_GPT.exists()} sov={PROD_SOV.exists()}")

    # v5 全量权重（若已下载到本地）
    v5s = sorted(V5.glob("**/SoVITS_weights_v5_full_e32/*.pth"))
    v5g = sorted(V5.glob("**/GPT_weights_gpt*/*.ckpt"))
    if v5s:
        print(f"  找到 v5 SoVITS 权重: {len(v5s)} 个")
    else:
        print("  v5 全量权重不在本地（在云端 23.5GB）-> 只评估生产版")
        print("     要用 v5 权重请先下载，或用已导出的 wav 做离线对比")

    results = []
    for label, version, gpt, sov in CANDIDATES:
        print()
        print("=" * 92)
        print(f"评估: {label}")
        print("=" * 92)
        try:
            tts = build_tts(version, gpt, sov)
        except Exception as e:  # noqa: BLE001
            print(f"  加载失败: {type(e).__name__}: {e}")
            continue

        sims, cers, cents, hfrs, durs, n_chars = [], [], [], [], [], []
        for i, text in enumerate(TESTS):
            try:
                gen = tts.run({
                    "text": text, "text_lang": "zh",
                    "ref_audio_path": str(REF), "prompt_text": REF_TXT,
                    "prompt_lang": "zh", "top_k": 15, "temperature": 1.0,
                    "repetition_penalty": 1.35, "text_split_method": "cut1",
                    "batch_size": 1, "speed_factor": 1.0, "seed": SEED,
                    "parallel_infer": True,
                })
                chunks, sr = [], None
                for item in gen:
                    if item is None:
                        continue
                    sr, a = item
                    chunks.append(a)
                if not chunks:
                    continue
                audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
            except Exception as e:  # noqa: BLE001
                print(f"  {i+1:2d} 合成失败 {type(e).__name__}: {str(e)[:60]}")
                continue

            tmp = OUT / f"_tmp_{label.replace('(','_').replace(')','')}_{i:02d}.wav"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(tmp), audio, sr)

            # 相似度
            try:
                e = emb16(tmp)
                sim = torch.einsum("id,jd->ij", e, tgt).max().item()
                sims.append(sim)
            except Exception as ex:  # noqa: BLE001
                print(f"  {i+1:2d} 相似度失败 {type(ex).__name__}")

            # 物理指标
            x = np.asarray(audio, dtype=np.float64)
            if x.ndim > 1:
                x = x.mean(axis=1)
            c, h = spectral(x, sr)
            cents.append(c)
            hfrs.append(h)
            n_chars.append(len(re.sub(r"[，。？！、…,.?!\s]", "", text)))
            durs.append(len(x) / sr)
            if i < 3 or i == len(TESTS) - 1:
                print(f"  {i+1:2d} sim={sims[-1]:.4f}  {len(x)/sr:5.2f}s  "
                      f"质心={c:5.0f}Hz  高频={h:.3f}")

            # CER
            if asr is not None:
                try:
                    segs, _ = asr.transcribe(asr_np(tmp), language="zh",
                                             beam_size=5, vad_filter=False)
                    hyp = "".join(s.text for s in segs)
                    cers.append(cer_zh(text, hyp))
                except Exception:  # noqa: BLE001
                    pass

        if not sims:
            print("  无有效样本")
            continue
        # ⚠️ 语速的分母/分子必须**一一对应**（只算成功合成的那些句子）
        n_ok = len(durs)
        row = {
            "label": label, "version": version,
            "sim_mean": float(np.mean(sims)),
            "sim_std": float(np.std(sims)),
            "sim_min": float(np.min(sims)),
            "sim_max": float(np.max(sims)),
            "cer_mean": float(np.mean(cers)) if cers else None,
            "cer_zero_rate": float(np.mean([c < 0.01 for c in cers])) if cers else None,
            "centroid": float(np.mean(cents)),
            "hf_ratio": float(np.mean(hfrs)),
            "speed": float(sum(n_chars[:n_ok]) / sum(durs)) if durs else 0.0,
            "n": len(sims),
            "ceiling": ceiling_val,
        }
        results.append(row)
        print()
        print(f"  **{label} 汇总**")
        print(f"    泛化相似度 : {row['sim_mean']:.4f} ± {row['sim_std']:.4f}  "
              f"(min {row['sim_min']:.4f} / max {row['sim_max']:.4f})")
        print(f"    天然上限   : {ceiling_val:.4f}   "
              f"-> 达成率 {row['sim_mean']/ceiling_val*100:.1f}%")
        if row["cer_mean"] is not None:
            print(f"    念错率 CER : {row['cer_mean']:.3f}  "
                  f"（完全正确率 {row['cer_zero_rate']*100:.0f}%）")
        print(f"    谱质心     : {row['centroid']:.0f} Hz")
        print(f"    高频占比   : {row['hf_ratio']:.3f}")
        print(f"    语速       : {row['speed']:.2f} 字/秒  (正常 3~5)")

    # ---------- 报告 ----------
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "ceiling": ceiling_val,
        "baseline_cer_ref": baseline_cer,
        "n_targets": len(targets),
        "results": results,
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["=" * 76, "客观评估报告", "=" * 76, ""]
    lines.append(f"留出集目标 : {len(targets)} 条（训练集外，同角色）")
    lines.append(f"天然上限   : {ceiling_val:.4f}  （真实录音 vs 真实录音）")
    lines.append(f"参考音频CER: {baseline_cer:.3f}  （ASR 基线）")
    lines.append("")
    lines.append(f"{'模型':<16}{'相似度':>9}{'±':>7}{'达成率':>9}"
                 f"{'CER':>8}{'质心':>8}{'高频':>8}{'语速':>7}")
    lines.append("-" * 76)
    for r in results:
        cer = f"{r['cer_mean']:.3f}" if r["cer_mean"] is not None else "  n/a"
        lines.append(
            f"{r['label']:<16}{r['sim_mean']:>9.4f}{r['sim_std']:>7.4f}"
            f"{r['sim_mean']/ceiling_val*100:>8.1f}%{cer:>8}"
            f"{r['centroid']:>8.0f}{r['hf_ratio']:>8.3f}{r['speed']:>7.2f}")
    lines.append("")
    lines.append("判读：")
    lines.append("  · 相似度接近天然上限 -> 该维度已触顶，不必再折腾")
    lines.append("  · 达成率 <80%       -> 还有提升空间")
    lines.append("  · 语速应在 3~5 字/秒，超出说明韵律异常")
    (OUT / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print("\n".join(lines))
    print(f"\n  -> {OUT / 'report.json'}")
    print(f"  -> {OUT / 'report.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
