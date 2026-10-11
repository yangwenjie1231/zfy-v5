# -*- coding: utf-8 -*-
"""GPT 三方对比 —— **只换 GPT，SoVITS 固定**。

## 实验设计（关键）
要回答"GPT 微调有没有用"，必须**固定其他所有变量**：

    SoVITS  : 固定用 v2Pro 生产权重（zfh_v2pro_e12_s384.pth）
    参考音频 : 固定 ref/default.wav
    推理参数 : 固定 top_k 15 / temp 1.0 / rep 1.35 / seed 42
    文本     : 固定 20 句（**不与之前实验复用**，避免重复计数）
    ─────────────────────────────────────────────
    唯一变量 : GPT 权重

## 三组
    base  : ref/zfh-e4.ckpt                    （生产 GPT，v2 底模微调）
    A     : GPT_weights_gptA/zfy_gptA-e{N}.ckpt （s1v3 底模微调）
    B     : GPT_weights_gptB/zfy_gptB-e{N}.ckpt （v2 底模微调，复刻生产路径）

## 为什么固定 v2Pro SoVITS
· 要测的是 **GPT 对语气/韵律的影响**
· SoVITS 管音色 —— 固定它才能隔离变量
· v2Pro 是**生产在用**的，有基准意义

## 看什么指标（GPT 影响韵律，不影响音色）
    · 语速（字/秒）—— 正常 3~5
    · 时长方差       —— GPT 变好 -> 同类句子时长更一致
    · 时长比值       —— 实际 ÷ (字数/4)，<0.6 = 提前终止
    · 停顿分布       —— 用静音段数与总静音时长近似

**不要期待相似度提升** —— 那是 SoVITS 的维度。

## 用法
    python compare_gpt.py                          # e8，全部 3 组
    python compare_gpt.py --epochs 2,4,8           # 多轮次
    python compare_gpt.py --only base,A            # 只跑两组
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")
GSV = V5 / "GPT-SoVITS"
OUT = V5 / "dist" / "gpt_ab"

# 固定项
SOVITS = GSV / "ref" / "v2pro_sovits.pth"
REF = GSV / "ref" / "default.wav"
REF_TXT_F = GSV / "ref" / "ref.txt"
PROD_GPT = GSV / "ref" / "zfh-e4.ckpt"
VERSION = "v2Pro"
SEED = 42

# 20 句测试文本（**新文本，不复用**）
TEXTS = [
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

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", default="8", help="逗号分隔，如 2,4,8")
ap.add_argument("--only", default="base,A,B")
ap.add_argument("--n", type=int, default=len(TEXTS))
args = ap.parse_args()
EPOCHS = [int(x) for x in args.epochs.split(",") if x.strip()]
ONLY = {x.strip() for x in args.only.split(",") if x.strip()}


def build_jobs() -> list[tuple[str, Path]]:
    """扫描所有 GPT_weights_* 目录，构建 (标签, 权重路径) 任务列表。

    ## 目录布局（实测存在的）
        GPT_weights_gptA_lr2e3       zfy_gptA-e{1..8}.ckpt   LR 2e-3（s1v3 底模）
        GPT_weights_gptB_lr2e3       zfy_gptB-e{1..8}.ckpt   LR 2e-3（v2 底模）
        GPT_weights_gptB_lr2e5       zfy_gptB-e{1..8}.ckpt   LR 2e-5（v2 底模）
        GPT_weights_gptB_lr2e3_e2    zfy_gptB-e{1..2}.ckpt   LR 2e-3 只训 2 轮
    旧脚本只认 `GPT_weights_gpt{A,B}`，所以新目录**全被漏掉**。

    ## 标签命名
        目录名后缀 `_lr2e5` / `_lr2e3_e2` -> 标签 `B_lr2e5_e8` / `B_lr2e3_e2`
    """
    jobs: list[tuple[str, Path]] = []
    if "base" in ONLY:
        jobs.append(("base", PROD_GPT))

    import re as _re
    for d in sorted(GSV.glob("GPT_weights_gpt*")):
        if not d.is_dir():
            continue
        name = d.name  # 形如 GPT_weights_gptB_lr2e5
        m = _re.match(r"GPT_weights_gpt([AB])(?:_(.+))?$", name)
        if not m:
            continue
        tag_letter, suffix = m.group(1), m.group(2) or "lr2e3"
        if tag_letter not in ONLY and f"{tag_letter}_all" not in ONLY:
            continue
        # 找出该目录里所有 epoch
        eps = []
        for p in d.glob("*.ckpt"):
            mm = _re.search(r"-e(\d+)\.ckpt$", p.name)
            if mm:
                eps.append((int(mm.group(1)), p))
        for e, p in sorted(eps):
            if e not in EPOCHS and EPOCHS != list(range(1, 9)):
                # EPOCHS 显式指定时才过滤；默认全取
                if e not in EPOCHS:
                    continue
            jobs.append((f"{tag_letter}_{suffix}_e{e}", p))
    return jobs


def silence_stats(x: np.ndarray, sr: int, thresh_db: float = -45.0):
    """静音段数与总静音时长（近似"停顿分布"）。"""
    nf = 256
    if len(x) < nf * 4:
        return 0, 0.0
    m = (len(x) // nf) * nf
    rmsf = np.sqrt((x[:m].reshape(-1, nf) ** 2).mean(axis=1))
    ref = float(np.sqrt((x ** 2).mean())) + 1e-12
    sil = rmsf < ref * (10 ** (thresh_db / 20))
    # 统计静音段数（连续的算一段）
    runs = 0
    prev = False
    for v in sil:
        if v and not prev:
            runs += 1
        prev = v
    return runs, float(sil.sum() * nf / sr)


def main() -> int:
    print("=" * 96)
    print("GPT 三方对比（**只换 GPT，SoVITS 固定 v2Pro**）")
    print("=" * 96)
    print(f"  SoVITS : {SOVITS.name}  ({SOVITS.stat().st_size/1e6:.1f} MB)")
    print(f"  参考   : {REF.name}")
    print(f"  版本   : {VERSION}")
    print(f"  文本   : {len(TEXTS[:args.n])} 句")
    print(f"  seed   : {SEED}")

    jobs = build_jobs()
    print()
    print(f"  任务（{len(jobs)} 个）:")
    for tag, p in jobs:
        print(f"    {tag:10s} {p.name}")

    if not jobs:
        print("  !! 无任务")
        return 1

    # 载入 TTS 骨架
    sys.path[:0] = [str(GSV), str(GSV / "GPT_SoVITS"), str(V5)]
    os.chdir(str(GSV))
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
    from GPT_SoVITS.process_ckpt import get_sovits_version_from_path_fast

    PRE = GSV / "GPT_SoVITS" / "pretrained_models"
    ref_txt = REF_TXT_F.read_text(encoding="utf-8").strip()
    print(f"\n  参考文本: {ref_txt[:44]}…")

    ver, mv, lora = get_sovits_version_from_path_fast(str(SOVITS))
    print(f"  SoVITS 版本探测: ver={ver} model_version={mv} if_lora_v3={lora}")

    # ---------- 逐任务合成 ----------
    OUT.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    for tag, gpt_path in jobs:
        print()
        print("=" * 96)
        print(f"生成: {tag}   GPT={gpt_path.name}")
        print("=" * 96)
        cfg = {"custom": {
            "device": "cuda", "is_half": False, "version": VERSION,
            "t2s_weights_path": str(gpt_path),
            "vits_weights_path": str(SOVITS),
            "bert_base_path": str(PRE / "chinese-roberta-wwm-ext-large"),
            "cnhuhbert_base_path": str(PRE / "chinese-hubert-base"),
        }}
        try:
            tts = TTS(TTS_Config(cfg))
        except Exception as e:  # noqa: BLE001
            print(f"  !! TTS 构造失败: {type(e).__name__}: {str(e)[:110]}")
            continue

        d = OUT / tag
        d.mkdir(parents=True, exist_ok=True)
        rows = []
        for i, text in enumerate(TEXTS[:args.n]):
            try:
                chunks, sr = [], None
                for item in tts.run({
                    "text": text, "text_lang": "zh",
                    "ref_audio_path": str(REF), "prompt_text": ref_txt,
                    "prompt_lang": "zh", "top_k": 15, "temperature": 1.0,
                    "repetition_penalty": 1.35, "text_split_method": "cut1",
                    "batch_size": 1, "speed_factor": 1.0, "seed": SEED,
                    "parallel_infer": True,
                }):
                    if item is None:
                        continue
                    sr, a = item
                    chunks.append(a)
                if not chunks:
                    print(f"  {i+1:2d} 无产出")
                    continue
                audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
                fp = d / f"{i+1:02d}.wav"
                sf.write(str(fp), audio, sr)

                x = np.asarray(audio, dtype=np.float64)
                if x.ndim > 1:
                    x = x.mean(axis=1)
                dur = len(x) / sr
                n_ch = len(re.sub(r"[，。？！、…,.?!\s]", "", text))
                runs, sil_t = silence_stats(x, sr)
                rows.append({
                    "i": i + 1, "dur": dur, "n_ch": n_ch,
                    "speed": n_ch / dur, "ratio": dur / (n_ch / 4.0),
                    "sil_runs": runs, "sil_sec": sil_t,
                })
                if i < 2 or i == args.n - 1:
                    print(f"  {i+1:2d} {dur:5.2f}s  {n_ch/dur:.2f}字/秒  "
                          f"停顿 {runs} 段/{sil_t:.2f}s")
            except Exception as e:  # noqa: BLE001
                print(f"  {i+1:2d} 失败 {type(e).__name__}: {str(e)[:70]}")

        if not rows:
            print("  无有效样本")
            continue
        durs = np.array([r["dur"] for r in rows])
        sps = np.array([r["speed"] for r in rows])
        results[tag] = {
            "n": len(rows),
            "dur_mean": float(durs.mean()),
            "dur_std": float(durs.std()),
            "dur_cv": float(durs.std() / durs.mean() * 100),
            "speed_mean": float(sps.mean()),
            "speed_std": float(sps.std()),
            "sil_runs": float(np.mean([r["sil_runs"] for r in rows])),
            "sil_sec": float(np.mean([r["sil_sec"] for r in rows])),
            "early_stop": int(sum(1 for r in rows if r["ratio"] < 0.6
                                  or r["speed"] > 7)),
            "rows": rows,
        }
        r = results[tag]
        print()
        print(f"  **{tag}**  {r['n']} 句")
        print(f"    平均时长 {r['dur_mean']:.2f}s ± {r['dur_std']:.2f}")
        print(f"    时长CV   {r['dur_cv']:.1f}%   （越低越稳定）")
        print(f"    平均语速 {r['speed_mean']:.2f} ± {r['speed_std']:.2f} 字/秒")
        print(f"    停顿     {r['sil_runs']:.1f} 段 / {r['sil_sec']:.2f}s")
        print(f"    提前终止 {r['early_stop']} 句")
        # 释放显存
        del tts
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    # ---------- 汇总 ----------
    print()
    print("=" * 96)
    print("汇总对比")
    print("=" * 96)
    keys = [k for k in ("base",) + tuple(
        f"{t}_e{e}" for e in EPOCHS for t in ("A", "B")) if k in results]
    if not keys:
        print("  无数据")
        return 1
    print(f"  {'组':10s} {'句数':>4s} {'平均时长':>9s} {'时长CV':>7s} "
          f"{'语速':>6s} {'语速σ':>6s} {'停顿段':>7s} {'静音s':>7s} {'早停':>5s}")
    print("  " + "-" * 80)
    for k in keys:
        r = results[k]
        print(f"  {k:10s} {r['n']:4d} {r['dur_mean']:8.2f}s {r['dur_cv']:6.1f}% "
              f"{r['speed_mean']:6.2f} {r['speed_std']:6.2f} "
              f"{r['sil_runs']:7.1f} {r['sil_sec']:7.2f} {r['early_stop']:5d}")

    # ---------- 与基线对比 ----------
    if "base" in results:
        b = results["base"]
        print()
        print("=" * 96)
        print("相对基线（base = 生产 zfh-e4.ckpt）的变化")
        print("=" * 96)
        print(f"  {'组':10s} {'时长':>9s} {'时长CV':>9s} {'语速':>9s} "
              f"{'语速σ':>9s} {'停顿段':>9s} {'早停':>7s}")
        print("  " + "-" * 74)
        for k in keys:
            if k == "base":
                continue
            r = results[k]
            dd = r["dur_mean"] - b["dur_mean"]
            dc = r["dur_cv"] - b["dur_cv"]
            ds = r["speed_mean"] - b["speed_mean"]
            dsd = r["speed_std"] - b["speed_std"]
            dr = r["sil_runs"] - b["sil_runs"]
            de = r["early_stop"] - b["early_stop"]
            print(f"  {k:10s} {dd:+8.2f}s {dc:+8.1f}% {ds:+8.2f} "
                  f"{dsd:+8.2f} {dr:+8.1f} {de:+7d}")
        print()
        print("  判读要点：")
        print("    · 时长CV **降低** = 同类句子更一致（GPT 变好）")
        print("    · 语速σ **降低** = 韵律更稳")
        print("    · 早停 **减少** = 更可靠")
        print("    · 语速超出 3~5 = 韵律异常")

    rep = {"epochs": EPOCHS, "n_texts": args.n, "results": {
        k: {kk: vv for kk, vv in v.items() if kk != "rows"}
        for k, v in results.items()}}
    (OUT / "gpt_ab_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"  -> {OUT / 'gpt_ab_report.json'}")
    print(f"  -> wav 在 {OUT}/<tag>/")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    sys.exit(main())
