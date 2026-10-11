# -*- coding: utf-8 -*-
"""评估 r128 重训结果：逐 epoch 扫 + 与 v2Pro / 旧 rank32 对比。

## 这一轮必须避开的坑（前两轮的教训）

1. **参考音频必须与 ref_text 对齐**
   上一轮用了数据集里错配的 ref.wav（前缀一致率 0%，语速 6.94 字/秒），
   导致两个模型都在"补完"参考文本，字数比 1.26/1.42（正常≈1.0）。
   本脚本**默认用生产参考** `zfh-models-v2pro/ref/default.wav`（已验证一致）。

2. **必须用同一个 GPT**
   生产 v2Pro 用微调过的 `zfh-e4.ckpt`，不是底模 `s1v3.ckpt`。
   上一轮对比时误用了 s1v3，不公平。本脚本统一用 zfh-e4。

3. **CER 样本要够**
   12 句对 CER 不够（双峰分布）。本脚本默认 20 句。

4. **SIM 的 epoch 间波动**
   上一轮 rank32 的 SIM 极差 0.0192，与模型间差距同量级 →
   单看 SIM 无法可靠区分 epoch。要**逐 epoch 都保留**，靠听感定。

用法：
    python eval_retrain.py --rank 128                 # 扫 r128 全部 epoch
    python eval_retrain.py --rank 128 --epochs 8,12,16,20,24
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

V5 = Path(r"D:\A\zfy-v5")
GSV = V5 / "GPT-SoVITS"
OLD = Path(r"D:\A\voiceclone\GPT-SoVITS")
OUT = V5 / "out" / "retrain_eval"

GOOD_REF = Path(r"D:\A\voiceclone\zfh-models-v2pro\ref\default.wav")
GOOD_REF_TEXT = "再好的武器，也得朝夕相处磨合上几天。等用惯了，我再和你说说感受，好吗？"
ZFH_GPT = r"D:\A\voiceclone\zfh-models-v2pro\torch_weights\zfh-e4.ckpt"
V2PRO_SOV = r"D:\A\voiceclone\zfh-models-v2pro\torch_weights\zfh_v2pro_e12_s384.pth"


def _ep_of(p: Path):
    """从权重文件名提取 epoch 号；提不到返回 None。

    兼容两类命名：
      v5 LoRA : zfy_v5_e8_s432_l32.pth
      全量微调: v5_full_e32_s1728_l0.pth
    """
    m = re.search(r"_e(\d+)_s\d+", p.name)
    if m:
        return int(m.group(1))
    m = re.search(r"_e(\d+)_", p.name)
    return int(m.group(1)) if m else None


# 20 句：覆盖短/中/长 + 省略号/问句/数字
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
SEED = 42


def use_v5():
    need = [str(GSV), str(GSV / "GPT_SoVITS"),
            str(GSV / "GPT_SoVITS" / "BigVGAN"),
            str(GSV / "tools"), str(GSV / "tools" / "asr"),
            str(GSV / "tools" / "uvr5")]
    for p in reversed(need):
        while p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    os.chdir(str(GSV))
    for m in list(sys.modules):
        if m.split(".")[0] in ("GPT_SoVITS", "module", "text", "tools",
                               "sv", "f5_tts", "TTS_infer_pack"):
            del sys.modules[m]


def use_old():
    need = [str(OLD), str(OLD / "GPT_SoVITS"),
            str(OLD / "GPT_SoVITS" / "BigVGAN"),
            str(OLD / "tools"), str(OLD / "tools" / "asr"),
            str(OLD / "tools" / "uvr5")]
    for p in reversed(need):
        while p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    os.chdir(str(OLD))
    for m in list(sys.modules):
        if m.split(".")[0] in ("GPT_SoVITS", "module", "text", "tools",
                               "sv", "f5_tts", "TTS_infer_pack"):
            del sys.modules[m]


def synth_v5(weights: Path, tag: str, texts, steps=32, cfg=1.30):
    use_v5()
    import torch
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
    d = OUT / tag
    d.mkdir(parents=True, exist_ok=True)
    cf = {"custom": {
        "device": "cuda", "is_half": True, "version": "v5dev",
        "t2s_weights_path": ZFH_GPT, "vits_weights_path": str(weights),
        "bert_base_path": str(GSV / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large"),
        "cnhuhbert_base_path": str(GSV / "GPT_SoVITS" / "pretrained_models" / "chinese-hubert-base"),
    }}
    tts = TTS(TTS_Config(cf))
    for i, t in enumerate(texts):
        p = d / f"{i:02d}.wav"
        if p.exists():
            continue
        ch, sr = [], None
        for it in tts.run({
            "text": t, "text_lang": "zh", "ref_audio_path": str(GOOD_REF),
            "prompt_text": GOOD_REF_TEXT, "prompt_lang": "zh",
            "top_k": 15, "temperature": 1.0, "repetition_penalty": 1.35,
            "text_split_method": "cut1", "batch_size": 1, "speed_factor": 1.0,
            "seed": SEED, "sample_steps": steps, "cfg_rate": cfg,
            "parallel_infer": True,
        }):
            if it is None:
                continue
            sr, a = it
            ch.append(np.asarray(a))
        sf.write(str(p), np.concatenate(ch), sr)
    del tts
    torch.cuda.empty_cache()


def synth_v2pro(tag: str, texts):
    use_old()
    import torch
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
    d = OUT / tag
    d.mkdir(parents=True, exist_ok=True)
    cf = {"custom": {
        "device": "cuda", "is_half": False, "version": "v2Pro",
        "t2s_weights_path": ZFH_GPT, "vits_weights_path": V2PRO_SOV,
        "bert_base_path": str(OLD / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large"),
        "cnhuhbert_base_path": str(OLD / "GPT_SoVITS" / "pretrained_models" / "chinese-hubert-base"),
    }}
    tts = TTS(TTS_Config(cf))
    for i, t in enumerate(texts):
        p = d / f"{i:02d}.wav"
        if p.exists():
            continue
        ch, sr = [], None
        for it in tts.run({
            "text": t, "text_lang": "zh", "ref_audio_path": str(GOOD_REF),
            "prompt_text": GOOD_REF_TEXT, "prompt_lang": "zh",
            "top_k": 15, "temperature": 1.0, "repetition_penalty": 1.35,
            "text_split_method": "cut1", "batch_size": 1, "speed_factor": 1.0,
            "seed": SEED, "parallel_infer": True,
        }):
            if it is None:
                continue
            sr, a = it
            ch.append(np.asarray(a))
        sf.write(str(p), np.concatenate(ch), sr)
    del tts
    torch.cuda.empty_cache()


def norm(s):
    s = re.sub(r"<\|[^|]*\|>", "", s or "")
    return re.sub(r"[^\u4e00-\u9fff0-9a-z]", "", s.lower())


def cer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    if not r:
        return 0.0
    dp = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(h) + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1,
                        prev + (0 if r[i - 1] == h[j - 1] else 1))
            prev = cur
    return dp[len(h)] / len(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--epochs", default=None, help="逗号分隔；默认全部")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--cfg", type=float, default=1.30)
    # ---- 新增：直接指定权重（用于评全量微调的 e8/e32 等）----
    ap.add_argument("--weights", default=None,
                    help="逗号分隔的 .pth 路径（覆盖 --rank 的自动发现）")
    ap.add_argument("--weights-dir", default=None,
                    help="权重目录（取其中所有 *.pth）")
    ap.add_argument("--tag-prefix", default=None,
                    help="输出子目录前缀；默认 r{rank} 或 full")
    ap.add_argument("--out", default=None,
                    help="输出根目录；默认 out/retrain_eval。不同实验请给不同目录，避免混算")
    ap.add_argument("--skip-v2pro", action="store_true",
                    help="不合成 v2Pro 基准（只在已生成过时用）")
    args = ap.parse_args()

    global OUT
    if args.out:
        OUT = Path(args.out)
    texts = TEXTS[: args.n]
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------- 选权重 ----------
    tag_prefix = args.tag_prefix
    if args.weights:
        all_ck = [Path(x.strip()) for x in args.weights.split(",") if x.strip()]
        tag_prefix = tag_prefix or "full"
    elif args.weights_dir:
        wdir = Path(args.weights_dir)
        all_ck = sorted(wdir.glob("*.pth"),
                        key=lambda p: _ep_of(p) if _ep_of(p) is not None else 0)
        tag_prefix = tag_prefix or "full"
    else:
        wdir = GSV / f"SoVITS_weights_v5_r{args.rank}"
        all_ck = sorted(wdir.glob("*.pth"),
                        key=lambda p: int(re.search(r"_e(\d+)_", p.name).group(1)))
        tag_prefix = tag_prefix or f"r{args.rank}"

    if args.epochs and not (args.weights or args.weights_dir):
        want = {int(x) for x in args.epochs.split(",")}
        all_ck = [p for p in all_ck
                  if int(re.search(r"_e(\d+)_", p.name).group(1)) in want]

    missing = [p for p in all_ck if not p.exists()]
    all_ck = [p for p in all_ck if p.exists()]
    for p in missing:
        print(f"  !! 权重不存在: {p}")
    if not all_ck:
        print("没有任何可用权重")
        return 1

    print(f"评估 {len(all_ck)} 个检查点（{len(texts)} 句，"
          f"steps={args.steps}, cfg={args.cfg}）")
    print(f"输出根目录: {OUT}")
    for p in all_ck:
        print(f"  {p.name}")

    for ck in all_ck:
        ep = _ep_of(ck)
        tag = f"{tag_prefix}_e{ep:02d}" if ep is not None else f"{tag_prefix}_{ck.stem}"
        print(f"\n合成 {tag} …", flush=True)
        synth_v5(ck, tag, texts, args.steps, args.cfg)

    if not args.skip_v2pro:
        print("\n合成 v2Pro（生产基准）…", flush=True)
        synth_v2pro("v2pro", texts)
    else:
        print("\n(跳过 v2Pro 合成)")

    # 评测
    use_v5()
    import torch
    from funasr import AutoModel
    import librosa
    from sv import SV

    asr = AutoModel(model="iic/SenseVoiceSmall", device="cuda:0",
                    disable_update=True, cache_dir=r"D:\A\voiceclone\models")
    sv = SV("cuda", False)

    def emb(w):
        t = torch.from_numpy(np.ascontiguousarray(w)).unsqueeze(0).to("cuda")
        with torch.no_grad():
            e = sv.compute_embedding3(t)
        return e.detach().cpu().flatten().numpy().astype(np.float64)

    ref_emb = emb(librosa.load(str(GOOD_REF), sr=32000, mono=True)[0])

    def sim(p):
        b = emb(librosa.load(str(p), sr=32000, mono=True)[0])
        a = ref_emb
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    rows = []
    # 只评本次生成的目录（tag_prefix 开头 或 v2pro），避免把历史结果混进来
    wanted = {d.name for d in OUT.iterdir()
              if d.is_dir() and (d.name.startswith(tag_prefix) or d.name == "v2pro")}
    for d in sorted(OUT.iterdir()):
        if not d.is_dir() or d.name not in wanted:
            continue
        ps = sorted(d.glob("*.wav"))
        if len(ps) < len(texts):
            print(f"跳过 {d.name}（只有 {len(ps)}/{len(texts)} 个）")
            continue
        cs, ss, ds, lr = [], [], [], []
        for i, p in enumerate(ps[:len(texts)]):
            x, sr = sf.read(str(p), dtype="float32", always_2d=True)
            x = x.mean(axis=1)
            ds.append(len(x) / sr)
            w16 = librosa.load(str(p), sr=16000, mono=True)[0]
            r = asr.generate(input=w16, cache={}, language="zh", use_itn=False,
                             batch_size_s=60, merge_vad=True, merge_length_s=15)
            hyp = r[0]["text"] if r else ""
            cs.append(cer(texts[i], hyp))
            lr.append(len(norm(hyp)) / max(len(norm(texts[i])), 1))
            ss.append(sim(p))
        rows.append({"tag": d.name, "n": len(cs),
                     "cer": round(float(np.mean(cs)), 4),
                     "sim": round(float(np.mean(ss)), 4),
                     "dur": round(float(np.mean(ds)), 3),
                     "len_ratio": round(float(np.mean(lr)), 3),
                     "_cer": cs, "_sim": ss})

    print("\n" + "=" * 82)
    print(f"{'检查点':16s} {'CER':>8s} {'SIM':>9s} {'时长':>8s} {'字数比':>8s}")
    print("-" * 82)
    for r in rows:
        print(f"{r['tag']:16s} {r['cer']*100:7.1f}% {r['sim']:9.4f} "
              f"{r['dur']:7.2f}s {r['len_ratio']:8.2f}")

    ref = next((r for r in rows if r["tag"] == "v2pro"), None)
    if ref:
        print("\n与 v2Pro 的配对检验：")
        for r in rows:
            if r["tag"] == "v2pro":
                continue
            ds_ = np.array(r["_sim"]) - np.array(ref["_sim"])
            dc_ = np.array(ref["_cer"]) - np.array(r["_cer"])
            n = len(ds_)
            ts = ds_.mean() / (ds_.std(ddof=1) / np.sqrt(n)) if ds_.std(ddof=1) > 0 else 0
            tc = dc_.mean() / (dc_.std(ddof=1) / np.sqrt(n)) if dc_.std(ddof=1) > 0 else 0
            crit = 2.093 if n >= 20 else 2.201
            print(f"  {r['tag']:16s} SIM差={ds_.mean():+.4f} (t={ts:5.2f}) "
                  f"CER差={dc_.mean():+.4f} (t={tc:5.2f}) 临界={crit}")

    (OUT / "eval.json").write_text(json.dumps(
        {"rows": [{k: v for k, v in r.items() if not k.startswith("_")}
                  for r in rows],
         "texts": texts, "tag_prefix": tag_prefix,
         "steps": args.steps, "cfg": args.cfg,
         "weights": [str(p) for p in all_ck]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果: {OUT/'eval.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
