# -*- coding: utf-8 -*-
"""v5 微调模型合成 + 与 v2Pro 生成盲听对比集。

为什么用 TTS_infer_pack 而不是命令行：
    `TTS_infer_pack.TTS.TTS` 是官方 API 的底层入口，能精确控制
    version / weights / sample_steps / cfg_rate / seed，且不依赖 webui。
    v5 的采样参数必须走 `v5_inference.resolve_sampling`：
      v5dev  -> 32 步, cfg 1.30
      v5turbo->  4 步, cfg 0.0
    这两个值不能乱给（validate_v5_sampling 只接受 4/8/16/32 步，cfg 0-2）。

对比设计（受控）：
    · 同一批句子
    · 同一参考音频（数据集自带的 ref.wav）
    · 同一 seed（42）
    · 亮度补偿不参与（v5 原生 48kHz，v2Pro 走插件才有补偿）
      —— 这里两边都取**模型原始输出**，避免引入插件侧差异

用法：
    python synth_v5.py --weights <v5权重.pth> --out <目录> [--tag v5]
    python synth_v5.py --compare     # 同时生成 v5 与 v2Pro 并拼盲听集
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

V5_ROOT = Path(r"D:\A\zfy-v5")
GSV = V5_ROOT / "GPT-SoVITS"
OUT = V5_ROOT / "out"
DATASET = Path(r"D:\Users\A\Downloads\仿通行证\新素材\音频\庄方宜_TTS训练集")

# 与 v2Pro 对比用同一批句子：覆盖短/中/长 + 标点/省略号/数字
TEXTS = [
    ("短", "让我想想。"),
    ("短", "好。"),
    ("中", "迎敌或是迎客，都得整装后再出发。"),
    ("中", "出门前，要不要先喝杯茶提提神？"),
    ("中", "嗯……我理一理。"),
    ("长", "再好的武器，也得朝夕相处磨合上几天。等用惯了，我再和你说说感受，好吗？"),
    ("长", "武陵城外有一片梨树林，每年春天都会开满白色的花。"),
    ("长", "她就在那阵花香里，只是他自己不知道。"),
    ("长", "作画者把心画在纸上，纸会烂。庄方宜把心砌在城里，城也会塌。"),
    ("数字", "这件事我三点五十分再去处理。"),
]

# ============================================================================
# ⚠️ 参考音频：必须用**生产验证过的** default.wav
#
# 数据集里的 `庄方宜_TTS训练集/ref_audio/ref.wav` 与它声称的 ref_text
# **错配**（实测：前缀匹配 0%、语速 6.94 字/秒 —— 正常 3~4.5）：
#     ref_text 声称："多亏你及时制止那些寨民袭击武陵城…我想回赠一些物资。"
#     实际说的是  ："那里常年监测着周边环境的深度毒数是抵抗秦世朝的第一线"
# 模型会去"补全"这个提示，导致字数和语调失控 —— 这是之前
# 「v5 语气差」的根因（详见 ROOTCAUSE_参考音频.md）。
#
# GPT 同理：生产 v2Pro 用**微调过的 zfh-e4.ckpt**，不是底模 s1v3.ckpt。
# 拿底模去比会不公平（语气是底模的语气）。
# ============================================================================
PROD_REF = Path(r"D:\A\voiceclone\zfh-models-v2pro\ref\default.wav")
PROD_REF_TEXT = "再好的武器，也得朝夕相处磨合上几天。等用惯了，我再和你说说感受，好吗？"
PROD_GPT = str(Path(r"D:\A\voiceclone\zfh-models-v2pro\torch_weights\zfh-e4.ckpt"))
S1V3_GPT = str(GSV / "GPT_SoVITS" / "pretrained_models" / "s1v3.ckpt")
BAD_REF = DATASET / "ref_audio" / "ref.wav"
BAD_REF_TEXT = "多亏你及时制止那些寨民袭击武陵城，为了感谢你们，我想回赠一些物资。"

# 默认 = 生产口径
REF = PROD_REF
REF_TEXT = PROD_REF_TEXT
T2S_GPT = PROD_GPT

SEED = 42


def build_tts(version: str, weights: str, is_half=True):
    """构造 TTS 实例。

    sys.path 布局必须复刻官方 `api_v2.py`（L114-115）+ `users.pth`：
      · `<GSV>`            —— 顶层包 GPT_SoVITS / tools
      · `<GSV>/GPT_SoVITS` —— module.* / text.* / sv.py / eres2net 等裸 import
      · `<GSV>/GPT_SoVITS/BigVGAN`
      · `<GSV>/tools`, `<GSV>/tools/asr`, `<GSV>/tools/uvr5`

    其中 `<GSV>` 必须排在**旧检出之前**：`site-packages/users.pth` 把
    `D:\\A\\voiceclone\\GPT-SoVITS\\GPT_SoVITS` 注入了每个进程，而 `f5_tts`
    是命名空间包（无 __init__.py），会把两边同名目录合并进 __path__。
    若不抢先，解析到的是**旧版** `f5_tts.DiT`（没有 use_step_embedding），
    训练/推理都会 TypeError。
    """
    gsv_root = str(GSV)
    need = [
        gsv_root,
        str(GSV / "GPT_SoVITS"),
        str(GSV / "GPT_SoVITS" / "BigVGAN"),
        str(GSV / "tools"),
        str(GSV / "tools" / "asr"),
        str(GSV / "tools" / "uvr5"),
    ]
    # 逆序插到最前，保证 need[0] 最终位于 sys.path[0]
    for p in reversed(need):
        while p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    # `GPT_SoVITS/sv.py:5` 用 `f"{os.getcwd()}/GPT_SoVITS/eres2net"` 拼路径，
    # `sv.py:6` 的 sv_path 也是相对路径 —— 所以**必须把 cwd 切到仓库根**，
    # 否则 ERes2NetV2 / 说话人向量都找不到（实测报 No module named
    # 'ERes2NetV2'）。
    os.chdir(str(GSV))
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

    cfg = {
        "custom": {
            "device": "cuda",
            "is_half": is_half,
            "version": version,
            "t2s_weights_path": T2S_GPT,
            "vits_weights_path": str(weights),
            "bert_base_path": str(GSV / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large"),
            "cnhuhbert_base_path": str(GSV / "GPT_SoVITS" / "pretrained_models" / "chinese-hubert-base"),
        }
    }
    return TTS(TTS_Config(cfg))


def collect(gen):
    """把 `TTS.run()` 的生成器收成 (sr, audio)。

    `TTS.run()` 是**生成器**（源码里是 `yield output_sr, audio`），
    直接 `sr, audio = tts.run(...)` 会报
    `ValueError: not enough values to unpack (expected 2, got 1)`。
    非流式模式下它会 yield 若干个 (sr, 片段)，这里按采样率分组拼接。
    """
    chunks, sr = [], None
    for item in gen:
        if item is None:
            continue
        s, a = item
        sr = s
        chunks.append(np.asarray(a))
    if not chunks:
        raise RuntimeError("TTS.run 没有产出任何音频")
    return sr, np.concatenate(chunks)


def synth(tts, texts, out_dir: Path, prefix: str,
          sample_steps=None, cfg_rate=None, tag="") -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    from GPT_SoVITS.module.v5_inference import resolve_sampling

    steps, cfg = resolve_sampling(tts.version if hasattr(tts, "version") else "v5dev",
                                  sample_steps, cfg_rate)
    print(f"  [{tag}] version={getattr(tts,'version','?')}  "
          f"steps={steps}  cfg={cfg}")
    paths = []
    for i, (_cat, text) in enumerate(texts):
        t0 = time.time()
        inputs = {
            "text": text,
            "text_lang": "zh",
            "ref_audio_path": str(REF),
            "prompt_text": REF_TEXT,
            "prompt_lang": "zh",
            "top_k": 15,
            "temperature": 1.0,
            "repetition_penalty": 1.35,
            "text_split_method": "cut1",
            "batch_size": 1,
            "speed_factor": 1.0,
            "seed": SEED,
            "sample_steps": steps,
            "cfg_rate": cfg,
            "parallel_infer": True,
        }
        sr, audio = collect(tts.run(inputs))
        p = out_dir / f"{prefix}_{i:02d}.wav"
        sf.write(str(p), audio, sr)
        paths.append(p)
        print(f"    {i:02d} {text[:24]:26s} sr={sr} "
              f"{len(audio)/sr:5.2f}s  {time.time()-t0:.1f}s")
    return paths


def load16(p: Path) -> np.ndarray:
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        k = int(round(len(x) * 16000 / sr))
        x = np.interp(np.linspace(0, len(x) - 1, k), np.arange(len(x)), x)
    return x.astype(np.float32)


def make_blind(a_files, b_files, out_path: Path, gap=0.7):
    """A/B 交替拼接，响度对齐，避免音量差干扰判断。"""
    g = np.zeros(int(gap * 16000), dtype=np.float32)
    parts = [np.zeros(int(0.5 * 16000), dtype=np.float32)]
    for a, b in zip(a_files, b_files):
        for f in (a, b):
            v = load16(f)
            r = float(np.sqrt(np.mean(v ** 2))) + 1e-9
            v = v * (0.06 / r)
            parts.append(v)
            parts.append(g)
    full = np.concatenate(parts)
    pk = float(np.max(np.abs(full)))
    if pk > 0.95:
        full *= 0.95 / pk
    sf.write(str(out_path), full, 16000, subtype="PCM_16")
    return len(full) / 16000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v5dev")
    ap.add_argument("--weights", default=None,
                    help="v5 微调权重；默认自动找 SoVITS_weights_v5 里最新的")
    ap.add_argument("--v2pro-weights",
                    default=r"D:\A\voiceclone\zfh-models-v2pro\torch_weights\zfh_v2pro_e12_s384.pth")
    ap.add_argument("--out", default=str(OUT / "compare_v5_vs_v2pro"))
    ap.add_argument("--tag", default="v5")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--cfg", type=float, default=None)
    ap.add_argument("--n", type=int, default=len(TEXTS))
    ap.add_argument("--only", default=None, choices=["v5", "v2pro"])
    # ---- 参考音频 / GPT 覆盖（默认已是生产口径）----
    ap.add_argument("--ref", default=None, help="参考音频；默认生产 default.wav")
    ap.add_argument("--ref-text", default=None, help="参考文本")
    ap.add_argument("--gpt", default=None,
                    help="GPT 权重；默认生产 zfh-e4.ckpt")
    ap.add_argument("--bad-ref", action="store_true",
                    help="用数据集里那个**错配**的 ref.wav（仅用于做对照，别用于正式评测）")
    args = ap.parse_args()

    # 应用参考/GPT 覆盖
    global REF, REF_TEXT, T2S_GPT
    if args.bad_ref:
        REF, REF_TEXT = BAD_REF, BAD_REF_TEXT
        print("!! 使用 --bad-ref：参考音频与文本是错配的，仅供对照实验")
    if args.ref:
        REF = Path(args.ref)
    if args.ref_text:
        REF_TEXT = args.ref_text
    if args.gpt:
        T2S_GPT = args.gpt

    print(f"参考音频: {REF}")
    print(f"参考文本: {REF_TEXT}")
    print(f"GPT    : {T2S_GPT}")
    print()

    texts = TEXTS[: args.n]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = args.weights
    if weights is None:
        cands = sorted((GSV / "SoVITS_weights_v5").glob("*.pth"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            print("找不到 v5 权重，先训练或用 --weights 指定")
            return 1
        weights = str(cands[0])
        print(f"自动选用最新权重: {Path(weights).name}")

    results = {}

    if args.only != "v2pro":
        print(f"\n=== 合成 {args.version} ===")
        print(f"权重: {Path(weights).name}")
        tts = build_tts(args.version, weights)
        a = synth(tts, texts, out_dir, "A", args.steps, args.cfg, args.version)
        results["v5"] = [str(p) for p in a]
        del tts
        import torch, gc
        gc.collect(); torch.cuda.empty_cache()

    if args.only != "v5":
        print("\n=== 合成 v2Pro（当前生产）===")
        b = synth(build_tts("v2Pro", args.v2pro_weights), texts, out_dir, "B",
                  tag="v2Pro")
        results["v2pro"] = [str(p) for p in b]

    if "v5" in results and "v2pro" in results:
        p = out_dir / "一次听完.wav"
        dur = make_blind([Path(x) for x in results["v5"]],
                         [Path(x) for x in results["v2pro"]], p)
        print(f"\n盲听集: {p}  ({dur:.1f}s)")
        print("  顺序：每句先 A(v5) 后 B(v2Pro)")
        (out_dir / "答案.txt").write_text(
            f"A = GPT-SoVITS {args.version} 微调（{Path(weights).name}）\n"
            f"B = GPT-SoVITS v2Pro（当前生产，zfh_v2pro_e12_s384.pth）\n\n"
            "每句先 A 后 B，句间 0.7s 静音。响度已对齐。\n\n"
            "句子清单:\n" + "\n".join(
                f"  {i+1}. [{c}] {t}" for i, (c, t) in enumerate(texts)),
            encoding="utf-8")

    (out_dir / "meta.json").write_text(json.dumps(
        {"version": args.version, "weights": weights,
         "texts": [t for _, t in texts], "seed": SEED,
         "ref": str(REF), "files": results},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
