# -*- coding: utf-8 -*-
"""验证 patch_infer_env.py —— 屏蔽 torchcodec 后仍能合成。

## 核心验证
云端 torch 2.12 的 `torchaudio.load` 委派给 **torchcodec**（未安装）：
    ImportError: TorchCodec is required for load_with_torchcodec.
-> 合成全部失败（`1/10 失败 ... 10/10 失败`）。

本脚本用**子进程 + 屏蔽 torchcodec** 模拟云端，验证：
  1. 补丁前：`_zfy_load_audio` 不存在 -> 会走 torchaudio.load -> 崩
  2. 补丁后：`_zfy_load_audio` 存在且能用 soundfile 读
  3. **端到端**：真的构造 TTS 并合成一句（这是最终判据）
  4. 与 `patch_v5_fullft_load.py` **共存**（两者不冲突）

## 为什么要单独写（而不是改 patch_v5_fullft_infer.py）
`patch_v5_fullft_infer.py` 含**已过时**的补丁 1，它改的
`if if_lora_v3 == False:` 那段现在由 `patch_v5_fullft_load.py` 用
**双向对齐**处理。两者改同一段 -> 后打的会失败 ->
**后面的补丁（torchaudio）没打上** -> 这就是云端报 torchcodec 的原因。
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_CANDS = [
    Path("/mnt/workspace/zfy-v5-fullft-pkg/GPT-SoVITS"),
    Path("/mnt/workspace/zfy-v5-fullft-pkg"),
    Path(r"D:\A\zfy-v5\GPT-SoVITS"),
    Path.cwd(),
]
GSV = next((p for p in _CANDS if (p / "GPT_SoVITS" / "utils.py").exists()),
           _CANDS[0])
_ap = argparse.ArgumentParser()
_ap.add_argument("--root", default=None)
_args, _ = _ap.parse_known_args()
if _args.root:
    GSV = Path(_args.root).resolve()
PY = sys.executable
PATCH = next((p for p in (GSV / "_tools" / "patch_infer_env.py",
                          Path(__file__).resolve().parent
                          / "patch_infer_env.py") if p.exists()),
             Path("patch_infer_env.py"))
LOADPATCH = next((p for p in (GSV / "_tools" / "patch_v5_fullft_load.py",
                              Path(__file__).resolve().parent
                              / "patch_v5_fullft_load.py") if p.exists()),
                 None)

print(f"  仓库根: {GSV}")
print(f"  补丁  : {PATCH}")


def run_patch(script: Path, *extra):
    r = subprocess.run([PY, str(script), "--root", str(GSV), *extra],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    for l in (r.stdout or "").split("\n"):
        if re.match(r"\s*\[", l) or "还原" in l or "已打过" in l \
                or "已应用" in l:
            print(f"      {l.strip()}")
    return r.returncode


print()
print("=" * 84)
print("0. 打补丁（load 补丁 + 环境补丁，顺序无关）")
print("=" * 84)
if LOADPATCH:
    print("  -- patch_v5_fullft_load.py --")
    run_patch(LOADPATCH)
print("  -- patch_infer_env.py --")
run_patch(PATCH)

print()
print("=" * 84)
print("1. 静态检查")
print("=" * 84)
tt = (GSV / "GPT_SoVITS" / "TTS_infer_pack" / "TTS.py").read_text(
    encoding="utf-8", errors="replace")
lg = (GSV / "GPT_SoVITS" / "text" / "LangSegmenter"
      / "langsegmenter.py").read_text(encoding="utf-8", errors="replace")
ch = (GSV / "GPT_SoVITS" / "text" / "chinese2.py").read_text(
    encoding="utf-8", errors="replace")

has_fn = "def _zfy_load_audio" in tt
n_call = len(re.findall(r"_zfy_load_audio\(", tt)) - (1 if has_fn else 0)
n_ta = len(re.findall(r"torchaudio\.load\(", tt))
has_lite = 'model="lite"' in lg
has_g2pw = "onnxruntime" in ch and "is_g2pw = False" in ch
# 双向对齐（来自 load 补丁）
has_dual = "_zfy_has_lora" in tt

print(f"  _zfy_load_audio 已插入      : {has_fn}")
print(f"  _zfy_load_audio 调用点      : {n_call}  (期望 2)")
print(f"  残留 torchaudio.load        : {n_ta}  (期望 0)")
print(f"  LangSegmenter 用 lite       : {has_lite}")
print(f"  G2PW 优雅降级               : {has_g2pw}")
print(f"  lora 双向对齐（load 补丁）  : {has_dual}")

print()
print("=" * 84)
print("2. 动态检查：屏蔽 torchcodec 后读音频 + 合成")
print("=" * 84)

PROBE = r'''
import os, sys, builtins
GSV = GSV_PATH
sys.path[:0] = [GSV, os.path.join(GSV, "GPT_SoVITS")]
os.chdir(GSV)

# 模拟云端：torchaudio.load 需要 torchcodec，而它没装
_ri = builtins.__import__
def _fake(name, *a, **k):
    if name == "torchcodec" or name.startswith("torchcodec."):
        raise ImportError("ZFY: torchcodec 被测试屏蔽（模拟云端）")
    return _ri(name, *a, **k)
builtins.__import__ = _fake

REF = os.path.join(GSV, "ref", "default.wav")
print(f"REF_EXISTS={os.path.exists(REF)}")

try:
    from GPT_SoVITS.TTS_infer_pack.TTS import _zfy_load_audio
    x, sr = _zfy_load_audio(REF)
    import numpy as np
    rms = float(np.sqrt((np.asarray(x, dtype=float) ** 2).mean()))
    # ⚠️ 输出必须是纯 k=v 形式，不能有裸标签 —— 否则解析器取不到
    print(f"LOAD=n{len(x)}_sr{sr}_rms{rms:.5f}")
except Exception as e:
    print(f"LOAD=fail_{type(e).__name__}_{str(e)[:60]}")

# 端到端：构造 TTS 并合成
import glob
cands = glob.glob(os.path.join(
    GSV, "SoVITS_weights_v5_full_e32_nofreeze", "*_e32_*.pth"))
if not cands:
    cands = glob.glob(os.path.join(GSV, "SoVITS_weights_v5*", "*.pth"))
if not cands:
    print("SYNTH=skip_noweights")
    raise SystemExit(0)
W = sorted(cands)[-1]
print(f"WEIGHT={os.path.basename(W)}")

try:
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
    PRE = os.path.join(GSV, "GPT_SoVITS", "pretrained_models")
    rtxt = open(os.path.join(GSV, "ref", "ref.txt"), encoding="utf-8").read().strip()
    cfg = {"custom": {
        "device": "cuda", "is_half": False, "version": "v5dev",
        "t2s_weights_path": os.path.join(GSV, "ref", "zfh-e4.ckpt"),
        "vits_weights_path": W,
        "bert_base_path": os.path.join(PRE, "chinese-roberta-wwm-ext-large"),
        "cnhuhbert_base_path": os.path.join(PRE, "chinese-hubert-base"),
    }}
    tts = TTS(TTS_Config(cfg))
    import numpy as np
    chunks, sr2 = [], None
    for item in tts.run({
        "text": "让我想想。", "text_lang": "zh",
        "ref_audio_path": REF, "prompt_text": rtxt, "prompt_lang": "zh",
        "top_k": 15, "temperature": 1.0, "repetition_penalty": 1.35,
        "text_split_method": "cut1", "batch_size": 1,
        "speed_factor": 1.0, "seed": 42, "parallel_infer": True,
    }):
        if item is None:
            continue
        sr2, a = item
        chunks.append(a)
    if not chunks:
        print("SYNTH=fail_novocal")
    else:
        audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        dur = len(audio) / sr2
        print(f"SYNTH=ok_dur{dur:.2f}_sr{sr2}")
except Exception as e:
    print(f"SYNTH=fail_{type(e).__name__}_{str(e)[:70]}")
'''


def probe(label: str) -> dict:
    code = PROBE.replace("GSV_PATH", repr(str(GSV)))
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=1800,
                       cwd=str(GSV))
    out = (r.stdout or "") + (r.stderr or "")
    d = {}
    print(f"\n--- {label} ---")
    for l in out.split("\n"):
        s = l.strip()
        if re.match(r"^(REF_EXISTS|LOAD|WEIGHT|SYNTH)=", s):
            print(f"    {s[:110]}")
            k, v = s.split("=", 1)
            d[k] = v
        elif "[ZFY]" in s:
            print(f"    {s[:110]}")
    return d


d = probe("屏蔽 torchcodec")

print()
print("=" * 84)
print("3. 断言")
print("=" * 84)
checks = [
    ("_zfy_load_audio 已插入", has_fn, ""),
    ("两处 torchaudio.load 都已替换", n_ta == 0 and n_call == 2,
     f"调用={n_call} 残留={n_ta}"),
    ("LangSegmenter 用 lite", has_lite, ""),
    ("G2PW 优雅降级", has_g2pw, ""),
    ("lora 双向对齐仍在（与 load 补丁共存）", has_dual, ""),
    ("屏蔽 torchcodec 后能读音频",
     str(d.get("LOAD", "")).startswith("n"), f"LOAD={d.get('LOAD')}"),
    ("屏蔽 torchcodec 后能合成",
     str(d.get("SYNTH", "")).startswith("ok"),
     f"SYNTH={d.get('SYNTH')}"),
]
bad = 0
for name, ok, extra in checks:
    bad += 0 if ok else 1
    print(f"  {'✓' if ok else '✗'} {name}" + (f"  ({extra})" if extra else ""))

print()
print("=" * 84)
if bad == 0:
    print("  ✓ 全部通过 —— 云端可以合成")
else:
    print(f"  ✗ {bad} 项失败")
print("=" * 84)
sys.exit(1 if bad else 0)
