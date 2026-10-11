#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只打"推理环境"补丁（不碰 lora 加载逻辑）。

## 为什么需要这个
`patch_v5_fullft_infer.py` 里包含 **5 处**补丁，其中**补丁 1 已过时**：
它改的是 `TTS.py` 的 `if_lora_v3` 分支，而那个位置现在由
`patch_v5_fullft_load.py` 用**双向对齐**的方式处理（更好，且不破坏 LoRA）。

**实测后果**：在云端先打 `patch_v5_fullft_load.py` 后，
`patch_v5_fullft_infer.py` 的补丁 1 找不到 `OLD_LORA` 片段 ->
脚本可能提前退出 -> **补丁 2/3（torchaudio 回退）没打上** ->
合成报 `ImportError: TorchCodec is required`。

## 本脚本只做这 4 处（都不碰 lora 逻辑）
  2  `_zfy_load_audio` 回退函数（soundfile -> torchaudio -> ffmpeg）
  3a `load_audio_gpu` 里的 `torchaudio.load` -> `_zfy_load_audio`
  3b 参考音频处的 `torchaudio.load` -> `_zfy_load_audio`
  4  LangSegmenter 用 lite 模型（零下载）
  5  G2PW 缺 onnxruntime 时优雅降级

## 用法
    python _tools/patch_infer_env.py --root $PWD
    python _tools/patch_infer_env.py --root $PWD --check
    python _tools/patch_infer_env.py --root $PWD --revert
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MARK = "ZFY_INFER_ENV_PATCH"

# ---------------------------------------------------------------- 补丁 2
T_REL = "GPT_SoVITS/TTS_infer_pack/TTS.py"

FALLBACK_FN = '''
# ---- ZFY_INFER_ENV_PATCH: torchaudio.load 回退 ----
# 云上 torch 2.12 的 torchaudio 把 load 委派给 **torchcodec**（未安装）：
#     ImportError: TorchCodec is required for load_with_torchcodec.
# 这里按 soundfile -> torchaudio -> ffmpeg 顺序回退。
# soundfile 对 wav/flac/ogg 足够，且不依赖 torchcodec。
def _zfy_load_audio(path, target_sr=None):
    """读音频 -> (float32 numpy [n], sample_rate)。

    返回 numpy（不是 tensor）—— 调用点自己转 torch。
    """
    errs = []
    # 1) soundfile（最稳）
    try:
        import soundfile as _sf
        x, sr = _sf.read(str(path), always_2d=True, dtype="float32")
        x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
        return x.astype("float32"), int(sr)
    except Exception as e:  # noqa: BLE001
        errs.append(f"soundfile: {type(e).__name__}: {e}")
    # 2) torchaudio（可能走 torchcodec，失败就继续）
    try:
        import torchaudio as _ta
        w, sr = _ta.load(str(path))
        w = w.mean(dim=0) if w.shape[0] > 1 else w[0]
        return w.numpy().astype("float32"), int(sr)
    except Exception as e:  # noqa: BLE001
        errs.append(f"torchaudio: {type(e).__name__}: {e}")
    # 3) ffmpeg CLI
    try:
        import subprocess as _sp
        import numpy as _np
        p = _sp.run(["ffmpeg", "-v", "error", "-i", str(path),
                     "-ac", "1", "-ar", "32000", "-f", "f32le", "pipe:1"],
                    capture_output=True, check=True)
        return (_np.frombuffer(p.stdout, dtype=_np.float32).copy(), 32000)
    except Exception as e:  # noqa: BLE001
        errs.append(f"ffmpeg: {type(e).__name__}: {e}")
    raise RuntimeError("无法读取音频 " + str(path) + "；已尝试: "
                       + " | ".join(errs))
# ---- /ZFY_INFER_ENV_PATCH ----
'''

# 在 import 段之后插入（锚定 torchaudio import）
ANCHOR = "import torchaudio\n"

OLD_LOAD1 = "    audio, source_sr = torchaudio.load(path)"
NEW_LOAD1 = ("    # [ZFY_INFER_ENV_PATCH] torchaudio.load 在 torch 2.12 上需要 "
             "torchcodec，改用回退\n"
             "    audio_np, source_sr = _zfy_load_audio(path)\n"
             "    audio = torch.from_numpy(audio_np).unsqueeze(0)")

OLD_LOAD2 = "        raw_audio, raw_sr = torchaudio.load(ref_audio_path)"
NEW_LOAD2 = ("        # [ZFY_INFER_ENV_PATCH] 同上\n"
             "        _raw_np, raw_sr = _zfy_load_audio(ref_audio_path)\n"
             "        raw_audio = torch.from_numpy(_raw_np).unsqueeze(0)")

# ---------------------------------------------------------------- 补丁 4
L_REL = "GPT_SoVITS/text/LangSegmenter/langsegmenter.py"
L_OLD = "fast_langdetect.infer._default_detector = fast_langdetect.infer.LangDetector(fast_langdetect.infer.LangDetectConfig(cache_dir=Path(__file__).parent.parent.parent / \"pretrained_models\" / \"fast_langdetect\"))"
# ⚠️ 幂等性陷阱：`L_NEW` 里**包含** `L_OLD` 两处（作为 except 分支的兜底），
#    所以第二次运行 `L_OLD in src` 仍为 True -> 再替换一遍 ->
#    嵌套出 `try: try: ... except: except:` -> IndentationError。
#    实测踩到：`IndentationError: expected an indented block after
#    'except' statement on line 15`。
#    解法：用带标记的整段替换，且标记检测放在**替换之前**（已有）。
#          这里再给 L_OLD 加上行首锚定，避免匹配到缩进后的副本。
L_OLD = ("\n" + L_OLD + "\n")
L_NEW = ("\n"
         "# [ZFY_INFER_ENV_PATCH] 用 lite 模型（916 KB，随 pip 包）而不是\n"
         "# full（125 MB，需联网下载）。实测两者分段结果逐条一致。\n"
         "_zfy_lsd_cfg = fast_langdetect.infer.LangDetectConfig(\n"
         "    cache_dir=Path(__file__).parent.parent.parent\n"
         "    / \"pretrained_models\" / \"fast_langdetect\",\n"
         "    model=\"lite\",\n"
         ")\n"
         "try:\n"
         "    fast_langdetect.infer._default_detector = \\\n"
         "        fast_langdetect.infer.LangDetector(_zfy_lsd_cfg)\n"
         "except Exception:\n"
         "    fast_langdetect.infer._default_detector = \\\n"
         "        fast_langdetect.infer.LangDetector(\n"
         "            fast_langdetect.infer.LangDetectConfig(\n"
         "                cache_dir=Path(__file__).parent.parent.parent\n"
         "                / \"pretrained_models\" / \"fast_langdetect\"))\n")

# ---------------------------------------------------------------- 补丁 5
C_REL = "GPT_SoVITS/text/chinese2.py"
# ⚠️ 同样有幂等陷阱：`is_g2pw = True` 是 `C_NEW` 的子串。
#    加行首锚定 + 完整行匹配。
C_OLD = "\nis_g2pw = True  # True if is_g2pw_str.lower() == 'true' else False\n"
C_NEW = ("\n"
         "# [ZFY_INFER_ENV_PATCH] 缺 onnxruntime 时优雅降级（G2PW 只是\n"
         "# 多音字消歧，没有它也能合成，只是多音字准确率下降）。\n"
         "try:\n"
         "    import onnxruntime as _zfy_ort  # noqa: F401\n"
         "    is_g2pw = True  # True if is_g2pw_str.lower() == 'true' else False\n"
         "except ImportError:\n"
         "    is_g2pw = False\n")


def patch_one(root: Path, rel: str, pairs: list[tuple[str, str]],
              check: bool, revert: bool, tag: str,
              insert_after: str | None = None,
              insert_text: str | None = None) -> int:
    p = root / rel
    if not p.exists():
        print(f"  !! 找不到 {p}")
        return 1
    src = p.read_text(encoding="utf-8", errors="replace")
    bak = p.with_suffix(p.suffix + ".bak_zfy_inferenv")

    if revert:
        if bak.exists():
            shutil.copy2(bak, p)
            print(f"  已还原 {rel}")
            return 0
        print(f"  未打过补丁 {rel}")
        return 0

    # ⚠️ 幂等性：**先查标记**，再尝试替换。
    #    我第一版是先替换再查标记，而 L_OLD 是 L_NEW 的子串 ->
    #    第二次运行时又替换一遍 -> 嵌套出 `try: try: ... except: except:`
    #    -> IndentationError（实测踩到）。
    if MARK in src:
        print(f"  已打过补丁: {rel}")
        return 0

    done = 0
    applied = []
    for old, new in pairs:
        if old not in src:
            continue
        src = src.replace(old, new, 1)
        done += 1
        applied.append(new.strip().split("\n")[0][:52])

    if insert_after and insert_text and "def _zfy_load_audio" not in src:
        if insert_after in src:
            src = src.replace(insert_after, insert_after + insert_text, 1)
            done += 1
            applied.append("插入 _zfy_load_audio 回退函数")

    if done == 0:
        print(f"  !! {rel} 找不到目标片段（{tag}）")
        return 1

    if check:
        print(f"  [{tag}] 将修改 {rel}（{done} 处）")
        return 0

    # ⚠️ **关键安全网**：打完立刻做语法检查。
    #    补丁改的是源码字符串，任何缩进/嵌套错误都会在这里暴露，
    #    而不是等到合成时才报 IndentationError（那时已浪费一轮）。
    import ast
    try:
        ast.parse(src)
    except SyntaxError as e:
        print(f"  !! {rel} 打完补丁后**语法错误**（已放弃写入）")
        print(f"     第 {e.lineno} 行: {e.msg}")
        print(f"     {str(e.text or '').strip()[:80]}")
        return 1

    if not bak.exists():
        shutil.copy2(p, bak)
    p.write_text(src, encoding="utf-8")
    print(f"  [{tag}] {rel} 已应用 {done} 处")
    for a in applied:
        print(f"        · {a}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    print("=" * 76)
    print(f"推理环境补丁（不含 lora 逻辑）: {root}")
    print("=" * 76)

    rc = 0
    rc |= patch_one(root, T_REL,
                    [(OLD_LOAD1, NEW_LOAD1), (OLD_LOAD2, NEW_LOAD2)],
                    args.check, args.revert, "torchaudio",
                    insert_after=ANCHOR, insert_text=FALLBACK_FN)
    print()
    rc |= patch_one(root, L_REL, [(L_OLD, L_NEW)],
                    args.check, args.revert, "LangSegmenter")
    print()
    rc |= patch_one(root, C_REL, [(C_OLD, C_NEW)],
                    args.check, args.revert, "G2PW")

    print()
    print("=" * 76)
    if rc == 0 and not args.check and not args.revert:
        print("""
  补丁完成（4 处，都不碰 lora 加载逻辑）。

  ## 与 patch_v5_fullft_load.py 的分工
    patch_v5_fullft_load.py  -> 只改 `if_lora_v3` 分支（双向对齐 lora_rank）
    本脚本                   -> 只改**推理环境**（torchaudio/LangSegmenter/G2PW）
  **两者互不冲突**，都要打。

  ## 为什么不直接用 patch_v5_fullft_infer.py
  它含**已过时**的补丁 1（改 `if_lora_v3` 分支），与
  `patch_v5_fullft_load.py` 改**同一段代码** -> 后打的会因找不到
  `OLD_LORA` 片段而失败，导致后面的补丁没打上。
  （实测：云端合成报 `ImportError: TorchCodec is required`）

  ## 验证
      python _tools/verify_infer_env.py

  ## 然后合成
      python _tools/cloud_synth_sweep.py \\
          --weights-dir SoVITS_weights_v5_full_e32_nofreeze \\
          --epochs 8,16,24,32 --no-v2pro --out out_sweep_nofreeze 2>&1 | tail -25
""")
    elif rc != 0:
        print("  有改动未应用，见上面的 !! 行")
    print("=" * 76)
    return rc


if __name__ == "__main__":
    sys.exit(main())
