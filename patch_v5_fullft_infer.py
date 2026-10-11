# -*- coding: utf-8 -*-
"""给 GPT-SoVITS 推理端打补丁，使其能正确加载 **v5 全量微调**权重。

## 为什么需要（云端合成失败的完整根因）

### 根因链
`process_ckpt.savee()` 只在两种情况写版本头：

    if lora_rank:                          my_save2(...)   # 写头
    elif "Pro" in model_version:           my_save2(...)   # 写头
    else:                                  my_save(...)    # **不写头**

我们的全量微调 `lora_rank=0` 且 `model_version="v5dev"`（不含 "Pro"）
-> 产出**纯 zip（PK 头）**。

于是 `get_sovits_version_from_path_fast` 落到「按文件大小猜」：

    if   size < 82978KB:        v1
    elif size < 700MB:          v2
    else:                       version=v2, model_version="**v3**"

736 MB -> **v3** -> `init_vocoder("v3")` -> `BigVGAN.from_pretrained(24kHz 目录)`
-> `HFValidationError`（那个 HF 缓存目录在云端不存在）

### 但只补版本头还不够
`b"07"` 的 `if_lora_v3=True`，会走 LoRA 分支：

    lora_rank = dict_s2["lora_rank"]     # 全量权重没这个键 -> KeyError

即使补个假值：`get_peft_model` 把键名变成 `cfm.base_model.model.*`，
我们的 `cfm.estimator.*` **匹配不上**，`strict=False` 会**静默忽略**
-> 实际加载的是**底模**而不是微调结果。**无声错误，比崩溃更危险。**

## 本补丁做三件事
  1. `if_lora_v3` 且 `dict_s2` 无 `"lora_rank"` -> 判定为**全量权重**，强制 False
  2. `torchaudio.load` -> 带回退的包装（soundfile -> torchaudio -> ffmpeg）
     （云端 torchaudio 新版委派给未安装的 torchcodec）
  3. 打印明确的加载日志，便于确认走对了分支

用法：
    python patch_v5_fullft_infer.py --root /mnt/workspace/zfy-v5-fullft-pkg
    python patch_v5_fullft_infer.py --root . --check     # 只检查
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MARK = "ZFY_V5FULLFT_PATCH"

# --------------------------------------------------------------------------
# 补丁 1：全量权重不当作 LoRA
# --------------------------------------------------------------------------
OLD_LORA = '''        if if_lora_v3 == False:
            print(
                f"Loading VITS weights from {weights_path}. {vits_model.load_state_dict(dict_s2['weight'], strict=False)}"
            )
        else:'''

NEW_LORA = '''        # ---- ZFY_V5FULLFT_PATCH: 全量微调权重没有 "lora_rank" 键 ----
        # 根因：process_ckpt.savee() 只在 lora_rank 非空或 model_version 含
        # "Pro" 时写版本头；全量微调两者都不满足 -> 无头 -> 推理端按文件
        # 大小猜成 v3，且 b"07" 的 if_lora_v3=True 会走 LoRA 路径。
        # 全量权重走 LoRA 路径的后果：KeyError('lora_rank')，或（补了假值后）
        # get_peft_model 把键名变成 cfm.base_model.model.*，与 cfm.estimator.*
        # 匹配不上 -> strict=False **静默忽略** -> 实际加载底模。
        if if_lora_v3 and "lora_rank" not in dict_s2:
            print(
                "[ZFY] 检测到全量微调权重（无 lora_rank 键），"
                "按全量权重加载，跳过 LoRA 路径"
            )
            if_lora_v3 = False
        # ---- /ZFY_V5FULLFT_PATCH ----

        if if_lora_v3 == False:
            print(
                f"Loading VITS weights from {weights_path}. {vits_model.load_state_dict(dict_s2['weight'], strict=False)}"
            )
        else:'''

# --------------------------------------------------------------------------
# 补丁 2：torchaudio.load 回退
# --------------------------------------------------------------------------
LOADER = '''

# ---- ZFY_V5FULLFT_PATCH: torchaudio.load 回退 ----
# 云上 torchaudio 新版把 load 委派给 torchcodec（未安装）：
#     ImportError: TorchCodec is required for load_with_torchcodec
# 这里按 soundfile -> torchaudio -> ffmpeg 顺序回退。
def _zfy_load_audio(path, target_sr=None):
    """返回 (waveform[T] float32, sample_rate)。单声道、numpy 后端。"""
    import numpy as _np
    errs = []
    # 1) soundfile（最稳，不依赖 torchcodec）
    try:
        import soundfile as _sf
        data, sr = _sf.read(str(path), dtype="float32", always_2d=True)
        wav = _np.asarray(data)
        if wav.ndim == 2 and wav.shape[1] > 1:
            wav = wav.mean(axis=1)
        else:
            wav = wav[:, 0]
        return wav, int(sr)
    except Exception as e:  # noqa: BLE001
        errs.append(f"soundfile: {type(e).__name__}: {e}")
    # 2) torchaudio（可能走 torchcodec，失败就继续）
    try:
        import torchaudio as _ta
        a, sr = _ta.load(str(path))
        if a.dim() == 2 and a.shape[0] > 1:
            a = a.mean(dim=0)
        elif a.dim() == 2:
            a = a[0]
        return a.numpy(), int(sr)
    except Exception as e:  # noqa: BLE001
        errs.append(f"torchaudio: {type(e).__name__}: {e}")
    # 3) ffmpeg CLI
    try:
        import subprocess as _sp, io as _io
        cmd = ["ffmpeg", "-v", "error", "-i", str(path),
               "-f", "f32le", "-ac", "1", "-ar", "32000", "-"]
        out = _sp.run(cmd, capture_output=True, check=True).stdout
        return _np.frombuffer(out, dtype=_np.float32), 32000
    except Exception as e:  # noqa: BLE001
        errs.append(f"ffmpeg: {type(e).__name__}: {e}")
    raise RuntimeError("无法读取音频 " + str(path) + "；已尝试: " + " | ".join(errs))
# ---- /ZFY_V5FULLFT_PATCH ----
'''

OLD_LOAD1 = "    audio, source_sr = torchaudio.load(path)"
NEW_LOAD1 = ("    audio_np, source_sr = _zfy_load_audio(path)\n"
             "    audio = torch.from_numpy(audio_np).unsqueeze(0)")

OLD_LOAD2 = "        raw_audio, raw_sr = torchaudio.load(ref_audio_path)"
NEW_LOAD2 = ("        _raw_np, raw_sr = _zfy_load_audio(ref_audio_path)\n"
             "        raw_audio = torch.from_numpy(_raw_np).unsqueeze(0)")


def patch_tts(root: Path, check: bool) -> int:
    p = root / "GPT_SoVITS" / "TTS_infer_pack" / "TTS.py"
    if not p.exists():
        print(f"  !! 找不到 {p}")
        return 1
    src = p.read_text(encoding="utf-8", errors="replace")

    if MARK in src:
        print(f"  已打过补丁: {p.name}")
        return 0

    n = 0
    # --- 补丁 1 ---
    if OLD_LORA in src:
        src = src.replace(OLD_LORA, NEW_LORA, 1)
        n += 1
        print("  [1/3] 已插入「全量权重不当作 LoRA」判断")
    else:
        print("  !! 找不到 if_lora_v3 分支，补丁 1 未应用")

    # --- 补丁 2：插入 loader 函数（放在 resample 之后、使用之前）---
    anchor = "def load_audio_gpu(path, target_sr, device, mono=True"
    if anchor in src:
        src = src.replace(anchor, LOADER.strip() + "\n\n\n" + anchor, 1)
        n += 1
        print("  [2/3] 已插入 _zfy_load_audio 回退函数")
    else:
        print("  !! 找不到 load_audio_gpu，补丁 2 未应用")

    # --- 补丁 3：替换两处调用 ---
    if OLD_LOAD1 in src:
        src = src.replace(OLD_LOAD1, NEW_LOAD1, 1)
        n += 1
        print("  [3a/3] 已替换 load_audio_gpu 里的 torchaudio.load")
    else:
        print("  !! 找不到 load_audio_gpu 里的 torchaudio.load")
    if OLD_LOAD2 in src:
        src = src.replace(OLD_LOAD2, NEW_LOAD2, 1)
        n += 1
        print("  [3b/3] 已替换参考音频处的 torchaudio.load")
    else:
        print("  !! 找不到参考音频处的 torchaudio.load")

    if check:
        print(f"  （--check 模式，未写盘；将应用 {n} 处改动）")
        return 0

    bak = p.with_suffix(".py.bak_zfy")
    if not bak.exists():
        shutil.copy2(p, bak)
        print(f"  备份: {bak.name}")
    p.write_text(src, encoding="utf-8")
    print(f"  已写入 {p}  ({n} 处改动)")
    return 0 if n >= 3 else 1


# --------------------------------------------------------------------------
# 补丁 3：LangSegmenter 用 lite 模型（不联网、不需要缓存目录）
# --------------------------------------------------------------------------
OLD_LANG = (
    'fast_langdetect.infer._default_detector = '
    'fast_langdetect.infer.LangDetector('
    'fast_langdetect.infer.LangDetectConfig('
    'cache_dir=Path(__file__).parent.parent.parent / "pretrained_models" '
    '/ "fast_langdetect"))'
)

NEW_LANG = '''# ---- ZFY_V5FULLFT_PATCH: 阻止联网下载 125MB 的 lid.176.bin ----
# 原代码只设了 cache_dir，而 model 默认 "auto" -> low_memory=False
# -> 去 cache_dir 找 lid.176.bin，找不到就**联网下载** 125MB：
#     https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
# cache_dir 目录不会自建，于是报：
#     FileNotFoundError: fast-langdetect: Cache directory not found: ...
#
# ⚠️ 但**只改 _default_detector 的 config 是没用的**（我第一版就错了）：
#     split_lang/detector.py:13 硬编码了
#         fast_langdetect.detect(text, model="full")
#     这个 `model="full"` **参数**会覆盖 config.model，
#     于是仍然走 low_memory=False -> 下载。
#     （实测调用栈确认：LangSegmenter.getTexts -> split_lang
#       -> detect_lang_combined -> fast_langdetect.detect(model="full")）
#
# 正确做法：把 `fast_langdetect.detect` 包一层，把 model 强制改成 "lite"。
# lite 模型（916KB）随 pip 包分发（resources/lid.176.ftz），**零下载**，
# 实测分段结果与 full 逐条一致。
import fast_langdetect as _zfy_fld

_ZFY_LANG_CACHE = Path(__file__).parent.parent.parent / "pretrained_models" / "fast_langdetect"
try:
    _ZFY_LANG_CACHE.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

_zfy_orig_detect = _zfy_fld.detect


def _zfy_detect_lite(text, model="lite", **kwargs):
    """强制 lite：忽略调用方传的 model（split_lang 传的是 "full"）。"""
    return _zfy_orig_detect(text, model="lite", **kwargs)


_zfy_fld.detect = _zfy_detect_lite

# 同时把默认 detector 也设成 lite（双保险）
try:
    fast_langdetect.infer._default_detector = fast_langdetect.infer.LangDetector(
        fast_langdetect.infer.LangDetectConfig(
            cache_dir=_ZFY_LANG_CACHE,
            model="lite",
        )
    )
except Exception:
    pass
# ---- /ZFY_V5FULLFT_PATCH ----'''


def patch_langsegmenter(root: Path, check: bool) -> int:
    """给 text/LangSegmenter/langsegmenter.py 打补丁，改用 lite 模型。"""
    p = (root / "GPT_SoVITS" / "text" / "LangSegmenter" / "langsegmenter.py")
    if not p.exists():
        print(f"  !! 找不到 {p}")
        return 1
    src = p.read_text(encoding="utf-8", errors="replace")
    if MARK in src:
        print("  已打过补丁（LangSegmenter）")
        return 0
    if OLD_LANG not in src:
        print("  !! 找不到 LangDetectConfig 那一行，补丁 3 未应用")
        return 1
    src = src.replace(OLD_LANG, NEW_LANG, 1)
    if check:
        print("  [4/4] 将把 LangSegmenter 改为 model='lite'（零下载）")
        return 0
    bak = p.with_suffix(".py.bak_zfy")
    if not bak.exists():
        shutil.copy2(p, bak)
        print(f"  备份: {bak.name}")
    p.write_text(src, encoding="utf-8")
    print("  [4/4] 已把 LangSegmenter 改为 model='lite'（零下载）")
    return 0


# --------------------------------------------------------------------------
# 补丁 4：G2PW（多音字消歧）优雅降级
# --------------------------------------------------------------------------
OLD_G2PW = '''# is_g2pw_str = os.environ.get("is_g2pw", "True")##默认开启
# is_g2pw = False#True if is_g2pw_str.lower() == 'true' else False
is_g2pw = True  # True if is_g2pw_str.lower() == 'true' else False
if is_g2pw:
    # print("当前使用g2pw进行拼音推理")
    from text.g2pw import G2PWPinyin, correct_pronunciation

    parent_directory = os.path.dirname(current_file_path)
    g2pw = G2PWPinyin(
        model_dir="GPT_SoVITS/text/G2PWModel",
        model_source=os.environ.get("bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"),
        v_to_u=False,
        neutral_tone_with_five=True,
    )'''

NEW_G2PW = '''# ---- ZFY_V5FULLFT_PATCH: G2PW 优雅降级 ----
# 原代码 is_g2pw = True 是**硬编码**的（上面两行环境变量读取被注释掉了），
# 于是 chinese2.py 一定会 import text.g2pw
#   -> g2pw/onnx_api.py:11 模块级 `import onnxruntime`
# 云上没装 onnxruntime 就直接崩：
#     ModuleNotFoundError: No module named 'onnxruntime'
# 而 G2PW 只是个**多音字消歧**的增强项（用 635MB 的 g2pW.onnx），
# 缺了它中文照样能合成（退回 jieba 词性 + 单音字表）。
#
# 改成：默认**尝试开启**，失败则打印警告并降级，不中断合成。
# 可用环境变量强制控制：ZFY_G2PW=0 关闭 / =1 强制开启（失败仍降级）
is_g2pw = os.environ.get("ZFY_G2PW", "auto")
if is_g2pw == "auto":
    is_g2pw = True
else:
    is_g2pw = is_g2pw.lower() in ("1", "true", "yes", "on")

if is_g2pw:
    try:
        from text.g2pw import G2PWPinyin, correct_pronunciation

        parent_directory = os.path.dirname(current_file_path)
        g2pw = G2PWPinyin(
            model_dir="GPT_SoVITS/text/G2PWModel",
            model_source=os.environ.get("bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"),
            v_to_u=False,
            neutral_tone_with_five=True,
        )
    except Exception as _zfy_g2pw_err:
        print(
            "[ZFY] G2PW 不可用，降级为 jieba 拼音（多音字可能不准）："
            f"{type(_zfy_g2pw_err).__name__}: {_zfy_g2pw_err}"
        )
        print("[ZFY] 如需开启：pip install onnxruntime  （并确保 "
              "GPT_SoVITS/text/G2PWModel/g2pW.onnx 存在）")
        is_g2pw = False
# ---- /ZFY_V5FULLFT_PATCH ----'''


def patch_g2pw(root: Path, check: bool) -> int:
    """让 G2PW 在缺 onnxruntime 时优雅降级，而不是让合成整个崩掉。"""
    p = root / "GPT_SoVITS" / "text" / "chinese2.py"
    if not p.exists():
        print(f"  !! 找不到 {p}")
        return 1
    src = p.read_text(encoding="utf-8", errors="replace")
    if MARK in src:
        print("  已打过补丁（chinese2/G2PW）")
        return 0
    if OLD_G2PW not in src:
        print("  !! 找不到 is_g2pw 那一段，补丁 5 未应用")
        return 1
    src = src.replace(OLD_G2PW, NEW_G2PW, 1)
    if check:
        print("  [5/5] 将让 G2PW 缺 onnxruntime 时优雅降级（不中断合成）")
        return 0
    bak = p.with_suffix(".py.bak_zfy")
    if not bak.exists():
        shutil.copy2(p, bak)
        print(f"  备份: {bak.name}")
    p.write_text(src, encoding="utf-8")
    print("  [5/5] 已让 G2PW 优雅降级（缺 onnxruntime 不再中断）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    print("=" * 74)
    print(f"给推理端打补丁: {root}")
    print("=" * 74)
    rc = patch_tts(root, args.check)
    rc2 = patch_langsegmenter(root, args.check)
    rc3 = patch_g2pw(root, args.check)
    print()
    print("=" * 74)
    if rc == 0 and rc2 == 0 and rc3 == 0 and not args.check:
        print("""
  补丁完成。现在需要**同时**给权重补版本头：

    python _tools/fix_weight_header.py \\
        --dir SoVITS_weights_v5_full_e32 --apply

  两者缺一不可：
    只补头 -> KeyError: 'lora_rank'（已被本补丁的 1 处理）
    只打补丁 -> 仍按大小猜成 v3 -> BigVGAN 报错（需补头）

  然后重跑合成（**注意 --weights-dir 要指到 e32 目录**）：
    python _tools/cloud_synth_sweep.py \\
        --weights-dir SoVITS_weights_v5_full_e32 --epochs 8,16,24,32

  可选：装 onnxruntime 开启多音字消歧（G2PW）
    pip install onnxruntime
    （g2pW.onnx 635MB 已在仓库里，装了就能用）
""")
    elif rc != 0 or rc2 != 0 or rc3 != 0:
        print("  有补丁未应用，请检查上面的 !! 行")
    print("=" * 74)
    return rc or rc2 or rc3


if __name__ == "__main__":
    sys.exit(main())
