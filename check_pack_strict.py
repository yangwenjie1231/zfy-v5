# -*- coding: utf-8 -*-
"""严格版打包覆盖检查：**逐目录对比文件数**，找出"只打包了一部分"的目录。

## 为什么要有这个

`check_pack_full.py` 只报了「0 个文件」的目录，于是漏掉了
`GPT_SoVITS/TTS_infer_pack` —— 它本地有 4 个文件，包里只有 1 个
（我只在 CODE_FILES 里列了 `TTS.py`）。

结果云端推理报：
    ModuleNotFoundError: No module named 'TTS_infer_pack.text_segmentation_method'

本脚本改成**比对数量**：本地 N 个 vs 包里 M 个，M < N 就报警。
这样"部分打包"再也不会溜过去。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\A\zfy-v5")
import pack_for_cloud as P  # noqa: E402

GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")
rels = {rel.replace("\\", "/") for _, rel in P.collect()}

# 需要**完整**上云的目录（其余目录如 webui/前端 故意不上）
WATCH = [
    "GPT_SoVITS/AR",
    "GPT_SoVITS/module",
    "GPT_SoVITS/f5_tts",
    "GPT_SoVITS/configs",
    "GPT_SoVITS/prepare_datasets",
    "GPT_SoVITS/text",
    "GPT_SoVITS/eres2net",
    "GPT_SoVITS/BigVGAN",
    "GPT_SoVITS/feature_extractor",
    "GPT_SoVITS/Accel",
    "GPT_SoVITS/TTS_infer_pack",
    "GPT_SoVITS/pretrained_models/gsv-v5-pretrained",
    "GPT_SoVITS/pretrained_models/sv",
    "tools",
    "tools/AP_BWE_main",
    "tools/asr",
    "tools/i18n",
    "tools/uvr5",
]

print("=" * 78)
print("逐目录文件数对比（本地 vs 打包）")
print("=" * 78)
bad = []
for rel in WATCH:
    d = GSV / rel
    if not d.is_dir():
        print(f"  跳过  {rel:44s} （本地不存在）")
        continue
    local = [f for f in d.rglob("*") if f.is_file()
             and "__pycache__" not in f.parts and f.suffix != ".pyc"]
    packed = [r for r in rels if r.startswith(rel + "/")]
    n_l, n_p = len(local), len(packed)
    if n_p == 0:
        tag = "**全缺**"
        bad.append((rel, n_l, n_p, "全缺"))
    elif n_p < n_l:
        tag = f"**部分缺 {n_l - n_p}**"
        bad.append((rel, n_l, n_p, "部分"))
    else:
        tag = "OK"
    print(f"  {tag:18s} {rel:44s} 本地 {n_l:4d}  包内 {n_p:4d}")

print()
print("=" * 78)
print("缺失明细（部分缺的目录里，具体少了哪些文件）")
print("=" * 78)
if not bad:
    print("  无")
for rel, n_l, n_p, kind in bad:
    d = GSV / rel
    local = {str(f.relative_to(GSV)).replace("\\", "/")
             for f in d.rglob("*")
             if f.is_file() and "__pycache__" not in f.parts
             and f.suffix != ".pyc"}
    miss = sorted(local - rels)
    print(f"\n  [{kind}] {rel}  本地 {n_l} / 包内 {n_p}")
    for m in miss[:15]:
        print(f"      - {m}")
    if len(miss) > 15:
        print(f"      … 还有 {len(miss)-15} 个")

print()
print("=" * 78)
print("顶层单文件检查（GPT_SoVITS/*.py）")
print("=" * 78)
for f in sorted((GSV / "GPT_SoVITS").glob("*.py")):
    rel = f"GPT_SoVITS/{f.name}"
    ok = rel in rels
    # 只关心推理/训练必需的
    need = f.name in {
        "sv.py", "s1_train.py", "process_ckpt.py", "utils.py",
        "s2_train_v5_full.py", "s2_train_v3_lora.py", "TTS.py",
    }
    if need or not ok:
        print(f"  {'OK  ' if ok else '缺失'} {rel}")

print()
print("=" * 78)
print("结论")
print("=" * 78)
if bad:
    print(f"  !! {len(bad)} 个目录未完整打包，需要修 pack_for_cloud.py")
    for rel, n_l, n_p, kind in bad:
        print(f"     {kind}: {rel}  ({n_p}/{n_l})")
else:
    print("  ✓ 受监控目录全部完整打包")
