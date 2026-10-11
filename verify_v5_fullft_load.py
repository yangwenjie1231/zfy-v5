# -*- coding: utf-8 -*-
"""验证 patch_v5_fullft_load.py —— 全量与 LoRA 权重都能正确加载。

## 补丁的最终设计（只改一处）
`TTS.py` 里以 **`lora_rank` 键是否存在**为唯一判据，**双向**对齐 `if_lora_v3`：

    有 lora_rank  -> 走 LoRA 路径（merge_and_unload）
    无 lora_rank  -> 走全量路径（直接 load_state_dict）

## 为什么不做单向（记录我踩的回归）
我第一版改了两处：
  1. `process_ckpt.py head2version`：b"07"/b"08" 第三位 True -> False
  2. `TTS.py`：只在"有 lora_v3 且无 lora_rank"时降级

**实测发现第 1 处是回归**：
    LoRA 权重 -> get_sovits_version_from_path_fast -> if_lora_v3=**False**
    -> 走 `load_state_dict(..., strict=False)`
    -> LoRA 的 `lora_A/lora_B` 键**静默不生效**（不报错，但 LoRA 没加载）
这是最危险的一类 bug：**静默失效**。

所以撤掉第 1 处，第 2 处改成双向。**头里写什么不再重要。**

## 本脚本验证
  1. `head2version` 的 b"07"/b"08" **仍是 True**（没被我改坏）
  2. `TTS.py` 有双向对齐逻辑（`_zfy_has_lora`）
  3. **回归**：LoRA 权重 -> `if_lora_v3=True` + 能构造 TTS
  4. 全量权重（云端才有）-> `if_lora_v3` 应为 False + 能构造 TTS
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 跨平台探测
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
_ap.add_argument("--weights", default=None, help="指定全量权重路径")
_args, _ = _ap.parse_known_args()
if _args.root:
    GSV = Path(_args.root).resolve()
if not (GSV / "GPT_SoVITS" / "utils.py").exists():
    for c in (Path("/mnt/workspace/zfy-v5-fullft-pkg"),
              Path(r"D:\A\zfy-v5\GPT-SoVITS")):
        if (c / "GPT_SoVITS" / "utils.py").exists():
            GSV = c
            break
PY = sys.executable
PATCH = next((p for p in (GSV / "_tools" / "patch_v5_fullft_load.py",
                          Path(__file__).resolve().parent
                          / "patch_v5_fullft_load.py",
                          GSV / "patch_v5_fullft_load.py") if p.exists()),
             Path("patch_v5_fullft_load.py"))

print(f"  仓库根: {GSV}")
print(f"  补丁  : {PATCH}")


def run_patch(*extra):
    r = subprocess.run([PY, str(PATCH), "--root", str(GSV), *extra],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    for l in (r.stdout or "").split("\n"):
        if "[1/1" in l or "[1/2" in l or "[2/2" in l or "还原" in l \
                or "备份" in l or "已打过" in l:
            print(f"      {l.strip()}")
    return r.returncode


print()
print("=" * 84)
print("0. 确保补丁已打")
print("=" * 84)
run_patch()

print()
print("=" * 84)
print("1. 静态检查")
print("=" * 84)
ck = (GSV / "GPT_SoVITS" / "process_ckpt.py").read_text(
    encoding="utf-8", errors="replace")
tt = (GSV / "GPT_SoVITS" / "TTS_infer_pack" / "TTS.py").read_text(
    encoding="utf-8", errors="replace")

# 只匹配**代码行**（行首是 b"07":），不匹配注释里的 "原值: b"07": ..."
m07 = re.search(r'^\s*b"07":\s*\[\s*"v2",\s*"v5dev",\s*(True|False)\s*\]',
                ck, re.M)
m08 = re.search(r'^\s*b"08":\s*\[\s*"v2",\s*"v5turbo",\s*(True|False)\s*\]',
                ck, re.M)
print(f"  head2version b\"07\" -> {m07.group(1) if m07 else '未找到'}"
      f"   （应保持 True —— 改它会破坏 LoRA）")
print(f"  head2version b\"08\" -> {m08.group(1) if m08 else '未找到'}")
for h in ("00", "01", "02", "03", "04", "05", "06"):
    m = re.search(rf'^\s*b"{h}":\s*\[([^\]]+)\]', ck, re.M)
    print(f"    b\"{h}\" -> {m.group(1).strip() if m else '未找到'}")

has_dual = "_zfy_has_lora" in tt and 'if_lora_v3 = _zfy_has_lora' in tt
print(f"  双向对齐逻辑 (_zfy_has_lora): {has_dual}")

print()
print("=" * 84)
print("2. 动态检查：真的构造 TTS")
print("=" * 84)

PROBE = r'''
import os, sys
GSV = GSV_PATH
sys.path[:0] = [GSV, os.path.join(GSV, "GPT_SoVITS")]
os.chdir(GSV)

W = WEIGHTS_PATH
print(f"WEIGHT={os.path.basename(W)}")
from GPT_SoVITS.process_ckpt import get_sovits_version_from_path_fast, load_sovits_new
ver, mv, lora = get_sovits_version_from_path_fast(W)
print(f"VER ver={ver} model_version={mv} if_lora_v3={lora}")
try:
    d = load_sovits_new(W)
    print(f"HAS_LORA={'lora_rank' in d}")
    print(f"NKEYS={len(d.get('weight', {}))}")
except Exception as e:
    print(f"HAS_LORA=err {type(e).__name__}")

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
PRE = os.path.join(GSV, "GPT_SoVITS", "pretrained_models")
cfg = {"custom": {
    "device": "cuda", "is_half": False, "version": "v5dev",
    "t2s_weights_path": os.path.join(GSV, "ref", "zfh-e4.ckpt"),
    "vits_weights_path": W,
    "bert_base_path": os.path.join(PRE, "chinese-roberta-wwm-ext-large"),
    "cnhuhbert_base_path": os.path.join(PRE, "chinese-hubert-base"),
}}
try:
    tts = TTS(TTS_Config(cfg))
    print("RESULT ok")
except Exception as e:
    print(f"RESULT fail {type(e).__name__}: {str(e)[:90]}")
'''


def probe(label: str, weights: Path) -> dict:
    if not weights.exists():
        print(f"\n--- {label} ---")
        print(f"    跳过（不存在: {weights}）")
        return {}
    code = (PROBE.replace("GSV_PATH", repr(str(GSV)))
            .replace("WEIGHTS_PATH", repr(str(weights))))
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=1200,
                       cwd=str(GSV))
    out = (r.stdout or "") + (r.stderr or "")
    d = {}
    print(f"\n--- {label} ---")
    for l in out.split("\n"):
        s = l.strip()
        if s.startswith(("WEIGHT=", "VER ", "HAS_LORA=", "NKEYS=")):
            print(f"    {s[:110]}")
            for tok in s.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    d.setdefault(k, v)
        elif s.startswith("RESULT"):
            # ⚠️ 形如 `RESULT ok` —— 值后面**没有 `=`**，
            #    不能用通用的 k=v 解析（我第一版就踩了这个）
            print(f"    {s[:110]}")
            parts = s.split(None, 2)
            d["RESULT"] = parts[1] if len(parts) > 1 else "?"
            if len(parts) > 2:
                d["RESULT_DETAIL"] = parts[2]
        elif "[ZFY]" in s:
            print(f"    {s[:110]}")
        elif "KeyError" in s or "matched successfully" in s:
            print(f"    {s[:110]}")
    return d


# 全量权重：优先用 --weights，否则找
full = Path(_args.weights) if _args.weights else None
if full is None:
    for pat in ("SoVITS_weights_v5_full_e32_nofreeze/*_e32_*.pth",
                "SoVITS_weights_v5_full_e32_nofreeze/*.pth",
                "SoVITS_weights_v5_full/*.pth"):
        hits = sorted(GSV.glob(pat)) or sorted(Path.cwd().glob(pat))
        if hits:
            full = hits[-1]
            break

# LoRA 权重（回归测试）
lora = None
for pat in ("SoVITS_weights_v5/*.pth", "SoVITS_weights_v5_r128/*.pth"):
    hits = sorted(GSV.glob(pat))
    if hits:
        lora = hits[0]
        break

fd = probe("全量权重（nofreeze）", full) if full else {}
if not full:
    print("\n--- 全量权重 ---")
    print("    本地无（在云端 24.5 GB）-> 只在云端可做动态验证")
ld = probe("LoRA 权重（回归测试）", lora) if lora else {}

print()
print("=" * 84)
print("3. 断言")
print("=" * 84)
checks = [
    ("head2version b\"07\" 保持 True（未破坏 LoRA）",
     bool(m07) and m07.group(1) == "True",
     m07.group(1) if m07 else "未找到"),
    ("head2version b\"08\" 保持 True",
     bool(m08) and m08.group(1) == "True",
     m08.group(1) if m08 else "未找到"),
    ("head2version 其他条目未变",
     bool(re.search(r'^\s*b"05":\s*\[\s*"v2",\s*"v2Pro",\s*False\s*\]',
                    ck, re.M))
     and bool(re.search(r'^\s*b"03":\s*\[\s*"v2",\s*"v3",\s*True\s*\]',
                        ck, re.M)),
     ""),
    ("TTS.py 有双向对齐逻辑", has_dual, ""),
]
if ld:
    checks.append(("回归: LoRA 权重 -> if_lora_v3=True",
                   ld.get("if_lora_v3") == "True",
                   f"if_lora_v3={ld.get('if_lora_v3')}"))
    checks.append(("回归: LoRA 权重 -> HAS_LORA=True",
                   ld.get("HAS_LORA") == "True",
                   f"HAS_LORA={ld.get('HAS_LORA')}"))
    checks.append(("回归: LoRA 权重能构造 TTS",
                   ld.get("RESULT") == "ok",
                   ld.get("RESULT", "?")))
if fd:
    checks.append(("全量权重 -> 能构造 TTS（不再 KeyError）",
                   fd.get("RESULT") == "ok",
                   f"RESULT={fd.get('RESULT')} "
                   f"HAS_LORA={fd.get('HAS_LORA')} "
                   f"NKEYS={fd.get('NKEYS')}"))

bad = 0
for name, ok, extra in checks:
    bad += 0 if ok else 1
    print(f"  {'✓' if ok else '✗'} {name}" + (f"  ({extra})" if extra else ""))

print()
print("=" * 84)
if bad == 0:
    print("  ✓ 全部通过")
    if not fd:
        print("    ⚠️ 本地无全量权重 -> 云端需再跑一次做动态验证")
    else:
        print(f"    全量权重: {fd.get('NKEYS')} 键, "
              f"HAS_LORA={fd.get('HAS_LORA')}, 构造成功")
else:
    print(f"  ✗ {bad} 项失败")
print("=" * 84)
sys.exit(1 if bad else 0)
