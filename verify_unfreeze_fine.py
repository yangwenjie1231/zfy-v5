# -*- coding: utf-8 -*-
"""验证 patch_unfreeze_fine.py —— 三种模式 + 关键：码本不漂移。

## 要验证的
  1. **未设置** `ZFY_UNFREEZE` -> 三组全冻结（与上游一致，零风险）
  2. **`enc_p,ssl_proj`**（推荐）-> 这两组解冻，**quantizer 冻结**
     + **forward 里 quantizer 仍 `.eval()`** <- 这是防漂移的关键
  3. **`all`** -> 三组全解冻（旧行为，码本会漂移）
  4. **未知项** -> 被忽略且有警告

## 为什么第 2 点的 "quantizer.eval()" 最关键
码本靠 **EMA** 更新（`core_vq.py:290 self.embed.data.copy_(...)`），
**与优化器无关**。只有 `.eval()` 能让 `if self.training:` 为 False。
所以"解冻 enc_p/ssl_proj 但 quantizer 保持 eval"才是正确组合。

## 附带验证：真的跑一次 forward，看码本是否变化
这是**端到端**的判据 —— 比只看 requires_grad 更硬。
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
PATCH = next((p for p in (GSV / "_tools" / "patch_unfreeze_fine.py",
                          Path(__file__).resolve().parent
                          / "patch_unfreeze_fine.py") if p.exists()),
             Path("patch_unfreeze_fine.py"))

print(f"  仓库根: {GSV}")
print(f"  补丁  : {PATCH}")

PROBE = r'''
import os, sys, json, contextlib
GSV = GSV_PATH
sys.path[:0] = [GSV, os.path.join(GSV, "GPT_SoVITS")]
os.chdir(GSV)

import torch
cfg = json.loads(open(os.path.join(GSV, "GPT_SoVITS", "configs", "s2.json"),
                      encoding="utf-8").read())
from module.models import SynthesizerTrnV3
m = SynthesizerTrnV3(
    cfg["data"]["filter_length"] // 2 + 1,
    cfg["train"]["segment_size"] // cfg["data"]["hop_length"],
    n_speakers=cfg["data"]["n_speakers"], **cfg["model"])
m.train()

def rg(mod):
    ps = list(mod.parameters())
    return sum(p.numel() for p in ps if p.requires_grad) / 1e6

print(f"FREEZE={m.freeze_quantizer}")
print(f"RG_ssl={rg(m.ssl_proj):.2f}")
print(f"RG_enc={rg(m.enc_p):.2f}")
print(f"RG_q={rg(m.quantizer):.2f}")

# 模拟 models.py:1336-1341 的真实调用路径
_zfy_ts = getattr(m, "_zfy_train_ssl_proj", False)
_zfy_tq = getattr(m, "_zfy_train_quantizer", False)
_zfy_te = getattr(m, "_zfy_train_enc_p", False)
print(f"FLAG_ssl={_zfy_ts} FLAG_q={_zfy_tq} FLAG_enc={_zfy_te}")
if m.freeze_quantizer:
    if not _zfy_ts:
        m.ssl_proj.eval()
    if not _zfy_tq:
        m.quantizer.eval()
    if not _zfy_te:
        m.enc_p.eval()
print(f"Q_TRAINING={m.quantizer.training}")

# ---- 端到端：跑一次 quantizer 前向，看码本是否变化 ----
x = torch.randn(1, 768, 200)
try:
    q, codes, commit, ql = m.quantizer(x, layers=[0])
    cb = m.quantizer.vq.layers[0]._codebook
    before = cb.embed.detach().clone()
    # 再跑几次（码本需要多次 EMA 才明显漂移）
    for _ in range(5):
        m.quantizer(torch.randn(1, 768, 200), layers=[0])
    after = cb.embed.detach().clone()
    delta = float((after - before).abs().max())
    print(f"CBM_DELTA={delta:.8f}")
    print(f"CBM_CHANGED={delta > 1e-6}")
except Exception as e:
    print(f"CBM_ERR={type(e).__name__}_{str(e)[:50]}")
'''


def run(label: str, env_extra: dict) -> dict:
    env = os.environ.copy()
    env.pop("ZFY_UNFREEZE", None)
    env.update(env_extra)
    code = PROBE.replace("GSV_PATH", repr(str(GSV)))
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env,
                       timeout=1800, cwd=str(GSV))
    out = (r.stdout or "") + (r.stderr or "")
    d = {}
    print(f"\n--- {label} ---")
    for l in out.split("\n"):
        s = l.strip()
        # ⚠️ 只解析**纯 k=v** 行，且键允许大小写混合
        #    （`RG_ssl` 含小写 —— 我第一版用 `^[A-Z][A-Z0-9_]*=`
        #      只匹配全大写，把 RG_ssl/RG_enc/RG_q 全漏了）
        if re.match(r"^[A-Za-z][A-Za-z0-9_]*=", s):
            print(f"    {s[:100]}")
            k, v = s.split("=", 1)
            d[k] = v
        elif "[ZFY]" in s:
            print(f"    {s[:100]}")
    return d


print()
print("=" * 88)
print("0. 打补丁")
print("=" * 88)
r = subprocess.run([PY, str(PATCH), "--root", str(GSV)],
                   capture_output=True, text=True, encoding="utf-8",
                   errors="replace")
for l in (r.stdout or "").split("\n"):
    if "[A]" in l or "[B]" in l or "备份" in l or "已打过" in l:
        print(f"      {l.strip()}")

print()
print("=" * 88)
print("1. 三种模式")
print("=" * 88)
d0 = run("未设置（应与上游一致：全冻结）", {})
d1 = run("ZFY_UNFREEZE=enc_p,ssl_proj （推荐）",
         {"ZFY_UNFREEZE": "enc_p,ssl_proj"})
d2 = run("ZFY_UNFREEZE=all （旧行为）", {"ZFY_UNFREEZE": "all"})
d3 = run("ZFY_UNFREEZE=enc_p,bogus （含未知项）",
         {"ZFY_UNFREEZE": "enc_p,bogus"})

print()
print("=" * 88)
print("2. 断言")
print("=" * 88)
checks = [
    ("未设置: freeze_quantizer 保持 True",
     d0.get("FREEZE") == "True", d0.get("FREEZE")),
    ("未设置: 三组全冻结（rg 都为 0）",
     d0.get("RG_ssl") == "0.00" and d0.get("RG_enc") == "0.00"
     and d0.get("RG_q") == "0.00",
     f"ssl={d0.get('RG_ssl')} enc={d0.get('RG_enc')} q={d0.get('RG_q')}"),
    ("未设置: 码本不变（quantizer.eval）",
     d0.get("CBM_CHANGED") == "False", f"delta={d0.get('CBM_DELTA')}"),
    ("推荐模式: enc_p 解冻",
     float(d1.get("RG_enc", "0") or 0) > 10, f"rg={d1.get('RG_enc')}M"),
    ("推荐模式: ssl_proj 解冻",
     float(d1.get("RG_ssl", "0") or 0) > 1, f"rg={d1.get('RG_ssl')}M"),
    ("推荐模式: **quantizer 仍冻结**",
     d1.get("RG_q") == "0.00", f"rg={d1.get('RG_q')}M"),
    ("推荐模式: **quantizer 仍 eval（码本不漂移）**",
     d1.get("Q_TRAINING") == "False", f"training={d1.get('Q_TRAINING')}"),
    ("推荐模式: **端到端验证码本未变**",
     d1.get("CBM_CHANGED") == "False", f"delta={d1.get('CBM_DELTA')}"),
    ("all 模式: 三组全解冻",
     float(d2.get("RG_enc", "0") or 0) > 10
     and float(d2.get("RG_ssl", "0") or 0) > 1,
     f"ssl={d2.get('RG_ssl')} enc={d2.get('RG_enc')}"),
    ("all 模式: quantizer 在 training（**会漂移**）",
     d2.get("Q_TRAINING") == "True", f"training={d2.get('Q_TRAINING')}"),
    ("all 模式: **端到端验证码本确实变了**",
     d2.get("CBM_CHANGED") == "True", f"delta={d2.get('CBM_DELTA')}"),
    ("未知项被忽略（enc_p 仍解冻）",
     float(d3.get("RG_enc", "0") or 0) > 10, f"rg={d3.get('RG_enc')}M"),
]
bad = 0
for name, ok, extra in checks:
    bad += 0 if ok else 1
    print(f"  {'✓' if ok else '✗'} {name}" + (f"  ({extra})" if extra else ""))

print()
print("=" * 88)
if bad == 0:
    print("  ✓ 全部通过")
    print()
    print("  **核心结论（端到端实测）**")
    print(f"    enc_p,ssl_proj 模式: 码本 delta = {d1.get('CBM_DELTA')}  (未变)")
    print(f"    all 模式          : 码本 delta = {d2.get('CBM_DELTA')}  (变了)")
    print("  -> 证明 '冻结 quantizer' 确实能阻止码本漂移，")
    print("     而 'all 模式' 的码本漂移就是 nofreeze 提前终止的根因")
else:
    print(f"  ✗ {bad} 项失败")
print("=" * 88)
sys.exit(1 if bad else 0)
