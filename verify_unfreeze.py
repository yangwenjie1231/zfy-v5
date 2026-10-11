# -*- coding: utf-8 -*-
"""验证 patch_v5_unfreeze.py：解冻真的生效，且默认行为不变。

## 要验证的四件事
  1. **默认（不设环境变量）** -> freeze_quantizer 保持 True，
     enc_p/ssl_proj/quantizer 的 requires_grad 全为 False
     —— 与上游**完全一致**（安全底线）
  2. **ZFY_UNFREEZE=1** -> freeze_quantizer 变 False，
     三组参数 requires_grad 全为 True
  3. 解冻后**参数量**增加 15.33M（进优化器）
  4. 解冻后 quantizer 处于 **training 模式**（码本会按 EMA 更新）

## 为什么必须验证第 4 点
`core_vq.py:246` 的码本更新是：
    if self.training:
        self.cluster_size.data.mul_(...)      # EMA
        self.embed.data.copy_(embed_normalized)
而 `models.py:1340` 在 freeze 时会调 `self.quantizer.eval()`。
所以"解冻"必须**同时**保证 quantizer 不在 eval 模式，否则码本不更新。

用子进程跑，才能干净地控制环境变量。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")
PY = sys.executable

PROBE = r'''
import os, sys, json
GSV = GSV_PATH
sys.path.insert(0, GSV)
sys.path.insert(0, GSV + r"\GPT_SoVITS")
os.chdir(GSV)

import torch
from module.models import SynthesizerTrnV3

cfg = json.loads(open(GSV + r"\GPT_SoVITS\configs\s2.json", encoding="utf-8").read())
m = cfg["model"]
# 构造模型（不加载权重，只看 requires_grad 与 training 标志）
model = SynthesizerTrnV3(
    cfg["data"]["filter_length"] // 2 + 1,
    cfg["train"]["segment_size"] // cfg["data"]["hop_length"],
    n_speakers=cfg["data"]["n_speakers"],
    **m,
)

def stat(mod, label):
    ps = list(mod.parameters())
    n = sum(p.numel() for p in ps)
    ng = sum(p.numel() for p in ps if p.requires_grad)
    print(f"  {label:12s} 参数 {n/1e6:7.2f}M  requires_grad {ng/1e6:7.2f}M  "
          f"training={mod.training}")

print(f"  freeze_quantizer = {model.freeze_quantizer}")
stat(model.enc_p, "enc_p")
stat(model.ssl_proj, "ssl_proj")
stat(model.quantizer, "quantizer")

tot = sum(p.numel() for p in model.parameters())
tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  {'合计':12s} 参数 {tot/1e6:7.2f}M  requires_grad {tr/1e6:7.2f}M")
print(f"  MARKER freeze={model.freeze_quantizer} "
      f"trainable={tr/1e6:.2f}M "
      f"quantizer_training={model.quantizer.training}")
'''


def probe(env_extra: dict) -> dict:
    env = os.environ.copy()
    env.pop("ZFY_UNFREEZE", None)
    env.update(env_extra)
    code = PROBE.replace("GSV_PATH", repr(str(GSV)))
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env,
                       timeout=600, cwd=str(GSV))
    out = (r.stdout or "") + (r.stderr or "")
    print("\n".join(f"  {l}" for l in out.strip().split("\n")
                    if l.strip() and "Warning" not in l))
    res = {}
    for line in out.split("\n"):
        if line.strip().startswith("MARKER "):
            parts = line.strip().split()
            # 格式: MARKER freeze=True trainable=367.49M quantizer_training=True
            # ⚠️ trainable 带 'M' 后缀，必须剥掉再转 float
            res = {
                "freeze": parts[1].split("=")[1],
                "trainable": float(parts[2].split("=")[1].rstrip("M")),
                "q_training": parts[3].split("=")[1],
            }
    if r.returncode != 0 and not res:
        print(f"  !! 探针失败 exit={r.returncode}")
    return res


print("=" * 80)
print("1. 打补丁前（上游行为）")
print("=" * 80)
before = probe({})

print()
print("=" * 80)
print("2. 打补丁")
print("=" * 80)
r = subprocess.run([PY, r"D:\A\zfy-v5\patch_v5_unfreeze.py",
                    "--root", str(GSV)],
                   capture_output=True, text=True, encoding="utf-8",
                   errors="replace")
for l in (r.stdout or "").split("\n"):
    if any(k in l for k in ("[1/1]", "[2/2]", "已插入", "已写出", "备份")):
        print(f"  {l.strip()}")

print()
print("=" * 80)
print("3. 打补丁后 —— 默认 vs 解冻")
print("=" * 80)
print("\n--- 默认（不设 ZFY_UNFREEZE）---")
d = probe({})
print("\n--- ZFY_UNFREEZE=1 ---")
u = probe({"ZFY_UNFREEZE": "1"})

print()
print("=" * 80)
print("4. 断言")
print("=" * 80)
checks = []

# 4.1 默认必须与上游一致
checks.append(("默认 freeze=True",
               d.get("freeze") == "True", f"{d.get('freeze')}"))
checks.append(("默认可训练参数 ≈ 367.8M（与上游一致）",
               abs(d.get("trainable", 0) - 367.8) < 1.0,
               f"{d.get('trainable')}M"))
if before:
    checks.append(("默认与打补丁前可训练参数量一致",
                   abs(d.get("trainable", 0) - before.get("trainable", -1)) < 0.01,
                   f"{d.get('trainable')} vs {before.get('trainable')}"))

# 4.2 解冻必须生效
checks.append(("解冻 freeze=False",
               u.get("freeze") == "False", f"{u.get('freeze')}"))
checks.append(("解冻可训练参数增加 ≈ 15.33M",
               abs((u.get("trainable", 0) - d.get("trainable", 0)) - 15.33) < 0.5,
               f"+{u.get('trainable',0)-d.get('trainable',0):.2f}M"))
checks.append(("解冻后 quantizer 处于 training 模式",
               u.get("q_training") == "True", f"{u.get('q_training')}"))
checks.append(("解冻后总可训练 ≈ 383.1M",
               abs(u.get("trainable", 0) - 383.08) < 1.0,
               f"{u.get('trainable')}M"))

bad = 0
for name, ok, extra in checks:
    bad += 0 if ok else 1
    print(f"  {'✓' if ok else '✗'} {name}" + (f"  ({extra})" if extra else ""))

print()
print("=" * 80)
print("结论")
print("=" * 80)
if bad == 0:
    print("  ✓ patch_v5_unfreeze.py 全部验证通过")
    print("    · 默认行为与上游一致（零风险）")
    print("    · ZFY_UNFREEZE=1 真的解冻了 enc_p/ssl_proj/quantizer")
    print("    · 解冻后 quantizer 不在 eval 模式 -> 码本会按 EMA 更新")
else:
    print(f"  ✗ {bad} 项失败")
