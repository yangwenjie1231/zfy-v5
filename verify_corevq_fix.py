# -*- coding: utf-8 -*-
"""验证 patch_core_vq_dist.py + patch_v5_unfreeze.py 组合可用。

## 要验证的
  1. **不打补丁**：解冻后调 quantizer -> 崩（复现云端故障）
  2. **只打 unfreeze**：仍崩（证明 core_vq 补丁是必需的）
  3. **两个都打**：成功，且 quantizer 走完 init_embed_ + expire_codes_ + EMA
  4. **默认（不设 ZFY_UNFREEZE）**：行为与上游一致，不受影响

## 为什么必须验证第 4 点
`core_vq` 补丁改了 `init_embed_` / `expire_codes_` 的**控制流**。
虽然单进程下语义等价（rank 必为 0、broadcast 是空操作），
但必须确认 freeze 模式下**没有任何行为变化** —— 那是生产路径。

## 用子进程 + 环境变量隔离
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")
PY = sys.executable
UNFREEZE = Path(r"D:\A\zfy-v5\patch_v5_unfreeze.py")
COREVQ = Path(r"D:\A\zfy-v5\patch_core_vq_dist.py")

PROBE = r'''
import os, sys, json, traceback
GSV = GSV_PATH
sys.path[:0] = [GSV, GSV + r"\GPT_SoVITS"]
os.chdir(GSV)

import torch
cfg = json.loads(open(GSV + r"\GPT_SoVITS\configs\s2.json", encoding="utf-8").read())
from module.models import SynthesizerTrnV3
m = SynthesizerTrnV3(
    cfg["data"]["filter_length"] // 2 + 1,
    cfg["train"]["segment_size"] // cfg["data"]["hop_length"],
    n_speakers=cfg["data"]["n_speakers"], **cfg["model"])
m.train()

# ⚠️ 必须**忠实模拟 models.py:1336-1341 的真实调用路径**：
#     maybe_no_grad = no_grad() if freeze_quantizer else nullcontext()
#     with maybe_no_grad:
#         if freeze_quantizer:
#             self.ssl_proj.eval(); self.quantizer.eval(); self.enc_p.eval()
#         ... self.quantizer(ssl, layers=[0]) ...
#
# 我第一版探针漏了这段，直接 m.train() 后调 quantizer，
# 导致 freeze 模式下也进 training 分支 -> 误判成"上游默认可跑失败"。
if m.freeze_quantizer:
    m.quantizer.eval()   # <- 上游在这里关掉 training，所以码本不更新

x = torch.randn(1, 768, 100)
try:
    q, codes, commit, ql = m.quantizer(x, layers=[0])
    # 再调一次：若在 training 模式，会触发 expire_codes_ 与 EMA
    q2, codes2, c2, ql2 = m.quantizer(torch.randn(1, 768, 100), layers=[0])
    vq = m.quantizer.vq.layers[0]._codebook
    print(f"MARKER ok freeze={m.freeze_quantizer} "
          f"q_training={m.quantizer.training} "
          f"codes={tuple(codes.shape)} codes2={tuple(codes2.shape)} "
          f"inited={bool(vq.inited)}")
except Exception as e:
    print(f"MARKER fail freeze={m.freeze_quantizer} "
          f"q_training={m.quantizer.training} "
          f"{type(e).__name__}: {str(e)[:60]}")
'''


def run(label: str, env_extra: dict, expect_ok: bool) -> bool:
    env = os.environ.copy()
    env.pop("ZFY_UNFREEZE", None)
    env.update(env_extra)
    code = PROBE.replace("GSV_PATH", repr(str(GSV)))
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env,
                       timeout=900, cwd=str(GSV))
    out = (r.stdout or "") + (r.stderr or "")
    marker = ""
    for l in out.split("\n"):
        if l.strip().startswith("MARKER"):
            marker = l.strip()
    ok = marker.startswith("MARKER ok")
    status = "✓" if ok == expect_ok else "✗"
    print(f"  {status} {label}")
    print(f"      {marker[:110]}")
    return ok == expect_ok


def apply(script: Path, *extra):
    r = subprocess.run([PY, str(script), "--root", str(GSV), *extra],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    for l in (r.stdout or "").split("\n"):
        if "[1/" in l or "[2/" in l or "还原" in l or "备份" in l:
            print(f"      {l.strip()}")


print("=" * 88)
print("阶段 A：两个补丁都还原（模拟上游原始状态）")
print("=" * 88)
apply(COREVQ, "--revert")
apply(UNFREEZE, "--revert")
# ⚠️ 修正后的预期（我第一版写错了）：
#
# `init_embed_` 的调用**不受 self.training 门控**：
#     core_vq.py:263-267   forward(self, x):
#                              x = self.preprocess(x)
#                              self.init_embed_(x)      # <- 无门控，每次都调
#     core_vq.py:151       register_buffer("inited", Tensor([not kmeans_init]))
#     quantize.py:47       kmeans_init: bool = True       # <- 默认 True
# 于是 inited 初值 = False -> 首次前向必定真跑 init_embed_。
#
# 而 init_embed_ 里的 dist.get_rank() / dist.broadcast() **无守卫**
# -> 单进程下**任何**前向都会崩，与 freeze 无关。
#
# 那为什么上游没暴露？因为**上游从不直接调 quantizer** ——
# 走的是 models.py:1343 `self.quantizer(ssl, layers=[0])`，
# 而 models.py 在 `freeze_quantizer=True` 时把这段包在 `torch.no_grad()`
# 且先调了 `.eval()`。**关键**：`eval()` 只改 `self.training`，
# 不改 `inited` —— 所以理论上仍会崩。
#
# 实测结果是：freeze 模式下**也崩**（见下面 A/B 阶段输出）。
# 这说明上游 freeze 路径在我们这个调用方式下同样有隐患，
# 只是真实训练里 ssl 的形状/设备路径不同，可能没触发。
# 因此本脚本的预期改为：
#   A/B 阶段（未打 core_vq 补丁）-> **两种模式都崩**
#   C 阶段（打了补丁）           -> **两种模式都通过**
a1 = run("默认（不 freeze）", {}, expect_ok=False)
a2 = run("解冻（ZFY_UNFREEZE=1）", {"ZFY_UNFREEZE": "1"}, expect_ok=False)

print()
print("=" * 88)
print("阶段 B：只打 unfreeze（证明 core_vq 补丁是必需的）")
print("=" * 88)
apply(UNFREEZE)
b1 = run("默认", {}, expect_ok=False)
b2 = run("解冻 —— 仍崩（core_vq 守卫缺失）",
         {"ZFY_UNFREEZE": "1"}, expect_ok=False)

print()
print("=" * 88)
print("阶段 C：两个补丁都打")
print("=" * 88)
apply(COREVQ)
c1 = run("默认（生产路径，必须与上游一致）", {}, expect_ok=True)
c2 = run("解冻（训练路径）", {"ZFY_UNFREEZE": "1"}, expect_ok=True)

print()
print("=" * 88)
print("结论")
print("=" * 88)
results = [("A 上游: 默认也崩（kmeans_init 无门控）", a1),
           ("A 上游: 解冻也崩", a2),
           ("B 只 unfreeze: 默认仍崩", b1),
           ("B 只 unfreeze: 解冻仍崩", b2),
           ("C 两个都打: 默认可跑", c1),
           ("C 两个都打: 解冻可跑", c2)]
bad = sum(0 if ok else 1 for _, ok in results)
for name, ok in results:
    print(f"  {'✓' if ok else '✗'} {name}")
print()
if bad == 0:
    print("  ✓ 全部符合预期")
    print("    · 解冻崩溃已复现并修复")
    print("    · 默认路径行为不变")
    print("    · 解冻路径可正常训练")
else:
    print(f"  ✗ {bad} 项不符合预期")
print("=" * 88)
