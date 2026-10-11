# -*- coding: utf-8 -*-
"""对比 v5dev 与 v5turbo 两个 SoVITS 底模。

## 已知（从代码读出）

  config.py:
    v5dev   -> s2Gv5dev.pth   (730.88 MiB)
    v5turbo -> s2Gv5turbo.pth (730.90 MiB)
    两者共用同一个 GPT: s1v3.ckpt

  v5_inference.py:
    sampling_defaults: v5turbo -> (4 步, cfg 0.0)
                       v5dev   -> (32 步, cfg 1.30)
    validate_v5_sampling: steps ∈ {4,8,16,32}，cfg ∈ [0,2]
    CFMV5.inference: cfg > 1e-5 时**每步多跑一次无条件前向**
                     -> v5dev 每步 2 次 = 64 次前向
                     -> v5turbo 每步 1 次 = 4 次前向
                     **相差 16 倍**

## 本脚本
比较两个 .pth 的键数/形状/参数量/权重差异，
判断 turbo 是"同架构蒸馏"还是"另一套架构"。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PRE = Path(r"D:\A\zfy-v5\GPT-SoVITS\GPT_SoVITS\pretrained_models\gsv-v5-pretrained")
A = PRE / "s2Gv5dev.pth"
B = PRE / "s2Gv5turbo.pth"


def load(p: Path):
    d = torch.load(str(p), map_location="cpu", weights_only=False)
    if isinstance(d, dict):
        for k in ("weight", "state_dict", "model"):
            if k in d and isinstance(d[k], dict):
                return d[k], {kk: vv for kk, vv in d.items() if kk != k}
        return d, {}
    return {}, {}


print("=" * 78)
print("1. 顶层结构")
print("=" * 78)
for label, p in (("v5dev", A), ("v5turbo", B)):
    if not p.exists():
        print(f"  {label}: 不存在 {p}")
        continue
    d = torch.load(str(p), map_location="cpu", weights_only=False)
    print(f"  {label:8s} 顶层键: {list(d.keys()) if isinstance(d, dict) else type(d)}")
    if isinstance(d, dict):
        for k, v in d.items():
            if k != "weight" and not isinstance(v, dict):
                s = str(v)
                print(f"           {k} = {s[:120]}")

a, ameta = load(A)
b, bmeta = load(B)
if not a or not b:
    print("  加载失败")
    sys.exit(1)

print()
print("=" * 78)
print("2. 键集合对比")
print("=" * 78)
ka, kb = set(a), set(b)
print(f"  v5dev   键数: {len(ka)}")
print(f"  v5turbo 键数: {len(kb)}")
print(f"  仅 v5dev 有  : {len(ka - kb)}  {sorted(ka - kb)[:5]}")
print(f"  仅 v5turbo 有: {len(kb - ka)}  {sorted(kb - ka)[:5]}")

print()
print("=" * 78)
print("3. 参数量与形状一致性")
print("=" * 78)
na = sum(v.numel() for v in a.values() if hasattr(v, "numel"))
nb = sum(v.numel() for v in b.values() if hasattr(v, "numel"))
print(f"  v5dev   参数量: {na/1e6:.2f} M   ({na:,d})")
print(f"  v5turbo 参数量: {nb/1e6:.2f} M   ({nb:,d})")
print(f"  差值: {(nb-na):,d}")

shape_mismatch = []
for k in sorted(ka & kb):
    sa = getattr(a[k], "shape", None)
    sb = getattr(b[k], "shape", None)
    if sa != sb:
        shape_mismatch.append((k, tuple(sa), tuple(sb)))
print(f"  形状不一致的键: {len(shape_mismatch)}")
for k, sa, sb in shape_mismatch[:8]:
    print(f"    {k}")
    print(f"        dev  : {sa}")
    print(f"        turbo: {sb}")

print()
print("=" * 78)
print("4. 权重差异（同形状键）")
print("=" * 78)
common = [k for k in sorted(ka & kb)
          if getattr(a[k], "shape", None) == getattr(b[k], "shape", None)
          and a[k].is_floating_point()]
print(f"  可比张量: {len(common)}")
if common:
    rels = []
    for k in common:
        x, y = a[k].float(), b[k].float()
        rels.append(((x - y).abs().mean() / (y.abs().mean() + 1e-12)).item())
    rels.sort()
    n = len(rels)
    print(f"  相对差: 最小 {rels[0]*100:.2f}%  中位 {rels[n//2]*100:.2f}%  "
          f"最大 {rels[-1]*100:.2f}%")
    print()
    print("  差异最大的 6 个键（最可能体现蒸馏改动）:")
    pairs = []
    for k in common:
        x, y = a[k].float(), b[k].float()
        pairs.append((k, ((x - y).abs().mean() / (y.abs().mean() + 1e-12)).item()))
    pairs.sort(key=lambda t: -t[1])
    for k, r in pairs[:6]:
        print(f"    {k[:60]:62s} {r*100:7.2f}%")

print()
print("=" * 78)
print("5. 结论")
print("=" * 78)
if not shape_mismatch and ka == kb:
    print("""
  两个底模**结构完全相同**（键集合一致、形状全一致、参数量一致）。
  => v5turbo 是 v5dev 同架构的**步数蒸馏版**（step distillation），
     不是另一套网络。

  这也解释了为什么两者文件大小几乎一样（730.88 vs 730.90 MiB）——
     参数量相同，只是权重值不同。

  实际差异来自两处：
    1. CFM 权重不同（turbo 被训练成 4 步即可收敛）
    2. 采样配置不同：
         v5dev   32 步 + cfg 1.30 -> 每步 2 次前向 = 64 次
         v5turbo  4 步 + cfg 0.0  -> 每步 1 次前向 =  4 次
       **前向量相差 16 倍**
""")
else:
    print("  两者结构**不完全相同**，需进一步看 shape_mismatch 详情。")
