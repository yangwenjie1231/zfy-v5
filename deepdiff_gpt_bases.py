# -*- coding: utf-8 -*-
"""深挖两个 GPT 底模的差异，回答"越晚出的底模是否预训练得更好"。

## 已知（前一轮）
  · sha256 各不相同（不是同一文件改名）
  · 结构完全一致：295 键 / (732,512) vocab / 77.61 M 参数
  · 两者 info 都写着 pretrained_s1longer_v2

## 名字里的信息量
  v1 底模: s1bert25hz-**2kh**-longer-epoch=68e-step=50232        -> 2000 小时
  v2 底模: s1bert25hz-**5kh**-longer-epoch=12-step=369668        -> 5000 小时
  s1v3   : s1v3.ckpt（名字里没有数据量信息）

本脚本：
  1. 完整 dump 两个 checkpoint 的 config，逐字段 diff
  2. 对比权重统计（均值/标准差/范数）看是否显著不同
  3. 列出所有能拿到的元信息
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PRE = Path(r"D:\A\zfy-v5\GPT-SoVITS\GPT_SoVITS\pretrained_models")
SRC = [
    ("s1v3.ckpt（v3+ 通用底模）", PRE / "s1v3.ckpt"),
    ("v2 底模（生产微调用的起点）",
     PRE / "gsv-v2final-pretrained" / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"),
    ("zfh-e4.ckpt（生产微调结果）",
     Path(r"D:\A\voiceclone\zfh-models-v2pro\torch_weights\zfh-e4.ckpt")),
]

print("=" * 78)
print("1. 完整元信息 dump")
print("=" * 78)
loaded = {}
for label, p in SRC:
    print(f"\n### {label}")
    if not p.exists():
        print("  不存在")
        continue
    d = torch.load(str(p), map_location="cpu", weights_only=False)
    loaded[label] = d
    print(f"  顶层键: {list(d.keys())}")
    for k in ("info", "version", "epoch", "step"):
        if k in d:
            print(f"  {k}: {d[k]}")
    cfg = d.get("config")
    if isinstance(cfg, dict):
        import json
        print("  config（完整）:")
        print("    " + json.dumps(cfg, ensure_ascii=False,
                                   default=str).replace("\n", "\n    "))

print()
print("=" * 78)
print("2. config 逐字段 diff（s1v3 vs v2 底模）")
print("=" * 78)
a = loaded.get(SRC[0][0], {}).get("config") or {}
b = loaded.get(SRC[1][0], {}).get("config") or {}


def flat(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{prefix}{k}."))
    else:
        out[prefix[:-1]] = d
    return out


fa, fb = flat(a), flat(b)
keys = sorted(set(fa) | set(fb))
same, diff = 0, []
for k in keys:
    va, vb = fa.get(k, "(无)"), fb.get(k, "(无)")
    if va == vb:
        same += 1
    else:
        diff.append((k, va, vb))
print(f"  相同字段: {same}")
print(f"  不同字段: {len(diff)}")
for k, va, vb in diff:
    print(f"    {k}")
    print(f"        s1v3 : {va}")
    print(f"        v2   : {vb}")

print()
print("=" * 78)
print("3. 权重统计对比（判断是否显著不同的训练）")
print("=" * 78)
stats = {}
for label, _ in SRC:
    d = loaded.get(label)
    if not d:
        continue
    sd = d.get("weight") or d.get("state_dict") or {}
    if not isinstance(sd, dict):
        continue
    tot = 0
    ssum = 0.0
    ssq = 0.0
    for v in sd.values():
        if hasattr(v, "numel") and v.is_floating_point():
            n = v.numel()
            tot += n
            ssum += float(v.float().sum())
            ssq += float((v.float() ** 2).sum())
    if tot:
        mean = ssum / tot
        var = ssq / tot - mean ** 2
        stats[label] = (mean, var ** 0.5)
        print(f"  {label[:34]:36s} 均值 {mean:+.6f}  标准差 {var**0.5:.6f}")

print()
print("=" * 78)
print("4. 逐张量差异（s1v3 vs v2 底模）")
print("=" * 78)
if len(stats) >= 2:
    d1 = loaded[SRC[0][0]].get("weight") or {}
    d2 = loaded[SRC[1][0]].get("weight") or {}
    common = [k for k in d1 if k in d2 and hasattr(d1[k], "shape")
              and d1[k].shape == d2[k].shape]
    print(f"  可比张量: {len(common)} / {len(d1)}")
    import torch as T
    rels = []
    for k in common[:12]:
        x, y = d1[k].float(), d2[k].float()
        diff = (x - y).abs().mean().item()
        base = y.abs().mean().item() + 1e-12
        rels.append((k, diff, diff / base))
    for k, ad, rd in rels:
        print(f"    {k[:52]:54s} 绝对差 {ad:.6f}  相对 {rd*100:6.2f}%")
    allrel = []
    for k in common:
        x, y = d1[k].float(), d2[k].float()
        allrel.append(((x - y).abs().mean() /
                       (y.abs().mean() + 1e-12)).item())
    allrel.sort()
    n = len(allrel)
    print()
    print(f"  全部 {n} 个张量的相对差: 中位 {allrel[n//2]*100:.2f}%  "
          f"最大 {allrel[-1]*100:.2f}%")

print()
print("=" * 78)
print("结论")
print("=" * 78)
print("""
  判断"越晚出的底模是否更好"需要看三点：
    1. 训练数据规模（名字里的 2kh/5kh，或 config 里的字段）
    2. 训练步数（文件名里的 step=）
    3. 公式/配方是否变化（info 字段）
""")
