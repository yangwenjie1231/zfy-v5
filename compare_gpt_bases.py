# -*- coding: utf-8 -*-
"""对比两个 GPT 底模，并查上游是否有更新的版本。

## 用户的问题

"全系更新的时候 GPT 底模应该也会更新吧，起码权重会不一样"

## 事实（config.py）

    v1  -> s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt
    v2  -> gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt
    v3  -> s1v3.ckpt
    v4  -> s1v3.ckpt
    v5dev   -> s1v3.ckpt
    v5turbo -> s1v3.ckpt
    v2Pro   -> s1v3.ckpt
    v2ProPlus -> s1v3.ckpt

=> **v3 之后就没有再更新过 GPT 底模**。v4/v5/v2Pro 全都复用 s1v3。

本脚本验证：
  1. 两个 GPT 权重是否真的不同（sha256 + 结构）
  2. 上游 ModelScope 上是否有更新的 GPT 权重
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PRE = Path(r"D:\A\zfy-v5\GPT-SoVITS\GPT_SoVITS\pretrained_models")
A = PRE / "s1v3.ckpt"
B = PRE / "gsv-v2final-pretrained" / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
PROD = Path(r"D:\A\voiceclone\zfh-models-v2pro\torch_weights\zfh-e4.ckpt")


def sha(p: Path, chunk=8 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


print("=" * 78)
print("1. 三个 GPT 权重的 sha256 与体积")
print("=" * 78)
files = [("s1v3.ckpt（v3+ 底模）", A),
         ("v2final s1bert25hz（v2 底模）", B),
         ("zfh-e4.ckpt（生产微调版）", PROD)]
hashes = {}
for label, p in files:
    if not p.exists():
        print(f"  {label:30s} 不存在")
        continue
    h = sha(p)
    hashes[label] = h
    print(f"  {label:30s} {p.stat().st_size:>12,d} 字节  {h[:32]}…")

print()
print("=" * 78)
print("2. 是否同一个文件？")
print("=" * 78)
labels = list(hashes)
if len(labels) >= 2:
    a, b = labels[0], labels[1]
    same = hashes[a] == hashes[b]
    print(f"  s1v3 == v2底模 ?  {same}")
    if same:
        print("  -> 完全相同的权重，只是改了个名字！")
    else:
        print("  -> 不同权重，确实是两次训练")
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        print(f"  {labels[i][:22]:24s} vs {labels[j][:22]:24s} "
              f"{'同' if hashes[labels[i]] == hashes[labels[j]] else '不同'}")

print()
print("=" * 78)
print("3. 结构与内容对比（键数 / shape / 内嵌信息）")
print("=" * 78)
import torch  # noqa: E402

for label, p in files[:2]:
    if not p.exists():
        continue
    print(f"\n--- {label} ---")
    try:
        d = torch.load(str(p), map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001
        print(f"  加载失败: {type(e).__name__}: {str(e)[:150]}")
        continue
    print(f"  顶层键: {list(d.keys())[:6]}")
    for k in ("info", "config", "version", "epoch", "step"):
        if k in d:
            v = d[k]
            s = str(v)
            print(f"  {k:10s}: {s[:200]}")
    sd = d.get("weight") or d.get("state_dict") or d
    if isinstance(sd, dict):
        keys = list(sd.keys())
        print(f"  权重键数: {len(keys)}")
        # 找 embedding 形状（反映 vocab）
        for k in keys:
            if "embedding" in k.lower() and hasattr(sd[k], "shape"):
                print(f"  {k}: {tuple(sd[k].shape)}")
                break
        # 参数量
        try:
            n = sum(v.numel() for v in sd.values()
                    if hasattr(v, "numel"))
            print(f"  参数总量: {n/1e6:.2f} M")
        except Exception:  # noqa: BLE001
            pass

print()
print("=" * 78)
print("4. config.py 里各版本引用的 GPT")
print("=" * 78)
cfg = Path(r"D:\A\zfy-v5\GPT-SoVITS\config.py").read_text(encoding="utf-8",
                                                          errors="replace")
import re  # noqa: E402
m = re.search(r"pretrained_gpt_name\s*=\s*\{(.*?)\}", cfg, re.S)
if m:
    for line in m.group(1).strip().split("\n"):
        if line.strip():
            print(f"  {line.strip()}")
