# -*- coding: utf-8 -*-
"""补齐检查：下载并检验 **v1 底模**（第一代 GPT）。

## 为什么

我先前只给了 A（s1v3）/ B（v2 底模）两条路线，
但上游官方仓库 `AI-ModelScope/GPT-SoVITS` 其实有 **3 个** GPT 底模：

  s1bert25hz-**2kh**-longer-epoch=68e-step=50232.ckpt   <- v1，本地**从未下载**
  gsv-v2final-pretrained/s1bert25hz-**5kh**-longer-...  <- v2
  s1v3.ckpt                                            <- v3+

## 硬门槛（必须先过）

`config.py` 里 v3/v4/v5/v2Pro 全用 s1v3，而 s1v3 的
`model.phoneme_vocab_size = 732`。
但 v1 时代的 `configs/s1longer.yaml` 写的是 **512**。

若 v1 底模真是 512，就与 v5 的文本前端（732 音素）**不兼容**，
直接排除，不用浪费时间训练。

## 本脚本
  1. 从 ModelScope 下载 v1 底模（若本地没有）
  2. 检查：键数 / 形状 / phoneme_vocab_size / 参数量 / info / config
  3. 与 s1v3、v2 底模逐项对比，给出兼容性结论
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PRE = Path(r"D:\A\zfy-v5\GPT-SoVITS\GPT_SoVITS\pretrained_models")
V1_REL = "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
V1 = PRE / V1_REL
V2 = PRE / "gsv-v2final-pretrained" / \
    "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
V3 = PRE / "s1v3.ckpt"
RID = "AI-ModelScope/GPT-SoVITS"


def download_v1() -> bool:
    if V1.exists():
        print(f"  已存在: {V1}  ({V1.stat().st_size/1e6:.1f} MB)")
        return True
    print(f"  本地没有，从 {RID} 下载 …")
    try:
        from modelscope import snapshot_download
        p = snapshot_download(
            RID, allow_patterns=[V1_REL],
            local_dir=str(PRE.parent.parent / "_gpt_bases"))
        got = Path(p) / V1_REL
        if got.exists():
            V1.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(got, V1)
            print(f"  下载完成 -> {V1}  ({V1.stat().st_size/1e6:.1f} MB)")
            return True
        print(f"  下载后找不到文件: {p}")
        return False
    except Exception as e:  # noqa: BLE001
        print(f"  下载失败: {type(e).__name__}: {str(e)[:300]}")
        return False


print("=" * 78)
print("1. 获取 v1 底模")
print("=" * 78)
ok = download_v1()

print()
print("=" * 78)
print("2. 三个底模的架构对比")
print("=" * 78)
BASES = [("v1 底模（2kh）", V1), ("v2 底模（5kh）", V2), ("s1v3（v3+ 通用）", V3)]
info = {}
for label, p in BASES:
    print(f"\n--- {label} ---")
    if not p.exists():
        print(f"  不存在: {p}")
        continue
    try:
        d = torch.load(str(p), map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001
        print(f"  加载失败: {type(e).__name__}: {str(e)[:150]}")
        continue
    sd = d.get("weight") or d.get("state_dict") or {}
    keys = list(sd.keys()) if isinstance(sd, dict) else []
    n = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
    # 关键：音素 embedding 的形状
    vocab_shape = None
    for k in keys:
        if "ar_text_embedding" in k and "weight" in k and hasattr(sd[k], "shape"):
            vocab_shape = tuple(sd[k].shape)
            break
    cfg = d.get("config") or {}
    pv = (cfg.get("model") or {}).get("phoneme_vocab_size")
    mv = (cfg.get("model") or {}).get("vocab_size")
    print(f"  文件大小   : {p.stat().st_size/1e6:.1f} MB")
    print(f"  info       : {d.get('info')}")
    print(f"  权重键数   : {len(keys)}")
    print(f"  参数量     : {n/1e6:.2f} M")
    print(f"  音素emb形状: {vocab_shape}   <- (phoneme_vocab, dim)")
    print(f"  config.phoneme_vocab_size: {pv}")
    print(f"  config.vocab_size        : {mv}")
    tc = cfg.get("train") or {}
    print(f"  config.train: epochs={tc.get('epochs')} batch={tc.get('batch_size')}")
    info[label] = {"keys": len(keys), "n": n, "vocab_shape": vocab_shape,
                   "pv": pv, "info": d.get("info")}

print()
print("=" * 78)
print("3. 兼容性判定（v5 需要 phoneme_vocab=732）")
print("=" * 78)
print(f"  {'底模':22s} {'音素emb':>14s} {'phoneme_vocab':>14s}  判定")
for label, _ in BASES:
    i = info.get(label)
    if not i:
        print(f"  {label:22s} {'--':>14s} {'--':>14s}  缺失")
        continue
    pv = i["pv"]
    shape_ok = i["vocab_shape"] is not None and i["vocab_shape"][0] == 732
    pv_ok = pv == 732
    verdict = "兼容" if (shape_ok or pv_ok) else f"**不兼容**"
    print(f"  {label:22s} {str(i['vocab_shape']):>14s} {str(pv):>14s}  {verdict}")

print()
print("=" * 78)
print("4. 键集合是否一致（能否直接载入同一套代码）")
print("=" * 78)
v3k = None
for label, p in BASES:
    if not p.exists():
        continue
    d = torch.load(str(p), map_location="cpu", weights_only=False)
    sd = d.get("weight") or {}
    ks = set(sd.keys())
    if label.startswith("s1v3"):
        v3k = ks
for label, p in BASES:
    if not p.exists() or v3k is None:
        continue
    d = torch.load(str(p), map_location="cpu", weights_only=False)
    sd = d.get("weight") or {}
    ks = set(sd.keys())
    print(f"  {label:22s} 键数 {len(ks):4d}  "
          f"与 s1v3 相比：多 {len(ks-v3k)} 少 {len(v3k-ks)}")
    if ks - v3k:
        print(f"      仅此底模有: {sorted(ks-v3k)[:4]}")
    if v3k - ks:
        print(f"      仅 s1v3 有: {sorted(v3k-ks)[:4]}")
