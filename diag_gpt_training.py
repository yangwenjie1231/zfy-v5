# -*- coding: utf-8 -*-
"""诊断 GPT 微调是否真的在学 —— 三个疑点逐个验证。

## 疑点
  1. `Epoch 7/7` + `v_num: 3.000` -> 是否 resume 了历史 checkpoint？
  2. `total_loss_step: 1401` -> 是累计值还是平均值？量级正常吗？
  3. 8 个 ckpt 大小完全相同 -> 权重真的更新了吗？

## 关键：ckpt 大小相同 ≠ 没训练
    fp16 权重大小固定，内容不同但字节数相同是**正常的**。
    必须比 **内容 hash** 或 **权重差异**。

## 本脚本（在云端跑）
  1. 算每个 ckpt 的 sha256（前 16 位）—— 全同则说明没训练
  2. 读 ckpt 的 `info` 字段（形如 "GPT-e4"）确认轮次
  3. 对比 e1 与 e8 的权重差异（逐层 max|Δ|）
  4. 检查是否有 resume 痕迹（logs/gptA/ckpt/ 下的历史文件）
  5. 打印 loss 的正确读法
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path("/mnt/workspace/zfy-v5-fullft-pkg")
if not (ROOT / "GPT_SoVITS").is_dir():
    ROOT = Path.cwd()

import torch  # noqa: E402


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


for tag in ("A", "B"):
    d = ROOT / f"GPT_weights_gpt{tag}"
    if not d.is_dir():
        print(f"  {d} 不存在")
        continue
    print("=" * 78)
    print(f"路线 [{tag}]  {d.name}")
    print("=" * 78)
    cks = sorted(d.glob("*.ckpt"),
                 key=lambda p: int("".join(c for c in p.stem.split("-e")[-1]
                                          if c.isdigit()) or 0))
    print(f"  {'文件':22s} {'大小':>12s} {'sha256[:16]':18s} {'info':12s}")
    hashes = {}
    for c in cks:
        h = sha16(c)
        hashes[c.name] = h
        try:
            obj = torch.load(str(c), map_location="cpu", weights_only=False)
            info = str(obj.get("info", "?"))
        except Exception as e:  # noqa: BLE001
            info = f"读取失败({type(e).__name__})"
        print(f"  {c.name:22s} {c.stat().st_size:12d} {h:18s} {info:12s}")

    uniq = len(set(hashes.values()))
    print()
    print(f"  不同 hash 数: {uniq} / {len(hashes)}")
    if uniq == 1:
        print("  !! **所有 ckpt 内容完全相同** -> 权重根本没更新！")
    elif uniq == len(hashes):
        print("  ✓ 每个 ckpt 都不同 -> 权重在更新")

    # e1 vs e8 逐层差异
    if len(cks) >= 2:
        a, b = cks[0], cks[-1]
        print()
        print(f"  逐层对比: {a.name}  vs  {b.name}")
        try:
            da = torch.load(str(a), map_location="cpu",
                            weights_only=False)["weight"]
            db = torch.load(str(b), map_location="cpu",
                            weights_only=False)["weight"]
            diffs = []
            for k in list(da)[:400]:
                if k not in db:
                    continue
                x, y = da[k].float(), db[k].float()
                if x.shape != y.shape:
                    continue
                d = (x - y).abs().max().item()
                diffs.append((k, d))
            if diffs:
                diffs.sort(key=lambda t: -t[1])
                n_zero = sum(1 for _, d in diffs if d == 0.0)
                print(f"    对比了 {len(diffs)} 层；完全相同的层: {n_zero}")
                print("    差异最大的 5 层:")
                for k, d in diffs[:5]:
                    print(f"      {d:.6f}  {k[:70]}")
                mx = diffs[0][1]
                if mx < 1e-9:
                    print("    !! **所有层完全相同** -> 权重没变")
                else:
                    print(f"    ✓ 权重确实变了（最大差 {mx:.6f}）")
        except Exception as e:  # noqa: BLE001
            print(f"    对比失败: {type(e).__name__}: {e}")

    # resume 痕迹
    ck = ROOT / "logs" / f"gpt{tag}" / "ckpt"
    if ck.is_dir():
        fs = sorted(ck.glob("*.ckpt"))
        print()
        print(f"  Lightning 历史 ckpt ({ck.name}/): {len(fs)} 个")
        for f in fs[-5:]:
            print(f"    {f.name}  {f.stat().st_size} 字节")
    print()

print("=" * 78)
print("loss 的正确读法")
print("=" * 78)
print("""
  Lightning 的 total_loss_step / total_loss_epoch 是**累计值**（sum），
  不是平均值。所以 1401 / 2266 这种量级本身不代表"loss 很大"。

  要看**真实 loss**，得除以步数：
      每 epoch 103 步
      total_loss_epoch 2266 / 103 ≈ 22.0   <- 这个才是平均 loss

  对比基准（生产 zfh-e4 微调时）：
      正常 GPT 微调的 loss 应在 1~10 量级
      22 偏大，但**每个 epoch 都在降**就说明在学

  top_3_acc 0.585 是**有意义的指标**：
      表示"正确答案在前 3 的概率"= 58.5%
      随机猜是 3/1025 ≈ 0.3%
      -> 0.585 说明模型**确实在预测语义 token**，不是随机

  ⚠️ 但要注意：s1v3/v2 底模本身就有基础能力，
     0.585 有多少来自底模、多少来自微调，需要对比底模的 acc。
""")
