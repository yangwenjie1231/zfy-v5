# -*- coding: utf-8 -*-
"""验证解冻训练的权重**真的含 enc_p / ssl_proj**。

## 为什么这是唯一的硬判据
`ZFY_UNFREEZE=1` 只是让 `freeze_quantizer=False`，从而
`models.py:1325 set_no_grad(ssl_proj/quantizer/enc_p)` 不执行
-> 三组参数 `requires_grad=True` -> 进优化器。

但**进优化器 ≠ 进权重文件**：
`s2_train_v5_full.py` 导出时按 `requires_grad` 过滤键，
若某处逻辑没走通，权重可能仍然缺 enc_p。
**只有实际读权重文件数键数才算验证。**

## 判据
| | freeze 版（旧 e32） | 解冻版（期望） |
|---|---|---|
| 总键数 | 444 | **>= 660** |
| `enc_p.*` | 0 | **235** |
| `ssl_proj.*` | 0 | **2** |

## 权重文件的头部坑（会踩）
`process_ckpt.savee()` 只在 `lora_rank` 或 `"Pro" in model_version` 时写
2 字节版本头。**全量微调的权重没有头**，直接是 zip（`PK`）。
所以本脚本两种都处理：
  · 前 2 字节是 `PK` -> 直接 torch.load
  · 否则 -> 剥掉 2 字节头再 load

## 用法
    python _tools/verify_encp_in_weights.py --dir SoVITS_weights_v5_full_e32_nofreeze
    python _tools/verify_encp_in_weights.py --dir <旧目录>   # 对照组
"""
from __future__ import annotations

import argparse
import sys
from io import BytesIO
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch


def load_weight(p: Path) -> dict:
    """读 SoVITS 权重，兼容有/无 2 字节版本头。"""
    raw = p.read_bytes()
    if raw[:2] == b"PK":
        return torch.load(BytesIO(raw), map_location="cpu",
                          weights_only=False)
    # 有版本头：剥掉前 2 字节，补回 zip 的 "PK"
    return torch.load(BytesIO(b"PK" + raw[2:]), map_location="cpu",
                      weights_only=False)


def epoch_of(p: Path) -> int:
    import re
    m = re.search(r"_e(\d+)", p.stem)
    return int(m.group(1)) if m else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="SoVITS_weights_v5_full_e32_nofreeze")
    ap.add_argument("--root", default=".")
    ap.add_argument("--max", type=int, default=3,
                    help="最多检查几个（按 epoch 排序取最早的，够验证了）")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    d = root / args.dir
    print("=" * 88)
    print(f"验证权重是否含 enc_p: {d}")
    print("=" * 88)

    if not d.is_dir():
        print(f"  !! 目录不存在")
        print(f"     现有权重目录:")
        for x in sorted(root.glob("SoVITS_weights*")):
            n = len(list(x.glob("*.pth")))
            print(f"       {x.name}  ({n} 个 .pth)")
        return 1

    ps = sorted(d.glob("*.pth"), key=epoch_of)
    print(f"  权重文件: {len(ps)} 个")
    if not ps:
        print("  !! 没有 .pth")
        return 1
    print(f"  epoch 范围: e{epoch_of(ps[0])} ~ e{epoch_of(ps[-1])}")
    print()

    # 检查最早的几个（解冻若生效，第 1 个就该有 enc_p）
    bad = 0
    all_ok = True
    for p in ps[: args.max]:
        try:
            obj = load_weight(p)
        except Exception as e:  # noqa: BLE001
            print(f"  {p.name}: 读取失败 {type(e).__name__}: {str(e)[:70]}")
            bad += 1
            continue

        w = obj.get("weight", obj)
        if not isinstance(w, dict):
            print(f"  {p.name}: 结构异常 {type(w)}")
            bad += 1
            continue

        keys = list(w.keys())
        n_total = len(keys)
        n_encp = sum(1 for k in keys if k.startswith("enc_p"))
        n_ssl = sum(1 for k in keys if k.startswith("ssl_proj"))
        n_quant = sum(1 for k in keys if k.startswith("quantizer"))
        n_cfm = sum(1 for k in keys if "cfm" in k)
        info = obj.get("info", "?") if isinstance(obj, dict) else "?"

        ok = (n_total >= 660) and (n_encp > 0) and (n_ssl > 0)
        all_ok = all_ok and ok
        mark = "✓" if ok else "✗"
        print(f"  {mark} {p.name}")
        print(f"      总键数   {n_total:5d}   (freeze 版 444, 期望 >=660)")
        print(f"      enc_p    {n_encp:5d}   (期望 235)")
        print(f"      ssl_proj {n_ssl:5d}   (期望 2)")
        print(f"      quantizer{n_quant:5d}")
        print(f"      cfm*     {n_cfm:5d}")
        print(f"      info     {info}")
        if not ok:
            bad += 1

    print()
    print("=" * 88)
    print("结论")
    print("=" * 88)
    if all_ok:
        print("  ✓ **解冻成功** —— enc_p / ssl_proj 已在权重里")
        print()
        print("  含义：推理时无需再跑 merge_base_encp.py 补底模值，")
        print("        而是用**已在新语料上微调过**的 enc_p。")
        print()
        print("  下一步：")
        print("    1. 给权重加版本头（否则推理会当 v3 加载 24k BigVGAN）")
        print("       python _tools/fix_weight_header.py --dir "
              f"{args.dir} --apply")
        print("    2. 合成对比（与 v2Pro / e16_merged 用同一套参数）")
        print("    3. 跑 python _tools/eval_from_wav.py 看相似度能否")
        print("       突破 0.7895 这个平台期")
    else:
        print(f"  ✗ 有 {bad} 个权重**不含** enc_p —— 解冻没生效")
        print()
        print("  排查：")
        print("    1. 环境变量是否传进子进程？启动命令里 ZFY_UNFREEZE=1")
        print("       必须在 nohup 之前。检查日志开头有无")
        print("       [ZFY] ZFY_UNFREEZE=1 -> 强制 freeze_quantizer=False")
        print("    2. 配置里 freeze_quantizer 是否为 false？")
        print("       python -c \"import json;print(json.load("
              "open('tmp_full_e32_nofreeze.json'))['model']['freeze_quantizer'])\"")
        print("    3. exp_dir 是否复用了旧目录（会 resume 到 freeze 版）？")
    print("=" * 88)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
