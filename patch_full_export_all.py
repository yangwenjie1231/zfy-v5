# -*- coding: utf-8 -*-
"""修 v5 full-FT 导出：**保存完整模型全部键**，包括冻结的 quantizer。

## 问题
`s2_train_v5_full.py` 继承了 LoRA/冻结训练的简化导出逻辑：

    for key in ckpt:
        if key not in no_grad_names:
            sim_ckpt[key] = ckpt[key].half().cpu()

`no_grad_names` 是所有 `requires_grad=False` 参数。最终 fine-grained 方案中：

    ssl_proj   解冻 -> 会导出
    enc_p      解冻 -> 会导出
    quantizer  冻结 -> **仍被排除**

推理端 `load_state_dict(strict=False)` 遇到缺失 quantizer 会保留随机初始化，
仍不是一个可独立发布的完整权重。

## 正确做法
full-FT 的发布权重必须**自包含**，无论参数是否参与梯度更新，全部保存：

    for key in ckpt:
        sim_ckpt[key] = ckpt[key].half().cpu()

冻结 quantizer 只是“不更新”，不等于“不保存”。保存的是官方底模原值。

## 用法
    python patch_full_export_all.py --root <检出根>
    python patch_full_export_all.py --root <检出根> --check
    python patch_full_export_all.py --root <检出根> --revert
"""
from __future__ import annotations

import argparse
import ast
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REL = "GPT_SoVITS/s2_train_v5_full.py"
MARK = "ZFY_FULL_EXPORT_ALL_PATCH"
OLD = '''            for key in ckpt:
                # if "cfm"not in key:
                #     print(key)
                if key not in no_grad_names:
                    sim_ckpt[key] = ckpt[key].half().cpu()
'''
NEW = '''            # ---- ZFY_FULL_EXPORT_ALL_PATCH ----
            # full-FT 发布权重必须自包含：冻结参数也必须保存。
            # 细粒度方案里 quantizer 保持冻结（防 EMA 漂移），但若按
            # no_grad_names 排除，它会在推理端随机初始化。
            # 冻结 = 不更新；**不等于不导出**。
            for key in ckpt:
                sim_ckpt[key] = ckpt[key].half().cpu()
            # ---- /ZFY_FULL_EXPORT_ALL_PATCH ----
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    p = root / REL
    if not p.exists():
        print(f"!! 找不到 {p}")
        return 1
    src = p.read_text(encoding="utf-8", errors="replace")
    bak = p.with_suffix(".py.bak_zfy_export_all")

    if args.revert:
        if bak.exists():
            shutil.copy2(bak, p)
            print(f"✓ 已从 {bak.name} 还原")
        else:
            print("~ 没有备份，未改")
        return 0

    if MARK in src:
        print(f"✓ 已打过补丁: {p.name}")
        return 0
    if OLD not in src:
        print("!! 找不到精确导出片段；脚本版本可能变化")
        return 1
    out = src.replace(OLD, NEW, 1)
    ast.parse(out)
    if args.check:
        print("✓ 能精确匹配导出逻辑；将改为保存全部 state_dict 键")
        return 0
    if not bak.exists():
        shutil.copy2(p, bak)
    p.write_text(out, encoding="utf-8", newline="\n")
    print(f"✓ 已应用: {p}")
    print("  full-FT 导出现在包含训练与冻结参数（quantizer 不再缺失）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
