#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""应急修复：把底模的 enc_p/ssl_proj/quantizer 合并进全量微调权重。

## 为什么需要
全量微调权重**缺了 15.23M 参数**（enc_p + ssl_proj），因为它们被
`freeze_quantizer=True` 设成了 requires_grad=False，导出时被
`s2_train_v5_full.py:406` 的 `if key not in no_grad_names` 跳过。

推理时 `load_state_dict(strict=False)` **静默保留随机初始化**
-> `decode_encp()` 先验错乱 -> 「让我想想。」11.40s（正常 1.2s）。

## 本脚本
从底模 `s2Gv5dev.pth` 取出缺失的键，**合并**进我们的权重文件：
  · 保留我们训练好的键（cfm/flow/dec/ref_enc 等）
  · 补上底模的 enc_p / ssl_proj / quantizer
  · 写出**新的**权重文件（不覆盖原件）

注意：这样补的是**底模原值**（freeze 期间没被训练）。
对"时长恢复正常"足够；若要 enc_p 也适配新语料，需重训（freeze_quantizer=false）。

用法：
    python merge_base_encp.py --weights <训练权重.pth> --base <底模.pth> \\
        --out <输出.pth> [--apply]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def ensure_utils_importable(base_hint: Path | None = None) -> None:
    """把 GPT_SoVITS 目录加进 sys.path —— torch.load 需要能 import `utils`。

    ## 为什么
    权重里的 `config` 是 `utils.HParams` 对象，pickle 会记录
    `utils\\nHParams`。反序列化时 Python 必须能 import 名为 `utils` 的模块，
    否则报：
        ModuleNotFoundError: No module named 'utils'

    仓库里 `GPT_SoVITS/utils.py` 就是它。TTS 推理时能正常加载，是因为
    sys.path 里有 `GPT_SoVITS`；而本脚本独立运行，需要自己加。

    ## 注意
    这个 `utils` 是 **GPT_SoVITS/utils.py**，不是 `AR/models/utils.py`
    也不是 `text/g2pw/utils.py` —— 顺序必须对，否则 import 到错的。
    """
    cands: list[Path] = []
    if base_hint is not None:
        # 从底模路径往上找 GPT_SoVITS
        for p in [base_hint, *base_hint.parents]:
            if (p / "utils.py").exists() and p.name == "GPT_SoVITS":
                cands.append(p)
                break
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents, Path.cwd(), *Path.cwd().parents]:
        g = p / "GPT_SoVITS"
        if (g / "utils.py").exists():
            cands.append(g)
        if p.name == "GPT_SoVITS" and (p / "utils.py").exists():
            cands.append(p)
    for c in ("/mnt/workspace/zfy-v5-fullft-pkg/GPT_SoVITS",):
        cp = Path(c)
        if (cp / "utils.py").exists():
            cands.append(cp)

    seen = set()
    for c in cands:
        s = str(c)
        if s in seen:
            continue
        seen.add(s)
        # 插到最前面，保证优先命中 GPT_SoVITS/utils.py
        sys.path.insert(0, s)
        print(f"  sys.path 加入: {s}")
    if not seen:
        print("  !! 找不到 GPT_SoVITS/utils.py —— torch.load 可能失败")
        print("     可手动：export PYTHONPATH=$PWD/GPT_SoVITS:$PYTHONPATH")


def load_any(p: Path):
    """用 load_sovits_new 的逻辑读（它会补 PK 头），失败则退回 torch.load。"""
    import torch
    from io import BytesIO
    with p.open("rb") as f:
        head = f.read(2)
    if head == b"PK":
        return torch.load(str(p), map_location="cpu", weights_only=False)
    data = b"PK" + p.read_bytes()[2:]
    return torch.load(BytesIO(data), map_location="cpu", weights_only=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="训练出的权重")
    ap.add_argument("--base", required=True, help="底模 s2Gv5dev.pth")
    ap.add_argument("--out", required=True, help="输出路径")
    ap.add_argument("--apply", action="store_true", help="实际写出（默认只报告）")
    ap.add_argument("--header", default="07",
                    help="版本头（v5dev=07），默认 07")
    args = ap.parse_args()

    W, B, O = Path(args.weights), Path(args.base), Path(args.out)
    for p in (W, B):
        if not p.exists():
            print(f"!! 不存在: {p}")
            return 1

    print("=" * 78)
    print("合并底模的缺失键")
    print("=" * 78)
    print(f"  训练权重: {W.name}  ({W.stat().st_size/1e6:.1f} MB)")
    print(f"  底模    : {B.name}  ({B.stat().st_size/1e6:.1f} MB)")

    # torch.load 需要能 import `utils`（权重里的 config 是 utils.HParams）
    ensure_utils_importable(B)

    dw = load_any(W)
    db = load_any(B)
    ww, wb = dw["weight"], db["weight"]

    missing = [k for k in wb if k not in ww]
    extra = [k for k in ww if k not in wb]

    print()
    print(f"  训练权重键数: {len(ww)}")
    print(f"  底模键数    : {len(wb)}")
    print(f"  **缺失**    : {len(missing)}")

    # 按前缀分组统计
    from collections import Counter
    grp = Counter(k.split(".")[0] for k in missing)
    print(f"  缺失分组    : {dict(grp)}")

    # 形状校验
    bad = [k for k in missing if ww.get(k) is not None
           and ww[k].shape != wb[k].shape]
    if bad:
        print(f"  !! {len(bad)} 个键形状不匹配，中止")
        for k in bad[:5]:
            print(f"     {k}: {tuple(ww[k].shape)} vs {tuple(wb[k].shape)}")
        return 1

    n_missing = sum(wb[k].numel() for k in missing)
    print(f"  补入参数量  : {n_missing/1e6:.2f}M")
    print(f"  合并后总参数: "
          f"{(sum(v.numel() for v in ww.values()) + n_missing)/1e6:.2f}M")

    if extra:
        print(f"  额外键（训练有、底模无）: {len(extra)}  "
              f"示例 {extra[:3]}")

    if not args.apply:
        print()
        print("  （未加 --apply，只报告）")
        return 0

    # 合并：训练权重优先，缺失的用底模
    merged = dict(ww)
    for k in missing:
        merged[k] = wb[k].half()
    dw["weight"] = merged
    dw["info"] = (str(dw.get("info", "")) + " + base enc_p/ssl_proj merged").strip()

    import torch
    from io import BytesIO
    bio = BytesIO()
    torch.save(dw, bio)
    data = bio.getvalue()
    head = args.header.encode("ascii")[:2]
    O.parent.mkdir(parents=True, exist_ok=True)
    with O.open("wb") as f:
        f.write(head + data[2:])

    print()
    print(f"  已写出: {O}")
    print(f"  大小  : {O.stat().st_size/1e6:.1f} MB")
    print(f"  头    : {O.read_bytes()[:2]!r}")

    # 复核
    dv = load_any(O)
    still = [k for k in wb if k not in dv["weight"]]
    print(f"  复核缺失键: {len(still)}  （应为 0）")
    return 0 if not still else 1


if __name__ == "__main__":
    sys.exit(main())
