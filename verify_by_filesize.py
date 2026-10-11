# -*- coding: utf-8 -*-
"""用**文件大小**非侵入地验证 enc_p/ssl_proj 是否进了权重。

## 思路
fp16 权重每个参数固定 2 字节。所以：

    Δ文件大小 = Δ参数数 × 2 字节

已知：
  · freeze 版（旧 e32）: 735,663,517 字节
  · 解冻后应多出 enc_p(14.15M) + ssl_proj(1.18M) = 15.33M 参数
  · 预期增量 = 15.33e6 × 2 = 30.66e6 字节 ≈ 30.66 MB

**若实测增量吻合 -> 非侵入地证明解冻生效**（不用读 766 MB 的文件）。

## 为什么这个验证有价值
1. 训练还在跑时就能做（只 `stat`，不读内容 -> 不会读到半个文件）
2. 与"键数 >= 660"是**两个独立证据**，互相印证
3. 能提前发现"解冻没生效"（增量会接近 0）

## 用法
    python _tools/verify_by_filesize.py \
        --freeze SoVITS_weights_v5_full_e32 \
        --nofreeze SoVITS_weights_v5_full_e32_nofreeze
    python _tools/verify_by_filesize.py --root ./GPT-SoVITS ...
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from statistics import median

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 参数数（来自 verify_unfreeze.py 实测）
ENC_P_PARAMS = 14_150_000
SSL_PROJ_PARAMS = 1_180_000
PARAMS_ADDED = ENC_P_PARAMS + SSL_PROJ_PARAMS
BYTES_PER_PARAM = 2          # fp16


def sizes(d: Path) -> dict[int, int]:
    out = {}
    for p in sorted(d.glob("*.pth")):
        m = re.search(r"_e(\d+)", p.stem)
        if m:
            out[int(m.group(1))] = p.stat().st_size
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--freeze", default="SoVITS_weights_v5_full_e32")
    ap.add_argument("--nofreeze",
                    default="SoVITS_weights_v5_full_e32_nofreeze")
    args = ap.parse_args()
    root = Path(args.root).resolve()

    fz, nf = root / args.freeze, root / args.nofreeze
    print("=" * 84)
    print("用文件大小验证 enc_p/ssl_proj 是否进了权重（非侵入）")
    print("=" * 84)

    for tag, d in (("freeze 版", fz), ("nofreeze 版", nf)):
        if not d.is_dir():
            print(f"  !! {tag} 目录不存在: {d}")
            return 1

    sf, sn = sizes(fz), sizes(nf)
    print(f"  freeze  : {len(sf)} 个    epoch {min(sf) if sf else '-'}~"
          f"{max(sf) if sf else '-'}")
    print(f"  nofreeze: {len(sn)} 个    epoch {min(sn) if sn else '-'}~"
          f"{max(sn) if sn else '-'}")

    if not sf or not sn:
        print("  !! 有一边没有权重")
        return 1

    # 用中位数比较（避免个别文件因 metadata 长度不同而抖动）
    mf = int(median(sf.values()))
    mn = int(median(sn.values()))
    # 同 epoch 配对比较（最严谨）
    common = sorted(set(sf) & set(sn))
    pairs = [(e, sf[e], sn[e]) for e in common]

    print()
    print(f"  中位数大小 : freeze {mf:,}  nofreeze {mn:,}")
    print(f"  差值       : {mn - mf:+,} 字节  ({(mn-mf)/1e6:+.2f} MB)")

    if pairs:
        print()
        print(f"  同 epoch 配对比较（{len(pairs)} 对）:")
        print(f"    {'epoch':>6s} {'freeze':>14s} {'nofreeze':>14s} {'差值':>12s}")
        diffs = []
        for e, a, b in pairs[:12]:
            print(f"    {e:6d} {a:14,} {b:14,} {b-a:+12,}")
            diffs.append(b - a)
        med_diff = int(median(diffs))
        print(f"    中位差值: {med_diff:+,} 字节 ({med_diff/1e6:+.2f} MB)")
    else:
        med_diff = mn - mf
        print(f"    （无同名 epoch 可配对，用中位数差）")

    expected = PARAMS_ADDED * BYTES_PER_PARAM
    print()
    print("=" * 84)
    print("判据")
    print("=" * 84)
    print(f"  预期增量 = (enc_p {ENC_P_PARAMS/1e6:.2f}M + ssl_proj "
          f"{SSL_PROJ_PARAMS/1e6:.2f}M) × {BYTES_PER_PARAM} 字节")
    print(f"           = {PARAMS_ADDED:,} × {BYTES_PER_PARAM}")
    print(f"           = {expected:,} 字节 ≈ {expected/1e6:.2f} MB")
    print(f"  实测增量 = {med_diff:,} 字节 ≈ {med_diff/1e6:.2f} MB")
    if expected:
        print(f"  吻合度   = {med_diff/expected*100:.1f}%")
    print()

    ratio = med_diff / expected if expected else 0
    if 0.9 <= ratio <= 1.1:
        print("  ✓ **解冻生效** —— 文件大小增量与 enc_p+ssl_proj 完全吻合")
        print()
        print("    两个独立证据一致：")
        print("      · 文件大小增量 ≈ +30.7 MB（本脚本）")
        print("      · 键数应 >= 660（verify_encp_in_weights.py）")
        print()
        print("    含义：推理时**不需要**再跑 merge_base_encp.py 补底模值。")
    elif abs(med_diff) < expected * 0.3:
        print("  ✗ **解冻未生效** —— 增量接近 0，权重里没有 enc_p")
        print()
        print("    排查：")
        print("      1. 训练进程的环境变量 ZFY_UNFREEZE=1 是否传到子进程")
        print("      2. 配置 freeze_quantizer 是否为 false")
        print("      3. nofreeze 目录是否其实是 freeze 版的重命名")
    else:
        print(f"  ~ 增量与预期不符（比值 {ratio:.2f}）")
        print("    可能：enc_p 只进了部分参数，或另有参数进/出权重")
        print("    -> 需用 verify_encp_in_weights.py 数键数确认")

    print()
    print("=" * 84)
    print("一致性检查（文件是否被并发写入损坏）")
    print("=" * 84)
    vals = list(sn.values())
    lo, hi = min(vals), max(vals)
    print(f"  nofreeze 文件大小范围: {lo:,} ~ {hi:,}")
    print(f"  极差: {hi-lo:,} 字节 ({(hi-lo)/1024:.1f} KB)")
    print()
    if hi - lo < 4096:
        print("  ✓ 极差 < 4 KB -> 只是 pickle 里 metadata 字符串长度不同，健康")
    else:
        print("  !! 极差偏大 -> 可能有文件被截断，建议逐个 torch.load 验证")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    sys.exit(main())
