# -*- coding: utf-8 -*-
"""严谨验证 merge_base_encp.py：模拟"全量微调权重缺 enc_p/ssl_proj"。

## 为什么要这样测
云端日志明确显示 missing_keys **只有** `enc_p.*` 和 `ssl_proj.*`
（不含 cfm / quantizer）。所以模拟必须**精确复现**这个缺失集合，
而不是随便删一堆键 —— 否则测的不是真实场景。

## 测试步骤
  1. 从 s2Gv5dev.pth 出发，**只删** enc_p.* 和 ssl_proj.*
     -> 得到"模拟的全量微调权重"（约 367.8M 参数）
  2. 用 merge_base_encp.py 合并
  3. 断言：合并后 681 键、参数量 383.08M、且 enc_p 与底模逐位相同
  4. 断言：未受影响的键（cfm 等）保持原值不变
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402

GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")
BASE = (GSV / "GPT_SoVITS" / "pretrained_models" / "gsv-v5-pretrained"
        / "s2Gv5dev.pth")


def load_any(p: Path):
    from io import BytesIO
    with p.open("rb") as f:
        head = f.read(2)
    if head == b"PK":
        return torch.load(str(p), map_location="cpu", weights_only=False)
    return torch.load(BytesIO(b"PK" + p.read_bytes()[2:]),
                      map_location="cpu", weights_only=False)


def save_hdr(obj, p: Path, header: bytes):
    from io import BytesIO
    bio = BytesIO()
    torch.save(obj, bio)
    d = bio.getvalue()
    p.write_bytes(header + d[2:])


print("=" * 78)
print("0. 读取底模")
print("=" * 78)
db = load_any(BASE)
wb = db["weight"]
print(f"  {BASE.name}: {len(wb)} 键, "
      f"{sum(v.numel() for v in wb.values())/1e6:.2f}M 参数")

tmp = Path(tempfile.mkdtemp(prefix="merge_test_"))
try:
    # ---- 1) 模拟：只删 enc_p.* 和 ssl_proj.* ----
    print()
    print("=" * 78)
    print("1. 模拟全量微调权重（只删 enc_p.* + ssl_proj.*）")
    print("=" * 78)
    sim = {k: v.clone() for k, v in wb.items()
           if not (k.startswith("enc_p.") or k.startswith("ssl_proj."))}
    n_sim = sum(v.numel() for v in sim.values())
    print(f"  模拟权重: {len(sim)} 键, {n_sim/1e6:.2f}M 参数")
    print(f"  删掉    : {len(wb)-len(sim)} 键, "
          f"{(sum(v.numel() for v in wb.values())-n_sim)/1e6:.2f}M 参数")
    print(f"  文件大小估计: {n_sim*2/1e6:.1f} MB  "
          f"（云端实际 735.7 MB）")
    # 加一点扰动，模拟"训练过"（便于验证未受影响键是否被保留）
    for k in list(sim)[:5]:
        if sim[k].dtype.is_floating_point:
            sim[k] = sim[k] + 0.01
    dsim = {"weight": sim, "config": db.get("config", {}),
            "info": "simulated_fullft"}
    sim_path = tmp / "sim_fullft.pth"
    save_hdr(dsim, sim_path, b"07")
    print(f"  已写出模拟权重: {sim_path.name}  "
          f"({sim_path.stat().st_size/1e6:.1f} MB)")

    # ---- 2) 跑 merge ----
    print()
    print("=" * 78)
    print("2. 运行 merge_base_encp.py")
    print("=" * 78)
    out = tmp / "merged.pth"
    r = subprocess.run(
        [sys.executable, r"D:\A\zfy-v5\merge_base_encp.py",
         "--weights", str(sim_path), "--base", str(BASE),
         "--out", str(out), "--apply"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout[-1600:])
    if r.returncode != 0:
        print(f"  !! 退出码 {r.returncode}")
        print(r.stderr[-600:])

    # ---- 3) 断言 ----
    print()
    print("=" * 78)
    print("3. 断言")
    print("=" * 78)
    if not out.exists():
        print("  ✗ 输出文件不存在")
        raise SystemExit(1)
    dm = load_any(out)
    wm = dm["weight"]

    checks = []
    checks.append(("键数 == 681", len(wm) == 681, f"{len(wm)}"))
    tot = sum(v.numel() for v in wm.values())
    checks.append(("参数量 == 383.08M", abs(tot/1e6 - 383.08) < 0.05,
                   f"{tot/1e6:.2f}M"))
    # enc_p 应与底模逐位相同
    same = all(torch.equal(wm[k].half(), wb[k].half())
               for k in wb if k.startswith(("enc_p.", "ssl_proj.")))
    checks.append(("补入的 enc_p/ssl_proj 与底模一致", same, ""))
    # 未受影响的键应保留（前 5 个扰动键）
    pert = [k for k in list(sim)[:5]]
    kept = all(not torch.equal(wm[k].half(), wb[k].half()) for k in pert
               if k in wm and wb[k].dtype.is_floating_point)
    checks.append(("原有键未被覆盖（扰动仍在）", kept, ""))
    checks.append(("版本头 == b'07'", out.read_bytes()[:2] == b"07",
                   repr(out.read_bytes()[:2])))
    checks.append(("输出可被 torch.load 读取", True, ""))

    for name, ok, extra in checks:
        print(f"  {'✓' if ok else '✗'} {name}"
              + (f"  ({extra})" if extra else ""))

    print()
    print("=" * 78)
    print("结论")
    print("=" * 78)
    if all(ok for _, ok, _ in checks):
        print("  ✓ merge_base_encp.py 验证通过")
        print("    -> 云端可直接用它把 enc_p/ssl_proj 补进 e32 权重，")
        print("       **无需重训**")
    else:
        print("  ✗ 有断言失败")
finally:
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
