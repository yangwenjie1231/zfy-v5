# -*- coding: utf-8 -*-
"""确认 GPT 微调的两条底模都在上云清单里，并做一次 dry-run。

## 背景
`--base both` 报「底模缺失 gsv-v2final-pretrained/s1bert25hz-5kh-longer-...ckpt」，
根因是 `pack_for_cloud.py` 的 MODEL_FILES 只列了 s1v3.ckpt，
**v2 底模整个目录都没打包**（本地其实有 148 MB）。

## 顺带修的第二个 bug
`train_gpt.py` 原先在任一底模缺失时直接 `return 1`，
导致 `--base both` 连**本来可跑的路线 A** 也不跑。
已改为"只跳过缺的那条，其余照跑"。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

V5 = Path(r"D:\A\zfy-v5")
sys.path.insert(0, str(V5))
import pack_for_cloud as P  # noqa: E402

rels = {rel.replace("\\", "/") for _, rel in P.collect()}

print("=" * 78)
print("1. 两个 GPT 底模是否都在清单里")
print("=" * 78)
GSV = V5 / "GPT-SoVITS"
for f in P.MODEL_FILES:
    ok = f in rels
    local = GSV / f
    sz = local.stat().st_size / 1e6 if local.exists() else 0
    mark = "OK  " if ok else "**缺**"
    print(f"  {mark} {f}")
    print(f"       本地 {sz:.1f} MB   在清单: {ok}")
print(f"\n  清单总文件数: {len(rels)}")

print()
print("=" * 78)
print("2. 对照 config.py 的 pretrained_gpt_name（确认这两条就是全部路线）")
print("=" * 78)
cfg = GSV / "config.py"
if cfg.exists():
    txt = cfg.read_text(encoding="utf-8", errors="replace")
    for i, line in enumerate(txt.split("\n"), 1):
        if "s1v3" in line or "s1bert25hz" in line:
            s = line.strip()
            if s.startswith('"') or s.startswith("i18n"):
                print(f"  config.py:{i}  {s[:110]}")

print()
print("=" * 78)
print("3. train_gpt.py --base both --dry-run（验证缺底模时不再全盘中止）")
print("=" * 78)
r = subprocess.run(
    [sys.executable, str(V5 / "train_gpt.py"), "--base", "both", "--dry-run",
     "--root", str(GSV)],
    capture_output=True, text=True, encoding="utf-8", errors="replace",
    cwd=str(GSV))
out = (r.stdout or "") + (r.stderr or "")
print("\n".join(out.split("\n")[-30:]))
print(f"\n  退出码: {r.returncode}")

print()
print("=" * 78)
print("4. 结论")
print("=" * 78)
allin = all(f in rels for f in P.MODEL_FILES)
print(f"  两个底模都在清单: {allin}")
if allin:
    print("  -> 下次 push 后云端就有 v2 底模，--base both 可跑全")
print(f"  dry-run 退出码 0 表示可跑: {r.returncode == 0}")
