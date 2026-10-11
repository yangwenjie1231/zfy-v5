# -*- coding: utf-8 -*-
"""校验上云包内容是否完整、是否含多余大文件。"""
from __future__ import annotations

import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\A\zfy-v5")
import pack_for_cloud as P  # noqa: E402

items = P.collect()
# Windows 路径分隔符归一，否则匹配 "f5_tts/model/..." 会全部落空
norm = [(p, rel.replace("\\", "/")) for p, rel in items]
d = defaultdict(lambda: [0, 0])
for p, rel in norm:
    k = "/".join(rel.split("/")[:2]) if rel.count("/") >= 1 else rel
    d[k][0] += 1
    d[k][1] += p.stat().st_size

print("=== 包内容（前两级目录）===")
for k, (n, s) in sorted(d.items(), key=lambda x: -x[1][1])[:14]:
    print(f"  {s/1e6:9.1f} MB  {n:5d} 文件  {k}")
total = sum(v[1] for v in d.values())
print(f"\n合计 {total/1e9:.2f} GB / {len(items)} 文件")

print("\n=== 关键文件是否在包内 ===")
KEY = ["s2_train_v5_full.py", "s2Gv5dev.pth", "s1v3.ckpt", "models_v5.py",
       "6-name2semantic.tsv", "zfy_v5.list", "vocoder.pth", "configs/s2.json",
       "f5_tts/model/backbones/dit.py", "process_ckpt.py",
       "module/models.py", "s2_train_v3_lora.py",
       "_tools/notebook_preflight.py", "_tools/patch_models_for_torch212.py",
       "_tools/MODELSCOPE_NOTEBOOK_GUIDE.md"]
for key in KEY:
    hit = [rel for _, rel in norm if key in rel]
    status = ("OK   " + hit[0][:56]) if hit else "**缺失**"
    print(f"  {key:32s} {status}")

print("\n=== models.py 是否已含 torch212 补丁 ===")
mp = [p for p, rel in norm if rel.endswith("module/models.py")]
if mp:
    src = mp[0].read_text(encoding="utf-8", errors="replace")
    print(f"  含 torch212-patch: {'torch212-patch' in src}")
    print(f"  含 torch.amp     : {'from torch.amp import' in src}")
else:
    print("  **未找到 models.py**")

print("\n=== 不应包含的内容（应全部为空）===")
BAD = ["logs_s2_", "SoVITS_weights", "G_2", "G2PWModel_1.1.zip",
       "ja_userdic", "s2Gv5turbo", ".bak"]
for b in BAD:
    hit = [rel for _, rel in norm if b in rel]
    print(f"  {b:24s} {len(hit)} 个  {'正确' if not hit else '**多余!**'}")
