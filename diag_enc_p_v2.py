# -*- coding: utf-8 -*-
"""修正单位错误，确认权重是否真的缺 enc_p + ssl_proj；并查 v5 推理路径。

## 上一版的单位 bug
我写了 `MB*1024*1024/2`（把 MB 当 MiB），而 fix_weight_header.py 打印的
735.7 MB 是 `sz/1e6`（十进制 MB）。差 4.9%，正好把 15M 的缺口掩盖成 -2.6M。

正确算法：
    735.7e6 字节 / 2 字节每参数 = 367.85M 参数
    完整模型 383.08M - 367.85M = 15.23M
    enc_p(14.15M) + ssl_proj(1.18M) = 15.33M   <-- 吻合！
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")
sys.path.insert(0, str(GSV))
sys.path.insert(0, str(GSV / "GPT-SoVITS"))
sys.path.insert(0, str(GSV / "GPT_SoVITS"))

from process_ckpt import load_sovits_new  # noqa: E402

print("=" * 78)
print("1. 修正单位后重算")
print("=" * 78)
d = load_sovits_new(str(GSV / "GPT_SoVITS" / "pretrained_models"
                        / "gsv-v5-pretrained" / "s2Gv5dev.pth"))
w = d["weight"]
tot = sum(v.numel() for v in w.values())
enc = sum(v.numel() for k, v in w.items() if k.startswith("enc_p"))
ssl = sum(v.numel() for k, v in w.items() if k.startswith("ssl_proj"))
print(f"  完整模型         : {tot/1e6:.2f}M 参数")
print(f"  enc_p            : {enc/1e6:.2f}M")
print(f"  ssl_proj         : {ssl/1e6:.2f}M")
print(f"  enc_p + ssl_proj : {(enc+ssl)/1e6:.2f}M")

BYTES = 735.7e6          # fix_weight_header.py 用 sz/1e6 打印的十进制 MB
saved = BYTES / 2        # fp16
print(f"\n  云端权重文件     : 735.7 MB (十进制) -> {saved/1e6:.2f}M 参数")
print(f"  缺口             : {(tot-saved)/1e6:.2f}M 参数")
gap = tot - saved
print(f"  与 enc_p+ssl_proj 之差: {abs(gap-(enc+ssl))/1e6:.2f}M")
print()
if abs(gap - (enc + ssl)) < 1.0e6:
    print("  ** 确认：权重缺的正是 enc_p + ssl_proj（误差 <1M，来自 config 等开销）**")
else:
    print("  仍不吻合，需继续查")

print()
print("=" * 78)
print("2. v5 推理走哪条路径（用不用 enc_p）")
print("=" * 78)
tts = GSV / "GPT_SoVITS" / "TTS_infer_pack" / "TTS.py"
txt = tts.read_text(encoding="utf-8", errors="replace")
lines = txt.split("\n")
for i, line in enumerate(lines, 1):
    if ("synthesize_v5_mel" in line or "decode_encp" in line
            or "vits_model.decode" in line or "self.vits_model(" in line
            or "synthesize_v5" in line):
        print(f"  TTS.py:{i}  {line.strip()[:108]}")

print()
mv5 = GSV / "GPT_SoVITS" / "module" / "models_v5.py"
print(f"  --- {mv5.name} 的 synthesize_v5_mel ---")
t5 = mv5.read_text(encoding="utf-8", errors="replace")
l5 = t5.split("\n")
start = None
for i, line in enumerate(l5):
    if "def synthesize_v5_mel" in line:
        start = i
        break
if start is not None:
    for i in range(start, min(start + 46, len(l5))):
        print(f"    {i+1:4d}: {l5[i][:104]}")

print()
print("=" * 78)
print("3. 结论")
print("=" * 78)
print("""
  若 v5 走 synthesize_v5_mel（CFM 路径），**不用 enc_p** ->
  那 missing enc_p 就**不是**时长异常的原因，可以忽略。

  若走 decode()（用 enc_p），则缺失会导致先验错乱 -> 时长异常。
""")
