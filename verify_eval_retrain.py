# -*- coding: utf-8 -*-
"""验证 eval_retrain.py 的改造：权重发现 / epoch 解析 / 目录过滤。

不跑合成与评测（要 GPU + 几分钟），只验证选择逻辑。
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

spec = importlib.util.spec_from_file_location(
    "er", r"D:\A\zfy-v5\eval_retrain.py")
er = importlib.util.module_from_spec(spec)
# 阻止 __main__ 执行（文件末尾有 if __name__ == "__main__"）
spec.loader.exec_module(er)

print("=" * 70)
print("1. _ep_of 兼容两类命名")
print("=" * 70)
CASES = [
    "zfy_v5_e8_s432_l32.pth",              # v5 LoRA r32
    "zfy_v5_r128_e24_s1296_l128.pth",      # v5 LoRA r128
    "v5_full_e32_s1728_l0.pth",            # 全量微调
    "v5_full_e8_s432_l0.pth",              # 全量微调（旧）
    "smoke_e2_s96_l0.pth",                 # 冒烟
    "noepoch.pth",                         # 无 epoch
]
ok = True
for name in CASES:
    got = er._ep_of(Path(name))
    print(f"  {name:34s} -> {got}")
print()
print("=" * 70)
print("2. 目录过滤逻辑（tag_prefix 前缀 或 v2pro）")
print("=" * 70)
dirs = ["full_e08", "full_e32", "r128_e24", "v2pro", "smoke", "old_experiment"]
for prefix in ("full", "r128"):
    wanted = {d for d in dirs
              if d.startswith(prefix) or d == "v2pro"}
    print(f"  tag_prefix={prefix!r:8s} -> 评估 {sorted(wanted)}")

print()
print("=" * 70)
print("3. 生产口径检查（必须是生产参考 + 生产 GPT）")
print("=" * 70)
checks = [
    ("参考音频 = 生产 default.wav",
     str(er.GOOD_REF).endswith("zfh-models-v2pro\\ref\\default.wav")
     or str(er.GOOD_REF).endswith("zfh-models-v2pro/ref/default.wav")),
    ("参考文本 = 与 default.wav 一致（不是错配那个）",
     er.GOOD_REF_TEXT.startswith("再好的武器")),
    ("GPT = zfh-e4.ckpt（生产微调版，非底模）",
     str(er.ZFH_GPT).endswith("zfh-e4.ckpt")),
    ("v2Pro 权重 = 生产 zfh_v2pro_e12_s384.pth",
     str(er.V2PRO_SOV).endswith("zfh_v2pro_e12_s384.pth")),
    ("测试句数 = 20", len(er.TEXTS) == 20),
]
for name, good in checks:
    print(f"  {'OK  ' if good else 'FAIL'} {name}")
    ok = ok and good

print()
print("=" * 70)
print("4. 相关文件是否存在")
print("=" * 70)
for label, p in (("生产参考", er.GOOD_REF), ("生产 GPT", Path(er.ZFH_GPT)),
                 ("v2Pro 权重", Path(er.V2PRO_SOV))):
    e = Path(p).exists()
    print(f"  {'OK  ' if e else '缺失'} {label:10s} {p}")
    ok = ok and e

print()
print("=" * 70)
print("结论:", "✓ 改造正确，口径是生产口径" if ok else "✗ 有问题")
print("=" * 70)
