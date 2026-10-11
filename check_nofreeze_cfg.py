# -*- coding: utf-8 -*-
"""检查 `tmp_full_e32_nofreeze.json` 的关键字段是否都正确。

## 为什么必须检查
`patch_v5_unfreeze.py` 的 `make_unfreeze_config()` 会改 6 个字段。
任何一处错都会导致：
  · `freeze_quantizer` 没改      -> 白训（还是 freeze）
  · `exp_dir` 复用旧目录        -> **resume 到 freeze 版 checkpoint**（最危险）
  · `save_weight_dir` 复用      -> 覆盖旧的 e32 权重
  · `pretrained_s2G` 指错       -> 从错误的底模开始

## 本脚本（在云端跑，也兼容本地）
  1. 打印 6 个关键字段
  2. 与 freeze 版配置逐项对比
  3. 断言：exp_dir / save_weight_dir **必须不同**（否则会 resume/覆盖）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path("/mnt/workspace/zfy-v5-fullft-pkg")
if not (ROOT / "GPT_SoVITS").is_dir():
    ROOT = Path(r"D:\A\zfy-v5\GPT-SoVITS")

print("=" * 80)
print(f"根目录: {ROOT}")
print("=" * 80)

# 找两个配置
freeze_cfg = None
for name in ("tmp_full_e32.json", "tmp_full_e16.json"):
    p = ROOT / name
    if p.exists():
        freeze_cfg = p
        break
if freeze_cfg is None:
    cands = sorted(ROOT.glob("tmp_full_e*.json"))
    freeze_cfg = cands[-1] if cands else None

nofreeze_cfg = None
cands = sorted(ROOT.glob("tmp_full_e*_nofreeze.json"))
if cands:
    nofreeze_cfg = cands[-1]

print()
print("1) 找到的配置")
print(f"  freeze 版  : {freeze_cfg.name if freeze_cfg else '**无**'}")
print(f"  nofreeze 版: {nofreeze_cfg.name if nofreeze_cfg else '**无**'}")

if nofreeze_cfg is None:
    print("\n  !! 没有 nofreeze 配置 —— 补丁可能没成功生成")
    sys.exit(1)

nc = json.loads(nofreeze_cfg.read_text(encoding="utf-8"))
fc = json.loads(freeze_cfg.read_text(encoding="utf-8")) if freeze_cfg else {}

print()
print("2) nofreeze 配置的关键字段")
FIELDS = [
    ("model", "freeze_quantizer"),
    ("model", "version"),
    ("train", "epochs"),
    ("train", "batch_size"),
    ("train", "save_every_epoch"),
    ("train", "if_save_every_weights"),
    ("train", "grad_ckpt"),
    ("train", "fp16_run"),
    ("data", "exp_dir"),
    ("s2_ckpt_dir", None),
    ("save_weight_dir", None),
    ("name", None),
    ("pretrained_s2G", None),
]
for sec, key in FIELDS:
    if key is None:
        v = nc.get(sec, "**缺**")
        fv = fc.get(sec, "**缺**") if fc else "?"
        print(f"  {sec:22s} = {v}")
    else:
        v = (nc.get(sec) or {}).get(key, "**缺**")
        fv = (fc.get(sec) or {}).get(key, "?") if fc else "?"
        flag = ""
        if sec == "model" and key == "freeze_quantizer":
            flag = "  <- 必须是 False" if v is not False else "  ✓"
        print(f"  {sec}.{key:20s} = {v}{flag}")
    if fc:
        if str(v) != str(fv):
            print(f"  {'':24s}   (freeze 版是 {fv})")

print()
print("3) 断言（安全底线）")
checks = []
checks.append(("freeze_quantizer == False",
               nc.get("model", {}).get("freeze_quantizer") is False,
               str(nc.get("model", {}).get("freeze_quantizer"))))

n_exp = str(nc.get("data", {}).get("exp_dir", ""))
f_exp = str(fc.get("data", {}).get("exp_dir", "")) if fc else ""
checks.append(("exp_dir 与 freeze 版**不同**（否则会 resume 旧 checkpoint）",
               bool(n_exp) and n_exp != f_exp,
               f"{Path(n_exp).name} vs {Path(f_exp).name}"))

n_w = str(nc.get("save_weight_dir", ""))
f_w = str(fc.get("save_weight_dir", "")) if fc else ""
checks.append(("save_weight_dir 与 freeze 版不同（否则覆盖旧权重）",
               bool(n_w) and n_w != f_w,
               f"{n_w} vs {f_w}"))

checks.append(("name 含 nofreeze",
               "nofreeze" in str(nc.get("name", "")),
               str(nc.get("name"))))

pg = str(nc.get("pretrained_s2G", ""))
checks.append(("pretrained_s2G 指向 v5dev 底模",
               "s2Gv5dev" in pg, Path(pg).name))

bad = 0
for name, ok, extra in checks:
    bad += 0 if ok else 1
    print(f"  {'✓' if ok else '✗'} {name}" + (f"  ({extra})" if extra else ""))

print()
print("=" * 80)
print("4) 训练启动后日志必须出现这行")
print("=" * 80)
print("""
  [ZFY] ZFY_UNFREEZE=1 -> 强制 freeze_quantizer=False（enc_p/ssl_proj/quantizer 参与训练并进入权重）

  若没有这行 -> 环境变量没传进去 -> 白训。

  监控命令：
      tail -f /mnt/workspace/zfy-v5-fullft-pkg/train_nofreeze.log
""")
print("=" * 80)
if bad == 0:
    print("  ✓ 配置全部正确，可以放心训练")
else:
    print(f"  ✗ {bad} 项失败 —— 建议先修再训")
