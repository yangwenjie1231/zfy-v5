# -*- coding: utf-8 -*-
"""验证 train_gpt.py 的 YAML 标量写出能被 PyYAML **正确回读**。

## 背景
云端 GPT 训练报：
    TypeError: unsupported operand type(s) for -: 'float' and 'str'
根因：`str(0.00001)` = `'1e-05'`，而 YAML 1.1 不认这种科学计数法，
PyYAML 把它读成**字符串** -> `lr_init` 变成 str -> 相减报错。

## 本脚本
  1. 测 `_yaml_val` 对各类值的输出 + 回读类型
  2. 模拟完整配置的写出/回读，逐项比对
  3. 反向验证：确认旧写法（str(v)）确实会失败
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import yaml

spec = importlib.util.spec_from_file_location(
    "tg", r"D:\A\zfy-v5\train_gpt.py")
tg = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(tg)
except SystemExit:
    pass

print("=" * 78)
print("1. _yaml_val 逐值测试")
print("=" * 78)
CASES = [0.00001, 0.01, 0.0001, 1e-5, 1e-4, 1e-7, 1234.0, 0.0,
         2000, 8, True, False, "abc", "/mnt/x/y.ckpt", 54]
print(f"  {'原值':>14s}  {'写出':>18s}  {'回读类型':10s}  结果")
bad = 0
for v in CASES:
    out = tg._yaml_val(v)
    back = yaml.full_load(f"v: {out}")["v"]
    ok = type(back) is type(v)
    if ok and isinstance(v, float):
        ok = abs(back - v) < 1e-12
    bad += 0 if ok else 1
    print(f"  {v!r:>14}  {out:>18}  {type(back).__name__:10s}  "
          f"{'OK' if ok else '**FAIL**'}")

print()
print("=" * 78)
print("2. 反向验证：旧写法 str(v) 会失败")
print("=" * 78)
for v in (0.00001, 1e-4):
    old = str(v)
    back = yaml.full_load(f"v: {old}")["v"]
    print(f"  str({v!r}) = {old!r}  -> 回读 {type(back).__name__} {back!r}"
          f"   {'**旧写法确实错**' if not isinstance(back, float) else ''}")

print()
print("=" * 78)
print("3. 完整配置写出 + 回读校验")
print("=" * 78)
cfg = tg.build_config(
    Path("/mnt/workspace/zfy-v5-fullft-pkg/GPT-SoVITS"),
    epochs=8, batch=8, warmup=2000, decay=40000,
    precision="16-mixed", max_sec=54, num_workers=4,
    base_rel="GPT_SoVITS/pretrained_models/s1v3.ckpt",
    exp="zfy_gptA", half_dir="GPT_weights_gptA",
    out_dir=Path("/mnt/workspace/zfy-v5-fullft-pkg/logs/gptA"),
    phoneme=Path("/mnt/workspace/zfy-v5-fullft-pkg/logs/zfy_v5/2-name2text.txt"),
    semantic=Path("/mnt/workspace/zfy-v5-fullft-pkg/logs/zfy_v5/6-name2semantic.tsv"),
)
out = Path(r"C:\Users\A\AppData\Local\Temp\_zfy_test_s1.yaml")
# write_yaml 内含回读校验，会自己抛 SystemExit 或打印通过
try:
    tg.write_yaml(cfg, out)
    ok_write = True
except SystemExit as e:
    print(f"  write_yaml 中止: {e}")
    ok_write = False

print()
print("  生成的 YAML:")
for l in out.read_text(encoding="utf-8").split("\n"):
    if l.strip():
        print(f"    {l}")

print()
print("=" * 78)
print("4. 重点核对 optimizer 段（报错的地方）")
print("=" * 78)
back = yaml.full_load(out.read_text(encoding="utf-8"))
for k, want in cfg["optimizer"].items():
    got = back["optimizer"][k]
    ok = type(got) is type(want) and (
        not isinstance(want, float) or abs(got - want) < 1e-12)
    print(f"  optimizer.{k:14s} 写入 {want!r:12} 回读 {got!r:12} "
          f"({type(got).__name__})  {'OK' if ok else '**FAIL**'}")

print()
print("=" * 78)
print("结论")
print("=" * 78)
print(f"  _yaml_val 逐值测试: {'全部正确' if bad == 0 else f'{bad} 个错误'}")
print(f"  write_yaml 回读校验: {'通过' if ok_write else '未通过'}")
print(f"""
  修复后 `lr_init: 0.00001`（定点十进制），PyYAML 会读成 float，
  `peak_lr - init_lr` 不再报 TypeError。

  ⚠️ 顺带说明：`lr_init/lr_end` 实际上**没用** ——
     lr_schedulers.py:58 会把 lr 硬锁成 2e-3（warmup/cosine 是死代码）。
     但类型必须正确，否则 __init__ 阶段就崩。
""")
