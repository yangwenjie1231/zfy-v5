# -*- coding: utf-8 -*-
"""关键诊断：全量微调权重**缺少 enc_p / ssl_proj**（推理要用！）。

## 线索
云端日志的 missing_keys 里全是 enc_p.* 和 ssl_proj.*：
    _IncompatibleKeys(missing_keys=['enc_p.ssl_proj.weight', ...,
                                    'enc_p.proj.weight', 'enc_p.proj.bias',
                                    'ssl_proj.weight', 'ssl_proj.bias'],
                      unexpected_keys=[])

**unexpected_keys 是空的** -> 我们的权重没有多余键，但**缺了这些**。
load_state_dict(strict=False) 会**静默保留随机初始化** —— 无声错误！

## 参数规模佐证
    权重文件 735.7 MB (fp16) -> 367.85M 参数
    v5 完整模型               -> 383.08M 参数
    差                       -> 15.23M  <-- 正好是 enc_p + ssl_proj 量级

## 为什么会缺
s2_train_v5_full.py:402-407
    sim_ckpt = od()
    for key in ckpt:
        if key not in no_grad_names:        # <-- 跳过 requires_grad=False 的
            sim_ckpt[key] = ckpt[key].half().cpu()

no_grad_names 收集所有 requires_grad=False 的参数。
若模型定义里 enc_p/ssl_proj 被冻结，它们就**不会进权重文件**。

本脚本：
  1. 在模型定义里找 requires_grad / freeze
  2. 加载 s2Gv5dev.pth，确认 enc_p 存在且统计参数量
  3. 确认 v5 推理路径是否真的用 enc_p
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")
sys.path.insert(0, str(GSV))
sys.path.insert(0, str(GSV / "GPT_SoVITS"))

print("=" * 78)
print("1. 模型定义里的 requires_grad / freeze")
print("=" * 78)
for f in ("module/models.py", "module/models_v5.py", "module/modules.py"):
    p = GSV / "GPT_SoVITS" / f
    if not p.exists():
        continue
    txt = p.read_text(encoding="utf-8", errors="replace")
    hits = []
    for i, line in enumerate(txt.split("\n"), 1):
        if "requires_grad" in line or "freeze" in line.lower():
            hits.append((i, line.strip()))
    if hits:
        print(f"\n  【{f}】")
        for i, s in hits:
            print(f"    L{i}: {s[:110]}")

print()
print("=" * 78)
print("2. 加载 s2Gv5dev.pth，统计 enc_p / ssl_proj / 总量")
print("=" * 78)
from process_ckpt import load_sovits_new  # noqa: E402

p = (GSV / "GPT_SoVITS" / "pretrained_models" / "gsv-v5-pretrained"
     / "s2Gv5dev.pth")
d = load_sovits_new(str(p))
w = d["weight"]
print(f"  文件: {p.name}   info={d.get('info')}")
print(f"  总键数: {len(w)}")

groups = {"enc_p": 0, "ssl_proj": 0, "quantizer": 0, "cfm": 0, "dec": 0,
          "flow": 0, "ref_enc": 0, "其他": 0}
params = {k: 0 for k in groups}
total = 0
for k, v in w.items():
    n = v.numel()
    total += n
    for g in ("enc_p", "ssl_proj", "quantizer", "cfm", "dec", "flow", "ref_enc"):
        if k.startswith(g) or (g == "enc_p" and k.startswith("enc_p")):
            groups[g] += 1
            params[g] += n
            break
    else:
        groups["其他"] += 1
        params["其他"] += n

print(f"\n  {'组':12s} {'键数':>6s} {'参数(M)':>12s}")
for g in groups:
    if groups[g]:
        print(f"  {g:12s} {groups[g]:6d} {params[g]/1e6:12.2f}")
print(f"  {'合计':12s} {len(w):6d} {total/1e6:12.2f}")

print()
print("=" * 78)
print("3. 关键算术：权重文件大小 vs 参数量")
print("=" * 78)
MB = 735.7
print(f"  云端 v5 全量微调权重: {MB} MB (fp16)")
print(f"    -> {MB*1024*1024/2/1e6:.2f}M 参数")
print(f"  完整模型            : {total/1e6:.2f}M 参数")
miss = total - MB*1024*1024/2
print(f"    差值              : {miss/1e6:.2f}M 参数")
print(f"  enc_p + ssl_proj    : "
      f"{(params['enc_p']+params['ssl_proj'])/1e6:.2f}M 参数")
print()
if abs(miss - (params["enc_p"] + params["ssl_proj"])) < 3e6:
    print("  ** 高度吻合 -> 权重确实缺了 enc_p + ssl_proj **")
else:
    print("  差异较大，需进一步查（可能有其他组也被排除）")

print()
print("=" * 78)
print("4. v5 推理路径是否用 enc_p")
print("=" * 78)
mv5 = GSV / "GPT_SoVITS" / "module" / "models_v5.py"
if mv5.exists():
    txt = mv5.read_text(encoding="utf-8", errors="replace")
    for i, line in enumerate(txt.split("\n"), 1):
        if "enc_p" in line or "def synthesize_v5_mel" in line or "def forward" in line:
            print(f"  models_v5.py:{i}  {line.strip()[:105]}")
print()
m = GSV / "GPT_SoVITS" / "module" / "models.py"
txt = m.read_text(encoding="utf-8", errors="replace")
for i, line in enumerate(txt.split("\n"), 1):
    if "def decode" in line or "def decode_latent" in line:
        print(f"  models.py:{i}  {line.strip()[:105]}")

print()
print("=" * 78)
print("5. 结论")
print("=" * 78)
print("""
  若确认缺 enc_p + ssl_proj，则**合成时长异常很可能就是它**：
  enc_p 保持随机初始化 -> 先验分布 m_p/logs_p 完全错乱 ->
  SoVITS 渲染出的时长与内容都不对（11.40s vs 1.2s）。

  修法：不要用 sim_ckpt（它按 no_grad_names 裁剪），
  直接从**完整 checkpoint** G_233333333333.pth 导出权重。
  那个文件含全部参数（含优化器状态）。

  下一步：先确认 v5 推理是否真用 enc_p；再改导出方式。
""")
