# -*- coding: utf-8 -*-
"""对比：生产 GPT 的训练语料 vs 现在的 v5 语料。

## 为什么

生产 `zfh-e4.ckpt` 的配置在
    D:\\A\\voiceclone\\data\\_tmp_cfg\\tmp_s1.yaml
指向 logs/zfh/{2-name2text.txt, 6-name2semantic.tsv}。

而 v5 语料在 logs/zfy_v5/ 同名文件。
本脚本量化两者规模差，判断"用新语料重训 GPT 值不值"。

同时列出现有可用的 GPT 底模，回答"有没有更好的选择"。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CORPORA = [
    ("生产 GPT 训练语料 (zfh)", Path(r"D:\A\voiceclone\GPT-SoVITS\logs\zfh")),
    ("v5 训练语料 (zfy_v5)", Path(r"D:\A\zfy-v5\GPT-SoVITS\logs\zfy_v5")),
]

print("=" * 78)
print("1. 两份语料的规模对比")
print("=" * 78)
results = {}
for name, d in CORPORA:
    print(f"\n--- {name} ---")
    print(f"  目录: {d}")
    if not d.is_dir():
        print("  不存在")
        continue
    info = {}
    for fn in ("2-name2text.txt", "6-name2semantic.tsv"):
        p = d / fn
        if not p.exists():
            print(f"  {fn:24s} 缺失")
            continue
        raw = p.read_text(encoding="utf-8", errors="replace")
        lines = [l for l in raw.strip().split("\n") if l.strip()]
        info[fn] = len(lines)
        print(f"  {fn:24s} {len(lines):5d} 行  {p.stat().st_size/1024:9.1f} KB")

    # semantic 的 token 总长度（更能反映训练量）
    sem = d / "6-name2semantic.tsv"
    if sem.exists():
        raw = sem.read_text(encoding="utf-8", errors="replace")
        toks, lens = 0, []
        for l in raw.strip().split("\n"):
            if not l.strip():
                continue
            parts = l.split("\t")
            if len(parts) >= 2:
                ids = [x for x in parts[1].split() if x.strip()]
                lens.append(len(ids))
                toks += len(ids)
        if lens:
            lens.sort()
            print(f"  semantic token 总数   {toks:,d}")
            print(f"  单条 token 中位/最长  {lens[len(lens)//2]} / {lens[-1]}")
            # 25 Hz -> 秒数
            print(f"  折算音频时长          {toks/25/60:.1f} 分钟")
            info["tokens"] = toks
    results[name] = info

print()
print("=" * 78)
print("2. 倍数关系")
print("=" * 78)
if len(results) == 2:
    a, b = list(results.values())
    for k in ("2-name2text.txt", "6-name2semantic.tsv", "tokens"):
        if k in a and k in b and a[k]:
            print(f"  {k:24s} {a[k]:>10,d} -> {b[k]:>10,d}   "
                  f"= {b[k]/a[k]:.2f} 倍")

print()
print("=" * 78)
print("3. 可用的 GPT 底模")
print("=" * 78)
PRE = Path(r"D:\A\zfy-v5\GPT-SoVITS\GPT_SoVITS\pretrained_models")
CANDS = [
    ("v1 底模", "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"),
    ("v2 底模", "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"),
    ("v3+ 底模 (v2Pro/v3/v4/v5 共用)", "s1v3.ckpt"),
]
for label, rel in CANDS:
    p = PRE / rel
    if p.exists():
        print(f"  OK   {label:32s} {p.stat().st_size/1e6:7.1f} MB  {rel}")
    else:
        print(f"  缺   {label:32s} 未下载     {rel}")

print()
print("  生产微调版:")
for p in [Path(r"D:\A\voiceclone\zfh-models-v2pro\torch_weights\zfh-e4.ckpt")]:
    if p.exists():
        print(f"  OK   zfh-e4.ckpt（从 v2 底模微调，e4）  "
              f"{p.stat().st_size/1e6:.1f} MB")

print()
print("=" * 78)
print("4. 结论")
print("=" * 78)
print("""
  关键事实：
    · config.py 里 v3/v4/v5dev/v5turbo/v2Pro/v2ProPlus **全部**用 s1v3.ckpt
      -> s1v3 就是"通用底模"，v5 也是它
    · 生产 zfh-e4.ckpt 是从 **v2 底模** 微调的（不是 s1v3）
    · GPT 与 SoVITS 必须配套：GPT 出 25Hz 语义 token，SoVITS 的
      quantizer 要能吃同一套 token。所以不能随便换个别的 TTS 模型当 GPT。

  可选项（按代价从低到高）：
    A. 用新语料从 s1v3.ckpt 微调 GPT   <- 最直接，数据已就绪
    B. 从 v2 底模微调（复刻生产路径，只是数据更多）
    C. 换整个 TTS 栈（CosyVoice/IndexTTS 等）—— 丢掉现有 SoVITS 音色
""")
