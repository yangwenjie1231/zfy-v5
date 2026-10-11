# -*- coding: utf-8 -*-
"""验证 `语音合集_分条` -> 训练集 的文本映射是否正确。

## 为什么必须验证
技能明确警告：`ref_text` 必须与参考音频**一字不差**，
写错了**整个音色提示就废了**（历史上的"语气差"就是这个原因）。

`map_ref_texts.py` 用「序号+名字」推断出 9 个映射，例如：
    语音合集: 052_休整
    训练集:   54_休整_chr_0030_zhuangfy_chrbark_rest_01
    -> 推断文本 "暂且歇歇吧,留着精力,更难的还在后面呢."

**但序号可能错位**（052 vs 54，相差 2）—— 必须验证。

## 验证方法：时长指纹
如果两条音频是**同一段录音**，时长必然相同（差 <0.01s）。
    052_休整.wav              vs  54_休整_chr_*.wav
若时长一致 -> 同一段音频 -> 文本可用
若不一致   -> 映射错误   -> 文本作废

**为什么可靠**：同一段语音无论怎么导出/重采样，时长不变。

## 输出
    dist/ref_best2/map_verify.txt
    dist/ref_best2/cands_verified.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

V5 = Path(r"D:\A\zfy-v5")
GSV = V5 / "GPT-SoVITS"
OUT = V5 / "dist" / "ref_best2"
TXT = GSV / "logs" / "zfy_v5" / "2-name2text.txt"
TRAIN = GSV / "logs" / "zfy_v5" / "5-wav32k"


def dur(p: Path) -> float | None:
    try:
        i = sf.info(str(p))
        return i.frames / float(i.samplerate)
    except Exception:  # noqa: BLE001
        return None


# 训练集时长表
train_d: dict[str, float] = {}
for w in TRAIN.glob("*.wav"):
    d = dur(w)
    if d is not None:
        train_d[w.stem] = d
print(f"  训练集时长表: {len(train_d)} 条")

# 文本表
tm: dict[str, str] = {}
for line in TXT.read_text(encoding="utf-8").splitlines():
    p = line.split("\t")
    if len(p) >= 4:
        tm[p[0].replace(".wav", "")] = p[3].strip()

rows = json.loads((OUT / "texts_mapped.json").read_text(encoding="utf-8"))

print()
print("=" * 104)
print("映射验证（时长指纹）")
print("=" * 104)
print(f"  {'#':>3s} {'候选':<24s} {'候选时长':>9s} {'匹配到的训练集':<40s} "
      f"{'训练集时长':>10s} {'Δ':>8s}  判定")
print("  " + "-" * 104)

verified = []
for r in rows:
    p = Path(r["path"])
    d_cand = dur(p)
    how = r["how"]
    if how in ("精确", "前缀"):
        # 文本直接来自训练集同 key -> 天然正确
        verified.append({**r, "verify": "文本直接来自训练集", "ok": True})
        print(f"  {r['rank']:3d} {r['stem'][:24]:<24s} {d_cand:8.3f}s "
              f"{'(同 key，无需验证)':<40s} {'':>10s} {'':>8s}  ✓ {how}")
        continue

    # 映射来的：从 how 里提取训练集 stem
    src = ""
    if "(" in how and ")" in how:
        src = how[how.index("(") + 1:how.index(")")]
    # 在训练集里找完整 stem
    full = next((k for k in train_d if k.startswith(src)), None)
    if not full:
        verified.append({**r, "verify": "找不到训练集对应", "ok": False})
        print(f"  {r['rank']:3d} {r['stem'][:24]:<24s} {d_cand:8.3f}s "
              f"{src[:40]:<40s} {'?':>10s} {'':>8s}  ✗ 无法验证")
        continue

    d_tr = train_d[full]
    delta = abs(d_cand - d_tr)
    ok = delta < 0.05
    verified.append({**r, "verify": f"时长Δ={delta:.4f}s", "ok": ok,
                     "train_stem": full, "train_dur": d_tr})
    print(f"  {r['rank']:3d} {r['stem'][:24]:<24s} {d_cand:8.3f}s "
          f"{full[:40]:<40s} {d_tr:9.3f}s {delta:+8.4f}  "
          f"{'✓ 同一段' if ok else '✗ 不同段'}")

good = [v for v in verified if v.get("ok")]
print()
print("=" * 104)
print(f"验证通过: **{len(good)} / {len(verified)}**")
print("=" * 104)
print()
print(f"  {'#':>3s} {'文件':<24s} {'相似度':>8s} {'尾平坦':>8s} {'尾/体':>8s} "
      f"{'时长':>7s}  文本")
print("  " + "-" * 104)
for v in sorted(good, key=lambda x: x["flat"]):
    t = v["text"][:36] + ("…" if len(v["text"]) > 36 else "")
    print(f"  {v['rank']:3d} {v['stem'][:24]:<24s} {v['sim']:8.4f} "
          f"{v['flat']:8.4f} {v['tb']:+8.2f} {v['dur']:6.2f}s  {t}")

# 按尾平坦排序，取最优几个
by_flat = sorted(good, key=lambda x: x["flat"])
print()
print("=" * 104)
print("按「尾音最干净」排序（用户的核心诉求）")
print("=" * 104)
for i, v in enumerate(by_flat[:8], 1):
    print(f"  {i}. {v['stem'][:30]:<32s} 尾平坦 {v['flat']:.4f}  "
          f"相似度 {v['sim']:.4f}  尾/体 {v['tb']:+.2f}dB  "
          f"时长 {v['dur']:.2f}s")

# 导出 CANDS：orig + 尾最干净的 2 个 + 相似度最高的 1 个
sel = []
seen = set()
for v in by_flat[:2]:
    sel.append(v)
    seen.add(v["stem"])
best_sim = max(good, key=lambda x: x["sim"])
if best_sim["stem"] not in seen:
    sel.append(best_sim)
    seen.add(best_sim["stem"])
# 再补一个相似度高的
for v in sorted(good, key=lambda x: -x["sim"]):
    if v["stem"] not in seen:
        sel.append(v)
        seen.add(v["stem"])
        break

lines = ["CANDS = [",
         '    ("orig", GSV / "ref" / "default.wav",',
         '     (GSV / "ref" / "ref.txt").read_text(encoding="utf-8").strip()),']
for v in sel:
    tag = f"r{v['rank']:02d}"
    lines.append(f'    ("{tag}", Path(r"{v["path"]}"),')
    lines.append(f'     {v["text"]!r}),   # flat={v["flat"]:.4f} '
                 f'sim={v["sim"]:.4f} {v["verify"]}')
lines.append("]")
(OUT / "cands_verified.py").write_text("\n".join(lines) + "\n", encoding="utf-8")

print()
print("=" * 104)
print("最终 CANDS（用于合成对比）")
print("=" * 104)
print("\n".join(lines))
print()
print(f"  -> {OUT / 'cands_verified.py'}")

rep = ["=" * 80, "映射验证报告", "=" * 80, ""]
rep.append(f"验证通过 {len(good)}/{len(verified)}")
rep.append("")
for v in verified:
    rep.append(f"{v['rank']:3d} {v['stem'][:34]:<36s} "
               f"{'OK' if v.get('ok') else 'FAIL':<5s} {v.get('verify','')}")
rep.append("")
rep.append("按尾平坦排序:")
for i, v in enumerate(by_flat[:12], 1):
    rep.append(f"{i:2d}. {v['stem'][:34]:<36s} flat={v['flat']:.4f} "
               f"sim={v['sim']:.4f} tb={v['tb']:+.2f}")
(OUT / "map_verify.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
print(f"  -> {OUT / 'map_verify.txt'}")
print("=" * 104)
