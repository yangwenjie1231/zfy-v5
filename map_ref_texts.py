# -*- coding: utf-8 -*-
"""反查 `语音合集_分条` 候选的参考文本（通过训练集映射）。

## 问题
`show_ref_texts.py` 显示 TOP10 里只有 2 个能拿到文本：
    top01 剧情_e10m2_001   ✓ "管理员…我们不见不散."
    top04 剧情_e9m4_010    ✓ "…潘工,真不用帮忙?…"
    top03/05/06/07/08/09/10  ✗ 未找到（都是 `语音合集_分条`）

**而 `语音合集_分条` 恰恰是尾音最干净的**：
    052_休整       尾平坦 0.0350   <- 最干净
    004_编入队伍1   尾平坦 0.0312
    005_编入队伍2   尾平坦 0.0477
    051_消除侵蚀    尾平坦 0.0424

## 为什么没文本
`2-name2text.txt` 的 key 是**训练集文件名**（如 `01_行动准备1_chr_0030_zhuangfy_chrbark_squad_01`），
而 `语音合集_分条` 的文件名是**短名**（`002_行动准备1`）。

**但它们是同一批音频**（之前已确认 `002_行动准备1` = `02_行动准备1_chr_*`）。
所以可以**按序号+名字映射**。

## 映射规则
    语音合集_分条: 002_行动准备1
    训练集:        02_行动准备1_chr_0030_zhuangfy_chrbark_squad_02
                   ^^ 序号(补零到2位) + 名字

## 输出
    dist/ref_best2/texts_mapped.json
    dist/ref_best2/cands_full.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

V5 = Path(r"D:\A\zfy-v5")
GSV = V5 / "GPT-SoVITS"
OUT = V5 / "dist" / "ref_best2"
TXT = GSV / "logs" / "zfy_v5" / "2-name2text.txt"
TRAIN = GSV / "logs" / "zfy_v5" / "5-wav32k"

# 文本表：key = 训练集 stem
tm: dict[str, str] = {}
for line in TXT.read_text(encoding="utf-8").splitlines():
    p = line.split("\t")
    if len(p) >= 4:
        tm[p[0].replace(".wav", "")] = p[3].strip()

# 训练集 stem 索引： 序号 -> [stems]
by_idx: dict[str, list[str]] = {}
by_name: dict[str, list[str]] = {}
for st in tm:
    m = re.match(r"^(\d+)_([^_]+)", st)
    if m:
        by_idx.setdefault(m.group(1).lstrip("0") or "0", []).append(st)
        by_name.setdefault(m.group(2), []).append(st)

print("=" * 100)
print("反查 `语音合集_分条` 候选的文本")
print("=" * 100)
print(f"  训练集 stem: {len(tm)} 个")
print(f"  按序号索引: {len(by_idx)} 组")
print()

cands = json.loads((OUT / "candidates.json").read_text(encoding="utf-8"))

# 训练集文件名（用于排除验证）
train_stems = {w.stem for w in TRAIN.glob("*.wav")}

rows = []
for i, s in enumerate(cands[:12], 1):
    p = Path(s["path"])
    st = p.stem
    text, how = "", ""

    # ① 精确
    if st in tm:
        text, how = tm[st], "精确"
    # ② 前缀模糊
    if not text:
        for k, v in tm.items():
            if k.startswith(st[:16]) or st.startswith(k[:16]):
                text, how = v, "前缀"
                break
    # ③ 语音合集_分条 映射： "023_帝江号闲聊4" -> 序号 23 + 名字 帝江号闲聊4
    if not text:
        m = re.match(r"^(\d+)_(.+)$", st)
        if m:
            idx = str(int(m.group(1)))
            name = m.group(2)
            # 先按名字找
            hits = by_name.get(name, [])
            # 再按序号+名字找
            if not hits:
                hits = [k for k in by_idx.get(idx, []) if name[:3] in k]
            if not hits:
                hits = by_idx.get(idx, [])
            if hits:
                text = tm[hits[0]]
                how = f"映射({hits[0][:32]})"

    rows.append({"rank": i, "stem": st, "path": str(p), "text": text,
                 "how": how, "sim": s["sim"], "flat": s["flat"],
                 "tb": s["tb"], "dur": s["dur"]})

print(f"  {'#':>3s} {'文件':<30s} {'相似度':>8s} {'尾平坦':>8s} "
      f"{'文本来源':<26s} 文本")
print("  " + "-" * 100)
for r in rows:
    t = (r["text"][:34] + "…") if len(r["text"]) > 34 else r["text"]
    print(f"  {r['rank']:3d} {r['stem'][:30]:<30s} {r['sim']:8.4f} "
          f"{r['flat']:8.4f} {r['how'][:26]:<26s} {t or '(无)'}")

usable = [r for r in rows if r["text"]]
print()
print(f"  可用于合成: **{len(usable)} / {len(rows)}**")

# ---------- 验证文本是否真的对应 ----------
print()
print("=" * 100)
print("⚠️ 文本对应性验证（技能警告：ref_text 必须与音频一字不差）")
print("=" * 100)
print("""
  映射来的文本**有风险** —— 序号+名字匹配可能错位。
  验证方法：用 ASR 转写音频，与文本比对。
  但本地没有可靠的 ASR（`_qwen3tts_research` 是空的）。

  ## 保守策略
  只采用**来源可靠**的候选：
    · `精确` 匹配（key 完全一致）      -> 安全
    · `前缀` 匹配（前 16 字符一致）    -> 较安全
    · `映射` 匹配（序号+名字推断）     -> **需人工核对**
""")

safe = [r for r in rows if r["how"] in ("精确", "前缀")]
risky = [r for r in rows if r["how"].startswith("映射")]
print(f"  安全（精确/前缀）: {len(safe)} 个")
for r in safe:
    print(f"    top{r['rank']:02d} {r['stem'][:34]:<36s} "
          f"sim={r['sim']:.4f} flat={r['flat']:.4f}  «{r['text'][:40]}»")
print(f"  需核对（映射）: {len(risky)} 个")
for r in risky:
    print(f"    top{r['rank']:02d} {r['stem'][:34]:<36s} "
          f"sim={r['sim']:.4f} flat={r['flat']:.4f}  «{r['text'][:40]}»")

# ---------- 导出 ----------
(OUT / "texts_mapped.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

lines = ["CANDS = [",
         '    ("orig", GSV / "ref" / "default.wav",',
         '     (GSV / "ref" / "ref.txt").read_text(encoding="utf-8").strip()),']
for r in usable[:5]:
    tag = f"best{r['rank']:02d}"
    lines.append(f'    ("{tag}", Path(r"{r["path"]}"),')
    lines.append(f'     {r["text"]!r}),   # {r["how"]}  '
                 f'sim={r["sim"]:.4f} flat={r["flat"]:.4f}')
lines.append("]")
(OUT / "cands_full.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
print()
print(f"  -> {OUT / 'texts_mapped.json'}")
print(f"  -> {OUT / 'cands_full.py'}")
print()
print("=" * 100)
print("生成的 CANDS")
print("=" * 100)
print("\n".join(lines))
print("=" * 100)
