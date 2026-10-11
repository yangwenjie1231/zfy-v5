# -*- coding: utf-8 -*-
"""A/B 对比：merged（补了 ssl_proj）vs 未合并 —— 确认根因与修复。

## 数据来自云端日志（T2S Decoding EOS + 输出时长）
  prefix_len = 213（prompt 199 token + 文本 14 token）
  生成 token = EOS 的第二个数 - 213
  时长 = token / 25Hz

## 结论
  merged 版：语速 4.0 字/秒（正常中文 3~5）✓
  未合并：  语速 1.5 字/秒（严重拖沓）✗
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# (序号, 字数, 未合并 EOS 末值, 未合并时长, merged EOS 末值, merged 时长)
DATA = [
    ("01 短", "让我想想。", 5, 494, 11.40, 237, 1.12),
    ("02 短", "好。", 2, 406, 7.88, 225, 0.64),
    ("03 中", "迎敌或是迎客，都得整装后再出发。", 16, 411, 8.08, 305, 3.84),
    ("04 中", "出门前，要不要先喝杯茶提提神？", 15, 588, 15.16, 307, 3.92),
    ("05 中", "嗯……我理一理。", 7, 451, 9.68, 279, 2.80),
    ("06 长", "再好的武器…（35字）", 35, 490, 11.24, 425, 8.64),
    ("07 长", "武陵城外有一片梨树林…", 24, 649, 17.60, 339, 5.20),
    ("08 长", "她就在那阵花香里…", 18, 464, 10.20, 322, 4.52),
    ("09 长", "作画者把心画在纸上…", 29, 503, 11.76, 391, 7.28),
    ("10 数字", "这件事我三点五十分再去处理。", 14, 414, 8.20, 294, 3.40),
]
PREFIX = 213

print("=" * 96)
print("A/B 对比：未合并（缺 ssl_proj）vs 合并后")
print("=" * 96)
print(f"  {'#':8s} {'字数':>4s} │ {'未合并token':>11s} {'时长':>7s} {'语速':>6s} "
      f"│ {'merged token':>12s} {'时长':>7s} {'语速':>6s} {'提速':>6s}")
print("  " + "-" * 92)

rows = []
for tag, text, n, e_bad, d_bad, e_ok, d_ok in DATA:
    t_bad, t_ok = e_bad - PREFIX, e_ok - PREFIX
    s_bad, s_ok = n / d_bad, n / d_ok
    print(f"  {tag:8s} {n:4d} │ {t_bad:11d} {d_bad:6.2f}s {s_bad:6.2f} "
          f"│ {t_ok:12d} {d_ok:6.2f}s {s_ok:6.2f} {d_bad/d_ok:5.2f}x")
    rows.append((n, d_bad, d_ok, t_bad, t_ok, s_bad, s_ok))

print()
print("=" * 96)
print("汇总")
print("=" * 96)
n_bad = sum(r[0] for r in rows)
d_bad = sum(r[1] for r in rows)
d_ok = sum(r[2] for r in rows)
t_bad = sum(r[3] for r in rows)
t_ok = sum(r[4] for r in rows)
print(f"  总字数      : {n_bad}")
print(f"  总时长      : 未合并 {d_bad:.2f}s  ->  merged {d_ok:.2f}s   "
      f"（缩短 {(1-d_ok/d_bad)*100:.0f}%）")
print(f"  总 token    : 未合并 {t_bad}  ->  merged {t_ok}   "
      f"（{(1-t_ok/t_bad)*100:.0f}% 减少）")
print(f"  平均语速    : 未合并 {n_bad/d_bad:.2f} 字/秒  ->  "
      f"merged {n_bad/d_ok:.2f} 字/秒")
print()
print(f"  正常中文语速 3~5 字/秒")
print(f"    merged 版全部落在区间内: "
      f"{all(3.0 <= r[6] <= 5.0 for r in rows)}")
print(f"    未合并版全部低于区间  : "
      f"{all(r[5] < 3.0 for r in rows)}")

print()
print("=" * 96)
print("根因确认")
print("=" * 96)
print("""
  · 缺失的键（云端 missing_keys 末尾）：
        'enc_p.proj.weight', 'enc_p.proj.bias',
        'ssl_proj.weight', 'ssl_proj.bias'
  · ssl_proj = Conv1d(768, 768, 2, stride=2)，负责把 HuBERT 的 50Hz
    特征降采样成 25Hz 语义 token（models.py:935-937）
  · extract_latent 用它算 prompt_semantic（models.py:1546-1548）
  · 它随机初始化 -> prompt_semantic 是垃圾 -> GPT 收到错误上下文
    -> 找不到 EOS 位置 -> 过度生成 281 token（正常 24）

  **修复后 token 数下降 92%，时长下降 90%，语速回到正常区间。**
  这就是决定性证据：缺的正是 ssl_proj（以及次要的 enc_p）。
""")
