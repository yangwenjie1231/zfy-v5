# -*- coding: utf-8 -*-
"""由日志时间戳精确推算 e32 剩余时间，并分析 loss 趋势。"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 用户提供的日志时间戳（epoch 结束时刻 = "====> Epoch: N"）
DONE = [
    ("18", "20:13:40"),
    ("19", "20:14:47"),
    ("20", "20:15:58"),
]


def sec(t: str) -> int:
    h, m, s = (int(x) for x in t.split(":"))
    return h * 3600 + m * 60 + s


print("=" * 74)
print("1. 最近几轮的实测耗时")
print("=" * 74)
durs = []
for i in range(1, len(DONE)):
    ep, t = DONE[i]
    d = sec(t) - sec(DONE[i - 1][1])
    durs.append(d)
    print(f"  epoch {ep}: {d} s")
avg = sum(durs) / len(durs)
print(f"  最近 {len(durs)} 轮平均: **{avg:.0f} s/epoch**")

print()
print("=" * 74)
print("2. 剩余时间")
print("=" * 74)
TOTAL = 32
last = int(DONE[-1][0])
remain_ep = TOTAL - last
print(f"  已完成 {last}/{TOTAL}，剩余 {remain_ep} 轮")
for label, sps in (("按最近平均", avg),
                   ("再快些（60s）", 60.0),
                   ("保守（80s）", 80.0)):
    r = remain_ep * sps
    print(f"    {label:16s} 每轮 {sps:5.1f}s -> 剩余 {r/60:5.1f} 分钟")

print()
print("  对比：epoch 2 时是 115 s/轮（缓存冷），现在 {:.0f} s/轮".format(avg))
print("        -> 速度提升了 {:.1f} 倍".format(115 / avg))

print()
print("=" * 74)
print("3. loss 趋势")
print("=" * 74)
LOSS = [(0, 0.25007), (100, 0.05656), (1000, 0.04284)]
print(f"  {'step':>6s} {'loss':>10s}")
for s, v in LOSS:
    print(f"  {s:6d} {v:10.5f}")
print()
print(f"  step 0 -> 1000: {LOSS[0][1]:.4f} -> {LOSS[-1][1]:.4f}  "
      f"= 降到 1/{LOSS[0][1]/LOSS[-1][1]:.1f}")
print(f"  step 100 -> 1000: {LOSS[1][1]:.4f} -> {LOSS[-1][1]:.4f}  "
      f"（又降 {(1-LOSS[-1][1]/LOSS[1][1])*100:.0f}%）")
print()
print("""
  ⚠️ 注意：log_interval=100，这些是**单点值**；而分桶采样器按长度分桶，
     每批样本长度不同 -> loss 天然有波动。所以不要逐点比较，
     看**量级趋势**即可：0.25 -> 0.04 量级，训练确实在收敛。

  与上一轮 e8（432 步）对比：
     e8   step 400 时 loss 0.1292
     e32  step 1000 时 loss 0.0428
  同为 0.04~0.13 区间，但 e32 的采样点更低。这**弱支持**了
  "e8 训练量不足"的假设，但要等盲听才能确认。
""")

print("=" * 74)
print("4. 跑完之后立刻要做的事")
print("=" * 74)
print("""
  权重会出现在：SoVITS_weights_v5_full_e32/
    v5_full_e32_e1_s54_l0.pth ... v5_full_e32_e32_s1728_l0.pth
    共 32 个 x 约 736 MB = 约 23.6 GB

  下一步（云端合成音频，含 v2Pro 基准）：
    python _tools/cloud_synth_sweep.py --epochs 8,16,24,32

  然后本机下载（几 MB）自己听：
    python -c "from modelscope import snapshot_download as d; \\
      d('yangwenjie1231/zfy-v5-fullft-pkg', repo_type='dataset', \\
        allow_patterns=['out_sweep/**'], local_dir=r'D:\\A\\zfy-v5\\dist\\sweep')"

  建议先挑 4 个 epoch 听（e8/e16/e24/e32），而不是 32 个全合成 ——
  每轮合成要重新加载 TTS（BERT+HuBERT+SoVITS），32 轮会比较久。
""")
