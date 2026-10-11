# -*- coding: utf-8 -*-
"""汇总 v5 vs v2Pro 的全部客观指标，并给出选点结论。

数据来源（都已落盘）：
  out/fair_compare/metrics.json   逐句 SIM / CER / band + 配对检验
  out/sweep_v5/sweep.json         8 个 epoch 的逐 epoch 指标
  out/hf_check/hf_check.json      16-24kHz 真实性核查
  out/spectrum_fair/spectrum.json 全频带对比

本脚本只做汇总与判定，不重新测量。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

V5 = Path(r"D:\A\zfy-v5\out")


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> int:
    fair = load(V5 / "fair_compare" / "metrics.json")
    sweep = load(V5 / "sweep_v5" / "sweep.json")
    hf = load(V5 / "hf_check" / "hf_check.json")
    spec = load(V5 / "spectrum_fair" / "spectrum.json")

    print("=" * 82)
    print("一、逐 epoch 指标（12 句，同参考/同 seed）")
    print("=" * 82)
    if sweep:
        print(f"{'epoch':>6s} {'12-16k/8-12k':>14s} {'CER':>8s} {'SIM':>8s}")
        print("-" * 44)
        for r in sweep["rows"]:
            print(f"{r['epoch']:6d} {r['band_ratio']:14.2f} "
                  f"{r['cer']*100:7.1f}% {r['sim']:8.4f}")
        # 逐 epoch 的 SIM 波动幅度
        sims = [r["sim"] for r in sweep["rows"]]
        cers = [r["cer"] for r in sweep["rows"]]
        print(f"\n  SIM 波动: {min(sims):.4f} ~ {max(sims):.4f}  "
              f"(极差 {max(sims)-min(sims):.4f})")
        print(f"  CER 波动: {min(cers)*100:.1f}% ~ {max(cers)*100:.1f}%  "
              f"(极差 {(max(cers)-min(cers))*100:.1f}pp)")
        print("  注：时长与 GPT 无关（GPT 未微调），故各 epoch 时长完全相同。")

    print("\n" + "=" * 82)
    print("二、v5 vs v2Pro 配对显著性（同 12 句）")
    print("=" * 82)
    if fair:
        for tag, m in fair["metrics"].items():
            print(f"{tag:10s} band={np.mean(m['band']):7.2f}  "
                  f"CER={np.mean(m['cer'])*100:5.1f}%  "
                  f"SIM={np.mean(m['sim']):.4f}  sr={m['sr']}")
        print()
        for k, v in fair["paired_tests"].items():
            print(f"{k}:")
            for mk, mv in v.items():
                print(f"   {mk:5s} 差={mv['mean']:+.4f}  t={mv['t']:6.2f}  "
                      f"临界={mv['crit']}  "
                      f"{'显著' if mv['sig'] else '不显著'}  "
                      f"(胜{mv['win']}/负{mv['lose']})")

    print("\n" + "=" * 82)
    print("三、高频真实性核查（16-24kHz 是信号还是噪声）")
    print("=" * 82)
    if hf:
        print(f"{'来源':18s} {'16-20k':>8s} {'平坦度':>8s} {'包络相关':>9s}")
        print("-" * 46)
        for r in hf:
            lv = r["levels"].get("16-20k")
            fl = r["flatness_16_20k"]
            ec = r["envcorr_16_20k"]
            print(f"{r['tag']:18s} "
                  f"{(f'{lv:8.2f}' if lv is not None else '--'.rjust(8))} "
                  f"{(f'{fl:8.4f}' if fl is not None else '--'.rjust(8))} "
                  f"{(f'{ec:+9.4f}' if ec is not None else '--'.rjust(9))}")
        print("\n  判读：平坦度 « 1 且包络相关 > 0.9 → 是真实谐波，不是噪声")

    print("\n" + "=" * 82)
    print("四、结论")
    print("=" * 82)
    if fair and spec:
        v5g = next((g for g in spec["groups"] if g["tag"].startswith("v5_e")), None)
        v2g = next((g for g in spec["groups"] if g["tag"] == "v2pro"), None)
        t_sim = fair["paired_tests"].get("v5_e8_vs_v2pro", {}).get("sim", {})
        t_cer = fair["paired_tests"].get("v5_e8_vs_v2pro", {}).get("cer", {})

        print(f"""
1) 带宽：v5 输出 48kHz、有效带宽约 {v5g['bandwidth_mean']:.0f} Hz；
   v2Pro 输出 32kHz、硬上限 {v2g['bandwidth_mean']:.0f} Hz。
   v5 的 16-20k 内容包络相关 +0.99（真实谐波），不是声码器噪声。

2) 12-16kHz：v5 {v5g['bands']['12-16k']} dB vs v2Pro {v2g['bands']['12-16k']} dB
   —— 绝对量级相当（差 {abs(v5g['bands']['12-16k']-v2g['bands']['12-16k']):.2f} dB）。
   注意：我之前用「12-16k / 8-12k 比值」比较得出"v5 差 5dB"是**错误指标** ——
   v5 的 8-12k 本身比 v2Pro 亮 4.1dB，比值法等于惩罚了"低频段更亮"。
   正确做法是看绝对频带能量（本表）。

3) 可懂度（CER）：v5 {np.mean(fair['metrics']['v5_e8']['cer'])*100:.1f}% vs
   v2Pro {np.mean(fair['metrics']['v2pro']['cer'])*100:.1f}%，
   配对差 {t_cer.get('mean', 0)*100:+.1f}pp，
   t={t_cer.get('t')} → {'显著' if t_cer.get('sig') else '不显著'}。

4) 音色相似度（同 ERes2NetV2 验证器）：v5 {np.mean(fair['metrics']['v5_e8']['sim']):.4f} vs
   v2Pro {np.mean(fair['metrics']['v2pro']['sim']):.4f}，
   配对差 {t_sim.get('mean', 0):+.4f}，t={t_sim.get('t')} →
   {'v2Pro 显著更好' if t_sim.get('sig') and t_sim.get('mean', 0) < 0 else '不显著'}。
""")

    best = sweep["best_epoch"] if sweep else None
    print(f"5) 选点：sweep 综合评分最高为 epoch {best}。")
    print("   但注意 SIM 的 epoch 间极差仅 "
          f"{max(r['sim'] for r in sweep['rows'])-min(r['sim'] for r in sweep['rows']):.4f}，"
          "与 v5-v2Pro 的差距同量级，")
    print("   说明**单看 SIM 无法可靠区分 epoch**。建议以听感为主、指标为辅。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
