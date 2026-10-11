# -*- coding: utf-8 -*-
"""v5 LoRA 重训：用更高秩 + 更多 epoch，并**修正参考音频相关配置**。

## 为什么重训

上一轮 v5 LoRA(rank32, 8 epoch) 的音色相似度 SIM=0.8370，
低于 v2Pro 全量微调的 0.8957。已核实：

    v2 / v2Pro  -> s2_train.py            = 全量微调（673/678 键）
    v5dev/v5turbo -> s2_train_v3_lora.py  = LoRA 微调（258 键）

v5 的官方训练路径**只有 LoRA**（`s2_train.py` 用的是 `models.SynthesizerTrn`，
而 v5 的 DiT 在 `models_v5.py`，不兼容），所以不能直接全量微调。
能做的是**把 LoRA 的能力用满**：

  1. **提高秩**：32 -> 128（webui 允许 16/32/64/128）
     显存峰值 rank32 时是 5.2/8.1 GB，还有余量
  2. **增加 epoch**：8 -> 24，每 epoch 保存，事后逐 epoch 扫
  3. 保持 grad_ckpt=True

## 已知的评估陷阱（上一轮的教训）

  · 参考音频必须与 ref_text 对齐 —— 上一轮用了错配的 ref.wav，
    导致两个模型都在"补完参考文本"（字数比 1.26/1.42，正常应≈1.0）
  · 12 句样本对 CER 不够（双峰分布），配对 t 检验常常不显著
  · 必须用**同一个 GPT** 对比（生产 v2Pro 用 zfh-e4，不是 s1v3）

用法：
    python retrain_v5.py --lora-rank 128 --epochs 24
    python retrain_v5.py --lora-rank 64  --epochs 16    # 显存不够时降级
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

V5 = Path(r"D:\A\zfy-v5")
GSV = V5 / "GPT-SoVITS"
PY = r"C:\Users\A\miniconda3\envs\voice\python.exe"
PRE = GSV / "GPT_SoVITS" / "pretrained_models"

WORK = V5 / "data" / "zfy_v5"
OPT_DIR = GSV / "logs" / "zfy_v5"
S2CONFIG = GSV / "GPT_SoVITS" / "configs" / "s2.json"
VERSION = "v5dev"


def run(cmd: str, env: dict | None = None, label: str = "") -> float:
    e = os.environ.copy()
    # 必须让 v5 根目录优先，否则 f5_tts 命名空间包会解析到旧检出的 DiT
    # （无 use_step_embedding）-> TypeError
    prev = e.get("PYTHONPATH", "")
    e["PYTHONPATH"] = str(GSV) + (os.pathsep + prev if prev else "")
    if env:
        e.update({k: str(v) for k, v in env.items()})
    print(f"\n{'='*74}\n[{label}] {cmd}\n{'='*74}", flush=True)
    print(f"  PYTHONPATH[0] = {GSV}", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, shell=True, cwd=str(GSV), env=e)
    dt = time.time() - t0
    print(f"[{label}] 退出码 {p.returncode}  用时 {dt:.1f}s", flush=True)
    if p.returncode != 0:
        raise SystemExit(f"步骤失败: {label}")
    return dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora-rank", type=int, default=128,
                    choices=[16, 32, 64, 128])
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--tag", default=None,
                    help="输出目录后缀，默认 rank{rank}")
    args = ap.parse_args()

    tag = args.tag or f"r{args.lora_rank}"
    pre_s2g = PRE / "gsv-v5-pretrained" / f"s2G{VERSION}.pth"
    if not pre_s2g.exists():
        raise SystemExit(f"缺底模: {pre_s2g}")

    with open(S2CONFIG, encoding="utf-8") as f:
        data = json.load(f)

    data["train"].update({
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "text_low_lr_rate": 0.4,
        "pretrained_s2G": str(pre_s2g),
        "pretrained_s2D": "",
        "if_save_latest": True,
        "if_save_every_weights": True,
        "save_every_epoch": args.save_every,
        "gpu_numbers": "0",
        "grad_ckpt": True,          # 8GB 显存硬要求
        "lora_rank": args.lora_rank,
    })
    data["model"]["version"] = VERSION
    data["data"]["exp_dir"] = data["s2_ckpt_dir"] = str(OPT_DIR)
    data["save_weight_dir"] = f"SoVITS_weights_v5_{tag}"
    data["name"] = f"zfy_v5_{tag}"
    data["version"] = VERSION

    (GSV / "logs" / "zfy_v5" / f"logs_s2_{VERSION}_lora_{args.lora_rank}").mkdir(
        parents=True, exist_ok=True)
    (GSV / f"SoVITS_weights_v5_{tag}").mkdir(parents=True, exist_ok=True)

    cfg = WORK / f"tmp_s2_{VERSION}_{tag}.json"
    cfg.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    print(f"重训配置:")
    print(f"  version    = {VERSION}")
    print(f"  lora_rank  = {args.lora_rank}   (上一轮 32)")
    print(f"  epochs     = {args.epochs}      (上一轮 8)")
    print(f"  batch_size = {args.batch_size}")
    print(f"  grad_ckpt  = True")
    print(f"  底模       = {pre_s2g.name}")
    print(f"  权重输出   = SoVITS_weights_v5_{tag}/")
    print(f"  日志       = logs_s2_{VERSION}_lora_{args.lora_rank}/")

    run(f'"{PY}" -s GPT_SoVITS/s2_train_v3_lora.py --config "{cfg}"',
        None, f"s2_train_v3_lora({VERSION}, rank={args.lora_rank})")

    wdir = GSV / f"SoVITS_weights_v5_{tag}"
    ck = sorted(wdir.glob("*.pth"))
    print(f"\n完成，产出 {len(ck)} 个权重:")
    for p in ck:
        print(f"  {p.name:34s} {p.stat().st_size/1e6:6.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
