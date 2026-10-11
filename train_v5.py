# -*- coding: utf-8 -*-
"""GPT-SoVITS v5 微调驱动（数据准备 → SoVITS LoRA 训练）

设计依据（全部来自对 origin/main 源码的实测阅读，不是猜测）
-----------------------------------------------------------
1. **v5 只需要换 SoVITS 侧**：`config.py` 里 v5dev/v5turbo 的 GPT 都是
   `s1v3.ckpt`，与 v2Pro/v3/v4 共用。所以本脚本**不重训 GPT**。

2. **v5 不需要 SV 步骤**：`webui.py` 里 `2-get-sv.py` 的调用条件是
   `if "Pro" in version`，而 v5dev/v5turbo 不含 "Pro" → 跳过。
   （对比：v2Pro 必须跑，因为 gin_channels=1024 需要 ERes2NetV2 嵌入。）

3. **v5 强制走 LoRA**：`webui.py` 的分支是
   `if version in ["v1","v2","v2Pro","v2ProPlus"] -> s2_train.py
    else -> s2_train_v3_lora.py`
   v5dev/v5turbo 落在 else，所以训练脚本是 `s2_train_v3_lora.py`，
   配置里读 `train.lora_rank`。**这正好满足「爆显存就降秩」的要求。**

4. **v5 用 `configs/s2.json`**：webui 的选择逻辑是
   `s2.json if version not in {"v2Pro","v2ProPlus"} else s2{version}.json`。
   v5 用 s2.json（sampling_rate=32000, hop=640）。

5. **音频不必预重采样**：`tools/my_utils.load_audio` 用 ffmpeg 的
   `ar=` 直接输出目标采样率，48kHz 源可直用。但为减少每次训练的
   解码开销，本脚本仍预转 32kHz 到独立目录。

6. **list 格式**：`音频名|说话人|语言|文本`（4 段，`|` 分隔）。
   `1-get-text.py` 里 `line.split("|")` 要求**恰好 4 段**，
   所以文本里不能含 `|`。

7. **产物合并**：单卡时 prepare_datasets 写出 `X-0.<ext>`，
   必须合并成 `X.<ext>`（webui 在多卡时做，单卡也要做）。

用法：
    python train_v5.py --stage data              # 只准备数据
    python train_v5.py --stage sovits            # 只训 SoVITS LoRA
    python train_v5.py --stage all               # 全流程
    python train_v5.py --stage sovits --lora-rank 16 --batch-size 2
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------- 路径
# 跨平台：优先用环境变量，其次自动探测，最后回退本机默认值。
# 云上（魔搭 Notebook）只需：
#     export GSV_ROOT=/mnt/workspace/zfy-v5-fullft-pkg
#     export DATASET_DIR=...   （可选；数据准备已在本地做完，通常不需要）
def _detect_gsv_root() -> Path:
    """找含 config.py + GPT_SoVITS/ 的那一层（检出根）。"""
    env = os.environ.get("GSV_ROOT")
    if env and Path(env).is_dir():
        return Path(env).resolve()
    here = Path(__file__).resolve().parent
    for up in (here, *here.parents):
        if (up / "config.py").exists() and (up / "GPT_SoVITS").is_dir():
            return up
    for c in ("/mnt/workspace/zfy-v5-fullft-pkg", "/mnt/workspace/zfy-v5",
              r"D:\A\zfy-v5\GPT-SoVITS"):
        p = Path(c)
        if (p / "config.py").exists() and (p / "GPT_SoVITS").is_dir():
            return p
    return Path(r"D:\A\zfy-v5\GPT-SoVITS")


GSV = _detect_gsv_root()
# 解释器：云上用当前解释器，避免指向本机的 conda 路径
PY = os.environ.get("PYTHON_BIN") or (
    sys.executable if os.name != "nt" or "miniconda3" not in sys.executable
    else r"C:\Users\A\miniconda3\envs\voice\python.exe")
if not Path(PY).exists():
    PY = sys.executable
PRE = GSV / "GPT_SoVITS" / "pretrained_models"

# 原始素材目录（只在本地重建语料时需要；云上特征已算好，不用它）
DATASET = Path(os.environ.get(
    "DATASET_DIR",
    r"D:\Users\A\Downloads\仿通行证\新素材\音频\庄方宜_TTS训练集"))


def _detect_work() -> Path:
    """定位 list + wav32k 所在目录。

    两种布局（实测）：
      本机: D:\\A\\zfy-v5\\data\\zfy_v5\\           （检出根之外）
      云上: <检出根>/data/zfy_v5/                  （检出根之内）
    所以按候选顺序探测，取第一个含 zfy_v5.list 的。
    """
    env = os.environ.get("WORK_DIR")
    if env and (Path(env) / "zfy_v5.list").exists():
        return Path(env)
    for c in (GSV.parent / "data" / "zfy_v5",      # 云上：检出根的兄弟目录
              GSV / "data" / "zfy_v5",             # 备用
              Path(r"D:\A\zfy-v5\data\zfy_v5")):   # 本机默认
        if (c / "zfy_v5.list").exists():
            return c
    # 都没有就返回云上默认（首次生成时用）
    return GSV.parent / "data" / "zfy_v5"


WORK = _detect_work()
WAV32 = WORK / "wav32k"
LIST = WORK / "zfy_v5.list"
OPT_DIR = GSV / "logs" / "zfy_v5"

EXP_NAME = "zfy_v5"
VERSION = "v5dev"          # v5dev: 32步/cfg1.30（质量）；v5turbo: 4步/cfg0（速度）

PRETRAINED_S2G = PRE / "gsv-v5-pretrained" / "s2Gv5dev.pth"
PRETRAINED_S1 = PRE / "s1v3.ckpt"
BERT_DIR = PRE / "chinese-roberta-wwm-ext-large"
HUBERT_DIR = PRE / "chinese-hubert-base"
S2CONFIG = GSV / "GPT_SoVITS" / "configs" / "s2.json"

SPK = "zfy"
LANG = "zh"

# 8GB 显存（RTX 5060 Laptop）：
#   · grad_ckpt=True 是硬要求
#   · v5 走 LoRA，可训练参数很少，但 CFM 的中间激活仍吃显存
#   · batch_size 先用 2，跑通后再看能否升
DEFAULT_BATCH = 2
DEFAULT_EPOCHS = 8
DEFAULT_LORA_RANK = 32
GRAD_CKPT = True


def run(cmd: str, env: dict | None = None, label: str = "") -> float:
    e = os.environ.copy()
    # ---------------------------------------------------------------
    # 必须把 v5 检出根目录放进 PYTHONPATH，否则训练必然报
    #   TypeError: DiT.__init__() got an unexpected keyword argument
    #              'use_step_embedding'
    #
    # 根因（已实测确认，不是猜测）：
    #   1. `f5_tts` 是**命名空间包**（无 __init__.py），Python 会把
    #      sys.path 上所有同名目录合并进 __path__。
    #   2. `site-packages/users.pth` 把**旧检出**
    #      `D:\A\voiceclone\GPT-SoVITS\GPT_SoVITS` 注入了每个进程。
    #   3. v5 的 `GPT_SoVITS/f5_tts/model/__init__.py` 写的是
    #      `from GPT_SoVITS.f5_tts... import DiT`，
    #      而 `GPT_SoVITS` 这个顶层包会解析到旧检出（因为 v5 的根目录
    #      不在 sys.path 上，旧检出的根在）。
    #   → 结果是拿到**旧版 DiT**（没有 use_step_embedding 参数）。
    #
    # 修法：把 v5 根目录放到 PYTHONPATH 最前，让 v5 的 GPT_SoVITS 先被找到。
    # 实测这样 DiT 就解析到 v5 那份（含 use_step_embedding）。
    # ---------------------------------------------------------------
    v5_root = str(GSV)
    prev = e.get("PYTHONPATH", "")
    e["PYTHONPATH"] = v5_root + (os.pathsep + prev if prev else "")
    if env:
        e.update({k: str(v) for k, v in env.items()})
    print(f"\n{'='*72}\n[{label}] {cmd}\n{'='*72}", flush=True)
    print(f"  PYTHONPATH[0] = {v5_root}", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, shell=True, cwd=str(GSV), env=e)
    dt = time.time() - t0
    print(f"[{label}] 退出码 {p.returncode}  用时 {dt:.1f}s", flush=True)
    if p.returncode != 0:
        raise SystemExit(f"步骤失败: {label} (exit {p.returncode})")
    return dt


# ---------------------------------------------------------------- 数据准备
def build_list(min_sec=1.5, max_sec=30.0):
    """从清洗好的数据集生成 GPT-SoVITS list，并把音频统一转 32kHz。

    为什么过滤时长：
      · <1.5s：切片后不足以提供稳定音色（GPT-SoVITS 训练侧按
        3-10s 滑窗切，过短片段会被丢弃）
      · >30s：语义 token 序列过长，8GB 显存下容易 OOM
      数据集自带 `train_raw_core.jsonl` 就是 1.5-30s 的 825 条子集，
      这里按 manifest 的实测时长自己筛，避免依赖它的口径。
    """
    import csv
    WORK.mkdir(parents=True, exist_ok=True)
    WAV32.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(DATASET / "manifest.csv", encoding="utf-8-sig")))
    keep = []
    for r in rows:
        if r.get("处理") != "保留":
            continue
        try:
            dur = float(r["新时长"])
        except (KeyError, ValueError):
            continue
        text = (r.get("文本清洗") or "").strip()
        if not text or "|" in text:
            continue
        if not (min_sec <= dur <= max_sec):
            continue
        # 文件名 = 原文件名的 basename
        name = Path(r["原文件"]).name
        src = DATASET / "audio" / name
        if not src.exists():
            continue
        keep.append((name, text, dur))

    print(f"筛选: {len(keep)} 条 / "
          f"{sum(d for _, _, d in keep)/60:.1f} 分钟 "
          f"(时长 {min_sec}-{max_sec}s)")

    # 转 32kHz（ffmpeg 一次到位，避免训练时反复解码 48k）
    import shutil
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("找不到 ffmpeg")
    todo = [(n, t, d) for n, t, d in keep
            if not (WAV32 / n).exists()]
    print(f"需转码 {len(todo)} 个（已存在 {len(keep)-len(todo)} 个）")
    for i, (n, _, _) in enumerate(todo, 1):
        dst = WAV32 / n
        subprocess.run(
            [ffmpeg, "-nostdin", "-y", "-loglevel", "error",
             "-i", str(DATASET / "audio" / n),
             "-ac", "1", "-ar", "32000", "-sample_fmt", "s16", str(dst)],
            check=True)
        if i % 200 == 0:
            print(f"  {i}/{len(todo)} …", flush=True)

    # 写 list：音频名|说话人|语言|文本
    with open(LIST, "w", encoding="utf-8") as f:
        for n, t, _ in keep:
            f.write(f"{n}|{SPK}|{LANG}|{t}\n")
    print(f"list 写入 {LIST}  ({len(keep)} 行)")

    rep = {"n": len(keep), "minutes": round(sum(d for _, _, d in keep)/60, 2),
           "min_sec": min_sec, "max_sec": max_sec,
           "list": str(LIST), "wav32k": str(WAV32)}
    (WORK / "dataset_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    return keep


def stage_data():
    if not LIST.exists() or not WAV32.exists():
        build_list()
    base = {
        "inp_text": str(LIST),
        "inp_wav_dir": str(WAV32),
        "exp_name": EXP_NAME,
        "opt_dir": str(OPT_DIR),
        "i_part": "0",
        "all_parts": "1",
        "_CUDA_VISIBLE_DEVICES": "0",
        "is_half": "True",
        "version": VERSION,
    }
    run(f'"{PY}" -s GPT_SoVITS/prepare_datasets/1-get-text.py',
        {**base, "bert_pretrained_dir": str(BERT_DIR)}, "1-get-text")
    run(f'"{PY}" -s GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py',
        {**base, "cnhubert_base_dir": str(HUBERT_DIR)}, "2-get-hubert-wav32k")
    # v5 不需要 2-get-sv（webui 里条件是 "Pro" in version）
    run(f'"{PY}" -s GPT_SoVITS/prepare_datasets/3-get-semantic.py',
        {**base, "pretrained_s2G": str(PRETRAINED_S2G),
         "s2config_path": str(S2CONFIG)}, "3-get-semantic")

    merge_parts()
    verify_outputs()


def merge_parts():
    """单卡也要合并：prepare_datasets 写出 `X-0.<ext>`。"""
    src = OPT_DIR / "2-name2text-0.txt"
    if src.exists():
        lines = src.read_text(encoding="utf-8").strip("\n").split("\n")
        (OPT_DIR / "2-name2text.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
        src.unlink()
        print(f"合并 2-name2text: {len(lines)} 行")

    src = OPT_DIR / "6-name2semantic-0.tsv"
    if src.exists():
        lines = src.read_text(encoding="utf-8").strip("\n").split("\n")
        (OPT_DIR / "6-name2semantic.tsv").write_text(
            "item_name\tsemantic_audio\n" + "\n".join(lines) + "\n",
            encoding="utf-8")
        src.unlink()
        print(f"合并 6-name2semantic: {len(lines)} 行")


def verify_outputs():
    need = ["2-name2text.txt", "3-bert", "4-cnhubert", "5-wav32k",
            "6-name2semantic.tsv"]
    print("\n产物校验:")
    ok = True
    for n in need:
        p = OPT_DIR / n
        if p.exists():
            if p.is_dir():
                print(f"  OK   {n}/  ({len(list(p.iterdir()))} 项)")
            else:
                print(f"  OK   {n}  ({p.stat().st_size/1024:.0f} KB)")
        else:
            print(f"  缺失 {n}")
            ok = False
    if not ok:
        raise SystemExit("数据准备产物不完整")


# ---------------------------------------------------------------- 训练
def stage_sovits(batch_size: int, epochs: int, lora_rank: int,
                 version: str = VERSION):
    pre_s2g = PRE / "gsv-v5-pretrained" / f"s2G{version}.pth"
    if not pre_s2g.exists():
        raise SystemExit(f"缺底模: {pre_s2g}")

    with open(S2CONFIG, encoding="utf-8") as f:
        data = json.load(f)

    # v5 是 LoRA-only，没有独立的 s2D 判别器底模（webui 里 v3v4set 的
    # pretrained_s2D 文本框是空的）
    data["train"].update({
        "batch_size": batch_size,
        "epochs": epochs,
        "text_low_lr_rate": 0.4,
        "pretrained_s2G": str(pre_s2g),
        "pretrained_s2D": "",
        "if_save_latest": True,
        "if_save_every_weights": True,
        "save_every_epoch": 1,
        "gpu_numbers": "0",
        "grad_ckpt": GRAD_CKPT,
        "lora_rank": lora_rank,
    })
    data["model"]["version"] = version
    data["data"]["exp_dir"] = data["s2_ckpt_dir"] = str(OPT_DIR)
    data["save_weight_dir"] = "SoVITS_weights_v5"
    data["name"] = EXP_NAME
    data["version"] = version

    (GSV / "logs" / EXP_NAME / f"logs_s2_{version}").mkdir(
        parents=True, exist_ok=True)
    (GSV / "SoVITS_weights_v5").mkdir(parents=True, exist_ok=True)

    cfg = WORK / f"tmp_s2_{version}.json"
    cfg.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"SoVITS(v5 LoRA) 配置: {cfg}")
    print(f"  version={version}  lora_rank={lora_rank}")
    print(f"  batch_size={batch_size}  epochs={epochs}  "
          f"grad_ckpt={GRAD_CKPT}")
    print(f"  底模 G = {pre_s2g.name}")
    print(f"  sampling_rate={data['data']['sampling_rate']}  "
          f"hop={data['data']['hop_length']}")

    run(f'"{PY}" -s GPT_SoVITS/s2_train_v3_lora.py --config "{cfg}"',
        None, f"s2_train_v3_lora({version})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["data", "sovits", "all", "list"])
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK,
                    choices=[16, 32, 64, 128])
    ap.add_argument("--version", default=VERSION,
                    choices=["v5dev", "v5turbo"])
    ap.add_argument("--min-sec", type=float, default=1.5)
    ap.add_argument("--max-sec", type=float, default=30.0)
    args = ap.parse_args()

    if args.stage == "list":
        build_list(args.min_sec, args.max_sec)
        return 0
    if args.stage in ("data", "all"):
        stage_data()
    if args.stage in ("sovits", "all"):
        stage_sovits(args.batch_size, args.epochs, args.lora_rank, args.version)
    print("\n完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
