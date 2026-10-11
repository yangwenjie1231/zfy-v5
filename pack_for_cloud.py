# -*- coding: utf-8 -*-
"""打包 v5 全量微调所需的全部文件，供上传到云算力机。

## 为什么要打包

数据准备（text / hubert / semantic 特征）已在本机算好，**不需要在云上重算**：
    logs/zfy_v5/
      2-name2text.txt          825 行（音素序列）
      3-bert/                  825 个 .pt
      4-cnhubert/              825 个 .pt
      5-wav32k/                825 个 wav（32kHz）
      6-name2semantic.tsv      825 行（语义 token）

云上只需要：代码 + 底模 + 特征 + 音频。

## 包内容（预计 4-6 GB）

  GPT-SoVITS/                 代码（含 v5 模块 + 我们生成的 s2_train_v5_full.py）
    GPT_SoVITS/
      module/                 models.py / models_v5.py / v5_inference.py ...
      f5_tts/                 v5 版 DiT（必须带，否则命名空间包会解析错）
      configs/                s2.json 等
      prepare_datasets/       数据准备脚本（如需重算）
      pretrained_models/
        s1v3.ckpt             GPT（v5 与 v2Pro 共用）
        gsv-v5-pretrained/    v5 SoVITS 底模 + vocoder
        chinese-roberta-wwm-ext-large/
        chinese-hubert-base/
    logs/zfy_v5/              特征（本机已算好）
    data/zfy_v5/wav32k/       音频

用法：
    python pack_for_cloud.py                    # 打包
    python pack_for_cloud.py --dry-run          # 只看清单和体积
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

V5 = Path(r"D:\A\zfy-v5")
GSV = V5 / "GPT-SoVITS"
OUT_ZIP = V5 / "dist" / "zfy_v5_fullft_cloud.zip"

# 代码：训练 + 推理必需的子目录（不含 .git / 前端 / Colab 演示）
#
# ⚠️ 教训：最初这份清单是**只为训练**裁剪的，漏了推理专用的目录，
#    导致云端跑合成时报 ModuleNotFoundError: No module named 'BigVGAN'。
#    推理入口 `TTS_infer_pack/TTS.py` 的模块级 import 需要：
#      L29 AR.models.t2s_lightning_module -> pytorch_lightning + torchmetrics
#      L30 BigVGAN.bigvgan                -> GPT_SoVITS/BigVGAN
#      L31 feature_extractor.cnhubert     -> GPT_SoVITS/feature_extractor
#      L35 peft                           -> pip 包
#      L39 tools.audio_sr.AP_BWE          -> tools/AP_BWE_main（已含）
#      L43 sv                             -> GPT_SoVITS/sv.py
#    用 check_pack_full.py 可一次性核对覆盖情况。
CODE_DIRS = [
    "GPT_SoVITS/module",
    "GPT_SoVITS/f5_tts",
    "GPT_SoVITS/configs",
    "GPT_SoVITS/prepare_datasets",
    "GPT_SoVITS/text",
    "GPT_SoVITS/eres2net",
    "GPT_SoVITS/AR",
    "GPT_SoVITS/BigVGAN",             # ★ 推理：v3/v4/v5 的声码器
    "GPT_SoVITS/feature_extractor",   # ★ 推理：CNHubert 特征提取
    "GPT_SoVITS/Accel",               # ★ 推理：加速模块
    # ★ 推理入口目录 —— 必须**整目录**打包！
    #   我第一版只在 CODE_FILES 里列了 TTS.py 一个文件，漏掉另外 3 个，
    #   云端报 ModuleNotFoundError: No module named
    #   'TTS_infer_pack.text_segmentation_method'
    #   （TTS.py:41-42 会 import text_segmentation_method / TextPreprocessor）
    "GPT_SoVITS/TTS_infer_pack",
    "tools",
]
CODE_FILES = [
    "GPT_SoVITS/s2_train_v5_full.py",      # 我们生成的全量训练脚本
    "GPT_SoVITS/s2_train_v3_lora.py",      # 留作对照
    "GPT_SoVITS/s1_train.py",              # ★ GPT 微调（用户可能要迭代 GPT）
    "GPT_SoVITS/sv.py",                    # ★ 推理：说话人向量（TTS.py:43 用）
    "GPT_SoVITS/process_ckpt.py",
    "GPT_SoVITS/utils.py",
    "GPT_SoVITS/inference_webui.py",
    "config.py",
    "requirements.txt",
]
MODEL_DIRS = [
    "GPT_SoVITS/pretrained_models/gsv-v5-pretrained",
    "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    "GPT_SoVITS/pretrained_models/chinese-hubert-base",
    # ★ v2Pro 的说话人向量模型（ERes2NetV2）—— 合成 v2Pro 基准必需，
    #   sv.py 会按 cwd 拼 "GPT_SoVITS/pretrained_models/sv/..." 去找
    "GPT_SoVITS/pretrained_models/sv",
]
# 工作区根目录下的辅助脚本（相对 V5，不是 GSV）
EXTRA_FILES = [
    "train_v5.py",                      # 数据准备 + LoRA 训练驱动
    "notebook_preflight.py",            # 环境自检（下包前就能发现问题）
    "patch_models_for_torch212.py",     # torch2.12 兼容补丁
    "verify_torch212_patch.py",         # 补丁验证
    "MODELSCOPE_NOTEBOOK_GUIDE.md",     # 操作手册
    "CLOUD_GUIDE.md",
    "zfy_v5_fullft.ipynb",              # ★ 可直接运行的 notebook
    "setup_venv.sh",                    # ★ cell 3 失败时的 venv 退路
    "install_deps_minimal.sh",          # ★ 最小依赖（修 numpy ABI）
    "fix_numpy_abi.py",                 # ★ numpy ABI 诊断与修复
    "install_deps_v3.sh",               # ★ 最小依赖 v3（不碰 transformers）
    "check_lfs_ready.py",               # ★ 训练前 LFS 体积检查
    "verify_weights_hash.py",            # ★ 体积+sha256 双重校验
    "diag_load_audio.py",                # ★ 诊断 load audio error
    "fix_and_verify_lfs.py",             # ★ LFS 全面校验（git 方式）
    "sdk_repair_lfs.py",                 # ★ 用 SDK 修复 LFS（推荐）
    "patch_strict_audio.py",             # ★ 数据加载失败不再静默
    "install_ffmpeg.sh",                  # ★ 安装 ffmpeg 可执行文件
    "cloud_synth_sweep.py",               # ★ 云端批量合成（供盲听挑 epoch）
    "install_infer_deps.sh",              # ★ 补装推理依赖（peft/lightning/resampy）
    "train_gpt.py",                       # ★ GPT(s1) 微调 A/B 两条底模路线
    "patch_s1_train_rocm.py",             # ★ s1_train 的 nccl->gloo 补丁
    "cloud_push_weights.py",              # ★ 把训练好的权重回传数据集
    "install_all_deps.sh",                # ★ 一次装全三条链路依赖（推荐）
    "recover_instance.sh",                # ★ 新实例一键恢复（ffmpeg+依赖+盘点）
    "recover_after_reset.sh",             # ★ 实例重置后恢复（含解开 _tools 暂存改动）
    "fix_synth_deps.sh",                        # ★ 修复合成依赖（钉版本+验证+端到端探针）
    "diag_hf_conflict.py",                      # ★ 诊断 HF 版本互斥
    "run_gpt.sh",                           # ★ GPT 训练一键启动（带前置检查）
    "push_sweep_audio.py",                    # ★ 推送合成音频（rebase 方案）
    "diag_status_2561.py",                    # ★ 诊断 2500+ 项 M 真假
    "diag_gpt_training.py",                   # ★ 诊断 GPT 微调是否真在学
    "patch_v5_unfreeze.py",                   # ★ 解冻 enc_p/ssl_proj/quantizer
    "verify_unfreeze.py",                     # ★ 验证解冻生效
    "explain_gpt_loss.py",                    # ★ 解释 GPT loss 量级
    "fix_deps_patch.sh",                  # ★ 依赖修复补丁（HfFolder + 相对导入漏包）
    "run_sovits_e32.sh",                  # ★ 一键启动 SoVITS 全量微调
    "fix_weight_header.py",                 # ★ 给全量权重补版本头（b"07"）
    "patch_v5_fullft_infer.py",             # ★ 推理端补丁（LoRA 判定 + 音频读取回退）
    "verify_langsegmenter_lite.py",          # ★ 验证 LangSegmenter 零下载
    "verify_g2pw_fallback.py",                # ★ 验证 G2PW 优雅降级
    "merge_base_encp.py",                    # ★ 应急：把底模 enc_p/ssl_proj 合并进全量权重
    "verify_merge_encp.py",                  # ★ 验证 merge 脚本
    "check_gpt_bases_pack.py",                # ★ 检查两条 GPT 底模是否都打包
    "confirm_ssl_proj_root_cause.py",         # ★ 确认 ssl_proj 是真根因
    "verify_lr_schedule_dead.py",             # ★ 验证 GPT 的 LR 调度是死代码
    "test_weight_regex.py",                   # ★ 权重名正则的 14 个用例
    "resolve_git_conflict.py",                # ★ 一键解决云端 git 冲突
    "diag_lfs_status.py",                      # ★ 诊断 2500+ 项 M 是否 LFS 假象
    "verify_lfs_install.py",                   # ★ lfs install 后的验收（含关 smudge）
    "fix_push_stuck.py",                      # ★ 修云端 push 卡住（stash+pull --rebase+push）
    "compare_merged_ab.py",                   # ★ A/B 对比 merged vs 未合并
    "verify_yaml_roundtrip.py",               # ★ YAML 标量回读校验
    "diag_yaml_float.py",                     # ★ 定位 1e-05 被当字符串
    "patch_s1_lr.py",                         # ★ s1 的 LR 可控开关（默认不变）
    "verify_s1_lr_patch.py",                  # ★ 验证三种 LR 模式
]
# 参考音频 + 生产 GPT 权重 -> 仓库 ref/
# 这些是"生产已验证"的资产，用来在云端合成试听样本。
# 用**绝对源路径**引用，不在工作区复制（省 148 MB）。
REF_FILES = [
    # (仓库内相对路径, 本地绝对源路径)
    ("ref/default.wav", r"D:\A\voiceclone\zfh-models-v2pro\ref\default.wav"),
    ("ref/ref.txt", r"D:\A\voiceclone\zfh-models-v2pro\ref\default.txt"),
    ("ref/zfh-e4.ckpt",
     r"D:\A\voiceclone\zfh-models-v2pro\torch_weights\zfh-e4.ckpt"),
    # ★ 生产 v2Pro 的 SoVITS 权重 —— 云端合成 v2Pro 基准必需
    #   （没有它就只能拿 v5 自己跟自己比）
    ("ref/v2pro_sovits.pth",
     r"D:\A\voiceclone\zfh-models-v2pro\torch_weights\zfh_v2pro_e12_s384.pth"),
]
MODEL_FILES = [
    "GPT_SoVITS/pretrained_models/s1v3.ckpt",
    # ★ GPT 微调路线 B 的底模 —— 复刻生产 zfh-e4 的起点（5kh 预训练）
    #   踩过：--base both 时因这个文件缺失，整批中止，连路线 A 也没跑成。
    #   （train_gpt.py 已改为"只跳过缺的那条"，但底模本身也该打包。）
    "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/"
    "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
]
DATA_DIRS = [
    "logs/zfy_v5",              # 特征（相对 GSV）
]
DATA_FILES = [
    "data/zfy_v5/zfy_v5.list",  # 相对 V5
]

# 排除：这些不需要上云，或体积过大
EXCLUDE_PATTERNS = [
    # 训练中间状态（G_*.pth 每个 2GB，是完整模型+优化器，云上会重新生成）
    "logs_s2_",
    "G_", "D_",
    # 本机已产出的 LoRA 权重（云上是全量微调，不用这些）
    "SoVITS_weights",
    # 无关的大文件
    "G2PWModel_1.1.zip",       # 588MB，g2pW.onnx 已解压
    "ja_userdic",              # 日文用户词典 17MB，训练中文不需要
    # v5turbo 底模（只用 v5dev）
    "s2Gv5turbo.pth",
    # 缓存/日志/备份
    ".gitignore", ".gitkeep",
    ".bak", "_bak", ".orig",   # 补丁备份，云上不需要
]


def _excluded(rel: str) -> bool:
    return any(pat in rel for pat in EXCLUDE_PATTERNS)


def collect() -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []
    skip_ext = {".pyc", ".pyo", ".log", ".tmp"}
    for d in CODE_DIRS + MODEL_DIRS:
        p = GSV / d
        if not p.exists():
            print(f"  [跳过] 不存在: {d}")
            continue
        for f in p.rglob("*"):
            if not f.is_file():
                continue
            rel = str(f.relative_to(GSV))
            if f.suffix in skip_ext or "__pycache__" in f.parts:
                continue
            if _excluded(rel):
                continue
            items.append((f, rel))
    for f in CODE_FILES + MODEL_FILES:
        p = GSV / f
        if p.exists() and not _excluded(f):
            items.append((p, str(p.relative_to(GSV))))
        elif not p.exists():
            print(f"  [跳过] 不存在: {f}")
    for d in DATA_DIRS:
        p = GSV / d
        if not p.exists():
            print(f"  [跳过] 不存在: {d}")
            continue
        for f in p.rglob("*"):
            if not f.is_file():
                continue
            rel = f"logs/{f.relative_to(GSV / 'logs')}"
            if f.suffix in skip_ext or _excluded(rel):
                continue
            items.append((f, rel))
    for f in DATA_FILES:
        p = V5 / f
        if p.exists():
            items.append((p, f))
        else:
            print(f"  [跳过] 不存在: {f}")
    # 工作区根目录的辅助脚本 -> 放到 _tools/
    for f in EXTRA_FILES:
        p = V5 / f
        if p.exists():
            items.append((p, f"_tools/{f}"))
        else:
            print(f"  [跳过] 不存在: {f}")
    # 生产参考音频 / GPT 权重 -> 放到 ref/
    for rel, src in REF_FILES:
        p = Path(src)
        if p.exists():
            items.append((p, rel))
        else:
            print(f"  [跳过] ref 不存在: {rel}  <- {src}")
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(OUT_ZIP))
    args = ap.parse_args()

    print("收集文件 …")
    items = collect()
    if not items:
        print("没有可打包的文件")
        return 1

    total = sum(p.stat().st_size for p, _ in items)
    by_top: dict[str, list[int]] = {}
    for p, rel in items:
        top = rel.split("/")[0] if "/" in rel else rel
        e = by_top.setdefault(top, [0, 0])
        e[0] += 1
        e[1] += p.stat().st_size

    print(f"\n{'分类':34s} {'文件数':>8s} {'体积':>12s}")
    print("-" * 58)
    for k, (n, s) in sorted(by_top.items(), key=lambda x: -x[1][1]):
        print(f"{k:34s} {n:8d} {s/1e6:11.1f} MB")
    print("-" * 58)
    print(f"{'合计':34s} {len(items):8d} {total/1e9:11.2f} GB")

    if args.dry_run:
        print("\n(dry-run，未打包)")
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n写入 {out} …")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for i, (p, rel) in enumerate(items, 1):
            z.write(p, rel)
            if i % 300 == 0:
                print(f"  {i}/{len(items)} …", flush=True)
    print(f"\n完成: {out}")
    print(f"  体积: {out.stat().st_size/1e9:.2f} GB")
    print(f"  文件: {len(items)}")
    print("""
注意：**不推荐再用这个 zip 上云**（见 push_via_git.py）。
魔搭数据集本身是 git 仓库，直传目录树更好：
  · 压缩只省约 10%（大头是已压缩的模型权重）
  · git 可增量同步，zip 改一个文件要重传整包
  · notebook 里 clone 完即可用，无需解压

本脚本仍保留，用于：
  · 本地留一份可离线分发的归档
  · 通过其他非 git 渠道传输
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
