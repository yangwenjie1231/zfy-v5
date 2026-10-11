# -*- coding: utf-8 -*-
"""验证 train_gpt.py 生成的 YAML 能被正确解析。

不真跑训练，只验证：
  1. YAML 语法合法
  2. 关键字段齐全（漏一个 s1_train.py 就会崩）
  3. model.phoneme_vocab_size == 732（与底模一致，模板里的 512 是错的）
  4. 路径字段指向真实存在的文件
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import yaml

spec = importlib.util.spec_from_file_location(
    "tg", r"D:\A\zfy-v5\train_gpt.py")
tg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tg)

ROOT = Path(r"D:\A\zfy-v5\GPT-SoVITS")
phoneme = ROOT / "logs" / "zfy_v5" / "2-name2text.txt"
semantic = ROOT / "logs" / "zfy_v5" / "6-name2semantic.tsv"

# s1_train.py / t2s_lightning_module / dataset 真正读取的键
REQUIRED = {
    "train": ["seed", "epochs", "batch_size", "save_every_n_epoch",
              "precision", "gradient_clip", "exp_name",
              "half_weights_save_dir", "if_save_latest",
              "if_save_every_weights"],
    "optimizer": ["lr", "lr_init", "lr_end", "warmup_steps", "decay_steps"],
    "data": ["max_eval_sample", "max_sec", "num_workers", "pad_val"],
    "model": ["vocab_size", "phoneme_vocab_size", "embedding_dim",
              "hidden_dim", "head", "linear_units", "n_layer", "dropout",
              "EOS", "random_bert"],
    "inference": ["top_k"],
}
TOP = ["output_dir", "pretrained_s1", "train_phoneme_path",
       "train_semantic_path"]

print("=" * 74)
print("生成 YAML 并校验（A / B 两条路线）")
print("=" * 74)

tmp = Path(tempfile.mkdtemp(prefix="gptcfg_"))
fails = []
for key in ("s1v3", "v2"):
    info = tg.BASES[key]
    tag = info["tag"]
    cfg = tg.build_config(
        root=ROOT, base_rel=info["rel"], exp=f"zfy_gpt{tag}",
        half_dir=str(ROOT / f"GPT_weights_gpt{tag}"),
        out_dir=ROOT / "logs" / f"gpt{tag}",
        phoneme=phoneme, semantic=semantic,
        epochs=8, batch=8, warmup=2000, decay=40000,
        precision="16-mixed", max_sec=54, num_workers=4)
    p = tmp / f"tmp_s1_{key}.yaml"
    tg.write_yaml(cfg, p)

    print(f"\n--- [{tag}] {key} -> {p.name} ---")
    try:
        loaded = yaml.safe_load(p.read_text(encoding="utf-8"))
        print("  YAML 解析: OK")
    except Exception as e:  # noqa: BLE001
        print(f"  YAML 解析失败: {e}")
        fails.append(f"{key}: yaml 解析失败")
        continue

    for sec, keys in REQUIRED.items():
        if sec not in loaded:
            fails.append(f"{tag}: 缺 section {sec}")
            continue
        for k in keys:
            if k not in loaded[sec]:
                fails.append(f"{tag}: 缺 {sec}.{k}")
    for k in TOP:
        if k not in loaded:
            fails.append(f"{tag}: 缺顶层 {k}")

    # 关键值
    pv = loaded["model"]["phoneme_vocab_size"]
    print(f"  phoneme_vocab_size = {pv}  "
          f"{'OK' if pv == 732 else '!! 应为 732'}")
    if pv != 732:
        fails.append(f"{tag}: phoneme_vocab_size={pv} != 732")

    bp = ROOT / loaded["pretrained_s1"]
    print(f"  pretrained_s1 存在: {bp.exists()}  ({loaded['pretrained_s1'][-46:]})")
    if not bp.exists():
        fails.append(f"{tag}: 底模不存在")

    for k in ("train_phoneme_path", "train_semantic_path"):
        v = Path(loaded[k])
        print(f"  {k}: {'存在' if v.exists() else '**缺失**'}")
        if not v.exists():
            fails.append(f"{tag}: {k} 不存在")

    hd = Path(loaded["train"]["half_weights_save_dir"])
    print(f"  half_weights_save_dir: {hd}")

print()
print("=" * 74)
if fails:
    print(f"✗ {len(fails)} 个问题：")
    for f in fails:
        print(f"    {f}")
    sys.exit(1)
print("✓ 配置字段齐全、数值正确、路径存在")
print("=" * 74)

print("\n生成的 YAML 内容（A 路线）:")
print((tmp / "tmp_s1_s1v3.yaml").read_text(encoding="utf-8"))
import shutil
shutil.rmtree(tmp, ignore_errors=True)
