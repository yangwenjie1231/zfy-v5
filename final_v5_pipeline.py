#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""庄方宜第二代（底层 GPT-SoVITS v5）最终 full-FT 一键流水线。

动作：
  doctor    环境/数据/底模体检
  prepare   应用补丁 + 生成最终配置（e8/e16/e24/e32）
  launch    后台启动训练
  status    查看 PID / 日志 / 权重进度
  validate  补权重头 + 完整 key / quantizer / enc_p / ssl_proj 验收
  upload    上传正式权重到 ModelScope 模型仓库

正式训练模式：
  ZFY_UNFREEZE=enc_p,ssl_proj
  quantizer 保持冻结且进入导出权重
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

TAG = "v5_full_e32_final"
WEIGHT_DIR_NAME = f"SoVITS_weights_{TAG}"
EXP_DIR_NAME = f"logs/{TAG}"
CFG_NAME = f"tmp_{TAG}.json"
LOG_NAME = f"train_{TAG}.log"
PID_NAME = f"train_{TAG}.pid"
MODEL_REPO = "yangwenjie1231/zhuang-fangyi-voice-v2"
EPOCHS = 32
BATCH = 16
SAVE_EVERY = 8
EXPECTED_EPOCHS = [8, 16, 24, 32]
EXPECTED_KEYS_APPROX = 681


def find_root() -> Path:
    cands = [Path.cwd(), Path("/mnt/workspace/zfy-v5-fullft-pkg")]
    cands += list(Path.cwd().parents)
    for c in cands:
        if (c / "GPT_SoVITS" / "configs" / "s2.json").is_file() and \
                (c / "logs" / "zfy_v5").is_dir():
            return c.resolve()
    raise SystemExit("找不到训练根：需要 GPT_SoVITS/configs/s2.json + logs/zfy_v5/")


ROOT = find_root()
GSV = ROOT / "GPT_SoVITS"
TOOLS = ROOT / "_tools" if (ROOT / "_tools").is_dir() else ROOT
WEIGHT_DIR = ROOT / WEIGHT_DIR_NAME
EXP_DIR = ROOT / EXP_DIR_NAME
CFG = ROOT / CFG_NAME
LOG = ROOT / LOG_NAME
PID = ROOT / PID_NAME


def tool(name: str) -> Path:
    p = TOOLS / name
    if not p.exists():
        raise SystemExit(f"缺工具: {p}")
    return p


def run(cmd, *, check=True, env=None, cwd=ROOT):
    print("$", " ".join(map(str, cmd)))
    r = subprocess.run(list(map(str, cmd)), cwd=str(cwd), env=env,
                       text=True, encoding="utf-8", errors="replace")
    if check and r.returncode:
        raise SystemExit(r.returncode)
    return r


def link_features(exp: Path):
    src = ROOT / "logs" / "zfy_v5"
    for name in ("2-name2text.txt", "3-bert", "4-cnhubert", "5-wav32k",
                 "6-name2semantic.tsv"):
        s, d = src / name, exp / name
        if not s.exists():
            raise SystemExit(f"缺训练特征: {s}")
        if d.exists() or d.is_symlink():
            continue
        try:
            d.symlink_to(s, target_is_directory=s.is_dir())
        except OSError:
            # NFS/平台不支持软链时复制；仅兜底
            if s.is_dir():
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
        print(f"  ✓ {name}")


def parse_epoch(name: str) -> int:
    m = re.search(r"_e(\d+)_s\d+_l\d+", name)
    return int(m.group(1)) if m else 0


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def doctor() -> int:
    print("=" * 90)
    print("最终 v5 full-FT 体检")
    print("=" * 90)
    print("ROOT:", ROOT)
    print("Python:", sys.version.split()[0])
    try:
        import torch
        print("torch:", torch.__version__, "hip=", getattr(torch.version, "hip", None))
        print("GPU 可用:", torch.cuda.is_available(), "数量:", torch.cuda.device_count())
        if torch.cuda.is_available():
            print("GPU0:", torch.cuda.get_device_name(0),
                  f"显存 {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    except Exception as e:
        print("!! torch:", type(e).__name__, e)
        return 1

    checks = {
        "训练脚本": GSV / "s2_train_v5_full.py",
        "v5 底模": GSV / "pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth",
        "v5 vocoder": GSV / "pretrained_models/gsv-v5-pretrained/vocoder.pth",
        "语义表": ROOT / "logs/zfy_v5/6-name2semantic.tsv",
        "文本表": ROOT / "logs/zfy_v5/2-name2text.txt",
        "BERT 特征": ROOT / "logs/zfy_v5/3-bert",
        "HuBERT 特征": ROOT / "logs/zfy_v5/4-cnhubert",
        "wav32k": ROOT / "logs/zfy_v5/5-wav32k",
    }
    bad = 0
    for label, p in checks.items():
        ok = p.exists()
        print(f"  {'✓' if ok else '✗'} {label:12s} {p}")
        bad += 0 if ok else 1
    sem = checks["语义表"]
    if sem.exists():
        n = len([x for x in sem.read_text(encoding="utf-8", errors="replace").splitlines()
                 if x.strip()]) - 1
        print("  语料条数:", n, "（期望约 825）")
        bad += 1 if n < 800 else 0
    print("  目标输出:", WEIGHT_DIR_NAME, "epochs", EXPECTED_EPOCHS)
    print("  解冻模式: enc_p,ssl_proj；quantizer 冻结")
    return 1 if bad else 0


def apply_patches():
    scripts = [
        "patch_models_for_torch212.py",
        "patch_core_vq_dist.py",
        "patch_ckpt_atomic.py",
        "patch_unfreeze_fine.py",
        "patch_full_export_all.py",
        "patch_v5_fullft_load.py",
        "patch_no_ffmpeg.py",
        "patch_infer_env.py",
    ]
    for name in scripts:
        p = TOOLS / name
        if not p.exists():
            if name == "patch_models_for_torch212.py":
                print("  ~ 可选补丁不存在，跳过:", name)
                continue
            raise SystemExit(f"缺补丁: {p}")
        run([sys.executable, p, "--root", ROOT])


def write_config(force=False):
    if (WEIGHT_DIR.exists() or EXP_DIR.exists()) and not force:
        raise SystemExit(
            f"目标已存在：{WEIGHT_DIR if WEIGHT_DIR.exists() else EXP_DIR}\n"
            "为防覆盖，确认要清空后加 prepare --force")
    if force:
        for p in (WEIGHT_DIR, EXP_DIR):
            if p.exists():
                print("  删除旧目标:", p)
                shutil.rmtree(p)
    WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    link_features(EXP_DIR)

    cfg = json.loads((GSV / "configs/s2.json").read_text(encoding="utf-8"))
    cfg["train"].update({
        "batch_size": BATCH,
        "epochs": EPOCHS,
        "pretrained_s2G": str(GSV / "pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth"),
        "pretrained_s2D": "",
        "if_save_latest": True,
        "if_save_every_weights": True,
        "save_every_epoch": SAVE_EVERY,
        "gpu_numbers": "0",
        "grad_ckpt": True,
        "lora_rank": 0,
        "fp16_run": True,
    })
    cfg["model"]["version"] = "v5dev"
    # 必须保持 True；细粒度补丁根据环境变量只解冻 enc_p/ssl_proj
    cfg["model"]["freeze_quantizer"] = True
    cfg["data"]["exp_dir"] = cfg["s2_ckpt_dir"] = str(EXP_DIR)
    cfg["save_weight_dir"] = WEIGHT_DIR.name
    cfg["name"] = TAG
    cfg["version"] = "v5dev"
    CFG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    back = json.loads(CFG.read_text(encoding="utf-8"))
    assert back["model"]["freeze_quantizer"] is True
    assert back["train"]["save_every_epoch"] == 8
    print("✓ 配置:", CFG)
    print("  batch=16 epochs=32 save_every=8 fp16=True")
    print("  quantizer 配置保持冻结；运行时只解冻 enc_p+ssl_proj")


def prepare(force=False) -> int:
    if doctor():
        return 1
    apply_patches()
    # 验证 fine 模式：会端到端检查码本 delta
    run([sys.executable, tool("verify_unfreeze_fine.py"), "--root", ROOT])
    write_config(force=force)
    # 静态标记检查
    models = (GSV / "module/models.py").read_text(encoding="utf-8", errors="replace")
    train = (GSV / "s2_train_v5_full.py").read_text(encoding="utf-8", errors="replace")
    assert "ZFY_UNFREEZE_FINE_PATCH" in models
    assert "ZFY_UNFREEZE_PATCH" not in models
    assert "ZFY_FULL_EXPORT_ALL_PATCH" in train
    print("✓ 预检：fine patch + export-all patch 均存在，旧 all-unfreeze marker 不存在")
    print("\n下一步：python _tools/final_v5_pipeline.py launch")
    return 0


def launch() -> int:
    if not CFG.exists():
        raise SystemExit("缺配置；先执行 prepare")
    if PID.exists():
        try:
            pid = int(PID.read_text().strip())
            os.kill(pid, 0)
            raise SystemExit(f"训练似乎已在运行 pid={pid}")
        except ProcessLookupError:
            PID.unlink(missing_ok=True)
        except ValueError:
            PID.unlink(missing_ok=True)
    env = os.environ.copy()
    env.update({
        "ZFY_UNFREEZE": "enc_p,ssl_proj",
        "ZFY_KEEP_CKPTS": "2",
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONPATH": str(ROOT),
        "PYTHONUNBUFFERED": "1",
    })
    logf = LOG.open("ab", buffering=0)
    cmd = [sys.executable, "-s", str(GSV / "s2_train_v5_full.py"),
           "--config", str(CFG)]
    p = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=logf,
                         stderr=subprocess.STDOUT, start_new_session=True)
    PID.write_text(str(p.pid), encoding="ascii")
    print(f"✓ 已启动 pid={p.pid}")
    print("  日志:", LOG)
    print("  必须看到: unfrozen ['enc_p', 'ssl_proj']; still frozen ['quantizer']")
    return 0


def status() -> int:
    alive = False
    pid = None
    if PID.exists():
        try:
            pid = int(PID.read_text().strip())
            os.kill(pid, 0)
            alive = True
        except Exception:
            alive = False
    print(f"PID: {pid}  alive={alive}")
    ws = sorted(WEIGHT_DIR.glob("*.pth"), key=lambda p: parse_epoch(p.name)) \
        if WEIGHT_DIR.is_dir() else []
    print("权重:", [(parse_epoch(p.name), round(p.stat().st_size/1e6, 1)) for p in ws])
    if LOG.exists():
        raw = LOG.read_bytes().decode("utf-8", errors="replace")
        lines = [x for x in re.split(r"[\r\n]+", raw) if x.strip()]
        epochs = [int(m.group(1)) for x in lines
                  for m in [re.search(r"====>\s*Epoch:\s*(\d+)", x)] if m]
        print("训练 epoch:", (min(epochs), max(epochs)) if epochs else "尚未识别")
        print("--- 日志末尾 ---")
        for x in lines[-20:]:
            print(x[:220])
    return 0


def load_weight(path: Path):
    sys.path.insert(0, str(GSV))
    from process_ckpt import load_sovits_new
    return load_sovits_new(str(path))


def validate() -> int:
    ws = sorted(WEIGHT_DIR.glob("*.pth"), key=lambda p: parse_epoch(p.name))
    epochs = [parse_epoch(p.name) for p in ws]
    if epochs != EXPECTED_EPOCHS:
        print("✗ 权重 epoch:", epochs, "期望", EXPECTED_EPOCHS)
        return 1
    # 补头（幂等）
    run([sys.executable, tool("fix_weight_header.py"), "--dir", WEIGHT_DIR,
         "--apply"])
    base = GSV / "pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth"
    db = load_weight(base)
    base_keys = set(db["weight"])
    report = []
    bad = 0
    for p in ws:
        d = load_weight(p)
        keys = set(d["weight"])
        missing = sorted(base_keys - keys)
        groups = {g: sum(k.startswith(g + ".") or k.startswith(g) for k in keys)
                  for g in ("enc_p", "ssl_proj", "quantizer", "cfm")}
        ok = (not missing and groups["enc_p"] > 0 and groups["ssl_proj"] > 0
              and groups["quantizer"] > 0 and len(keys) >= EXPECTED_KEYS_APPROX - 10)
        bad += 0 if ok else 1
        item = {"epoch": parse_epoch(p.name), "file": p.name,
                "size": p.stat().st_size, "sha256": sha256(p),
                "keys": len(keys), "missing_vs_base": len(missing),
                "groups": groups, "header": p.read_bytes()[:2].decode("ascii", "replace"),
                "ok": ok}
        report.append(item)
        print(f"{'✓' if ok else '✗'} e{item['epoch']:02d} {item['size']/1e6:.1f}MB "
              f"keys={item['keys']} missing={item['missing_vs_base']} {groups} header={item['header']}")
    out = ROOT / "final_v5_manifest.json"
    out.write_text(json.dumps({"tag": TAG, "mode": "enc_p+ssl_proj train, quantizer frozen",
                               "weights": report}, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print("manifest:", out)
    return 1 if bad else 0


def upload() -> int:
    """上传完整可用模型包。

    正式推理只需要 e32；e8/e16/e24 放 checkpoints 供回归与选优。
    GPT 与三种参考音频一并上传，保证 model repo 自包含。
    """
    if validate():
        return 1
    ws = {parse_epoch(p.name): p for p in WEIGHT_DIR.glob("*.pth")}
    items = []
    # 正式 SoVITS 权重
    items.append((ws[32], f"torch_weights/{ws[32].name}"))
    # 中间 epoch（不重复上传 e32）
    for ep in (8, 16, 24):
        items.append((ws[ep], f"checkpoints/v5_fullft_final/{ws[ep].name}"))
    # GPT / 默认参考 / 三种风格
    items += [
        (ROOT / "ref/zfh-e4.ckpt", "torch_weights/zfh-e4.ckpt"),
        (ROOT / "ref/default.wav", "ref/default.wav"),
        (ROOT / "ref/ref.txt", "ref/default.txt"),
        (ROOT / "ref/styles", "ref/styles"),
        (ROOT / "final_v5_manifest.json", "manifest.json"),
        (CFG, "training/final_config.json"),
        (LOG, "training/train_v5_full_e32_final.log"),
    ]
    for src, remote in items:
        if not src.exists():
            print("~ 跳过不存在:", src)
            continue
        run(["modelscope", "upload", MODEL_REPO, src, remote])
    print("✓ ModelScope 完整模型包上传完成:", MODEL_REPO)
    print("  正式 SoVITS: torch_weights/" + ws[32].name)
    print("  GPT: torch_weights/zfh-e4.ckpt")
    print("  风格: ref/styles/{normal,firm,gentle}.wav")
    print("GitHub Release 在本机执行 publish_v5_release.py 即可。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="action", required=True)
    sub.add_parser("doctor")
    p = sub.add_parser("prepare")
    p.add_argument("--force", action="store_true")
    sub.add_parser("launch")
    sub.add_parser("status")
    sub.add_parser("validate")
    sub.add_parser("upload")
    args = ap.parse_args()
    return {"doctor": doctor, "prepare": lambda: prepare(args.force),
            "launch": launch, "status": status, "validate": validate,
            "upload": upload}[args.action]()


if __name__ == "__main__":
    raise SystemExit(main())
