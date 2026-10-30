#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 ModelScope 下载正式 v5 e32，发布到 GitHub Release。

训练完成且 `final_v5_pipeline.py upload` 成功后，在本机运行：
    python publish_v5_release.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ID = "yangwenjie1231/zhuang-fangyi-voice-v2"
GH_REPO = "yangwenjie1231/dsh-zhuang-fangyi-voice-v2"
TAG = "model-v2-v5-fullft"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=r"D:\A\zfy-v5\dist\final-v5-release")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)

    from modelscope import snapshot_download
    root = Path(snapshot_download(
        REPO_ID,
        allow_patterns=["torch_weights/v5_full_e32_final_e32_*.pth", "manifest.json"],
        local_dir=str(out),
    ))
    weights = list((root / "torch_weights").glob("v5_full_e32_final_e32_*.pth"))
    manifest = root / "manifest.json"
    if len(weights) != 1 or not manifest.exists():
        print("!! 下载结果不完整", weights, manifest.exists())
        return 1
    w = weights[0]
    d = json.loads(manifest.read_text(encoding="utf-8"))
    item = next((x for x in d.get("weights", []) if x.get("epoch") == 32), None)
    if not item:
        print("!! manifest 缺 e32")
        return 1
    import hashlib
    h = hashlib.sha256(w.read_bytes()).hexdigest()
    if h != item["sha256"]:
        print("!! SHA256 不一致")
        return 1
    print("✓", w.name, f"{w.stat().st_size/1e6:.1f}MB", h[:16])
    if args.dry_run:
        return 0

    exists = subprocess.run(["gh", "release", "view", TAG, "--repo", GH_REPO],
                            capture_output=True).returncode == 0
    if not exists:
        subprocess.run([
            "gh", "release", "create", TAG, "--repo", GH_REPO,
            "--title", "庄方宜第二代语音模型 · GPT-SoVITS v5 full-FT",
            "--notes", "底层 GPT-SoVITS v5；enc_p+ssl_proj 微调；quantizer 冻结；完整自包含权重。模型权重与插件代码分开发布。",
        ], check=True)
    subprocess.run(["gh", "release", "upload", TAG, str(w), str(manifest),
                    "--repo", GH_REPO, "--clobber"], check=True)
    print("✓ GitHub Release:", f"https://github.com/{GH_REPO}/releases/tag/{TAG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
