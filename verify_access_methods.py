# -*- coding: utf-8 -*-
"""验证两种取包方式在我们的仓库结构上是否可用。

用户给了两种官方方式：
  SDK: api.login(token) + MsDataset.load('owner/repo')
  GIT: git lfs install + git clone https://oauth2:<token>@...

但我们的仓库**不是标准 ML 数据集**：
  · 根目录是 GPT_SoVITS/ logs/ data/ _tools/ —— 代码 + 模型权重 + 特征
  · 没有数据集常见的 data/*.csv / *.jsonl 之类可解析的数据文件
  · 大头是 .pth/.ckpt 模型权重（走 LFS）

`MsDataset.load()` 会尝试按数据集格式解析仓库内容，
对这种"代码+模型"结构**很可能失败或行为异常**。

本脚本：
  1. 用本地凭据（不打印明文）测试 git clone 是否可用
  2. 尝试 MsDataset.load，如实报告结果
  3. 给出推荐方式

用法：
    python verify_access_methods.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NS, REPO = "yangwenjie1231", "zfy-v5-fullft-pkg"
RID = f"{NS}/{REPO}"


def local_token() -> str:
    p = Path.home() / ".modelscope" / "credentials" / "git_token"
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


print("=" * 74)
print("1) GIT 方式（推荐）")
print("=" * 74)
tok = local_token()
if not tok:
    print("  无本地凭据，跳过")
else:
    url = f"https://oauth2:{tok}@www.modelscope.cn/datasets/{RID}.git"
    tmp = Path(tempfile.mkdtemp(prefix="ms_acc_"))
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    try:
        r = subprocess.run(
            ["git", "ls-remote", url, "HEAD"],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", env=env, timeout=90)
        if r.returncode == 0 and r.stdout.strip():
            print(f"  OK  ls-remote 成功: {r.stdout.split()[0][:12]}…")
        else:
            print(f"  失败: {r.stderr[-200:]}")
        # 只读 clone 结构（不拉 LFS）
        r2 = subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none",
             "--no-checkout", url, "r"],
            cwd=str(tmp), capture_output=True, text=True,
            encoding="utf-8", errors="replace", env=env, timeout=180)
        if r2.returncode == 0:
            n = subprocess.run(
                ["git", "ls-tree", "-r", "--name-only", "HEAD"],
                cwd=str(tmp / "r"), capture_output=True, text=True,
                encoding="utf-8", errors="replace").stdout
            cnt = len([x for x in n.split("\n") if x.strip()])
            print(f"  OK  clone 成功，仓库内 {cnt} 个文件")
            print(f"  → git clone 方式**可用**")
        else:
            print(f"  clone 失败: {r2.stderr[-200:]}")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

print()
print("=" * 74)
print("2) SDK 方式 MsDataset.load()")
print("=" * 74)
try:
    from modelscope.msdatasets import MsDataset
    print("  MsDataset 可导入")
except Exception as e:  # noqa: BLE001
    print(f"  MsDataset 导入失败: {type(e).__name__}: {e}")
    MsDataset = None

if MsDataset is not None:
    print("\n  尝试 load（只探结构，不实际下载大文件）…")
    try:
        ds = MsDataset.load(RID, download_mode="force_redownload")
        print(f"  结果: {type(ds)}")
        print(f"  → MsDataset.load 返回了对象（但内容语义需确认）")
    except Exception as e:  # noqa: BLE001
        print(f"  失败: {type(e).__name__}")
        print(f"  {str(e)[:300]}")
        print("\n  说明：我们的仓库是「代码 + 模型权重 + 特征」的目录树，")
        print("        不是标准 ML 数据集（无可解析的数据文件）。")
        print("        MsDataset 期望的是 dataset 结构，因此不适用。")

print()
print("=" * 74)
print("结论")
print("=" * 74)
print("""
  推荐 **git clone**：
    git lfs install
    git clone https://oauth2:<你的token>@www.modelscope.cn/datasets/\\
yangwenjie1231/zfy-v5-fullft-pkg.git zfy-v5

  理由：
    · 仓库是「代码 + 模型权重 + 特征」，git clone 能完整还原目录树
    · LFS 自动拉取 .pth/.ckpt 大文件
    · 不需要任何额外的格式解析

  若只想先看结构、暂不拉 3GB 权重：
    GIT_LFS_SKIP_SMUDGE=1 git clone <url> zfy-v5
    cd zfy-v5 && git lfs pull      # 需要时再拉

  注意：**不要**把带 token 的 URL 提交进任何仓库或粘贴到公开场合。
""")
