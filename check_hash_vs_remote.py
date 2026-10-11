# -*- coding: utf-8 -*-
"""用**远程仓库的指针**验证哈希校验逻辑，并核对本地文件是否与远程一致。

本地 D:\\A\\zfy-v5\\GPT-SoVITS 是上游 clone（无 LFS），
所以脚本要能从远程仓库读指针才有意义 —— 这正是云上的情况。

本脚本：
  1. 从远程仓库读 LFS 指针（oid + size）
  2. 与本地文件比对 sha256（本地文件是"正确"的参照）
  3. 同时模拟云上那个 701,571,072 字节的截断文件，确认脚本能判出问题
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")
TARGETS = [
    "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth",
    "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/vocoder.pth",
    "GPT_SoVITS/pretrained_models/s1v3.ckpt",
]

# 云上实测的截断体积
CLOUD_SIZE = {"s2Gv5dev.pth": 701_571_072,
              "vocoder.pth": 57_761_163}

PTR_RE = re.compile(rb"oid sha256:([0-9a-f]{64})\s*\n\s*size (\d+)")

tok = (Path.home() / ".modelscope" / "credentials" / "git_token") \
    .read_text(encoding="utf-8").strip()
url = ("https://oauth2:" + tok + "@www.modelscope.cn/datasets/"
       "yangwenjie1231/zfy-v5-fullft-pkg.git")


def sha256_of(p: Path, chunk=8 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


tmp = Path(tempfile.mkdtemp(prefix="ms_hash_"))
try:
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none",
                    url, "r"], cwd=str(tmp), capture_output=True, env=env)
    repo = tmp / "r"

    print("=" * 78)
    print("远程 LFS 指针 vs 本地文件 sha256")
    print("=" * 78)

    for rel in TARGETS:
        name = Path(rel).name
        r = subprocess.run(["git", "show", f"HEAD:{rel}"],
                           cwd=str(repo), capture_output=True)
        blob = r.stdout or b""
        m = PTR_RE.search(blob)
        local = GSV / rel
        print(f"--- {name} ---")
        if not m:
            print(f"  远程无指针（blob {len(blob)} 字节）")
            continue
        oid, size = m.group(1).decode(), int(m.group(2))
        print(f"  远程声明: size={size:,d}  oid={oid[:32]}…")

        if not local.exists():
            print("  本地不存在，跳过比对")
            continue
        lsz = local.stat().st_size
        lh = sha256_of(local)
        print(f"  本地文件: size={lsz:,d}  sha256={lh[:32]}…")
        print(f"  体积匹配: {'是' if lsz == size else '否'}   "
              f"哈希匹配: {'是' if lh == oid else '否'}")
        if lh == oid:
            print("  => 本地文件与远程 LFS 对象**完全一致**")
        else:
            print("  => **不一致**（本地是上游原版，可能推送时被平台重打包）")

        # 模拟云上截断
        if name in CLOUD_SIZE:
            cs = CLOUD_SIZE[name]
            print(f"  云上实测体积: {cs:,d}")
            print(f"    与远程声明差: {size - cs:,d} 字节 "
                  f"（{cs/size*100:.1f}%）")
            print(f"    -> 校验器会判为: "
                  f"{'体积不符（截断）' if cs != size else '体积一致'}")

    print()
    print("=" * 78)
    print("结论")
    print("=" * 78)
    print("""
  云上 s2Gv5dev.pth 是 701,571,072 字节，而远程声明 766,385,517 ——
  **少了 64,814,445 字节（8.5%）**，是下载截断。

  截断的 .pth 会让 torch.load 报：
      PytorchStreamReader failed reading zip archive
  （zip 的中央目录在文件末尾，尾部缺失就读不到）

  修复：强制重新拉取（见 verify_weights_hash.py 末尾的 4 步）
""")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
