# -*- coding: utf-8 -*-
"""权重文件哈希校验 —— 判断 LFS 内容是否**完整且正确**。

## 为什么不能只看体积

用户云上 `s2Gv5dev.pth` 是 701,571,072 字节，
而 LFS 指针里记录的 `size` 是 **766,385,517** 字节 —— 只有 91.5%。
典型的下载截断。但体积对比是间接证据，**sha256 才是定论**。

## 本脚本的巧妙之处

LFS 指针里**本来就记着内容的 sha256**：

    version https://git-lfs.github.com/spec/v1
    oid sha256:16c11fc588a9229a479316a11fc0f518087e79773f5271666a90b39d50460407
    size 766385517

所以不需要我预先硬编码哈希 —— 从 git 里读指针，再算工作区文件的 sha256，
**两者对比即可**。指针 = 权威来源。

## 判据

  · 工作区文件 < 1 KB 且以 "version https://git-lfs" 开头 -> 未拉取
  · 体积 != 指针里的 size                                    -> 截断/损坏
  · sha256 != 指针里的 oid                                   -> 内容错误

用法（云上，检出根）：
    python _tools/verify_weights_hash.py
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
if not (ROOT / "GPT_SoVITS").is_dir():
    for c in (Path("/mnt/workspace/zfy-v5-fullft-pkg"),
              Path("/mnt/workspace/zfy-v5"),
              Path(r"D:\A\zfy-v5\GPT-SoVITS")):
        if (c / "GPT_SoVITS").is_dir():
            ROOT = c
            break

# 训练必需的大文件
TARGETS = [
    "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth",
    "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/vocoder.pth",
    "GPT_SoVITS/pretrained_models/s1v3.ckpt",
    "GPT_SoVITS/pretrained_models/chinese-hubert-base/pytorch_model.bin",
    "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/pytorch_model.bin",
]

PTR_RE = re.compile(rb"oid sha256:([0-9a-f]{64})\s*\n\s*size (\d+)")


def sha256_of(p: Path, chunk=8 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def pointer_from_git(rel: str) -> tuple[str, int] | None:
    """从 git 读该路径的 blob；若是指针则返回 (oid, size)。"""
    try:
        r = subprocess.run(["git", "show", f"HEAD:{rel}"],
                           cwd=str(ROOT), capture_output=True, timeout=120)
        blob = r.stdout or b""
    except Exception:  # noqa: BLE001
        return None
    m = PTR_RE.search(blob)
    if not m:
        return None
    return m.group(1).decode(), int(m.group(2))


print("=" * 78)
print("权重完整性校验（体积 + sha256 双重比对）")
print("=" * 78)
print(f"检出根: {ROOT}")
try:
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          cwd=str(ROOT), capture_output=True, text=True)
    print(f"提交:   {head.stdout.strip()}")
except Exception:  # noqa: BLE001
    pass
print()

bad: list[tuple[str, str, str]] = []   # (路径, 问题, 修复提示)
ok_n = 0
no_ptr = 0

for rel in TARGETS:
    name = Path(rel).name
    p = ROOT / rel
    print(f"--- {name} ---")

    if not p.exists():
        print("  文件不存在")
        bad.append((rel, "缺失", "git lfs pull"))
        continue

    sz = p.stat().st_size
    head_bytes = p.read_bytes()[:80]
    if sz < 1024 and head_bytes.startswith(b"version https://git-lfs"):
        print(f"  未拉取（{sz} 字节的 LFS 指针）")
        bad.append((rel, "指针未拉取", "git lfs pull"))
        continue

    ref = pointer_from_git(rel)
    if ref is None:
        print(f"  git 里取不到指针（可能是普通文件），体积 {sz/1e6:.1f} MB")
        no_ptr += 1
        continue
    exp_oid, exp_size = ref

    print(f"  体积   实际 {sz:>12,d}  期望 {exp_size:>12,d}  "
          f"{'一致' if sz == exp_size else '**不一致**'}")
    if sz != exp_size:
        pct = sz / exp_size * 100
        print(f"         -> 只有 {pct:.1f}%，下载被截断")
        bad.append((rel, f"体积不符（{pct:.1f}%）", "强制重新拉取"))
        continue

    print("  计算 sha256 …", flush=True)
    got = sha256_of(p)
    same = got == exp_oid
    print(f"  sha256 实际 {got[:24]}…  期望 {exp_oid[:24]}…  "
          f"{'一致' if same else '**不一致**'}")
    if same:
        print("  完整且正确")
        ok_n += 1
    else:
        bad.append((rel, "sha256 不符（内容错误）", "强制重新拉取"))

print()
print("=" * 78)
print("结论")
print("=" * 78)
print(f"  通过 {ok_n} / 共 {len(TARGETS)}"
      + (f"（{no_ptr} 个无指针，按普通文件处理）" if no_ptr else ""))

if not bad:
    print("\n  全部权重完整且哈希正确，可以训练")
    sys.exit(0)

print(f"\n  {len(bad)} 个文件有问题：")
for rel, why, fix in bad:
    print(f"    [{why}] {Path(rel).name}")

print("""
  修复步骤（在检出根执行）：

    # 1) 让 git-lfs 忘记本地缓存的对象，强制重下
    git lfs install --local
    rm -rf .git/lfs/objects .git/lfs/incomplete

    # 2) 删除工作区里的坏文件，让它从 HEAD 重新检出
    git checkout --force -- GPT_SoVITS/pretrained_models/

    # 3) 重新拉取（这次会真的下载）
    git lfs pull

    # 4) 再校验一次
    python _tools/verify_weights_hash.py
""")
sys.exit(1)
