# -*- coding: utf-8 -*-
"""把上云包上传到**私有**魔搭数据集（供 notebook 内下载）。

## 为什么走数据集

魔搭 notebook 的持久化目录挂载在数据集仓库上，且 `ms upload` 支持
`--repo-type dataset`。把包上传到私有数据集后，notebook 里可以直接
用 `snapshot_download` 或 `ms download` 取用。

## 版权处理（重要）

包里含 825 个《明日方舟：终末地》语音（245 MB），版权归鹰角网络。
用户确认：**数据集是自己做的，设为私密即可**。
所以本脚本强制使用 `visibility=5`（私有），并在创建后**实测验证**
匿名不可访问，才继续上传。

用法：
    python upload_to_modelscope.py --dry-run     # 只看计划
    python upload_to_modelscope.py               # 建私有数据集并上传
    python upload_to_modelscope.py --check       # 只检查已有仓库可见性
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NS = "yangwenjie1231"
REPO = "zfy-v5-fullft-pkg"
ZIP = Path(r"D:\A\zfy-v5\dist\zfy_v5_fullft_cloud.zip")

# 魔搭 visibility 取值（**实测**，非猜测）：
#   用「上传纯文本 + 匿名下载」定论测试得到：
#     visibility=1 -> 匿名 HTTP 404        -> 私有
#     visibility=5 -> 匿名 HTTP 200 且取到内容 -> 公开
#   见 probe_visibility_definitive.py
# 本脚本按 1 建（私有），并在上传前**强制匿名验证**。
VIS_PRIVATE = 1
VIS_PUBLIC = 5


def anon_visible(repo: str) -> tuple[int, bool]:
    """匿名（不带 token）访问仓库文件 API。

    判据用**文件列表**接口而不是仓库信息接口：
    实测发现仓库信息接口对私有仓也返回 HTTP 200（只是 Data 为 None），
    而文件列表接口对私有仓返回 404/405。用后者才能可靠区分。
    """
    url = (f"https://modelscope.cn/api/v1/datasets/{repo}/repo/files"
           f"?Revision=master&Recursive=False")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, r.status == 200
    except urllib.error.HTTPError as e:
        return e.code, False
    except Exception:  # noqa: BLE001
        return -1, False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--zip", default=str(ZIP))
    args = ap.parse_args()
    rid = f"{NS}/{args.repo}"

    from modelscope.hub.api import HubApi
    api = HubApi()

    if args.check:
        code, pub = anon_visible(rid)
        print(f"仓库 {rid}")
        print(f"  匿名访问 HTTP {code}  -> "
              f"{'**公开可见**' if pub else '非公开（需登录）'}")
        return 0

    z = Path(args.zip)
    if not z.exists():
        print(f"找不到包: {z}")
        print("请先运行: python pack_for_cloud.py")
        return 1
    print(f"包: {z}")
    print(f"  体积: {z.stat().st_size/1e9:.2f} GB")

    print(f"\n计划:")
    print(f"  数据集 : {rid}   (visibility={VIS_PRIVATE} 私有)")
    print(f"  上传   : {z.name} -> 仓库根目录")
    if args.dry_run:
        print("\n(dry-run，未执行)")
        return 0

    # 1) 建仓（已存在则复用）
    print(f"\n=== 1) 创建私有数据集 ===")
    try:
        url = api.create_dataset(
            args.repo, namespace=NS, visibility=VIS_PRIVATE,
            chinese_name="庄方宜 v5 全量微调工作包（私有）",
            description="GPT-SoVITS v5 全量微调所需：代码+底模+已算好的特征。"
                        "含游戏语音，版权归鹰角网络，仅个人研究用，禁止再分发。")
        print(f"  创建成功: {url}")
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "exist" in msg.lower() or "已存在" in msg:
            print(f"  仓库已存在，复用: {rid}")
        else:
            print(f"  创建失败: {type(e).__name__}: {msg[:200]}")
            return 1

    # 2) **强制**验证私有性
    print(f"\n=== 2) 验证私有性（必须匿名不可访问）===")
    time.sleep(3)
    code, pub = anon_visible(rid)
    print(f"  匿名访问 HTTP {code}  -> "
          f"{'**公开可见**' if pub else '非公开（需登录）'}")
    if pub:
        print("\n  !! 仓库是公开的，含版权语料，**中止上传**")
        print(f"  请手动改为私有: https://modelscope.cn/datasets/{rid}/settings")
        return 1
    print("  ✓ 确认非公开，继续上传")

    # 3) 上传
    print(f"\n=== 3) 上传 {z.stat().st_size/1e9:.2f} GB ===")
    print("  （大文件，可能较慢；可用 Ctrl+C 中断，支持断点续传）")
    t0 = time.time()
    try:
        api.upload_folder(
            repo_id=rid, repo_type="dataset",
            folder_path=str(z.parent),
            allow_patterns=[z.name],
            commit_message="add v5 fullft cloud package (private)",
        )
        dt = time.time() - t0
        print(f"  上传完成，用时 {dt:.0f}s "
              f"({z.stat().st_size/1e6/max(dt,0.01):.1f} MB/s)")
    except Exception as e:  # noqa: BLE001
        print(f"  上传失败: {type(e).__name__}: {str(e)[:300]}")
        print(f"  可改用 CLI 重试:")
        print(f'    ms upload {rid} "{z.parent}" --repo-type dataset')
        return 1

    print(f"\n完成。notebook 内下载：")
    print(f'  from modelscope import snapshot_download')
    print(f'  snapshot_download("{rid}", repo_type="dataset", '
          f'local_dir="/tmp/pkg")')
    return 0


if __name__ == "__main__":
    sys.exit(main())
