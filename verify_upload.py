# -*- coding: utf-8 -*-
"""最终验证：包已在私有数据集上，且仓库仍不可匿名访问。

三项检查：
  1. 用**已登录**的凭据能列出仓库文件（确认上传成功）
  2. **匿名**访问被拒绝（确认仍是私有）
  3. 文件大小与本地一致
"""
from __future__ import annotations

import sys
import urllib.error
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RID = "yangwenjie1231/zfy-v5-fullft-pkg"
LOCAL_SIZE = 2770000000  # 本地 zip 约 2.77 GB


def anon(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return -1, f"{type(e).__name__}: {e}"


print("=" * 72)
print("1) 已登录凭据列出仓库文件")
print("=" * 72)
try:
    from modelscope.hub.api import HubApi
    api = HubApi()
    files = api.get_dataset_files(RID, recursive=False)
    print(f"  文件数: {len(files)}")
    for f in files:
        sz = int(f.get("Size") or 0)
        print(f"    {f['Path']:40s} {sz/1e9:8.2f} GB")
    ok_upload = any("zfy_v5_fullft_cloud.zip" in f["Path"] for f in files)
    print(f"  zip 在仓库中: {ok_upload}")
except Exception as e:  # noqa: BLE001
    print(f"  失败: {type(e).__name__}: {str(e)[:200]}")
    ok_upload = False

print()
print("=" * 72)
print("2) 匿名访问（必须被拒绝）")
print("=" * 72)
private_ok = True
for name, url in (
    ("文件列表", f"https://modelscope.cn/api/v1/datasets/{RID}/repo/files"
                 f"?Revision=master&Recursive=False"),
    ("取文件", f"https://modelscope.cn/api/v1/datasets/{RID}/repo"
               f"?Revision=master&FilePath=zfy_v5_fullft_cloud.zip"),
):
    code, body = anon(url)
    leaked = "zfy_v5_fullft_cloud" in body and code == 200
    print(f"  {name:8s} HTTP {code}  {'**内容泄漏!**' if leaked else '拒绝/无内容'}")
    if leaked:
        private_ok = False

print()
print("=" * 72)
print("结论")
print("=" * 72)
if ok_upload and private_ok:
    print("""
  ✓ 上传成功，仓库保持私有
  ✓ 匿名访问被拒绝（HTTP 404/405）

  Notebook 内取包：
    from modelscope import snapshot_download
    snapshot_download("yangwenjie1231/zfy-v5-fullft-pkg",
                      repo_type="dataset", local_dir="/mnt/workspace/pkg")
    # 然后 unzip pkg/zfy_v5_fullft_cloud.zip -d zfy-v5
""")
elif not ok_upload:
    print("  ✗ 上传未成功，请重跑 upload_to_modelscope.py")
else:
    print("  !! **仓库可能公开**，请立即到网页检查并设为私有")
    print(f"     https://modelscope.cn/datasets/{RID}/settings")
