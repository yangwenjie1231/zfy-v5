# -*- coding: utf-8 -*-
"""修正：把工作包数据集建成**私有**（visibility=1）。

## 实测结论（与直觉相反）

用「上传纯文本 + 匿名下载」定论测试（`probe_visibility_definitive.py`）：

    visibility=1  ->  匿名 HTTP 404  ->  **私有**
    visibility=5  ->  匿名 HTTP 200 且取到内容  ->  **公开**

我原先假设「5=私有」是**错的**。先前建的 `zfy-v5-fullft` 用的是 5，
即**公开** —— 幸好只建了空仓，未上传任何版权内容。

## 本脚本

  1. 若已存在同名仓，**先删除**（避免留下公开仓）
  2. 用 visibility=1 重建
  3. **匿名验证必须 404**，否则中止
  4. 通过后才允许上传（上传由 upload_to_modelscope.py 负责）

用法：
    python fix_dataset_private.py --recreate      # 删旧建新并验证
    python fix_dataset_private.py --verify        # 只验证现有仓
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NS = "yangwenjie1231"
REPO = "zfy-v5-fullft"
VIS_PRIVATE = 1          # 实测：1 = 私有
VIS_PUBLIC = 5           # 实测：5 = 公开

PROBE_URLS = [
    "https://modelscope.cn/api/v1/datasets/{rid}/repo"
    "?Revision=master&FilePath=probe.txt",
    "https://modelscope.cn/api/v1/datasets/{rid}/repo/files"
    "?Revision=master&Recursive=False",
]


def anon(url: str) -> tuple[int, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            return r.status, r.read().decode("utf-8", "replace")[:200]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")[:200]
    except Exception as e:  # noqa: BLE001
        return -1, f"{type(e).__name__}: {e}"


def is_private(rid: str) -> tuple[bool, list[str]]:
    """匿名探测；只要有一个 URL 返回 200 就算公开。"""
    log = []
    public = False
    for tpl in PROBE_URLS:
        u = tpl.format(rid=rid)
        code, body = anon(u)
        log.append(f"HTTP {code}  {u.split('?')[0][-40:]}")
        if code == 200:
            public = True
    return (not public), log


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--repo", default=REPO)
    args = ap.parse_args()
    rid = f"{NS}/{args.repo}"

    from modelscope.hub.api import HubApi
    api = HubApi()

    if args.verify:
        priv, log = is_private(rid)
        print(f"仓库 {rid}")
        for l in log:
            print(f"  {l}")
        print(f"  => {'私有（正确）' if priv else '**公开（危险）**'}")
        return 0 if priv else 1

    if args.recreate:
        print(f"=== 1) 删除可能存在的旧仓 {rid} ===")
        try:
            api.delete_dataset(rid)
            print("  已删除")
        except Exception as e:  # noqa: BLE001
            print(f"  删除跳过（可能不存在）: {type(e).__name__}")
        time.sleep(2)

        print(f"\n=== 2) 用 visibility={VIS_PRIVATE}（私有）重建 ===")
        try:
            url = api.create_dataset(
                args.repo, namespace=NS, visibility=VIS_PRIVATE,
                chinese_name="庄方宜 v5 全量微调工作包（私有）",
                description="GPT-SoVITS v5 全量微调所需：代码+底模+已算好的特征。"
                            "含游戏语音，版权归鹰角网络，仅个人研究，禁止再分发。")
            print(f"  创建成功: {url}")
        except Exception as e:  # noqa: BLE001
            print(f"  创建失败: {type(e).__name__}: {str(e)[:200]}")
            return 1

    print(f"\n=== 3) 匿名验证私有性 ===")
    time.sleep(4)
    priv, log = is_private(rid)
    for l in log:
        print(f"  {l}")
    if priv:
        print("\n  ✓ 确认私有，可以安全上传")
        print(f"\n  下一步:")
        print(f"    python upload_to_modelscope.py")
        return 0
    print("\n  !! **仍是公开**，禁止上传版权语料")
    print(f"  请到网页手动设为私有: "
          f"https://modelscope.cn/datasets/{rid}/settings")
    return 1


if __name__ == "__main__":
    sys.exit(main())
