# -*- coding: utf-8 -*-
"""用「上传小文件 + 匿名下载」定论测试 visibility 取值（1 vs 5）。

## 为什么必须实测

刚发现：已知**公开**的 `modelscope/clue` 的 API 响应里 `Visibility=5`，
与我建的仓库相同。这说明我原先「5=私有」的假设**很可能是错的**。

包里含 825 个鹰角版权语音，如果误传到公开仓库 = 公开再分发。
所以必须定论，不能猜。

## 做法

建两个**只含一个纯文本文件**的测试仓（不含任何版权内容）：
  · probe-vis-1  用 visibility=1
  · probe-vis-5  用 visibility=5
然后对两个仓都做**匿名下载**该文件：
  · 匿名下载成功 -> 公开
  · 匿名下载失败 -> 私有
用完立即删除。

用法：
    python probe_visibility_definitive.py
"""
from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NS = "yangwenjie1231"
TMP = Path(r"D:\A\zfy-v5\dist\_vis_probe")
TMP.mkdir(parents=True, exist_ok=True)
PAYLOAD = TMP / "probe.txt"
PAYLOAD.write_text("visibility probe payload - no copyrighted content\n",
                   encoding="utf-8")


def anon_get(url: str) -> tuple[int, str]:
    """不带任何凭据的 GET。"""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            return r.status, r.read().decode("utf-8", "replace")[:200]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")[:200]
    except Exception as e:  # noqa: BLE001
        return -1, f"{type(e).__name__}: {e}"


def main() -> int:
    from modelscope.hub.api import HubApi
    api = HubApi()
    created = []

    for vis in (1, 5):
        name = f"zfh-vis-probe-{vis}"
        rid = f"{NS}/{name}"
        print("=" * 72)
        print(f"visibility={vis}  ->  {rid}")
        print("=" * 72)
        try:
            api.create_dataset(name, namespace=NS, visibility=vis,
                               chinese_name=f"visibility probe {vis}",
                               description="temporary probe, safe to delete")
            created.append((rid, name))
            print("  创建 OK")
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "exist" in msg.lower() or "已存在" in msg:
                created.append((rid, name))
                print("  已存在，复用")
            else:
                print(f"  创建失败: {type(e).__name__}: {msg[:120]}")
                continue

        # 上传一个纯文本文件
        try:
            api.upload_folder(repo_id=rid, repo_type="dataset",
                              folder_path=str(TMP),
                              allow_patterns=["probe.txt"],
                              commit_message="probe")
            print("  上传 probe.txt OK")
        except Exception as e:  # noqa: BLE001
            print(f"  上传失败: {type(e).__name__}: {str(e)[:120]}")
            continue

        time.sleep(3)

        # 匿名下载测试（多种 URL 形式）
        print("  匿名访问测试:")
        urls = [
            f"https://modelscope.cn/api/v1/datasets/{rid}/repo?Revision=master&FilePath=probe.txt",
            f"https://www.modelscope.cn/api/v1/datasets/{rid}/repo?Revision=master&FilePath=probe.txt",
        ]
        verdict = None
        for u in urls:
            code, body = anon_get(u)
            got = "visibility probe payload" in body
            print(f"    HTTP {code}  取到内容={got}")
            if got:
                verdict = "公开"
                break
            elif code in (401, 403, 404):
                verdict = "私有"
        print(f"  => visibility={vis} 判定: {verdict}")
        print()

    # 清理
    print("=" * 72)
    print("清理测试仓")
    print("=" * 72)
    for rid, name in created:
        try:
            api.delete_dataset(rid)
            print(f"  已删除 {rid}")
        except Exception as e:  # noqa: BLE001
            print(f"  删除失败 {rid}: {type(e).__name__}")
            print(f"    请手动删: https://modelscope.cn/datasets/{rid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
