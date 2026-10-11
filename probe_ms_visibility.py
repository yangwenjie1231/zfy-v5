# -*- coding: utf-8 -*-
"""实测魔搭 create_dataset 的 visibility 取值语义（1 = 私有还是公开）。

为什么不猜：包里含 825 个鹰角版权 .wav（245MB），
数据集 README 明确写「请勿用于商业用途或再分发」。
如果误建成**公开**仓库，就是把版权语料公开再分发了。

做法：建一个小的测试数据集，分别用 visibility=1 / 5，
然后请求仓库页面看它是否匿名可见。
用完立刻删除。

用法：
    python probe_ms_visibility.py            # 只看，不建仓
    python probe_ms_visibility.py --create   # 实际建测试仓并检测
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NS = "yangwenjie1231"


def http_status(url: str) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:  # noqa: BLE001
        return -1


def api_json(url: str):
    import json
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--create", action="store_true")
    ap.add_argument("--name", default="zfh-visibility-probe")
    args = ap.parse_args()

    from modelscope.hub.api import HubApi
    api = HubApi()

    # 先看一个已知仓库的公开可见性作为对照
    print("=== 对照：已知公开数据集 ===")
    try:
        d = api_json("https://modelscope.cn/api/v1/datasets/AI-ModelScope/GPT-SoVITS")
        vis = (d.get("Data") or {}).get("Visibility")
        print(f"  AI-ModelScope/GPT-SoVITS  Visibility = {vis}")
    except Exception as e:  # noqa: BLE001
        print(f"  查询失败: {type(e).__name__}")

    if not args.create:
        print("\n(未指定 --create，不做实测)")
        print("提示：魔搭 API 里 Visibility 常见取值 1=公开 / 5=私有，"
              "但本脚本不猜测，需实测确认。")
        return 0

    print(f"\n=== 实测 visibility=1（建 {NS}/{args.name}）===")
    try:
        url = api.create_dataset(args.name, namespace=NS, visibility=1,
                                 chinese_name="visibility 探测（临时）",
                                 description="临时测试仓，用完即删")
        print(f"  创建成功: {url}")
    except Exception as e:  # noqa: BLE001
        print(f"  创建失败: {type(e).__name__}: {e}")
        return 1

    time.sleep(3)
    # 匿名访问仓库 API，看是否需要登录
    probe = f"https://modelscope.cn/api/v1/datasets/{NS}/{args.name}"
    code = http_status(probe)
    print(f"  匿名访问 {probe}")
    print(f"  HTTP {code}  -> "
          f"{'公开可见（匿名可访问）' if code == 200 else '非公开/需登录'}")

    print(f"\n=== 清理：删除测试仓 ===")
    try:
        api.delete_dataset(f"{NS}/{args.name}")
        print("  已删除")
    except Exception as e:  # noqa: BLE001
        print(f"  删除失败（请手动删）: {type(e).__name__}: {e}")
        print(f"  地址: https://modelscope.cn/datasets/{NS}/{args.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
