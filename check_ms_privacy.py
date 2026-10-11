# -*- coding: utf-8 -*-
"""准确判断魔搭数据集是否**真的公开**（HTTP 200 可能是假信号）。

## 问题

刚建的私有数据集，匿名请求返回 HTTP 200，看着像公开。
但魔搭 API 对**私有仓库**也可能返回 200 + 空 Data，
所以不能只看状态码。

## 判据（三重）

  1. **响应体**：公开仓库的 Data 里有 Name/Visibility 等字段；
     私有/不存在则 Data 为 None 或 code 是错误码
  2. **对照组**：拿一个**确定公开**的数据集比响应结构
  3. **文件列表 API**：匿名能否列出仓库文件（私有会 401/403/404）

用法：
    python check_ms_privacy.py
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TARGET = "yangwenjie1231/zfy-v5-fullft"


def probe(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            body = r.read().decode("utf-8", "replace")
            try:
                j = json.loads(body)
            except Exception:  # noqa: BLE001
                j = None
            return {"status": r.status, "json": j, "raw": body[:300]}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        try:
            j = json.loads(body)
        except Exception:  # noqa: BLE001
            j = None
        return {"status": e.code, "json": j, "raw": body[:300]}
    except Exception as e:  # noqa: BLE001
        return {"status": -1, "json": None, "raw": f"{type(e).__name__}: {e}"}


def summarize(tag: str, r: dict):
    j = r.get("json") or {}
    data = j.get("Data")
    print(f"\n=== {tag} ===")
    print(f"  HTTP {r['status']}")
    print(f"  Success={j.get('Success')}  Code={j.get('Code')}  "
          f"Message={str(j.get('Message'))[:60]}")
    if isinstance(data, dict):
        keys = [k for k in ("Name", "Path", "Visibility", "ChineseName",
                            "Downloads", "CreatedTime") if k in data]
        print(f"  Data 字段: {keys}")
        if "Visibility" in data:
            print(f"  **Visibility = {data['Visibility']}**")
        if "Name" in data:
            print(f"  Name = {data['Name']}")
    else:
        print(f"  Data = {data!r}")


print("=" * 74)
print("A) 目标仓库（应私有）")
print("=" * 74)
summarize(TARGET, probe(f"https://modelscope.cn/api/v1/datasets/{TARGET}"))

print()
print("=" * 74)
print("B) 对照组：确定公开的数据集")
print("=" * 74)
for ref in ("modelscope/clue", "AI-ModelScope/GPT-SoVITS"):
    summarize(ref, probe(f"https://modelscope.cn/api/v1/datasets/{ref}"))

print()
print("=" * 74)
print("C) 文件列表 API（私有应拒绝匿名访问）")
print("=" * 74)
for tag, rid in (("目标", TARGET), ("对照(公开)", "modelscope/clue")):
    u = (f"https://modelscope.cn/api/v1/datasets/{rid}/repo/files"
         f"?Revision=master&Recursive=False")
    r = probe(u)
    j = r.get("json") or {}
    files = ((j.get("Data") or {}).get("Files")
             if isinstance(j.get("Data"), dict) else None)
    print(f"  {tag:12s} HTTP {r['status']}  "
          f"Success={j.get('Success')}  可列文件={files is not None}"
          + (f" ({len(files)} 个)" if files else ""))

print()
print("=" * 74)
print("判读")
print("=" * 74)
print("""
  · 若目标仓库的 Data 为 None / Success=False，而对照组有完整 Data
    -> 说明**目标确实是私有的**，HTTP 200 只是 API 的统一响应格式
  · 若目标仓库的 Data 里有 Visibility 字段且值等于公开仓库的值
    -> 那是**真公开**，必须改成私有再上传
""")
