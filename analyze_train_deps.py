# -*- coding: utf-8 -*-
"""分析：s2 全量微调路径**到底需不需要 transformers**？

## 为什么这个问题是关键

云上预装了 vllm 0.28.0，它硬性要求：
    transformers>=5.5.3
    huggingface_hub>=1.27.0

而 GPT-SoVITS 要求：
    transformers>=4.51,<5
    （间接）huggingface_hub<1.0

**两者不可共存。** 除非我们根本不需要 transformers。

## 已知

  · `3-bert/` 特征**已在本机算好**并随包上传 -> 不需要在线跑 BERT
  · `1-get-text.py` 用 transformers，但那是**数据准备**，已做完
  · s2 训练只读 `logs/zfy_v5/` 里的现成特征

## 本脚本

从 `s2_train_v5_full.py` 出发，递归追出所有本地 import，
找出训练真正需要的第三方包。特别关注 transformers。
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")
ENTRY = GSV / "GPT_SoVITS" / "s2_train_v5_full.py"

# 本地模块根（这些前缀的 import 是我们自己的代码，要递归）
LOCAL_PREFIXES = ("module", "text", "tools", "utils", "process_ckpt",
                  "f5_tts", "AR", "GPT_SoVITS", "config", "inference")

# 这些是本地模块，但搜不到文件（在 sys.path 各处），用映射兜底
SEARCH_ROOTS = [GSV, GSV / "GPT_SoVITS", GSV / "tools",
                GSV / "GPT_SoVITS" / "module", GSV / "GPT_SoVITS" / "AR"]


def resolve(mod: str) -> Path | None:
    """把模块名解析成本地文件路径（找不到返回 None）。"""
    parts = mod.split(".")
    for root in SEARCH_ROOTS:
        p = root.joinpath(*parts)
        for cand in (p.with_suffix(".py"), p / "__init__.py"):
            if cand.exists():
                return cand
    # 单级模块名再扫一遍常见目录
    if len(parts) == 1:
        for root in SEARCH_ROOTS:
            c = root / (parts[0] + ".py")
            if c.exists():
                return c
    return None


def imports_of(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:  # noqa: BLE001
        return set()
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                out.add(node.module)
    return out


def main() -> int:
    if not ENTRY.exists():
        print(f"找不到入口: {ENTRY}")
        return 1

    seen: set[str] = set()
    queue = [ENTRY]
    third_party: dict[str, set[str]] = {}   # 包名 -> 谁引入的
    local_files: set[Path] = set()

    while queue:
        f = queue.pop()
        if f in local_files:
            continue
        local_files.add(f)
        for mod in imports_of(f):
            top = mod.split(".")[0]
            if mod in seen:
                continue
            seen.add(mod)
            # 是不是本地模块？
            if top in LOCAL_PREFIXES:
                rp = resolve(mod)
                if rp is not None:
                    queue.append(rp)
                    continue
                # 本地前缀但找不到 -> 也算第三方（可能真缺）
            third_party.setdefault(top, set()).add(str(f.relative_to(GSV)))

    print("=" * 76)
    print("s2_train_v5_full.py 的完整第三方依赖（递归追踪本地 import）")
    print("=" * 76)
    print(f"  追踪到本地文件: {len(local_files)} 个")
    print(f"  第三方顶层包  : {len(third_party)} 个")
    print()
    for pkg in sorted(third_party):
        srcs = sorted(third_party[pkg])[:3]
        print(f"  {pkg:26s} <- {', '.join(srcs)}")

    print()
    print("=" * 76)
    print("关键判定")
    print("=" * 76)
    KEY = ["transformers", "huggingface_hub", "gradio", "pandas", "peft",
           "pytorch_lightning", "torchmetrics", "x_transformers"]
    for k in KEY:
        hit = third_party.get(k)
        mark = "**需要**" if hit else "不需要"
        print(f"  {k:22s} {mark}")
        if hit:
            for s in sorted(hit)[:4]:
                print(f"      <- {s}")

    print()
    print("=" * 76)
    print("结论")
    print("=" * 76)
    need_tf = "transformers" in third_party
    if need_tf:
        print("  s2 训练**需要** transformers —— 与云上 vllm 冲突，必须用 venv 隔离")
    else:
        print("""
  ✓ s2 训练**不需要** transformers！

  => 云上不必降 transformers/huggingface-hub，
     也就不必和 vllm 的约束打架。

  需要做的是：确认 gradio / pandas 的**当前版本**能被 my_utils.py import
  （它们只在模块级 import，训练时不用其功能）。
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
