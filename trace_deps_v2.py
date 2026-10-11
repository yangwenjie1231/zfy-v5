# -*- coding: utf-8 -*-
"""依赖追踪 v2 —— **支持相对导入**（v1 的重大缺陷）。

## v1 的 bug

    elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:

只处理绝对导入。而 `text/LangSegmenter/__init__.py` 用的是

    from .langsegmenter import LangSegmenter     # level=1

相对导入被整个跳过 -> `langsegmenter.py` 没被追踪 -> 它里面的
`import fast_langdetect` 也就没进依赖清单 -> 云端推理报
`ModuleNotFoundError: No module named 'fast_langdetect'`

## v2 修法

按 Python 语义解析相对导入：level=N 表示向上 N-1 层，从当前包的
`__package__` 出发定位目标模块。

同时输出**每个第三方包由谁引入**，便于人工复核。
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")
ROOTS = [GSV, GSV / "GPT_SoVITS", GSV / "tools",
         GSV / "GPT_SoVITS" / "module", GSV / "GPT_SoVITS" / "AR"]

STD = set("""os sys re json math time random logging warnings traceback typing
pathlib collections functools itertools subprocess shutil glob copy hashlib io
struct argparse platform datetime contextlib dataclasses enum abc inspect
textwrap unicodedata ctypes threading queue tempfile uuid base64 urllib socket
string operator pickle importlib gc signal stat errno csv zipfile numbers types
locale _distutils_hack concurrent multiprocessing asyncio heapq bisect array
atexit codecs difflib fnmatch getpass html http ipaddress keyword linecache
mail mimetypes mmap netrc ntpath optparse pdb plistlib pprint profile pty pwd
py_compile pydoc pytz quopri rlcompleter runpy sched secrets selectors shelve
shlex site smtplib sqlite3 ssl statistics symtable tabnanny tarfile termios
this tkinter token tokenize trace tracemalloc tty unittest uu venv wave weakref
webbrowser xml xmlrpc zipapp zipimport zlib zoneinfo builtins __future__
typing_extensions""".split())


def _module_of(path: Path) -> str:
    """把文件路径转成"包内模块名"（用于解析相对导入）。"""
    for root in (GSV / "GPT_SoVITS", GSV, GSV / "tools"):
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        parts = list(rel.parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1][:-3]      # 去掉 .py
        return ".".join(parts)
    return path.stem


def resolve(mod: str) -> Path | None:
    parts = [p for p in mod.split(".") if p]
    for root in ROOTS:
        p = root.joinpath(*parts)
        for c in (p.with_suffix(".py"), p / "__init__.py"):
            if c.exists():
                return c
    if len(parts) == 1:
        for root in ROOTS:
            c = root / (parts[0] + ".py")
            if c.exists():
                return c
    return None


def imports_of(path: Path) -> list[tuple[str, int]]:
    """返回 [(module, level)]，level>0 表示相对导入。"""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:  # noqa: BLE001
        return []
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                out.append((a.name, 0))
        elif isinstance(n, ast.ImportFrom):
            out.append((n.module or "", n.level))
    return out


def resolve_relative(cur_file: Path, mod: str, level: int) -> str | None:
    """按 Python 语义解析相对导入。"""
    if level == 0:
        return mod or None
    pkg = _module_of(cur_file)
    # __init__.py 本身就在包内；普通模块要退到它的包
    if cur_file.name != "__init__.py":
        pkg = pkg.rsplit(".", 1)[0] if "." in pkg else ""
    parts = pkg.split(".") if pkg else []
    up = level - 1
    if up > 0:
        parts = parts[:-up] if up <= len(parts) else []
    if mod:
        parts = parts + mod.split(".")
    return ".".join(parts) if parts else None


def trace(entry: Path, label: str, verbose=True):
    seen, queue = set(), [entry]
    third: dict[str, set[str]] = {}
    while queue:
        f = queue.pop()
        if f in seen or not f.exists():
            continue
        seen.add(f)
        for mod, level in imports_of(f):
            full = resolve_relative(f, mod, level)
            if not full:
                continue
            top = full.split(".")[0]
            if top in STD:
                continue
            rp = resolve(full)
            if rp is not None:
                queue.append(rp)
            else:
                third.setdefault(top, set()).add(str(f.relative_to(GSV)))
    if verbose:
        print("=" * 78)
        print(label)
        print("=" * 78)
        print(f"  入口: {entry.relative_to(GSV)}")
        print(f"  本地文件: {len(seen)} 个")
        print(f"  第三方  : {sorted(third)}")
    return seen, third


print("#" * 78)
print("# 依赖追踪 v2（支持相对导入）")
print("#" * 78)

chains = {
    "GPT 训练": GSV / "GPT_SoVITS" / "s1_train.py",
    "SoVITS 训练": GSV / "GPT_SoVITS" / "s2_train_v5_full.py",
    "推理": GSV / "GPT_SoVITS" / "TTS_infer_pack" / "TTS.py",
}
results = {}
for label, entry in chains.items():
    print()
    seen, third = trace(entry, label)
    results[label] = third

print()
print("#" * 78)
print("# 各链路第三方依赖汇总")
print("#" * 78)
allp = sorted(set().union(*[set(t) for t in results.values()]))
print(f"  {'包':24s} {'GPT训练':>8s} {'SoVITS训练':>11s} {'推理':>6s}")
for p in allp:
    a = "需要" if p in results["GPT 训练"] else "-"
    b = "需要" if p in results["SoVITS 训练"] else "-"
    c = "需要" if p in results["推理"] else "-"
    print(f"  {p:24s} {a:>8s} {b:>11s} {c:>6s}")

print()
print("#" * 78)
print("# 推理链路的第三方 —— 由谁引入（便于定位）")
print("#" * 78)
for p in sorted(results["推理"]):
    print(f"  {p}")
    for src in sorted(results["推理"][p])[:2]:
        print(f"      <- {src}")

print()
print("#" * 78)
print("# 对照：v1 追踪器漏掉的（含相对导入才能发现）")
print("#" * 78)
print("  v1 只处理 level==0，因此漏掉了这些相对导入链：")
print("    text/LangSegmenter/__init__.py  from .langsegmenter import ...")
print("      -> langsegmenter.py  import fast_langdetect   <-- v1 漏掉")
print()
print("  v2 结果里是否含 fast_langdetect:",
      "是" if "fast_langdetect" in results["推理"] else "**否**")
