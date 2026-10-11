# -*- coding: utf-8 -*-
"""把 GPT 微调链路的**每个文件**列出来，逐个查是否 import transformers / 缺的包。

用户质疑："你上传时是按不使用 Transformer 搞的，但微调 GPT 也会用 Transformer 吧"

前面 trace_gpt_deps.py 已给出"不需要 transformers"的结论，
但那个脚本没打印**具体追到哪些文件**。本脚本补上，便于人工复核。
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


def imports_of(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:  # noqa: BLE001
        return set()
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            out.add(n.module)
    return out


def resolve(mod: str) -> Path | None:
    parts = mod.split(".")
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


seen, queue = set(), [GSV / "GPT_SoVITS" / "s1_train.py"]
third: dict[str, set[str]] = {}
while queue:
    f = queue.pop()
    if f in seen or not f.exists():
        continue
    seen.add(f)
    for mod in imports_of(f):
        top = mod.split(".")[0]
        if top in STD:
            continue
        rp = resolve(mod)
        if rp is not None:
            queue.append(rp)
        else:
            third.setdefault(top, set()).add(str(f.relative_to(GSV)))

print("=" * 78)
print(f"GPT(s1) 微调链路追踪到的 {len(seen)} 个本地文件")
print("=" * 78)
for f in sorted(seen):
    print(f"  {f.relative_to(GSV)}")

print()
print("=" * 78)
print("第三方依赖 —— 以及**是哪个文件引入的**")
print("=" * 78)
for pkg in sorted(third):
    print(f"  {pkg}")
    for src in sorted(third[pkg])[:3]:
        print(f"      <- {src}")

print()
print("=" * 78)
print("重点：transformers 是否出现在 GPT 链路里")
print("=" * 78)
tf = [p for p in third if p in ("transformers", "huggingface_hub")]
if tf:
    print(f"  **出现了**: {tf}")
else:
    print("  未出现 —— GPT 微调不需要 HF transformers")
    print("  （GPT-SoVITS 的 GPT 是自研 PyTorch 实现，见 AR/modules/）")

print()
print("=" * 78)
print("缺的包：谁引入的")
print("=" * 78)
for pkg in ("matplotlib", "gradio", "pandas", "pytorch_lightning",
            "torchmetrics"):
    srcs = sorted(third.get(pkg, []))
    in_gpt = bool(srcs)
    print(f"  {pkg:20s} {'GPT链路需要' if in_gpt else 'GPT链路不需要'}")
    for s in srcs[:3]:
        print(f"      <- {s}")
