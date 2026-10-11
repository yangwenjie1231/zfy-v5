# -*- coding: utf-8 -*-
"""追踪 **GPT(s1) 微调**链路的真实依赖（AST 递归）。

## 用户的质疑
"当时你上传这个是按不使用 Transformer 搞的，但是现在我需要微调 GPT 的话，
 也会使用 Transformer 吧"

## 为什么必须实测而不是猜
上次我给 SoVITS 训练裁剪依赖清单时，是**按 import 分析**做的，
结果漏了推理专用的 BigVGAN / feature_extractor / sv.py。
这次对 GPT 训练同样要**递归追到底**，尤其确认 transformers 是否需要。

## 入口
  GPT_SoVITS/s1_train.py
    -> AR.data.data_module.Text2SemanticDataModule
    -> AR.models.t2s_lightning_module.Text2SemanticLightningModule
    -> AR.models.t2s_model ...
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")
ROOTS = [GSV, GSV / "GPT_SoVITS", GSV / "tools",
         GSV / "GPT_SoVITS" / "module", GSV / "GPT_SoVITS" / "AR"]

STD = {
    "os", "sys", "re", "json", "math", "time", "random", "logging", "warnings",
    "traceback", "typing", "pathlib", "collections", "functools", "itertools",
    "subprocess", "shutil", "glob", "copy", "hashlib", "io", "struct",
    "argparse", "platform", "datetime", "contextlib", "dataclasses", "enum",
    "abc", "inspect", "textwrap", "unicodedata", "ctypes", "threading",
    "queue", "tempfile", "uuid", "base64", "urllib", "socket", "string",
    "operator", "pickle", "importlib", "gc", "signal", "stat", "errno",
    "csv", "zipfile", "numbers", "types", "locale", "_distutils_hack",
    "concurrent", "multiprocessing", "asyncio", "heapq", "bisect", "array",
    "atexit", "codecs", "difflib", "filecmp", "fnmatch", "getpass",
    "html", "http", "ipaddress", "keyword", "linecache", "mail", "mimetypes",
    "mmap", "netrc", "ntpath", "optparse", "pdb", "plistlib", "pprint",
    "profile", "pty", "pwd", "py_compile", "pydoc", "pytz", "quopri",
    "rlcompleter", "runpy", "sched", "secrets", "selectors", "shelve",
    "shlex", "site", "smtplib", "sndhdr", "spwd", "sqlite3", "sre_compile",
    "sre_constants", "sre_parse", "ssl", "statistics", "stringprep",
    "subprocess", "symtable", "tabnanny", "tarfile", "telnetlib", "termios",
    "textwrap", "this", "tkinter", "token", "tokenize", "trace", "tracemalloc",
    "tty", "turtle", "turtledemo", "unittest", "uu", "venv", "warnings",
    "wave", "weakref", "webbrowser", "wsgiref", "xdrlib", "xml", "xmlrpc",
    "zipapp", "zipimport", "zlib", "zoneinfo", "builtins", "__future__",
    "dataclasses", "typing_extensions",
}


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


def trace(entry: Path, label: str):
    seen, queue = set(), [entry]
    localmods, third = set(), set()
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
                localmods.add(top)
                queue.append(rp)
            else:
                third.add(top)
    print("=" * 78)
    print(f"{label}")
    print("=" * 78)
    print(f"  入口: {entry.relative_to(GSV)}")
    print(f"  追踪本地文件: {len(seen)} 个")
    print(f"  本地模块    : {sorted(localmods)}")
    print(f"  第三方      : {sorted(third)}")
    print()
    return third, localmods


t_gpt, l_gpt = trace(GSV / "GPT_SoVITS" / "s1_train.py",
                     "GPT(s1) 微调链路")
t_s2, l_s2 = trace(GSV / "GPT_SoVITS" / "s2_train_v5_full.py",
                   "SoVITS(s2) 全量微调链路（对照）")
t_inf, l_inf = trace(GSV / "GPT_SoVITS" / "TTS_infer_pack" / "TTS.py",
                     "推理链路（对照）")

print("=" * 78)
print("关键判定：各链路是否需要这些包")
print("=" * 78)
PKGS = ["transformers", "huggingface_hub", "peft", "pytorch_lightning",
        "torchmetrics", "gradio", "pandas", "matplotlib", "resampy",
        "librosa", "soundfile", "x_transformers", "einops", "safetensors",
        "cn2an", "pypinyin", "jieba", "chardet", "opencc", "tensorboard",
        "ffmpeg", "yaml", "psutil", "numpy", "scipy", "torch", "torchaudio",
        "models", "datasets1", "kaldi", "ERes2NetV2", "Accel", "BigVGAN"]
print(f"  {'包':22s} {'GPT训练':>8s} {'SoVITS训练':>11s} {'推理':>6s}")
for p in PKGS:
    a = "需要" if p in t_gpt else "-"
    b = "需要" if p in t_s2 else "-"
    c = "需要" if p in t_inf else "-"
    print(f"  {p:22s} {a:>8s} {b:>11s} {c:>6s}")

print()
print("=" * 78)
print("结论")
print("=" * 78)
need_tf = "transformers" in t_gpt
print(f"  GPT 微调是否需要 transformers: {'**是**' if need_tf else '否'}")
if need_tf:
    where = []
    for f in sorted(set()):
        pass
    # 找出谁引入的
    for f in (GSV / "GPT_SoVITS").rglob("*.py"):
        if "transformers" in imports_of(f):
            where.append(str(f.relative_to(GSV)))
    print(f"    引入位置（前 6）:")
    for w in where[:6]:
        print(f"      {w}")
