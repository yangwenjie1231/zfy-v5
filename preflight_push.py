# -*- coding: utf-8 -*-
"""上传前守卫：检查所有要上云的 .sh 是否含 CRLF，以及内嵌 Python 语法。

## 为什么需要

我在 Windows 上用 Python 的 `Path.write_text()` 改了一个 .sh，
它默认把 `\\n` 写成 `\\r\\n`（Windows 换行）。结果云端 bash 报：

    run_sovits_e32.sh: line 35: syntax error near unexpected token `$'in\\r''

297 个 CRLF 把脚本毁了。而**本机 Windows 上看不出来** ——
Git Bash 容忍 CRLF，云上 Linux bash 不容忍。

这个脚本把"上云前必查"固化成一条命令：
  1. 所有 .sh 的换行必须是 LF
  2. 所有 .sh 里内嵌的 `python - <<'PYEOF'` 块语法必须正确
  3. 关键 .py 能 py_compile

用法：
    python preflight_push.py
    python preflight_push.py --fix     # 顺便修 CRLF
"""
from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

V5 = Path(r"D:\A\zfy-v5")
PY = r"C:\Users\A\miniconda3\envs\voice\python.exe"
BLOCK = re.compile(r"<<'?(\w+)'?\s*\n(.*?)\n\1", re.S)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true", help="自动把 CRLF 转成 LF")
    args = ap.parse_args()

    problems = []

    print("=" * 74)
    print("1. shell 脚本换行（必须是 LF）")
    print("=" * 74)
    for p in sorted(V5.glob("*.sh")):
        raw = p.read_bytes()
        n = raw.count(b"\r\n")
        if n:
            problems.append(f"{p.name}: {n} 个 CRLF")
            if args.fix:
                p.write_bytes(raw.replace(b"\r\n", b"\n"))
                print(f"  已修 {p.name}（{n} -> 0）")
            else:
                print(f"  **CRLF** {p.name}: {n} 个  "
                      f"（云上 bash 会报 syntax error）")
        else:
            print(f"  OK   {p.name}")

    print()
    print("=" * 74)
    print("2. shell 脚本内嵌 Python 的语法")
    print("=" * 74)
    for p in sorted(V5.glob("*.sh")):
        txt = p.read_text(encoding="utf-8", errors="replace")
        blocks = [b for tag, b in BLOCK.findall(txt) if tag.upper() == "PYEOF"]
        n_ok = 0
        for i, body in enumerate(blocks, 1):
            try:
                ast.parse(body)
                n_ok += 1
            except SyntaxError as e:
                problems.append(f"{p.name} 内嵌块{i}: {e.msg}")
                print(f"  FAIL {p.name} 第{i}块 行{e.lineno}: {e.msg}")
        if blocks:
            print(f"  OK   {p.name}: {n_ok}/{len(blocks)} 块")

    print()
    print("=" * 74)
    print("3. 关键 .py 编译")
    print("=" * 74)
    KEY_PY = ["cloud_synth_sweep.py", "cloud_push_weights.py", "train_gpt.py",
              "patch_s1_train_rocm.py", "patch_strict_audio.py",
              "pack_for_cloud.py", "eval_retrain.py", "synth_v5.py",
              "make_listen_any.py", "sdk_repair_lfs.py"]
    for name in KEY_PY:
        p = V5 / name
        if not p.exists():
            print(f"  --   {name} 不存在")
            continue
        r = subprocess.run([PY, "-m", "py_compile", str(p)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            print(f"  OK   {name}")
        else:
            problems.append(f"{name} 编译失败")
            print(f"  FAIL {name}")
            print(f"       {r.stderr[-200:]}")

    print()
    print("=" * 74)
    print("4. 上云清单里 .sh 的换行（实际会推送的那些）")
    print("=" * 74)
    try:
        sys.path.insert(0, str(V5))
        import pack_for_cloud as P
        items = P.collect()
        n_sh = 0
        for src, rel in items:
            if rel.endswith(".sh"):
                n_sh += 1
                raw = src.read_bytes()
                c = raw.count(b"\r\n")
                if c:
                    problems.append(f"待推送 {rel}: {c} 个 CRLF")
                    print(f"  **CRLF** {rel}: {c}")
        print(f"  待推送 .sh 共 {n_sh} 个")
    except Exception as e:  # noqa: BLE001
        print(f"  跳过（{type(e).__name__}: {str(e)[:80]}）")

    print()
    print("=" * 74)
    print("结论")
    print("=" * 74)
    if problems:
        print(f"  ✗ {len(problems)} 个问题:")
        for x in problems:
            print(f"     {x}")
        if not args.fix:
            print("\n  加 --fix 可自动修 CRLF")
        return 1
    print("  ✓ 换行、内嵌 Python、关键 .py 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
