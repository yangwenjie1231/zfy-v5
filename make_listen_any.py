# -*- coding: utf-8 -*-
"""通用盲听集生成：把 eval 输出目录里的多个候选拼成"听不出谁是谁"的对比文件。

## 为什么需要

现有的 `make_listen_rank.py` / `make_listen_retrain.py` 都**硬编码**到
旧实验目录（`out/r32_vs_r128`、`out/ref_fix` 等），没法用于新实验
（v5 全量 e8/e32 vs v2Pro）。

## 做法

  1. 从 `<eval-dir>/<tag>/*.wav` 读各候选的同一批句子
  2. **每条句子内部打乱候选顺序**（每句独立打乱，避免"总是第一个"这种规律）
  3. **响度对齐**：每段先归一到 TARGET_RMS（0.06）再拼接
     —— 否则音量差会主导判断（这是我们之前踩过的：先前的 16kHz 对比
        就把带宽差毁了）
  4. 段间留 0.7s 静音，开头留 0.5s
  5. **答案单独写在 answer_key.json**，不在文件名里泄露

## 用法

    python make_listen_any.py --eval-dir out/full_eval --tags full_e08,full_e32,v2pro
    python make_listen_any.py --eval-dir out/full_eval --auto --n 8
    python make_listen_any.py --eval-dir out/full_eval --tags a,b,c --n 5 --seed 7

产出：
    out/listen_any/
      01.wav .. 0N.wav     每句一个文件，内含所有候选（顺序已打乱）
      answer_key.json      每个文件里各段的标签顺序
      README.txt           怎么听、答案在哪
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SR = 48000
TARGET_RMS = 0.06     # 与 make_listen_rank.py 保持一致，便于跨轮对比
GAP = 0.7
HEAD = 0.5


def load_aligned(p: Path) -> np.ndarray:
    """读音频 -> 单声道 -> 重采样到 SR -> 归一到 TARGET_RMS。"""
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        k = max(1, int(round(len(x) * SR / sr)))
        x = np.interp(np.linspace(0, len(x) - 1, k), np.arange(len(x)), x)
    x = x.astype(np.float32)
    r = float(np.sqrt(np.mean(x ** 2))) + 1e-9
    return x * (TARGET_RMS / r)


def concat(parts: list[np.ndarray], gap: float = GAP) -> np.ndarray:
    g = np.zeros(int(gap * SR), dtype=np.float32)
    out = [np.zeros(int(HEAD * SR), dtype=np.float32)]
    for p in parts:
        out += [p, g]
    full = np.concatenate(out)
    pk = float(np.max(np.abs(full)))
    return full * (0.95 / pk) if pk > 0.95 else full


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True,
                    help="eval_retrain.py 的输出根目录（含各 tag 子目录）")
    ap.add_argument("--tags", default=None,
                    help="逗号分隔的标签；留空配合 --auto")
    ap.add_argument("--auto", action="store_true",
                    help="自动取目录下所有 tag（按 eval.json 的 sim 降序）")
    ap.add_argument("--out", default=None, help="输出目录；默认 <eval-dir>/../listen_any")
    ap.add_argument("--n", type=int, default=8, help="用前 N 句（默认 8）")
    ap.add_argument("--seed", type=int, default=42, help="打乱种子（固定=可复现）")
    ap.add_argument("--exclude", default="", help="逗号分隔，排除某些 tag")
    args = ap.parse_args()

    ev = Path(args.eval_dir)
    if not ev.is_dir():
        print(f"目录不存在: {ev}")
        return 1

    out = Path(args.out) if args.out else (ev.parent / "listen_any")
    out.mkdir(parents=True, exist_ok=True)

    excl = {x.strip() for x in args.exclude.split(",") if x.strip()}

    # ---- 决定用哪些 tag ----
    if args.tags:
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    else:
        sims = {}
        ej = ev / "eval.json"
        if ej.exists():
            try:
                for r in json.loads(ej.read_text(encoding="utf-8"))["rows"]:
                    sims[r["tag"]] = r.get("sim", 0)
            except Exception:  # noqa: BLE001
                pass
        tags = [p.name for p in sorted(ev.iterdir())
                if p.is_dir() and any(p.glob("*.wav"))]
        tags.sort(key=lambda t: -sims.get(t, 0))

    tags = [t for t in tags if t not in excl]
    if len(tags) < 2:
        print(f"至少需要 2 个候选，当前: {tags}")
        return 1

    print("=" * 70)
    print("通用盲听集生成")
    print("=" * 70)
    print(f"  来源: {ev}")
    print(f"  候选: {tags}")
    print(f"  输出: {out}")
    print(f"  响度对齐: 每段归一到 RMS {TARGET_RMS}（避免音量差干扰）")
    print(f"  顺序: 每句独立打乱（seed={args.seed}），答案单写在 answer_key.json")
    print()

    # ---- 收集每个 tag 的句子文件 ----
    files: dict[str, list[Path]] = {}
    for t in tags:
        fs = sorted((ev / t).glob("*.wav"))
        if not fs:
            print(f"  !! {t} 没有 wav，跳过")
            continue
        files[t] = fs

    tags = [t for t in tags if t in files]
    if len(tags) < 2:
        print("有效候选不足 2 个")
        return 1

    n = min(args.n, min(len(v) for v in files.values()))
    print(f"  每句用前 {n} 段")

    rng = random.Random(args.seed)
    key = {}
    for i in range(n):
        order = tags[:]
        rng.shuffle(order)                       # 每句独立打乱
        parts = [load_aligned(files[t][i]) for t in order]
        full = concat(parts)
        fn = out / f"{i + 1:02d}.wav"
        sf.write(str(fn), full, SR)
        key[fn.name] = order
        durs = [f"{len(p)/SR:.2f}s" for p in parts]
        print(f"  {fn.name}  {len(full)/SR:6.2f}s  "
              f"{len(order)} 段 {durs}  顺序={order}")

    (out / "answer_key.json").write_text(json.dumps(
        {"eval_dir": str(ev), "tags": tags, "n": n, "seed": args.seed,
         "order": key},
        ensure_ascii=False, indent=2), encoding="utf-8")

    # 客观指标也存一份，供听完后对照
    if (ev / "eval.json").exists():
        try:
            rows = json.loads((ev / "eval.json").read_text(encoding="utf-8"))["rows"]
            (out / "objective.json").write_text(json.dumps(
                rows, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    (out / "README.txt").write_text(
        "盲听说明\n"
        "========\n"
        f"每个 NN.wav 里是同一句话的 {len(tags)} 个版本，顺序已随机打乱。\n"
        "所有片段都做了响度对齐（RMS 0.06），所以音量差不应影响判断。\n"
        "\n"
        "听的时候请横跨多个文件判断，不要只听一个 —— 单句容易受巧合影响。\n"
        "本仓库的教训：CER 是双峰分布（≈0% 或 ≈25%），单句差异可达 20pp，\n"
        "纯属噪声。要分辨 10pp 级别的差异至少需要 30~50 句。\n"
        "\n"
        f"候选（字母顺序不代表实际顺序）：{', '.join(tags)}\n"
        "答案：answer_key.json（**先别打开**）\n"
        "客观指标：objective.json（听完再对照）\n",
        encoding="utf-8")

    print()
    print(f"✓ 生成 {n} 个盲听文件 -> {out}")
    print(f"  答案: {out/'answer_key.json'}（先别看）")
    print(f"  说明: {out/'README.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
