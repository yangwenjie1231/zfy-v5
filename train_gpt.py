# -*- coding: utf-8 -*-
"""GPT（s1）微调驱动 —— 支持 A / B 两条底模路线。

## 背景（已查证的事实）

  · `config.py` 里 v3/v4/v5dev/v5turbo/v2Pro/v2ProPlus **全部**用 `s1v3.ckpt`
    -> s1v3 就是"通用底模"，v3 之后没再更新过
  · 上游官方仓库（AI-ModelScope/GPT-SoVITS）**只有 3 个** GPT 底模
  · 生产 `zfh-e4.ckpt` 是从 **v2 底模** 微调的，不是 s1v3
  · 两个底模结构完全一致（295 键 / (732,512) / **77.61 M** 参数），
    但 sha256 不同、逐张量相对差中位 **21.66%** -> 是两次实质不同的训练
  · 两者 info 都写着 `pretrained_s1longer_v2`，config 只差 `train.epochs`
    （s1v3=20，v2=200）

## 两条路线

  A  `--base s1v3`  从 s1v3 微调  —— v5 原生配套
  B  `--base v2`    从 v2 底模微调 —— 复刻生产路径，只是数据多 13 倍

## 语料（已就绪，无需重新准备）

  logs/zfy_v5/2-name2text.txt     825 行（音素）
  logs/zfy_v5/6-name2semantic.tsv 826 行（语义 token）
  logs/zfy_v5/3-bert/             825 个 —— **dataset.py 会从 name2text
                                  的目录推导 3-bert，必须有**

  对比生产：片段 62 -> 825（13.3 倍），token 21,176 -> 95,210（4.5 倍）

## 用法（云上，检出根）

    python _tools/train_gpt.py --base both            # A + B 都跑
    python _tools/train_gpt.py --base s1v3            # 只跑 A
    python _tools/train_gpt.py --base v2 --epochs 6
    python _tools/train_gpt.py --base both --dry-run

产出：GPT_weights_gpt{A,B}/zfy_gpt{A,B}-e{N}.ckpt
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 两条路线的底模
BASES = {
    "s1v3": {
        "tag": "A",
        "rel": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
        "desc": "s1v3 底模（v5 原生配套，v3+ 全系共用）",
    },
    "v2": {
        "tag": "B",
        "rel": ("GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/"
                "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"),
        "desc": "v2 底模（复刻生产 zfh-e4 的起点，5kh 预训练）",
    },
}


def find_root() -> Path:
    """定位检出根（含 GPT_SoVITS/s1_train.py）。

    两种布局都要支持：
      云端: <检出根>/GPT_SoVITS/s1_train.py      （clone 出来的仓库根）
      本机: <检出根>/GPT-SoVITS/GPT_SoVITS/s1_train.py
    """
    here = Path.cwd()
    cands = []
    for c in (here, *here.parents):
        cands.append(c)
        cands.append(c / "GPT-SoVITS")     # 本机布局多一层
    cands.append(Path("/mnt/workspace/zfy-v5-fullft-pkg"))
    for c in cands:
        if (c / "GPT_SoVITS" / "s1_train.py").exists():
            return c
    return Path("/mnt/workspace/zfy-v5-fullft-pkg")


def build_config(root: Path, base_rel: str, exp: str, half_dir: str,
                 out_dir: Path, phoneme: Path, semantic: Path,
                 epochs: int, batch: int, warmup: int, decay: int,
                 precision: str, max_sec: int, num_workers: int) -> dict:
    """按 s1longer-v2 模板生成配置。

    ⚠️ 关键字段（缺一个就崩）：
      pretrained_s1        载入底模（t2s_lightning_module 读这个 key）
      train_phoneme_path   2-name2text.txt
      train_semantic_path  6-name2semantic.tsv
      half_weights_save_dir 输出目录（**必须已存在**，my_save 不会自建）
      model.phoneme_vocab_size 必须 732（与底模一致，不能用模板里的 512）
    """
    return {
        "train": {
            "seed": 1234,
            "epochs": epochs,
            "batch_size": batch,
            "save_every_n_epoch": 1,        # 每轮都存，便于盲听挑
            "precision": precision,
            "gradient_clip": 1.0,
            "exp_name": exp,
            "half_weights_save_dir": half_dir,
            "if_save_latest": True,
            "if_save_every_weights": True,
            "if_dpo": False,
        },
        "optimizer": {
            "lr": 0.01,
            "lr_init": 0.00001,
            "lr_end": 0.0001,
            "warmup_steps": warmup,
            "decay_steps": decay,
        },
        "data": {
            "max_eval_sample": 8,
            "max_sec": max_sec,
            "num_workers": num_workers,
            "pad_val": 1024,
        },
        "model": {
            "vocab_size": 1025,
            "phoneme_vocab_size": 732,      # ★ 与 s1v3/v2 底模一致
            "embedding_dim": 512,
            "hidden_dim": 512,
            "head": 16,
            "linear_units": 2048,
            "n_layer": 24,
            "dropout": 0,
            "EOS": 1024,
            "random_bert": 0,
        },
        "inference": {"top_k": 15},
        "output_dir": str(out_dir),
        "pretrained_s1": base_rel,
        "train_phoneme_path": str(phoneme),
        "train_semantic_path": str(semantic),
    }


def write_yaml(cfg: dict, path: Path) -> None:
    """手写 YAML（避免依赖 pyyaml 的格式细节；本配置全是标量）。

    ⚠️ 写完必须**回读校验** —— 手写 YAML 要自己保证标量格式合法，
    而 float 的科学计数法是经典陷阱（见 `_yaml_val` 的注释）。
    这里用 PyYAML 回读并逐项比对类型与数值，不一致就直接报错。
    """
    lines = ["# 由 train_gpt.py 生成"]
    for section in ("train", "optimizer", "data", "model", "inference"):
        lines.append(f"{section}:")
        for k, v in cfg[section].items():
            lines.append(f"  {k}: {_yaml_val(v)}")
    for k in ("output_dir", "pretrained_s1", "train_phoneme_path",
              "train_semantic_path"):
        lines.append(f"{k}: {_yaml_val(cfg[k])}")
    # ---- ZFY_WIN_ENCODING_PATCH ----
    # ⚠️ `newline="\n"`：Windows 下 `Path.write_text` 的 `newline=None`
    #    会把 `\n` 转成 `\r\n`（实测生成文件头是 `... 0d 0a`）。
    #    虽然 PyYAML 容忍 CRLF，但与 Linux 上生成的不一致，
    #    跨平台比对 YAML 时会误判"文件不同"。
    path.write_text("\n".join(lines) + "\n", encoding="utf-8",
                    newline="\n")

    # ---- 回读校验（防止 "1e-05 被当成字符串" 这类静默错误）----
    try:
        import yaml
    except ImportError:
        print("  ⚠️ 无 pyyaml，跳过回读校验")
        return
    back = yaml.full_load(path.read_text(encoding="utf-8"))
    bad = []
    for section in ("train", "optimizer", "data", "model", "inference"):
        for k, want in cfg[section].items():
            got = (back.get(section) or {}).get(k)
            if type(got) is not type(want):
                bad.append((f"{section}.{k}", want, got))
            elif isinstance(want, float) and abs(got - want) > 1e-12:
                bad.append((f"{section}.{k}", want, got))
    for k in ("output_dir", "pretrained_s1", "train_phoneme_path",
              "train_semantic_path"):
        got = back.get(k)
        if type(got) is not type(cfg[k]):
            bad.append((k, cfg[k], got))
    if bad:
        print()
        print("  !! YAML 回读校验失败（类型/数值不一致）：")
        for name, want, got in bad:
            print(f"     {name}: 写入 {want!r} ({type(want).__name__}) "
                  f"-> 回读 {got!r} ({type(got).__name__})")
        print(f"     文件: {path}")
        raise SystemExit("YAML 标量格式不合法，已中止（见上）")
    print(f"  ✓ YAML 回读校验通过（{len(bad)} 处不一致）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="both",
                    choices=["s1v3", "v2", "both"], help="底模路线")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--warmup-steps", type=int, default=2000)
    ap.add_argument("--decay-steps", type=int, default=40000)
    ap.add_argument("--precision", default="16-mixed",
                    choices=["16-mixed", "32", "bf16-mixed"])
    ap.add_argument("--max-sec", type=int, default=54)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--root", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root) if args.root else find_root()
    os.chdir(root)

    print("=" * 78)
    print("GPT（s1）微调 —— A/B 两条底模路线")
    print("=" * 78)
    print(f"检出根: {root}")

    # 语料检查
    phoneme = root / "logs" / "zfy_v5" / "2-name2text.txt"
    semantic = root / "logs" / "zfy_v5" / "6-name2semantic.tsv"
    bert = root / "logs" / "zfy_v5" / "3-bert"
    problems = []
    for label, p, isdir in (("2-name2text.txt", phoneme, False),
                            ("6-name2semantic.tsv", semantic, False),
                            ("3-bert/", bert, True)):
        if not p.exists():
            problems.append(f"{label} 不存在")
            print(f"  ✗ {label}")
        elif isdir:
            n = len(list(bert.iterdir()))
            print(f"  ✓ {label} {n} 个")
            if n < 100:
                problems.append(f"3-bert 只有 {n} 个")
        else:
            n = len([l for l in p.read_text(encoding="utf-8",
                                            errors="replace").strip().split("\n")
                     if l.strip()])
            print(f"  ✓ {label} {n} 行")
            if n < 10:
                problems.append(f"{label} 只有 {n} 行")
    if problems:
        print(f"\n  !! 语料有问题: {problems}")
        print("     先跑 python _tools/sdk_repair_lfs.py --check")
        return 1

    # 每 epoch 步数
    n_sem = len([l for l in semantic.read_text(encoding="utf-8",
                                               errors="replace").strip().split("\n")
                 if l.strip()])
    steps_per_epoch = max(1, (n_sem - 1) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    print(f"\n  样本 {n_sem - 1} 条 -> 每 epoch 约 {steps_per_epoch} 步，"
          f"共 {total_steps} 步")

    # LR 调度提示
    #
    # ⚠️ 重要更正：warmup / cosine **是死代码**。
    #    AR/modules/lr_schedulers.py:58
    #        self.lr = lr = self.end_lr = 0.002   ###锁定用线性###不听话，直接锁定！
    #    step() 里 warmup / cosine 两个分支算出的 lr 在最后一行被**覆盖**，
    #    set_lr 里也是 g["lr"] = self.end_lr。
    #    实测（verify_lr_schedule_dead.py）：无论传什么参数，实际 LR 恒为 2e-3。
    #
    #    所以 `--warmup-steps` / `--decay-steps` **没有任何作用**，
    #    之前那个「全程在 warmup，LR 只会升到 4.13e-3」的提示是**误导**。
    print()
    print(f"  LR: 实际恒为 **2e-3**（warmup/cosine 是死代码，见下）")
    print(f"    AR/modules/lr_schedulers.py:58")
    print(f"      self.lr = lr = self.end_lr = 0.002   # 覆盖掉 warmup/cosine 的结果")
    print(f"    -> --warmup-steps {args.warmup_steps} / --decay-steps "
          f"{args.decay_steps} 无作用")
    print(f"    本次共 {total_steps} 步；生产 zfh-e4 走的同一条路径，也是 2e-3")
    print(f"    ⚠️ 2e-3 对微调偏大（常见 1e-5~1e-4）；若发散/过拟合需改该文件")

    targets = ["s1v3", "v2"] if args.base == "both" else [args.base]

    print()
    print("=" * 78)
    print("待跑路线")
    print("=" * 78)
    plan = []
    skipped = []
    for key in targets:
        info = BASES[key]
        base_path = root / info["rel"]
        exists = base_path.exists()
        tag = info["tag"]
        exp = f"zfy_gpt{tag}"
        half_dir_name = f"GPT_weights_gpt{tag}"
        half_dir = root / half_dir_name
        out_dir = root / "logs" / f"gpt{tag}"
        print(f"  [{tag}] {info['desc']}")
        print(f"      底模: {info['rel']}")
        if exists:
            print(f"      存在: 是  ({base_path.stat().st_size/1e6:.1f} MB)")
        else:
            print("      存在: **否**  <- 跳过这条路线")
        print(f"      输出: {half_dir_name}/{exp}-e*.ckpt")
        if exists:
            plan.append((key, info, base_path, exp, half_dir, out_dir))
        else:
            skipped.append((key, info))

    # ⚠️ 缺底模时**只跳过那一条**，不要让整批中止。
    #    踩过：--base both 因路线 B 缺底模而直接 return 1，
    #    连本来可跑的路线 A 也没跑成。
    if skipped:
        print()
        print("  !! 以下路线因底模缺失被跳过（其余照跑）：")
        for key, info in skipped:
            print(f"     [{info['tag']}] {info['rel']}")
        print()
        print("  补底模（本地已有，只是没打包）：")
        print("     v2 底模在 GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/")
        print("     s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt  (148 MB)")

    if not plan:
        print("\n  !! 没有任何可跑的路线（所有底模都缺失）")
        return 1

    if args.dry_run:
        print(f"\n(dry-run，未训练；将跑 {len(plan)} 条路线)")
        return 0

    # 逐个训练
    results = []
    for key, info, base_path, exp, half_dir, out_dir in plan:
        tag = info["tag"]
        print()
        print("=" * 78)
        print(f"[{tag}] 从 {key} 底模微调")
        print("=" * 78)

        # 目录必须预建（my_save 只 shutil.move，不自建）
        half_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "ckpt").mkdir(parents=True, exist_ok=True)

        cfg = build_config(
            root=root,
            base_rel=info["rel"],
            exp=exp,
            half_dir=str(half_dir),
            out_dir=out_dir,
            phoneme=phoneme,
            semantic=semantic,
            epochs=args.epochs,
            batch=args.batch_size,
            warmup=args.warmup_steps,
            decay=args.decay_steps,
            precision=args.precision,
            max_sec=args.max_sec,
            num_workers=args.num_workers,
        )
        cfg_path = root / f"tmp_s1_{key}.yaml"
        write_yaml(cfg, cfg_path)
        print(f"  配置: {cfg_path}")

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{root}{os.pathsep}{root/os.sep/'GPT_SoVITS'}"
        env["hz"] = "25hz"      # dataset.py 用 os.environ['hz'] 推 token 帧率
        env["_CUDA_VISIBLE_DEVICES"] = "0"

        t0 = time.time()
        r = subprocess.run(
            [sys.executable, "-s", "GPT_SoVITS/s1_train.py", "-c", str(cfg_path)],
            cwd=str(root), env=env, capture_output=True, text=True)
        out = (r.stdout or "") + "\n" + (r.stderr or "")
        print(out[-3000:])

        got = sorted(half_dir.glob(f"{exp}-e*.ckpt"))
        ok = r.returncode == 0 and len(got) > 0
        results.append((tag, key, ok, len(got), time.time() - t0))
        print(f"  -> {'成功' if ok else '失败'}，产出 {len(got)} 个 ckpt，"
              f"用时 {time.time()-t0:.0f}s")
        if not ok:
            for kw in ("nccl", "gloo", "out of memory", "ModuleNotFound",
                       "FileNotFound", "Traceback"):
                hits = [l for l in out.split("\n") if kw.lower() in l.lower()]
                if hits:
                    print(f"     发现 '{kw}': {hits[0].strip()[:140]}")

    print()
    print("=" * 78)
    print("汇总")
    print("=" * 78)
    for tag, key, ok, n, secs in results:
        print(f"  [{tag}] {key:6s} {'✓' if ok else '✗'}  {n} 个 ckpt  "
              f"{secs:.0f}s")
    if all(r[2] for r in results):
        print("""
  下一步：把两条路线的 ckpt 加入合成对比。

  在本机合成时用 --gpt 指定：
      python synth_v5.py --weights <SoVITS权重> --gpt GPT_weights_gptA/zfy_gptA-e4.ckpt
""")
    return 0 if all(r[2] for r in results) else 1


def _yaml_val(v) -> str:
    """把 Python 值转成**PyYAML 能正确回读**的 YAML 标量。

    ## ⚠️ 踩过的坑：float 的科学计数法
    Python 的 `str(0.00001)` 是 `'1e-05'`，而 **YAML 1.1 不认这种写法**
    （必须写成 `1.0e-05` 或 `0.00001`）。PyYAML 会把 `1e-05` 解析成**字符串**，
    于是 `lr_init` 变成 str，训练时报：
        TypeError: unsupported operand type(s) for -: 'float' and 'str'
    实测：
        yaml.full_load("v: 1e-05")   -> str '1e-05'     <-- 错
        yaml.full_load("v: 0.00001") -> float 1e-05     <-- 对
        yaml.full_load("v: 1.0e-05") -> float 1e-05     <-- 对

    ## 修法
    用 `repr()` 拿到可 round-trip 的表示；若含 `e`/`E`，展开成定点十进制。
    """
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        # 路径里可能有反斜杠/冒号，统一加引号
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, float):
        s = repr(v)
        if "e" in s or "E" in s:
            # 科学计数法 -> 定点十进制（YAML 1.1 只认定点或 1.0e-5 形式）
            s = f"{v:.12f}".rstrip("0").rstrip(".")
            if s in ("", "-"):
                s = "0"
        return s
    return str(v)


if __name__ == "__main__":
    sys.exit(main())
