# -*- coding: utf-8 -*-
"""给 v5 全量微调权重补版本头（b"07" = v5dev），让推理端能正确识别。

## 根因（已定位到 process_ckpt.savee）

    def savee(ckpt, name, epoch, steps, hps, model_version=None, lora_rank=None):
        ...
        if lora_rank:
            my_save2(opt, path, model_version)            # 写版本头
        elif model_version != None and "Pro" in model_version:
            my_save2(opt, path, model_version)            # 写版本头
        else:
            my_save(opt, path)                            # ← 不写头！

我们的全量微调 lora_rank=0 且 model_version="v5dev"（不含 "Pro"）
-> 走 my_save -> 纯 zip（PK 头）。

而推理端 `get_sovits_version_from_path_fast`：
    1. hash 查表（我们的权重不在表里）
    2. 读头 2 字节；若是 "PK" 就跳过
    3. **按文件大小猜**：v3 约 750 MB，我们 736 MB -> **误判成 v3**
-> init_vocoder("v3") -> BigVGAN.from_pretrained(24kHz 目录)
-> HFValidationError（那目录本地不存在）

## 修法

给权重文件**补上正确的前 2 字节** b"07"：

    my_save2 的做法:  data = byte + data[2:]
    即把 zip 的头两字节 "PK" 替换成版本字节。

反向操作（本脚本）：把前 2 字节换回版本字节，其余不动。
因为 torch.save 的 zip 结构只依赖后续内容，头 2 字节被替换是官方设计。

用法：
    python fix_weight_header.py --dir <权重目录>            # 只检查
    python fix_weight_header.py --dir <权重目录> --apply    # 实际修改
    python fix_weight_header.py --file a.pth --apply
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# process_ckpt.model_version2byte
V2B = {"v3": b"03", "v4": b"04", "v2Pro": b"05", "v2ProPlus": b"06",
       "v5dev": b"07", "v5turbo": b"08"}
B2V = {v: k for k, v in V2B.items()}


def peek(p: Path) -> tuple[bytes, int]:
    with p.open("rb") as f:
        return f.read(2), p.stat().st_size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="权重目录")
    ap.add_argument("--file", default=None, help="单个权重")
    ap.add_argument("--version", default="v5dev",
                    choices=list(V2B), help="要写入的版本（默认 v5dev）")
    ap.add_argument("--apply", action="store_true", help="实际修改（默认只检查）")
    args = ap.parse_args()

    want = V2B[args.version]
    files: list[Path] = []
    if args.file:
        files = [Path(args.file)]
    elif args.dir:
        d = Path(args.dir)
        files = sorted(d.glob("*.pth"))
    else:
        print("需给 --dir 或 --file")
        return 1

    if not files:
        print("没找到 .pth")
        return 1

    print("=" * 76)
    print(f"补版本头 -> {args.version} ({want!r})   模式: "
          f"{'**实际修改**' if args.apply else '只检查'}")
    print("=" * 76)

    n_ok = n_fix = n_bad = 0
    for p in files:
        if not p.exists():
            print(f"  缺失 {p}")
            n_bad += 1
            continue
        head, size = peek(p)
        if head == want:
            n_ok += 1
            continue
        if head == b"PK":
            # 纯 zip -> 需要补头
            if args.apply:
                bak = p.with_suffix(p.suffix + ".bak_hdr")
                if not bak.exists():
                    shutil.copy2(p, bak)
                data = p.read_bytes()
                p.write_bytes(want + data[2:])
                new_head, _ = peek(p)
                ok = new_head == want
                print(f"  已修 {p.name}  PK -> {new_head!r}  "
                      f"{'OK' if ok else '**失败**'}  ({size/1e6:.1f} MB)")
                n_fix += 1 if ok else 0
                n_bad += 0 if ok else 1
            else:
                print(f"  待修 {p.name}  头={head!r} -> {want!r}  "
                      f"({size/1e6:.1f} MB)")
                n_fix += 1
        elif head in B2V:
            print(f"  !! {p.name} 头={head!r} 已是 {B2V[head]}，"
                  f"与目标 {args.version} 不同 —— 跳过（避免改错）")
            n_bad += 1
        else:
            print(f"  !! {p.name} 头={head!r} 未知 —— 跳过")
            n_bad += 1

    print()
    print("=" * 76)
    print(f"已是目标版本 {n_ok} 个 / 需修 {n_fix} 个 / 异常 {n_bad} 个")
    if not args.apply and n_fix:
        print("\n  加 --apply 实际修改（会自动备份为 *.bak_hdr）")
    print("=" * 76)
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
