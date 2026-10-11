# -*- coding: utf-8 -*-
"""修掉 checkpoint 的**覆盖写** + 原子保存（为续训做准备）。

## 为什么要改（3 个真实踩到的坑）

### 坑 1：`my_save` 的临时文件写在**当前工作目录**
    # utils.py:67-72  原实现
    def my_save(fea, path):
        dir  = os.path.dirname(path)
        name = os.path.basename(path)
        tmp_path = "%s.pth" % (ttime())     # <- **相对路径**
        torch.save(fea, tmp_path)           # <- 落在 $PWD
        shutil.move(tmp_path, "%s/%s" % (dir, name))

实测后果：配额满时崩在 `torch.save` ->
仓库根留下 **`1791697691.0237224.pth`，4.3 GB 纯垃圾**。

### 坑 2：`if_save_latest=1` 用**固定文件名**每轮覆盖
    # s2_train_v5_full.py:380-396
    if hps.train.if_save_latest == 0:
        ... "G_{}.pth".format(global_step)          # 唯一名
    else:
        ... "G_{}.pth".format(233333333333)         # 固定名，覆盖写

该文件含 optimizer state，**实测 4.58 GB**。覆盖写意味着：
  · 需要瞬时 **2 倍空间**（先写 tmp 再 move）
  · 崩在写入中间 -> **旧文件也毁了** -> 无法 resume

### 坑 3：固定名 `G_233333333333.pth` 会**永远**被当成"最新"
    # utils.py:112-117
    def latest_checkpoint_path(dir_path, regex="G_*.pth"):
        f_list = glob.glob(...)
        f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        x = f_list[-1]

`233333333333` 是 12 位数，比任何真实 global_step 都大 ->
排序后**永远排最后** -> 一旦混入唯一名的新 checkpoint，
resume 仍会选到那个哨兵文件。

## 修法（3 处补丁）

| # | 文件 | 改动 |
|---|---|---|
| 1 | `utils.py` `my_save` | 临时文件放**目标同目录**、`os.replace` 原子替换、`fsync`、失败时清理 |
| 2 | `utils.py` `latest_checkpoint_path` | 排序 key 只看 **basename**，并把哨兵值映射为 0（最旧） |
| 3 | `s2_train_v5_full.py` | 始终用唯一名 `G_{global_step}.pth` + **保留策略**（`ZFY_KEEP_CKPTS`，默认 2） |

## 兼容性
· 唯一名 = 上游 `if_save_latest=0` 的行为，`latest_checkpoint_path` 本来就能处理
· 保留策略默认 2：允许回退一个 epoch；`ZFY_KEEP_CKPTS=1` 更省空间
· 哨兵文件（若有）被视作最旧 -> 第 2 次保存后自动清掉
· **默认行为改变**：即使 `if_save_latest=1` 也用唯一名。
  这是**有意的** —— 覆盖写本身就是要修掉的 bug。

## 用法
    python _tools/patch_ckpt_atomic.py --root $PWD            # 打补丁
    python _tools/patch_ckpt_atomic.py --root $PWD --check    # 只检查
    python _tools/patch_ckpt_atomic.py --root $PWD --revert   # 还原
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MARK = "ZFY_CKPT_ATOMIC_PATCH"

# ---------------------------------------------------------------- 补丁 1
U_REL = "GPT_SoVITS/utils.py"

U_OLD_1 = '''def my_save(fea, path):  #####fix issue: torch.save doesn't support chinese path
    dir = os.path.dirname(path)
    name = os.path.basename(path)
    tmp_path = "%s.pth" % (ttime())
    torch.save(fea, tmp_path)
    shutil.move(tmp_path, "%s/%s" % (dir, name))
'''

U_NEW_1 = '''def my_save(fea, path):  #####fix issue: torch.save doesn't support chinese path
    # ---- ZFY_CKPT_ATOMIC_PATCH (1/2) ----
    # 原实现:
    #     tmp_path = "%s.pth" % (ttime())      # **相对路径** -> 落在 $PWD
    #     torch.save(fea, tmp_path)
    #     shutil.move(tmp_path, "%s/%s" % (dir, name))
    #
    # 三个问题:
    #   1. 临时文件写在工作目录 -> 崩溃时留下 4.3 GB 垃圾
    #      （实测: 1791697691.0237224.pth）
    #   2. 若目标在别的文件系统，shutil.move 退化成 copy ->
    #      4.58 GB 的文件需要 2 倍空间
    #   3. 没有 fsync -> 配额/断电中断时可能留下半截文件
    #
    # 修法:
    #   · 临时文件放**目标同目录**（同文件系统 -> rename 是原子的）
    #   · 传**文件对象**给 torch.save（顺带绕开中文路径问题）
    #   · 写前 fsync，成功后 os.replace 原子替换
    #   · 任何失败都清理临时文件
    import tempfile

    tdir = os.path.dirname(path) or "."
    os.makedirs(tdir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".zfy_tmp_", suffix=".pth",
                                    dir=tdir)
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(fea, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise
'''

U_OLD_2 = '''def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    print(x)
    return x
'''

U_NEW_2 = '''def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    # ---- ZFY_CKPT_ATOMIC_PATCH (2/2) ----
    # 原实现:
    #     f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    # 两个问题:
    #   1. digits 取自**完整路径**，目录名里的数字（logs/v5_full_e32_...）
    #      也被算进去。同目录下前缀相同，所以相对顺序侥幸正确，
    #      但一旦换目录就不可靠。
    #   2. 固定名 G_233333333333.pth 的 233333333333 是 12 位数，
    #      比任何真实 global_step 都大 -> **永远**排最后 ->
    #      一旦混入唯一名的新 checkpoint，resume 仍会选到哨兵文件。
    # 修法: key 只看 basename，且把哨兵值视为 0（最旧）。
    def _zfy_ckpt_key(p):
        base = os.path.basename(p)
        n = int("".join(filter(str.isdigit, base)) or 0)
        return 0 if n == 233333333333 else n

    f_list = glob.glob(os.path.join(dir_path, regex))
    if not f_list:
        raise FileNotFoundError(
            "no checkpoint matching %r in %s" % (regex, dir_path))
    f_list.sort(key=_zfy_ckpt_key)
    x = f_list[-1]
    print(x)
    return x
'''

# ---------------------------------------------------------------- 补丁 2
S_REL = "GPT_SoVITS/s2_train_v5_full.py"

S_OLD = '''    if epoch % hps.train.save_every_epoch == 0 and rank == 0:
        if hps.train.if_save_latest == 0:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(save_root, "G_{}.pth".format(global_step)),
            )
        else:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(save_root, "G_{}.pth".format(233333333333)),
            )
'''

S_NEW = '''    if epoch % hps.train.save_every_epoch == 0 and rank == 0:
        # ---- ZFY_CKPT_ATOMIC_PATCH ----
        # 原逻辑: `if_save_latest == 1`（本配置就是）时用**固定名**
        #         G_233333333333.pth 每轮覆盖写。
        # 问题（实测踩到）:
        #   · 该文件含 optimizer state，实测 **4.58 GB**
        #   · 覆盖写需要瞬时 2 倍空间（my_save 先写 tmp 再 move）
        #   · 配额满时崩在写入中间 -> **旧文件也毁了**，无法 resume
        #   · TensorBoard 也写同目录，配额满时连日志都写不进
        #     -> OSError: [Errno 122] Disk quota exceeded
        #
        # 修法: 始终用**唯一名** G_{global_step}.pth
        #       （等价于上游 if_save_latest=0 的命名，latest_checkpoint_path
        #         本来就能正确处理），并只保留最新 N 个。
        #
        # 为什么安全:
        #   · 新文件先以唯一名完整写完 -> 中途崩只会丢**这一个**，
        #     旧的仍然可用（不像覆盖写会毁掉旧文件）
        #   · 保留策略在**保存成功之后**才删旧的
        #
        # ZFY_KEEP_CKPTS 默认 2（允许回退一个 epoch）；
        # 设 1 更省空间（每次只用最新，旧的保存成功后即删）。
        utils.save_checkpoint(
            net_g,
            optim_g,
            hps.train.learning_rate,
            epoch,
            os.path.join(save_root, "G_{}.pth".format(global_step)),
        )

        try:
            _zfy_keep = int(os.environ.get("ZFY_KEEP_CKPTS", "2"))
        except ValueError:
            _zfy_keep = 2
        if _zfy_keep > 0:
            def _zfy_key(fn):
                n = int("".join(c for c in fn if c.isdigit()) or 0)
                return 0 if n == 233333333333 else n

            _zfy_all = sorted(
                [f for f in os.listdir(save_root)
                 if f.startswith("G_") and f.endswith(".pth")],
                key=_zfy_key,
            )
            for _zfy_old in _zfy_all[:-_zfy_keep]:
                try:
                    os.remove(os.path.join(save_root, _zfy_old))
                    logger.info("[ZFY] 清理旧 checkpoint: %s" % _zfy_old)
                except OSError as _e:
                    logger.info("[ZFY] 清理失败 %s: %s" % (_zfy_old, _e))
'''


def patch_file(root: Path, rel: str, pairs: list[tuple[str, str]],
               check: bool, revert: bool, label: str) -> int:
    p = root / rel
    if not p.exists():
        print(f"  !! 找不到 {p}")
        return 1
    src = p.read_text(encoding="utf-8", errors="replace")
    bak = p.with_suffix(".py.bak_zfy_ckpt")

    if revert:
        if bak.exists():
            shutil.copy2(bak, p)
            print(f"  已从 {bak.name} 还原 ({rel})")
            return 0
        print(f"  未打过补丁 ({rel})")
        return 0

    if MARK in src:
        print(f"  已打过补丁: {rel}")
        return 0

    missing = [i + 1 for i, (old, _) in enumerate(pairs) if old not in src]
    if missing:
        print(f"  !! {rel} 找不到第 {missing} 处目标片段")
        return 1

    if check:
        for i, _ in enumerate(pairs, 1):
            print(f"  [{i}/{len(pairs)}] {label} {i}")
        return 0

    for old, new in pairs:
        src = src.replace(old, new, 1)
    if not bak.exists():
        shutil.copy2(p, bak)
        print(f"  备份: {bak.name}")
    p.write_text(src, encoding="utf-8")
    for i, _ in enumerate(pairs, 1):
        print(f"  [{i}/{len(pairs)}] {label} {i} 已应用")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    print("=" * 76)
    print(f"修 checkpoint 覆盖写 + 原子保存: {root}")
    print("=" * 76)

    rc = 0
    rc |= patch_file(root, U_REL,
                     [(U_OLD_1, U_NEW_1), (U_OLD_2, U_NEW_2)],
                     args.check, args.revert, "utils.py")
    print()
    rc |= patch_file(root, S_REL, [(S_OLD, S_NEW)],
                     args.check, args.revert, "s2_train_v5_full.py")

    print()
    print("=" * 76)
    if rc == 0 and not args.check and not args.revert:
        print("""
  补丁完成。

  ## 现在的行为
    · checkpoint 名: `G_{global_step}.pth`（**唯一**，不再覆盖）
    · 保留最新 2 个（`ZFY_KEEP_CKPTS=1` 可改成 1）
    · 保存是**原子**的：临时文件在目标同目录，`os.replace` 替换
    · 崩溃只丢**当前这一个**，旧 checkpoint 完好 -> 可 resume

  ## 续训命令
      cd /mnt/workspace/zfy-v5-fullft-pkg
      ZFY_KEEP_CKPTS=2 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD ZFY_UNFREEZE=1 \\
        nohup python -s GPT_SoVITS/s2_train_v5_full.py \\
        --config tmp_full_e32_nofreeze.json > train_nofreeze2.log 2>&1 &

  ## 启动后确认
      grep -E "Loaded checkpoint|start training from epoch" train_nofreeze2.log | head -3
      # 应显示: Loaded checkpoint '.../G_233333333333.pth' (iteration 26)
      #         start training from epoch 27

  ## 空间预算（每次保存的峰值）
      旧 checkpoint 4.58 GB + 新 checkpoint 4.58 GB（写入中）
      + 导出权重 766 MB
      ≈ 10 GB 峰值，稳定后 ≈ 9.2 GB（保留 2 个）

  ## 验证
      python _tools/verify_ckpt_atomic.py --root $PWD
""")
    elif rc != 0 and not args.check:
        print("  有改动未应用，见上面的 !! 行")
    print("=" * 76)
    return rc


if __name__ == "__main__":
    sys.exit(main())
