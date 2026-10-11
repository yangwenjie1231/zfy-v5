# -*- coding: utf-8 -*-
"""给 s1（GPT）的 LR 加可控开关 —— 默认行为不变，需要时才改。

## 为什么需要
`AR/modules/lr_schedulers.py:58` 把学习率**硬锁**成 0.002：

    def step(self):
        if self._current_step < self.warmup_steps:
            lr = self.init_lr + self._warmup_rate * self._current_step
        elif ...
        else:
            ...cosine...
        self.lr = lr = self.end_lr = 0.002   ###锁定用线性###不听话，直接锁定！
        self.set_lr(lr)

上面的 warmup / cosine **全是死代码** —— 结果被最后一行无条件覆盖。
`set_lr` 里也是 `g["lr"] = self.end_lr`。

后果：
  · `--warmup-steps` / `--decay-steps` / `lr_init` / `lr_end` 全部无效
  · 实际 LR 恒为 **2e-3**，这对微调**偏大**（常见 1e-5 ~ 1e-4）
  · 若训练发散/过拟合，**没有任何办法调**（除非改源码）

## 本补丁做什么
**默认行为完全不变**（仍是 2e-3），只在设置了环境变量时才生效：

    ZFY_S1_LR 未设置 / 空     -> 恒 2e-3（与上游一致，零风险）
    ZFY_S1_LR=1e-4            -> 恒 1e-4（常量）
    ZFY_S1_LR=5e-5            -> 恒 5e-5
    ZFY_S1_LR=schedule        -> **恢复真正的 warmup + cosine**
                                  （用 config 里的 lr_init/lr/warmup_steps/
                                    decay_steps，即上游本来的设计）

同时把 `set_lr` 改成使用**传入的 lr**（而不是锁死的 `self.end_lr`），
否则不管外面算什么，写进 optimizer 的永远是 end_lr。

## 用法
    python _tools/patch_s1_lr.py --root $PWD            # 打补丁
    python _tools/patch_s1_lr.py --root $PWD --check    # 只检查
    python _tools/patch_s1_lr.py --root $PWD --revert   # 还原

    然后训练时：
    ZFY_S1_LR=1e-4 python _tools/train_gpt.py --base both
    ZFY_S1_LR=schedule python _tools/train_gpt.py --base both
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MARK = "ZFY_S1_LR_PATCH"
REL = "GPT_SoVITS/AR/modules/lr_schedulers.py"

# ---------------------------------------------------------------- set_lr
OLD_SET_LR = '''    def set_lr(self, lr):
        self._last_lr = [g["lr"] for g in self.optimizer.param_groups]
        for g in self.optimizer.param_groups:
            # g['lr'] = lr
            g["lr"] = self.end_lr  ###锁定用线性
'''

NEW_SET_LR = '''    def set_lr(self, lr):
        # ---- ZFY_S1_LR_PATCH ----
        # 上游这里写的是 `g["lr"] = self.end_lr`（锁死），于是**不管外面
        # 算出什么 lr，写进 optimizer 的永远是 end_lr**。
        # 改成用传入的 lr —— 这样常量/调度两种模式都能真正生效。
        # 未设 ZFY_S1_LR 时，step() 传入的就是 0.002，行为与上游**完全一致**。
        # ---- /ZFY_S1_LR_PATCH ----
        self._last_lr = [g["lr"] for g in self.optimizer.param_groups]
        for g in self.optimizer.param_groups:
            # g['lr'] = lr
            g["lr"] = lr
'''

# ---------------------------------------------------------------- __init__
OLD_TAIL = '''        self._last_lr = [self.lr]
'''

NEW_TAIL = '''        self._last_lr = [self.lr]

        # ---- ZFY_S1_LR_PATCH ----
        # 解析 ZFY_S1_LR 环境变量，决定 LR 策略。
        # 未设置 -> "locked"，保持上游行为（恒 0.002），零风险。
        import os as _zfy_os

        _zfy_raw = _zfy_os.environ.get("ZFY_S1_LR", "").strip().lower()
        self._zfy_mode = "locked"
        self._zfy_lr = 0.002
        self._zfy_logged = False
        if _zfy_raw:
            if _zfy_raw in ("schedule", "cosine", "warmup"):
                self._zfy_mode = "schedule"
            else:
                try:
                    self._zfy_lr = float(_zfy_raw)
                    self._zfy_mode = "constant"
                except ValueError:
                    print(
                        f"[ZFY] ZFY_S1_LR={_zfy_raw!r} 无法解析为浮点数，"
                        f"退回默认 2e-3"
                    )
        # ---- /ZFY_S1_LR_PATCH ----
'''

# ---------------------------------------------------------------- step 锁定行
OLD_LOCK = '''        self.lr = lr = self.end_lr = 0.002  ###锁定用线性###不听话，直接锁定！
        self.set_lr(lr)
        self.lr = lr
        self._current_step += 1
        return self.lr
'''

NEW_LOCK = '''        # ---- ZFY_S1_LR_PATCH ----
        # 上游这行把 warmup/cosine 的结果**无条件覆盖**成 0.002。
        # 现在按模式选择：
        #   locked   -> 0.002（上游行为，默认）
        #   constant -> self._zfy_lr（环境变量指定）
        #   schedule -> 用上面分支算出的 lr（真正的 warmup + cosine）
        if self._zfy_mode == "schedule":
            pass  # lr 已由上方 warmup / cosine 分支算出
        elif self._zfy_mode == "constant":
            lr = self._zfy_lr
        else:
            lr = 0.002
        if not self._zfy_logged:
            self._zfy_logged = True
            if self._zfy_mode == "schedule":
                print(
                    f"[ZFY] s1 LR 模式=schedule（真正的 warmup+cosine）  "
                    f"init={self.init_lr} peak={self.peak_lr} "
                    f"end={self.end_lr} warmup={self.warmup_steps} "
                    f"total={self.total_steps}"
                )
            else:
                print(f"[ZFY] s1 LR 模式={self._zfy_mode}  恒定 {lr:g}")
        # ---- /ZFY_S1_LR_PATCH ----

        self.set_lr(lr)
        self.lr = lr
        self._current_step += 1
        return self.lr
'''


def apply_patch(root: Path, check: bool, revert: bool) -> int:
    p = root / REL
    if not p.exists():
        print(f"  !! 找不到 {p}")
        return 1

    src = p.read_text(encoding="utf-8", errors="replace")
    bak = p.with_suffix(".py.bak_zfy_lr")

    # ---- 还原 ----
    if revert:
        if bak.exists():
            shutil.copy2(bak, p)
            print(f"  已从 {bak.name} 还原")
            return 0
        # 没有备份时用标记反查（不实现自动反向替换，提示手动）
        if MARK in src:
            print("  找不到备份文件，无法自动还原。请手动删除 "
                  f"{MARK} 相关代码块。")
            return 1
        print("  未打过补丁，无需还原")
        return 0

    # ---- 幂等 ----
    if MARK in src:
        print(f"  已打过补丁: {p.name}")
        return 0

    n = 0
    if OLD_SET_LR in src:
        src = src.replace(OLD_SET_LR, NEW_SET_LR, 1)
        n += 1
        print("  [1/3] set_lr：改用传入的 lr（不再锁死 end_lr）")
    else:
        print("  !! 找不到 set_lr 原文")

    if OLD_TAIL in src:
        src = src.replace(OLD_TAIL, NEW_TAIL, 1)
        n += 1
        print("  [2/3] __init__：解析 ZFY_S1_LR 环境变量")
    else:
        print("  !! 找不到 __init__ 末尾的 _last_lr 行")

    if OLD_LOCK in src:
        src = src.replace(OLD_LOCK, NEW_LOCK, 1)
        n += 1
        print("  [3/3] step：按模式选择 LR（不再无条件覆盖成 0.002）")
    else:
        print("  !! 找不到 step 里的锁定行")

    if n < 3:
        print(f"  !! 只应用了 {n}/3 处，中止（避免改出半成品）")
        return 1

    if check:
        print(f"  （--check 模式，未写盘；将应用 {n} 处改动）")
        return 0

    if not bak.exists():
        shutil.copy2(p, bak)
        print(f"  备份: {bak.name}")
    p.write_text(src, encoding="utf-8")
    print(f"  已写入 {p}  ({n} 处改动)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    print("=" * 74)
    print(f"给 s1（GPT）的 LR 加可控开关: {root}")
    print("=" * 74)
    rc = apply_patch(root, args.check, args.revert)
    print()
    print("=" * 74)
    if rc == 0 and not args.check and not args.revert:
        print("""
  补丁完成。**默认行为不变**（仍恒 2e-3），需要时才用环境变量：

      # 默认（与上游一致，零风险）
      python _tools/train_gpt.py --base both

      # 恒定 1e-4（微调常用量级）
      ZFY_S1_LR=1e-4 python _tools/train_gpt.py --base both

      # 恢复真正的 warmup + cosine（用 config 的 lr_init/lr/warmup_steps）
      ZFY_S1_LR=schedule python _tools/train_gpt.py --base both

  启动后日志里会有一行确认实际生效的模式：
      [ZFY] s1 LR 模式=locked  恒定 0.002
      [ZFY] s1 LR 模式=constant  恒定 0.0001
      [ZFY] s1 LR 模式=schedule（真正的 warmup+cosine） ...

  还原：python _tools/patch_s1_lr.py --root $PWD --revert
""")
    elif rc != 0:
        print("  有改动未应用，请检查上面的 !! 行")
    print("=" * 74)
    return rc


if __name__ == "__main__":
    sys.exit(main())
