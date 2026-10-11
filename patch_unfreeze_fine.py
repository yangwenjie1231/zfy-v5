# -*- coding: utf-8 -*-
"""细粒度解冻 —— 只解冻 ssl_proj + enc_p，**冻结 quantizer**（保护码本）。

## 问题（实测发现）
`patch_v5_unfreeze.py` 是**三组一起解冻**（因为 `models.py` 的
`freeze_quantizer` 是全有全无）。结果 nofreeze 权重**严重抖动 + 提前终止**：

    !! e08   2 句异常: 04(1.56s, 8.3字/秒), 05(0.64s, 7.8字/秒)
    !! e16   2 句异常: 03(1.40s, 10.0字/秒), 09(3.16s, 7.9字/秒)
    !! e24   1 句异常: 04(1.48s, 8.8字/秒)
    !! e32   3 句异常: 03(0.80s, 17.5字/秒), 05(0.64s, 7.8字/秒)

对照组 `e16_merged` / `v2pro` **一句都没崩**。

## 根因：码本 EMA 漂移
    core_vq.py:274  if self.training:            # 解冻后为 True
                        ...
    core_vq.py:290      self.embed.data.copy_(embed_normalized)   # 码本被 EMA 更新

`quantizer` 的 1024 个码字**不是靠梯度更新，而是靠 EMA**：
    self.cluster_size.data.mul_(self.decay).add_(...)
    self.embed.data.copy_(embed_normalized)

**只要 `self.training == True`，码本就必然漂移** —— 与优化器无关，
所以"给量化器设小学习率"**没用**。

漂移的后果：
  · GPT 在 `6-name2semantic.tsv` 上学的是**旧码本**的 token
  · SoVITS 解码时用**漂移后的**码本 -> 语义不对应 -> 提前 EOS

## 为什么 e16_merged 稳定
它用**冻结的底模** `ssl_proj`/`quantizer`（`freeze_quantizer=True` 训出来的），
码本完全没动。

## 修法：把"全有全无"拆成三组独立开关
`models.py:1325-1328` 原逻辑：
    if self.freeze_quantizer == True:
        set_no_grad(self.ssl_proj)
        set_no_grad(self.quantizer)
        set_no_grad(self.enc_p)

改成读环境变量 `ZFY_UNFREEZE` 的**细粒度列表**：
    ZFY_UNFREEZE=enc_p,ssl_proj    -> 只解冻这两组，**quantizer 保持冻结**（推荐）
    ZFY_UNFREEZE=all               -> 三组全解冻（等于旧行为）
    ZFY_UNFREEZE=1                 -> 等价 all（向后兼容）

同时 `forward` 里的 `self.quantizer.eval()` 也要跟着：
**只有 quantizer 被解冻时才不 eval**，否则保持 eval（码本不漂移）。

## 用法
    python _tools/patch_unfreeze_fine.py --root $PWD
    python _tools/patch_unfreeze_fine.py --root $PWD --check
    python _tools/patch_unfreeze_fine.py --root $PWD --revert
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MARK = "ZFY_UNFREEZE_FINE_PATCH"
REL = "GPT_SoVITS/module/models.py"

# ---------------------------------------------------------------- 补丁 A：set_no_grad 拆分
OLD_A = """        if self.freeze_quantizer == True:
            set_no_grad(self.ssl_proj)
            set_no_grad(self.quantizer)
            set_no_grad(self.enc_p)
"""

NEW_A = """        # ---- ZFY_UNFREEZE_FINE_PATCH (A: 细粒度解冻) ----
        # 上游 `freeze_quantizer` 是**全有全无**，三组一起冻结/解冻。
        # 但实测发现"三组一起解冻"会导致**提前终止**：
        #     e32 有 3 句异常（03 句 14 字只出 0.80s = 17.5 字/秒）
        # 根因是 **quantizer 的码本靠 EMA 漂移**（core_vq.py:290
        # `self.embed.data.copy_(...)`），与优化器无关 ——
        # 只要 `self.training == True` 就必然漂移。
        # 码本一变，GPT 学到的旧 token 就不再对应 -> 提前 EOS。
        #
        # 所以拆成三组独立开关，用环境变量 ZFY_UNFREEZE 控制：
        #     ZFY_UNFREEZE=enc_p,ssl_proj   只解冻这两组（**推荐**，码本不动）
        #     ZFY_UNFREEZE=all              三组全解冻（旧行为，会漂移）
        #     ZFY_UNFREEZE=1                等价 all（向后兼容）
        # 未设置时行为**完全不变**（零风险）。
        import os as _zfy_os
        _zfy_raw = _zfy_os.environ.get("ZFY_UNFREEZE", "").strip().lower()
        if _zfy_raw in ("1", "true", "yes", "on"):
            _zfy_want = {"ssl_proj", "quantizer", "enc_p"}
        elif _zfy_raw in ("", "0", "false", "no", "off"):
            _zfy_want = set()
        elif _zfy_raw == "all":
            _zfy_want = {"ssl_proj", "quantizer", "enc_p"}
        else:
            _zfy_want = {s.strip() for s in _zfy_raw.split(",") if s.strip()}
            _zfy_bad = _zfy_want - {"ssl_proj", "quantizer", "enc_p"}
            if _zfy_bad:
                print(f"[ZFY] [WARN] unknown ZFY_UNFREEZE item(s) "
                      f"{sorted(_zfy_bad)} ignored "
                      f"(valid: ssl_proj/quantizer/enc_p/all)")
                _zfy_want -= _zfy_bad

        # 记录给 forward 用（决定 quantizer 是否 .eval()）
        self._zfy_train_quantizer = "quantizer" in _zfy_want
        self._zfy_train_ssl_proj = "ssl_proj" in _zfy_want
        self._zfy_train_enc_p = "enc_p" in _zfy_want

        if self.freeze_quantizer == True:
            if "ssl_proj" not in _zfy_want:
                set_no_grad(self.ssl_proj)
            if "quantizer" not in _zfy_want:
                set_no_grad(self.quantizer)
            if "enc_p" not in _zfy_want:
                set_no_grad(self.enc_p)
            if _zfy_want:
                _zfy_frozen = sorted({"ssl_proj", "quantizer", "enc_p"}
                                     - _zfy_want)
                print(f"[ZFY] unfrozen {sorted(_zfy_want)}; "
                      f"still frozen {_zfy_frozen}")
                if "quantizer" in _zfy_want:
                    # ⚠️ 不要用 emoji（如 ⚠️）—— Windows GBK 控制台下
                    #    print 会抛 UnicodeEncodeError，把整个构造过程搞崩。
                    #    实测：verify_unfreeze_fine.py 的 all 模式探针因此崩溃，
                    #    三条断言全 None。用纯 ASCII 标记。
                    print("[ZFY] [WARN] quantizer unfrozen -> codebook WILL "
                          "drift by EMA; may cause early EOS "
                          "(measured: e32 had 3 abnormal sentences)")
        # ---- /ZFY_UNFREEZE_FINE_PATCH (A) ----
"""

# ---------------------------------------------------------------- 补丁 B：forward 的 eval 分支
OLD_B = """            maybe_no_grad = torch.no_grad() if self.freeze_quantizer else contextlib.nullcontext()
            with maybe_no_grad:
                if self.freeze_quantizer:
                    self.ssl_proj.eval()  #
                    self.quantizer.eval()
                    self.enc_p.eval()
"""

NEW_B = """            # ---- ZFY_UNFREEZE_FINE_PATCH (B: eval 粒度) ----
            # 原逻辑：freeze_quantizer=True 时三组一起 .eval()。
            # 现在按**各组是否被解冻**决定：
            #   未解冻 -> .eval()（冻结行为，码本不漂移）
            #   已解冻 -> 保持 train（参与训练）
            # ⚠️ quantizer 必须"未解冻就 eval" —— 它的码本靠 EMA 更新
            #    （core_vq.py:274 `if self.training:`），
            #    eval() 是**唯一**能阻止码本漂移的手段（优化器管不到 EMA）。
            _zfy_tq = getattr(self, "_zfy_train_quantizer", False)
            _zfy_ts = getattr(self, "_zfy_train_ssl_proj", False)
            _zfy_te = getattr(self, "_zfy_train_enc_p", False)
            _zfy_any_frozen = (not _zfy_ts) or (not _zfy_tq) or (not _zfy_te)
            maybe_no_grad = (torch.no_grad() if (self.freeze_quantizer
                                                 and not _zfy_any_frozen)
                             else contextlib.nullcontext())
            with maybe_no_grad:
                if self.freeze_quantizer:
                    if not _zfy_ts:
                        self.ssl_proj.eval()
                    if not _zfy_tq:
                        self.quantizer.eval()
                    if not _zfy_te:
                        self.enc_p.eval()
            # ---- /ZFY_UNFREEZE_FINE_PATCH (B) ----
"""


def patch_file(root: Path, check: bool, revert: bool) -> int:
    p = root / REL
    if not p.exists():
        print(f"  !! 找不到 {p}")
        return 1
    src = p.read_text(encoding="utf-8", errors="replace")
    bak = p.with_suffix(".py.bak_zfy_fine")

    if revert:
        if bak.exists():
            shutil.copy2(bak, p)
            print(f"  已从 {bak.name} 还原")
            return 0
        print("  未打过补丁")
        return 0

    if MARK in src:
        print(f"  已打过补丁: {p.name}")
        return 0

    missing = []
    if OLD_A not in src:
        missing.append("A: set_no_grad 段")
    if OLD_B not in src:
        missing.append("B: forward 的 eval 段")
    if missing:
        print(f"  !! 找不到目标片段: {missing}")
        print("     （models.py 可能已被 patch_v5_unfreeze.py 改过 ——")
        print("       本补丁**不能**与它共存，二选一）")
        return 1

    if check:
        print("  [A] 把 set_no_grad 拆成三组独立开关")
        print("  [B] 把 forward 的 .eval() 改成按组决定")
        return 0

    src = src.replace(OLD_A, NEW_A, 1)
    src = src.replace(OLD_B, NEW_B, 1)

    import ast
    try:
        ast.parse(src)
    except SyntaxError as e:
        print(f"  !! 打完补丁后语法错误（已放弃写入）: 第 {e.lineno} 行 {e.msg}")
        return 1

    if not bak.exists():
        shutil.copy2(p, bak)
        print(f"  备份: {bak.name}")
    p.write_text(src, encoding="utf-8")
    print("  [A] set_no_grad 已拆成三组独立开关")
    print("  [B] forward 的 .eval() 已改成按组决定")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    print("=" * 78)
    print(f"细粒度解冻（保护码本）: {root}")
    print("=" * 78)
    rc = patch_file(root, args.check, args.revert)
    print()
    if rc == 0 and not args.check and not args.revert:
        print("""
  补丁完成。

  ## 三组开关（环境变量 ZFY_UNFREEZE）
      enc_p,ssl_proj   只解冻这两组，**quantizer 保持冻结**（推荐）
      all / 1          三组全解冻（旧行为，码本会漂移）
      （未设置）       全冻结（与上游一致）

  ## 为什么 quantizer 必须保持冻结
  它的码本**不是靠梯度更新，而是靠 EMA**：
      core_vq.py:274  if self.training:
      core_vq.py:290      self.embed.data.copy_(embed_normalized)
  所以给优化器设小学习率**没用** —— 只有 `.eval()` 能阻止漂移。

  码本一变，GPT 在 `6-name2semantic.tsv` 上学的旧 token 就不再对应
  -> SoVITS 解码错乱 -> **提前 EOS**（实测 e32 有 3 句异常，
     最极端 03 句 14 字只出 0.80s = 17.5 字/秒）。

  ## 训练命令
      cd /mnt/workspace/zfy-v5-fullft-pkg
      ZFY_UNFREEZE=enc_p,ssl_proj CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD \\
        nohup python -s GPT_SoVITS/s2_train_v5_full.py \\
        --config tmp_full_e32_fine.json > train_fine.log 2>&1 &

  启动后应看到：
      [ZFY] 解冻 ['enc_p', 'ssl_proj']；仍冻结 ['quantizer']

  ## ⚠️ 本补丁与 patch_v5_unfreeze.py 互斥
  两者改同一段代码。用本补丁前先：
      python _tools/patch_v5_unfreeze.py --root $PWD --revert
""")
    elif rc != 0:
        print("  有改动未应用，见上面的 !! 行")
    print("=" * 78)
    return rc


if __name__ == "__main__":
    sys.exit(main())
