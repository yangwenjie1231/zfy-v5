# -*- coding: utf-8 -*-
"""解冻 enc_p / ssl_proj / quantizer，做**真正的全量微调**。

## 问题回顾
`s2_train_v5_full.py` 导出权重时会跳过 `requires_grad=False` 的参数：

    for key in ckpt:
        if key not in no_grad_names:      # <- enc_p/ssl_proj 被排除
            sim_ckpt[key] = ckpt[key].half().cpu()

而 `freeze_quantizer=True`（v5 的默认配置）导致：

    models.py:1325-1328   SynthesizerTrnV3.__init__
        if self.freeze_quantizer == True:
            set_no_grad(self.ssl_proj)     # -> requires_grad = False
            set_no_grad(self.quantizer)
            set_no_grad(self.enc_p)

于是这 15.33M 参数**既没训练、也没进权重** -> 推理时随机初始化
-> prompt_semantic 错乱（时长 9 倍异常）+ 音质细节缺失。

## 本补丁做什么
把 `freeze_quantizer` 从 True 改成 **False**，并确认三处配套逻辑：

  1. `models.py:1325` 的 `set_no_grad` 不再执行 -> 三组参数 requires_grad=True
  2. `models.py:1336-1341` 的 `maybe_no_grad` 变成 nullcontext，
     且不再调 `.eval()` -> 量化器处于 training 模式
  3. **量化器码本用 EMA 更新**（core_vq.py:246-262）：
         if self.training:
             self.cluster_size.data.mul_(self.decay).add_(...)
             self.embed.data.copy_(embed_normalized)
     所以只要不 `.eval()`，码本就会**自动跟着新语料漂移**
     —— 这正是我们想要的（让量化器适配庄方宜的语料）

## ⚠️ 两个风险（必须知道）
  A. **码本漂移** —— EMA 会让 1024 个码字向新语料移动。
     如果语料太少（825 条 / 63.5 分钟），码本可能"过拟合"到这个小分布，
     导致泛化变差。
     缓解：`freeze_quantizer` 改成 False 但**给量化器设小学习率**不可行
     （EMA 不受优化器控制，由 decay 决定）。所以只能靠**少训几轮**。

  B. **显存** —— 多 15.33M 参数要算梯度，激活也会增加。
     云端 205 GB，完全够（估算从 21~31 GB 增到 25~36 GB）。

## 用法
    python _tools/patch_v5_unfreeze.py --root $PWD            # 打补丁
    python _tools/patch_v5_unfreeze.py --root $PWD --check    # 只检查
    python _tools/patch_v5_unfreeze.py --root $PWD --revert   # 还原
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MARK = "ZFY_UNFREEZE_PATCH"
REL = "GPT_SoVITS/module/models.py"


def patch_models(root: Path, check: bool, revert: bool) -> int:
    """给 models.py 的 set_no_grad 加环境变量开关。

    为什么改 models.py 而不是改 JSON 配置：
      `freeze_quantizer` 是从 config 传进来的，改 JSON 也能生效。
      但**改 JSON 会让所有依赖该配置的地方都变**（包括未来可能的 LoRA 训练）。
      这里加一个**环境变量优先**的开关，更可控：
          ZFY_UNFREEZE=1  -> 强制 freeze_quantizer=False
          未设置          -> 保持配置原样（零风险）
    """
    p = root / REL
    if not p.exists():
        print(f"  !! 找不到 {p}")
        return 1
    src = p.read_text(encoding="utf-8", errors="replace")
    bak = p.with_suffix(".py.bak_zfy_unfreeze")

    if revert:
        if bak.exists():
            shutil.copy2(bak, p)
            print(f"  已从 {bak.name} 还原")
            return 0
        print("  未打过补丁（或备份已丢）")
        return 0

    if MARK in src:
        print(f"  已打过补丁: {p.name}")
        return 0

    # ⚠️⚠️ 必须锚定到 **SynthesizerTrnV3**（训练用的那个类）！
    #
    # models.py 里有**两个**几乎一样的类：
    #     L1241  class SynthesizerTrnV3     <- s2_train_v5_full.py:41 导入的就是它
    #     L1392  class SynthesizerTrnV3b    <- 没被训练用到
    # 我第一版只按 `self.quantizer = ResidualVectorQuantizer(...)` 找片段，
    # 结果**改中了 V3b**，训练毫无变化（实测：解冻后 trainable 仍是 367.49M）。
    #
    # 所以这里用"从 class SynthesizerTrnV3 开始、到下一个 class 为止"的
    # 区间来限定替换范围。
    cls_start = src.find("class SynthesizerTrnV3(nn.Module):")
    if cls_start < 0:
        print("  !! 找不到 class SynthesizerTrnV3")
        return 1
    nxt = src.find("\nclass ", cls_start + 10)
    cls_end = nxt if nxt > 0 else len(src)
    region = src[cls_start:cls_end]

    OLD = """        self.quantizer = ResidualVectorQuantizer(dimension=ssl_dim, n_q=1, bins=1024)
        self.freeze_quantizer = freeze_quantizer
        inter_channels2 = 512"""

    NEW = """        self.quantizer = ResidualVectorQuantizer(dimension=ssl_dim, n_q=1, bins=1024)
        self.freeze_quantizer = freeze_quantizer

        # ---- ZFY_UNFREEZE_PATCH ----
        # 环境变量优先：ZFY_UNFREEZE=1 时强制解冻 enc_p/ssl_proj/quantizer。
        #
        # 为什么必须放在 **set_no_grad 之前**：
        #   下面紧接着就是
        #       if self.freeze_quantizer == True:
        #           set_no_grad(self.ssl_proj / quantizer / enc_p)
        #   所以只要在这里把 freeze_quantizer 改成 False，
        #   那段 set_no_grad 就不会执行 -> 三组参数 requires_grad=True
        #   -> 进优化器、进权重文件。
        #
        # 解冻后 quantizer 不再被 .eval()（forward 里的 if 分支不执行）
        # -> 码本按 EMA 自动适配新语料（core_vq.py:246 `if self.training:`）
        #
        # 未设环境变量时行为**完全不变**（零风险）。
        import os as _zfy_os
        if _zfy_os.environ.get("ZFY_UNFREEZE", "").strip() in ("1", "true", "yes", "on"):
            if self.freeze_quantizer:
                print("[ZFY] ZFY_UNFREEZE=1 -> 强制 freeze_quantizer=False "
                      "（enc_p/ssl_proj/quantizer 参与训练并进入权重）")
            self.freeze_quantizer = False
        # ---- /ZFY_UNFREEZE_PATCH ----

        inter_channels2 = 512"""

    if MARK in region:
        print(f"  已打过补丁（SynthesizerTrnV3）: {p.name}")
        return 0

    if OLD not in region:
        print("  !! 在 SynthesizerTrnV3 里找不到目标片段")
        print("     （该类的 __init__ 结构可能已变）")
        return 1

    new_region = region.replace(OLD, NEW, 1)
    src = src[:cls_start] + new_region + src[cls_end:]

    if check:
        print("  [1/1] 将在 **SynthesizerTrnV3** 里插入 ZFY_UNFREEZE 开关")
        return 0
    if not bak.exists():
        # ⚠️ 是 copy2(p, bak)（从源文件复制到备份），
        #    我第一版写成了 copy2(bak, bak) -> FileNotFoundError
        shutil.copy2(p, bak)
        print(f"  备份: {bak.name}")
    p.write_text(src, encoding="utf-8")
    print("  [1/1] 已在 **SynthesizerTrnV3** 里插入 ZFY_UNFREEZE 开关")
    return 0


def make_unfreeze_config(root: Path, check: bool) -> int:
    """生成一份 freeze_quantizer=false 的训练配置（作为环境变量的备份方案）。"""
    gsv = root / "GPT_SoVITS"
    src = gsv / "tmp_full_e32.json"
    if not src.exists():
        # 找任意 tmp_full_e*.json
        cands = sorted(root.glob("tmp_full_e*.json"))
        if not cands:
            print("  （找不到 tmp_full_e*.json，跳过配置生成）")
            return 0
        src = cands[-1]

    cfg = json.loads(src.read_text(encoding="utf-8"))
    epochs = cfg.get("train", {}).get("epochs", 32)
    out = root / f"tmp_full_e{epochs}_nofreeze.json"

    cfg["model"]["freeze_quantizer"] = False
    cfg["train"]["epochs"] = epochs
    tag = f"v5_full_e{epochs}_nofreeze"
    cfg["name"] = tag
    cfg["save_weight_dir"] = f"SoVITS_weights_{tag}"
    cfg["data"]["exp_dir"] = cfg["s2_ckpt_dir"] = str(root / "logs" / tag)
    # ⚠️ 不能复用旧 exp_dir，否则会 resume 到 freeze 版本的 checkpoint
    cfg["pretrained_s2G"] = str(
        gsv / "pretrained_models" / "gsv-v5-pretrained" / "s2Gv5dev.pth")

    if check:
        print(f"  [2/2] 将写出 {out.name}（freeze_quantizer=false, "
              f"{epochs} epochs）")
        return 0

    out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"  [2/2] 已写出 {out.name}")
    print(f"        输出权重目录: {cfg['save_weight_dir']}/")
    print(f"        实验目录    : logs/{tag}/")

    # ---- [3/3] 建 exp_dir + 特征软链 + 权重目录 ----
    #
    # ⚠️ 这一步**不能省**！
    # 踩过的坑：只写配置、不建软链 -> 新 exp_dir 是空目录 ->
    # 训练脚本找不到 2-name2text.txt / 3-bert/ 等 -> 直接崩。
    # （run_sovits_e32.sh 里做了这件事，但生成 nofreeze 配置时我漏了。）
    exp = root / "logs" / tag
    wdir = root / cfg["save_weight_dir"]
    exp.mkdir(parents=True, exist_ok=True)
    wdir.mkdir(parents=True, exist_ok=True)

    feat = root / "logs" / "zfy_v5"
    linked = 0
    for n in ("2-name2text.txt", "3-bert", "4-cnhubert", "5-wav32k",
              "6-name2semantic.tsv"):
        s, d = feat / n, exp / n
        if not s.exists():
            print(f"        !! 源特征不存在: {s}")
            continue
        if d.exists() or d.is_symlink():
            continue
        try:
            d.symlink_to(s)
            linked += 1
        except OSError as e:
            # ⚠️ Windows 建符号链接需要管理员权限或开发者模式
            #    （Linux 云端没这个限制）。
            #    降级方案：写一个 .txt 记录，并提示手动处理。
            print(f"        !! 建软链失败（{type(e).__name__}）: {n}")
            print(f"           Windows 需要管理员权限/开发者模式；"
                  f"Linux 云端不会有此问题")
            print(f"           手动: ln -s {s} {d}")
    print(f"  [3/3] 已建 {exp.relative_to(root)}/ 与 "
          f"{wdir.name}/，软链 {linked} 个特征")

    # 清掉可能存在的旧 checkpoint（否则会 resume 后立刻退出）
    ck = exp / "logs_s2_v5dev_full"
    if ck.exists():
        stale = list(ck.glob("G_*.pth")) + list(ck.glob("D_*.pth"))
        if stale:
            import shutil as _sh
            _sh.rmtree(ck)
            print(f"        已清除 {len(stale)} 个旧 checkpoint")
    ck.mkdir(parents=True, exist_ok=True)

    print()
    print("  ⚠️ 启动时必须带 **PYTHONPATH**，否则报")
    print("     ModuleNotFoundError: No module named 'tools'")
    print("     （data_utils.py:11 `from tools.my_utils import load_audio`，")
    print("       tools/ 在仓库根，脚本在 GPT_SoVITS/ 里）")
    print()
    print("  启动命令：")
    print(f"     cd {root}")
    print(f"     PYTHONPATH=$PWD ZFY_UNFREEZE=1 nohup python -s "
          f"GPT_SoVITS/s2_train_v5_full.py \\")
    print(f"         --config {out.name} > train_nofreeze.log 2>&1 &")
    print()
    print("  成功标志（日志前几行必须有这三条）：")
    print("     [ZFY] ZFY_UNFREEZE=1 -> 强制 freeze_quantizer=False ...")
    print("     loaded pretrained .../s2Gv5dev.pth")
    print("     Train Epoch: 1 [0%]")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    print("=" * 74)
    print(f"解冻 enc_p/ssl_proj/quantizer（真正的全量微调）: {root}")
    print("=" * 74)
    rc1 = patch_models(root, args.check, args.revert)
    rc2 = 0 if args.revert else make_unfreeze_config(root, args.check)
    print()
    print("=" * 74)
    if rc1 == 0 and rc2 == 0 and not args.check and not args.revert:
        print("""
  补丁完成。**默认行为不变**（仍 freeze），需要时用环境变量：

      # 默认（与上游一致，零风险）
      python -s GPT_SoVITS/s2_train_v5_full.py --config tmp_full_e32.json

      # 解冻（真正的全量微调）
      ZFY_UNFREEZE=1 python -s GPT_SoVITS/s2_train_v5_full.py \\
          --config tmp_full_e32_nofreeze.json

  启动后日志里应出现：
      [ZFY] ZFY_UNFREEZE=1 -> 强制 freeze_quantizer=False ...

  训练完**必须验证权重含 enc_p**：
      python _tools/diag_missing_enc_p.py --weights <新权重.pth>
      # 或直接：python _tools/fix_weight_header.py --file <新权重.pth>
      #   （若显示 "已是目标版本" 说明有头；再看键数应 >= 660）

  ⚠️ 两个风险：
     · 码本 EMA 漂移 —— 语料少时可能过拟合。建议先训 **8~16 epoch** 看效果
     · 显存增加 —— 205 GB 完全够

  还原：python _tools/patch_v5_unfreeze.py --root $PWD --revert
""")
    elif rc1 != 0 or rc2 != 0:
        print("  有改动未应用，请检查上面的 !! 行")
    print("=" * 74)
    return rc1 or rc2


if __name__ == "__main__":
    sys.exit(main())
