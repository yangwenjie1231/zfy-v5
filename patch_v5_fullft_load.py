# -*- coding: utf-8 -*-
"""修全量微调权重的推理加载 —— `b"07"` 头会让它走 LoRA 分支。

## 症状
加了 `b"07"`（v5dev）版本头后：
    Loading VITS pretrained weights from ..._e32_s1728_l0.pth. <All keys matched successfully>
    !! TTS 构造失败: KeyError: 'lora_rank'

**注意 `<All keys matched successfully>`** —— 权重本身是**完好的**
（681 个键全部匹配，含 enc_p/ssl_proj）。问题纯在**加载路径**。

## 根因：head2version 的第三位是 if_lora_v3
    # process_ckpt.py:77-87
    head2version = {
        b"05": ["v2", "v2Pro",      False],
        b"06": ["v2", "v2ProPlus",  False],
        b"07": ["v2", "v5dev",      True],   # <- True = 走 LoRA 分支！
        b"08": ["v2", "v5turbo",    True],
    }

`TTS.py:581` 用它决定加载方式：

    version, model_version, if_lora_v3 = get_sovits_version_from_path_fast(weights_path)

    if if_lora_v3 == False:                                  # TTS.py:647
        # 全量权重：直接 load_state_dict            <- **我们要走这条**
    else:                                                    # TTS.py:651
        vits_model.load_state_dict(load_sovits_new(path_sovits)['weight'], ...)
        lora_rank = dict_s2["lora_rank"]           # <- KeyError！全量权重没这键

**为什么上游把 v5dev/v5turbo 标成 lora=True**：
官方 v5 只提供 **LoRA** 微调（`s2_train_v3_lora.py` 里硬编码 `get_peft_model`）。
所以 `b"07"`/`b"08"` 的头是"LoRA 版 v5"的意思。

**我们的权重是"全量微调版"**，语义上更接近 `b"05"`（`v2Pro`，`False`）
或 `b"02"`（`v3`，`False`）—— 但版本号必须仍是 **v5dev**，
否则会加载错误的 BigVGAN（v3 是 24kHz）。

## 两处修法（都做了才完整）

### 修 1：权重头写 `b"07"` 但打补丁让加载器忽略 lora 标志
改 `process_ckpt.py` 的 `head2version`，把 `b"07"`/`b"08"` 的第三位改成
`False` **当且仅当**权重里没有 `lora_rank` 键。

### 修 2：给 `TTS.py` 加"缺 lora_rank 就降级走全量路径"的保护
`dict_s2["lora_rank"]` 改成 `dict_s2.get("lora_rank")`，
缺失时走 `if_lora_v3 == False` 那条全量加载路径。

**修 2 是根本性的**（不依赖权重头），修 1 是保险。

## 用法
    python _tools/patch_v5_fullft_load.py --root $PWD           # 打补丁
    python _tools/patch_v5_fullft_load.py --root $PWD --check
    python _tools/patch_v5_fullft_load.py --root $PWD --revert
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MARK = "ZFY_V5_FULLFT_LOAD_PATCH"

# ================================================================
# ⚠️ 我第一版改了两处，结果**破坏了 LoRA 路径**（实测发现的回归）
#
# 第一版修 1：把 head2version 的 b"07"/b"08" 第三位 True 改成 False。
#   意图：全量权重没 lora_rank 键，走 False 分支就不 KeyError 了。
#   实测后果：**LoRA 权重也被改成 False** ->
#       get_sovits_version_from_path_fast -> if_lora_v3=False
#     -> 走 `vits_model.load_state_dict(dict_s2['weight'], strict=False)`
#     -> LoRA 的键（如 `cfm.base_model.model...lora_A`）**静默不生效**
#     -> 等于**根本没加载 LoRA**，但不会报错！
#
# 这是最危险的一类 bug：**静默失效**。所以修 1 撤掉，只保留修 2。
# ================================================================

# 修 2（唯一改动）：以 `lora_rank` 键**是否存在**为准，双向覆盖头里的标志
T_REL = "GPT_SoVITS/TTS_infer_pack/TTS.py"

T_OLD = """        if if_lora_v3 == False:
            print(
                f"Loading VITS weights from {weights_path}. {vits_model.load_state_dict(dict_s2['weight'], strict=False)}"
            )
        else:
            print(
                f"Loading VITS pretrained weights from {weights_path}. {vits_model.load_state_dict(load_sovits_new(path_sovits)['weight'], strict=False)}"
            )
            lora_rank = dict_s2["lora_rank"]
"""

T_NEW = """        # ---- ZFY_V5_FULLFT_LOAD_PATCH ----
        # `if_lora_v3` 来自**权重头**（process_ckpt.py head2version 的第三位）。
        # 上游把 v5 全系列标成 True（b"07" v5dev / b"08" v5turbo），
        # 因为官方 v5 **只**提供 LoRA 微调。
        #
        # 但我们的权重是**全量微调版**（681 键，含 enc_p/ssl_proj）——
        # 没有 `lora_rank` 键 -> 上游 `dict_s2["lora_rank"]` 直接 KeyError：
        #     Loading VITS pretrained weights from ..._e32_...pth.
        #         <All keys matched successfully>      <- 权重完好！
        #     KeyError: 'lora_rank'                    <- 纯加载路径问题
        #
        # ## 修法：以 `lora_rank` 键**是否存在**为唯一判据（双向覆盖）
        #
        # 为什么不只单向降级：
        #   我第一版只做"有 lora_v3 且无 lora_rank -> 降级"，
        #   同时把 head2version 的 b"07"/b"08" 改成 False。
        #   结果 **LoRA 权重也被判成 False** -> 走 strict=False 的普通加载
        #   -> LoRA 的 `lora_A/lora_B` 键**静默不生效**（不报错，但 LoRA 没加载）。
        #
        # 所以必须**双向**对齐：有 `lora_rank` 就走 LoRA，没有就走全量。
        # 这样与头里写什么无关，且**不会破坏 LoRA 权重**。
        _zfy_has_lora = "lora_rank" in dict_s2
        if bool(if_lora_v3) != _zfy_has_lora:
            print(f"[ZFY] {os.path.basename(weights_path)}: "
                  f"头标志 if_lora_v3={if_lora_v3}, "
                  f"权重含 lora_rank={_zfy_has_lora} -> 以权重为准，"
                  f"走{'LoRA' if _zfy_has_lora else '全量'}加载路径")
            if_lora_v3 = _zfy_has_lora

        if if_lora_v3 == False:
            print(
                f"Loading VITS weights from {weights_path}. {vits_model.load_state_dict(dict_s2['weight'], strict=False)}"
            )
        else:
            print(
                f"Loading VITS pretrained weights from {weights_path}. {vits_model.load_state_dict(load_sovits_new(path_sovits)['weight'], strict=False)}"
            )
            lora_rank = dict_s2["lora_rank"]
"""


def patch_file(root: Path, rel: str, old: str, new: str,
               check: bool, revert: bool, tag: str) -> int:
    p = root / rel
    if not p.exists():
        print(f"  !! 找不到 {p}")
        return 1
    src = p.read_text(encoding="utf-8", errors="replace")
    bak = p.with_suffix(".py.bak_zfy_v5load")

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

    if old not in src:
        print(f"  !! {rel} 找不到目标片段（{tag}）")
        return 1

    if check:
        print(f"  [{tag}] 将修改 {rel}")
        return 0

    src = src.replace(old, new, 1)
    if not bak.exists():
        shutil.copy2(p, bak)
        print(f"  备份: {bak.name}")
    p.write_text(src, encoding="utf-8")
    print(f"  [{tag}] {rel} 已应用")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    print("=" * 76)
    print(f"修全量微调权重的推理加载: {root}")
    print("=" * 76)

    # 只改 TTS.py 一处（head2version 不动 —— 改它会破坏 LoRA）
    rc = patch_file(root, T_REL, T_OLD, T_NEW, args.check, args.revert,
                    "1/1 TTS.py")

    print()
    print("=" * 76)
    if rc == 0 and not args.check and not args.revert:
        print("""
  补丁完成（只改 TTS.py 一处）。

  ## 为什么只改一处
  我第一版还改了 `process_ckpt.py` 的 head2version（b"07"/b"08" 第三位
  True -> False）。**实测发现那是回归**：
      LoRA 权重 -> if_lora_v3=False -> 走 strict=False 的普通加载
      -> LoRA 的 lora_A/lora_B 键**静默不生效**（不报错，但 LoRA 没加载）
  所以撤掉，改为在 TTS.py 里**双向对齐**：
      有 lora_rank 键 -> 走 LoRA
      无 lora_rank 键 -> 走全量
  与头里写什么无关，且**不破坏 LoRA**。

  ## 两处必须同时满足（否则会静默错）
    全量权重（681 键，含 enc_p/ssl_proj，无 lora_rank）-> 全量路径 ✓
    LoRA 权重（258 键，有 lora_rank）              -> LoRA 路径  ✓

  ## 验证
      python _tools/verify_v5_fullft_load.py

  ## 然后合成
      python _tools/cloud_synth_sweep.py \\
          --weights-dir SoVITS_weights_v5_full_e32_nofreeze \\
          --epochs 8,16,24,32 --no-v2pro --out out_sweep_nofreeze
""")
    elif rc != 0:
        print("  有改动未应用，见上面的 !! 行")
    print("=" * 76)
    return rc


if __name__ == "__main__":
    sys.exit(main())
