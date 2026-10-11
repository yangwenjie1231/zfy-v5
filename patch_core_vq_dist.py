# -*- coding: utf-8 -*-
"""修 `core_vq.py` 的分布式守卫缺失 —— 解冻 enc_p 后单卡训练崩溃的根因。

## 症状（云端实测）
    INFO: Train Epoch: 1 [0%]
    INFO: [0.22658592462539673, 0, 9.99875e-05]
      7%|  | 4/54 [01:06<13:26, 16.14s/it]
    ...
    ValueError: Default process group has not been initialized,
                please make sure to call init_process_group.

**跑到第 4 步才崩** —— 不是启动期问题。

## 根因：`init_embed_` 的守卫不完整
`core_vq.py:156-178`：

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.inited:
            return
        if dist.is_available() and dist.is_initialized():   # ← 有守卫
            data = SyncFunction.apply(data)
        if dist.get_rank() == 0:                            # ← 无守卫！
            embed, cluster_size = kmeans(...)
        else:
            embed = torch.empty_like(self.embed)
            cluster_size = torch.empty_like(self.cluster_size)
        dist.broadcast(embed, src=0)                        # ← 无守卫！
        dist.broadcast(cluster_size, src=0)                 # ← 无守卫！
        ...

**上游只在 `dist.is_initialized()` 为真时才安全** ——
但 GPT-SoVITS 的训练脚本是单进程启动（`world_size=1`），
`torch.distributed.is_initialized()` 返回 **False**，
于是 `dist.get_rank()` 直接抛 ValueError。

## 为什么**只有解冻后**才崩（这是关键）
`init_embed_` 的第一行是 `if self.inited: return`。
`self.inited` 是 buffer，**只有在 `init_embed_` 成功跑过一次后才变 True**。

  · **freeze 模式**（上游默认）：`models.py:1340` 调 `self.quantizer.eval()`
    -> `core_vq.py:246 if self.training:` 为 False
    -> **码本永不更新**，`init_embed_` 也永不被触发
    -> 从未暴露这个 bug
  · **解冻模式**（本补丁链）：quantizer 处于 training
    -> 第一次前向就进 `forward` -> `self.init_embed_(x)`（L239）
    -> 触发无守卫的 `dist.get_rank()` -> **崩溃**

**所以这是上游一个潜伏的 bug，被"解冻"这个正确操作暴露出来。**

## 修法
给 L165/170/171 补上 `is_initialized()` 守卫，单进程时走本地 kmeans 路径：

    _zfy_dist_ok = dist.is_available() and dist.is_initialized()
    if _zfy_dist_ok:
        if dist.get_rank() == 0:
            embed, cluster_size = kmeans(...)
        else:
            embed = torch.empty_like(self.embed)
            cluster_size = torch.empty_like(self.cluster_size)
        dist.broadcast(embed, src=0)
        dist.broadcast(cluster_size, src=0)
    else:
        # 单进程：直接算，不涉及通信
        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)

**语义等价**：单进程时 rank 必为 0，broadcast 是空操作。
多进程时行为完全不变。

## 用法
    python _tools/patch_core_vq_dist.py --root $PWD           # 打补丁
    python _tools/patch_core_vq_dist.py --root $PWD --check   # 只检查
    python _tools/patch_core_vq_dist.py --root $PWD --revert  # 还原
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MARK = "ZFY_COREVQ_DIST_PATCH"
REL = "GPT_SoVITS/module/core_vq.py"

# ---------- 补丁 1：init_embed_ ----------
OLD = """    @torch.jit.ignore
    def init_embed_(self, data):
        if self.inited:
            return

        if dist.is_available() and dist.is_initialized():
            # [B * T * world_size, D]
            data = SyncFunction.apply(data)

        if dist.get_rank() == 0:
            embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        else:
            embed = torch.empty_like(self.embed)
            cluster_size = torch.empty_like(self.cluster_size)
        dist.broadcast(embed, src=0)
        dist.broadcast(cluster_size, src=0)

        self.embed.data.copy_(embed)
"""

NEW = """    @torch.jit.ignore
    def init_embed_(self, data):
        if self.inited:
            return

        # ---- ZFY_COREVQ_DIST_PATCH (1/2) ----
        # 上游只给 SyncFunction 那行加了 is_initialized() 守卫，
        # 但 dist.get_rank() / dist.broadcast() **没有**守卫。
        # 单进程训练（world_size=1，未初始化）时会直接抛
        #     ValueError: Default process group has not been initialized
        #
        # 为什么只有"解冻 quantizer"后才暴露：
        #   freeze 模式下 models.py:1340 调 quantizer.eval()
        #   -> core_vq.py:246 `if self.training:` 为 False
        #   -> 码本不更新，init_embed_ 永不被触发。
        #   解冻后 quantizer 处于 training，第一次前向就进来了。
        #
        # 单进程时 rank 必为 0、broadcast 是空操作，所以下面这条分支
        # **语义等价**；多进程时行为完全不变。
        _zfy_dist_ok = dist.is_available() and dist.is_initialized()
        if _zfy_dist_ok:
            # [B * T * world_size, D]
            data = SyncFunction.apply(data)
            if dist.get_rank() == 0:
                embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
            else:
                embed = torch.empty_like(self.embed)
                cluster_size = torch.empty_like(self.cluster_size)
            dist.broadcast(embed, src=0)
            dist.broadcast(cluster_size, src=0)
        else:
            # 单进程：本地 kmeans，无需通信
            embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        # ---- /ZFY_COREVQ_DIST_PATCH (1/2) ----

        self.embed.data.copy_(embed)
"""

# ---------- 补丁 2：expire_codes_ ----------
OLD2 = """        if is_distributed():
            # [B * T * world_size, D]
            batch_samples = SyncFunction.apply(batch_samples)

        if dist.get_rank() == 0:
            new_embeds = sample_vectors(batch_samples, expired_codes.sum())
        else:
            new_embeds = torch.zeros(expired_codes.sum(), self.embed.size(1), device=self.embed.device)
        dist.broadcast(new_embeds, src=0)
        self.embed.data[expired_codes] = new_embeds
        broadcast_tensors(self.buffers())
"""

NEW2 = """        # ---- ZFY_COREVQ_DIST_PATCH (2/2) ----
        # 与 init_embed_ 同样的守卫缺失。这一处是**实际崩的地方**：
        #   core_vq.py:284 forward -> expire_codes_(x) -> L214 dist.get_rank()
        # 由完整 traceback 定位（diag_process_group2.py）。
        _zfy_dist_ok2 = dist.is_available() and dist.is_initialized()
        if _zfy_dist_ok2:
            if is_distributed():
                # [B * T * world_size, D]
                batch_samples = SyncFunction.apply(batch_samples)
            if dist.get_rank() == 0:
                new_embeds = sample_vectors(batch_samples, expired_codes.sum())
            else:
                new_embeds = torch.zeros(expired_codes.sum(), self.embed.size(1), device=self.embed.device)
            dist.broadcast(new_embeds, src=0)
            self.embed.data[expired_codes] = new_embeds
            broadcast_tensors(self.buffers())
        else:
            # 单进程：本地采样替换死码字，无需通信
            new_embeds = sample_vectors(batch_samples, expired_codes.sum())
            self.embed.data[expired_codes] = new_embeds
        # ---- /ZFY_COREVQ_DIST_PATCH (2/2) ----
"""


def patch(root: Path, check: bool, revert: bool) -> int:
    p = root / REL
    if not p.exists():
        print(f"  !! 找不到 {p}")
        return 1
    src = p.read_text(encoding="utf-8", errors="replace")
    bak = p.with_suffix(".py.bak_zfy_corevq")

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

    missing = []
    if OLD not in src:
        missing.append("init_embed_")
    if OLD2 not in src:
        missing.append("expire_codes_")
    if missing:
        print(f"  !! 找不到目标片段: {missing}")
        print("     （core_vq.py 可能已被其他补丁改过）")
        return 1

    if check:
        print("  [1/2] init_embed_     : 补 dist.get_rank / dist.broadcast 守卫")
        print("  [2/2] expire_codes_   : 补 dist.get_rank / dist.broadcast 守卫")
        return 0

    src = src.replace(OLD, NEW, 1)
    src = src.replace(OLD2, NEW2, 1)
    if not bak.exists():
        shutil.copy2(p, bak)
        print(f"  备份: {bak.name}")
    p.write_text(src, encoding="utf-8")
    print("  [1/2] init_embed_   已补守卫（单进程走本地 kmeans）")
    print("  [2/2] expire_codes_ 已补守卫（单进程走本地采样）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    print("=" * 74)
    print(f"修 core_vq.py 的分布式守卫: {root}")
    print("=" * 74)
    rc = patch(root, args.check, args.revert)
    print()
    if rc == 0 and not args.check and not args.revert:
        print("""
  补丁完成。

  ## 为什么这个补丁是**必须的**
  解冻 enc_p/ssl_proj/quantizer 后，quantizer 处于 training 模式，
  第一次前向就会进 `core_vq.py:239 self.init_embed_(x)`，
  而 `init_embed_` 里 `dist.get_rank()` 无守卫 -> 单进程直接崩。

  这与 `patch_v5_unfreeze.py` 是**一对**：
      unfreeze 让 quantizer 训练 -> 暴露 core_vq 的守卫缺失
      core_vq 补丁修好守卫 -> 解冻才能跑起来

  ## 验证（本地）
      python _tools/diag_process_group.py
      预期：调 quantizer(x) 成功，codes.shape 正常

  ## 云端启动
      cd /mnt/workspace/zfy-v5-fullft-pkg
      PYTHONPATH=$PWD ZFY_UNFREEZE=1 nohup python -s \\
          GPT_SoVITS/s2_train_v5_full.py \\
          --config tmp_full_e32_nofreeze.json > train_nofreeze.log 2>&1 &
      sleep 120
      grep -cE "Default process group|load audio or ssl error" train_nofreeze.log
      # 必须是 0
""")
    elif rc != 0:
        print("  有改动未应用，见上面的 !! 行")
    print("=" * 74)
    return rc


if __name__ == "__main__":
    sys.exit(main())
