# -*- coding: utf-8 -*-
"""生成 v5 全量微调脚本（s2_train_v5_full.py）。

## 为什么要这个脚本

官方只给 v5 提供 LoRA 训练（`s2_train_v3_lora.py` 里
`net_g.cfm = get_peft_model(net_g.cfm, lora_config)` 是硬编码的）。
而 v2/v2Pro 用的是 `s2_train.py`（全量），但它 **import 的是
SynthesizerTrn，全文没有 models_v5** → 不认识 v5。

实测确认：
  · `SynthesizerTrnV3`（在 models.py L1227）**就是** v5 的完整模型
    （L1306-1308：version in {v5dev,v5turbo} → self.cfm = CFMV5(...)）
  · 构造成功，总参数 381.51M，cfm 335.79M
  · LoRA r128 只训练 23.07M（6.05%）→ 这是 v5 音色不如 v2Pro 的根因

所以本脚本的做法：**以 s2_train_v3_lora.py 为基底，去掉 peft**，
用全量参数训练。同时对 ROCm 做兼容处理。

## 对 ROCm 的改动（3 处）

  a. `torch.backends.cuda.matmul.allow_tf32` → try/except
     （ROCm 上没有这个属性，会 AttributeError）
  b. `torch.cuda.amp` → `torch.amp`（新路径，CUDA/ROCm 通用）
  c. `torch.cuda.is_available()` 在 ROCm 上返回 True（HIP 伪装），
     所以 .cuda() 调用可用，无需改

用法：
    python make_v5_fullft_script.py            # 生成到 GPT-SoVITS/GPT_SoVITS/
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GSV = Path(r"D:\A\zfy-v5\GPT-SoVITS")
SRC = GSV / "GPT_SoVITS" / "s2_train_v3_lora.py"
DST = GSV / "GPT_SoVITS" / "s2_train_v5_full.py"


def main() -> int:
    if not SRC.exists():
        print(f"源文件不存在: {SRC}")
        return 1
    src = SRC.read_text(encoding="utf-8")
    orig = src

    # ---- 1. 去掉 peft 依赖 ----
    src = src.replace(
        "from peft import LoraConfig, get_peft_model",
        "# [v5-fullft] 全量微调：不使用 peft\n"
        "# from peft import LoraConfig, get_peft_model")

    # ---- 2. 去掉 get_peft_model 调用（两处）----
    src = re.sub(
        r"^(\s*)net_g\.cfm = get_peft_model\(net_g\.cfm, lora_config\)",
        r"\1# [v5-fullft] 全量微调：不套 LoRA，训练整个 cfm\n"
        r"\1pass",
        src, flags=re.M)

    # ---- 3. 整块注释掉 LoraConfig 定义（逐行加 #，保持缩进合法）----
    def _comment_block(text: str, start_pat: str) -> str:
        lines = text.split("\n")
        out, i = [], 0
        while i < len(lines):
            if re.match(start_pat, lines[i]):
                indent = len(lines[i]) - len(lines[i].lstrip())
                # 从这一行开始，直到遇到缩进 <= 起始缩进的非空行
                j = i
                while j < len(lines):
                    ln = lines[j]
                    if j > i and ln.strip():
                        cur = len(ln) - len(ln.lstrip())
                        if cur <= indent and not ln.lstrip().startswith(")"):
                            break
                    out.append((" " * indent) + "# [v5-fullft] " + ln.strip()
                               if ln.strip() else ln)
                    if ln.strip() == ")" or ln.rstrip().endswith(")"):
                        j += 1
                        break
                    j += 1
                out.append(" " * indent + "lora_config = None")
                i = j
                continue
            out.append(lines[i])
            i += 1
        return "\n".join(out)

    src = _comment_block(src, r"\s*lora_config = LoraConfig\(")

    # ---- 4. ROCm: allow_tf32 加保护 ----
    src = src.replace(
        "torch.backends.cuda.matmul.allow_tf32 = True\n"
        "torch.backends.cudnn.allow_tf32 = True",
        "# [v5-fullft] ROCm 上没有 cuda.matmul / cudnn，加保护\n"
        "try:\n"
        "    torch.backends.cuda.matmul.allow_tf32 = True\n"
        "    torch.backends.cudnn.allow_tf32 = True\n"
        "except (AttributeError, RuntimeError):\n"
        "    print('[v5-fullft] 跳过 allow_tf32（ROCm/非 CUDA 后端）')")

    # ---- 4b. ROCm: 分布式后端 nccl -> gloo ----
    # 社区反馈：GPT-SoVITS 默认用 NCCL，AMD 平台会直接报错
    # （ncclInternalError）。gloo 跨平台，单卡/小规模完全够用。
    # 原代码是 `backend="gloo" if os.name == "nt" else "nccl"`，
    # 这里改成**检测 ROCm** 而不是只认 Windows。
    src = src.replace(
        'backend="gloo" if os.name == "nt" else "nccl",',
        '# [v5-fullft] ROCm 不支持 NCCL：检测到 ROCm 就用 gloo\n'
        'backend=("gloo" if (os.name == "nt" or getattr(torch.version, "hip", None))\n'
        '                 else "nccl"),')
    # cudnn 相关开关在 ROCm 上也不存在
    src = src.replace(
        "torch.backends.cudnn.benchmark = False\n"
        "torch.backends.cudnn.deterministic = False",
        "# [v5-fullft] ROCm 无 cudnn，加保护\n"
        "try:\n"
        "    torch.backends.cudnn.benchmark = False\n"
        "    torch.backends.cudnn.deterministic = False\n"
        "except (AttributeError, RuntimeError):\n"
        "    pass")
    # set_float32_matmul_precision 在 ROCm 上也应可调用，但加保护更稳
    src = src.replace(
        'torch.set_float32_matmul_precision("medium")',
        '# [v5-fullft] 加保护（部分后端无此 API）\n'
        'try:\n'
        '    torch.set_float32_matmul_precision("medium")\n'
        'except (AttributeError, RuntimeError):\n'
        '    pass')

    # ---- 5. torch.cuda.amp -> torch.amp ----
    src = src.replace(
        "from torch.cuda.amp import GradScaler, autocast",
        "# [v5-fullft] 用 torch.amp（CUDA/ROCm 通用）\n"
        "try:\n"
        "    from torch.amp import GradScaler, autocast\n"
        "except ImportError:\n"
        "    from torch.cuda.amp import GradScaler, autocast")
    # GradScaler 在新 API 需要 device 参数（保留原缩进）
    src = re.sub(
        r"^(\s*)scaler = GradScaler\(enabled=hps\.train\.fp16_run\)",
        r"\1try:\n"
        r"\1    scaler = GradScaler('cuda', enabled=hps.train.fp16_run)\n"
        r"\1except TypeError:\n"
        r"\1    scaler = GradScaler(enabled=hps.train.fp16_run)",
        src, flags=re.M)
    # autocast 新 API 需要 device_type
    src = src.replace(
        "with autocast(enabled=hps.train.fp16_run):",
        "with autocast(device_type='cuda', enabled=hps.train.fp16_run):")

    # ---- 6. lora_rank 相关：全量时目录名不带 lora_rank ----
    # 注意保留原缩进（4 空格），否则会 IndentationError
    src = re.sub(
        r"^(\s*)save_root = \"%s/logs_s2_%s_lora_%s\" % "
        r"\(hps\.data\.exp_dir, hps\.model\.version, hps\.train\.lora_rank\)",
        r'\1# [v5-fullft] 全量微调：目录不带 lora_rank\n'
        r'\1save_root = "%s/logs_s2_%s_full" % '
        r'(hps.data.exp_dir, hps.model.version)',
        src, flags=re.M)

    # ---- 7. 头部说明 ----
    src = src.replace(
        "# coding=utf-8",
        "# coding=utf-8\n"
        "# ============================================================\n"
        "# [v5-fullft] 由 make_v5_fullft_script.py 自动生成\n"
        "#\n"
        "# 与 s2_train_v3_lora.py 的唯一区别：**去掉 LoRA，全量微调**\n"
        "#   · v2/v2Pro 用的 s2_train.py 不支持 v5（import SynthesizerTrn，\n"
        "#     全文无 models_v5）\n"
        "#   · v5 的模型在 SynthesizerTrnV3 里（models.py L1306-1308）\n"
        "#   · 实测 v5 SoVITS 381.51M 参数，LoRA r128 只训练 23.07M(6.05%)\n"
        "#   · 本机 8GB 显存放不下（仅优化器状态就 6.10GB）→ 需云算力\n"
        "#\n"
        "# ROCm 兼容改动：allow_tf32 加保护、torch.cuda.amp → torch.amp\n"
        "# ============================================================")

    if src == orig:
        print("警告：没有任何替换生效，请检查源文件是否已改过")
        return 1

    DST.write_text(src, encoding="utf-8")

    # 校验
    n_peft_call = len(re.findall(r"^\s*net_g\.cfm = get_peft_model", src, re.M))
    print(f"已生成: {DST}")
    print(f"  文件大小      : {len(src)/1024:.1f} KB")
    print(f"  残留 get_peft_model 调用: {n_peft_call}  (应为 0)")
    print(f"  含 allow_tf32 保护      : {'[v5-fullft] 跳过 allow_tf32' in src}")
    print(f"  含 torch.amp            : {'from torch.amp import' in src}")
    print(f"  目录名去 lora_rank      : {'logs_s2_%s_full' in src}")
    return 0 if n_peft_call == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
