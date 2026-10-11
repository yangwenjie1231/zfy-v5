# -*- coding: utf-8 -*-
"""生成可直接在魔搭 Notebook 运行的 .ipynb。

设计原则（基于已实测的云环境）：
  · 云上 torch 2.12.0+rocm7.2 已装好，**不要重装 torch**
  · torch.cuda.amp 在 2.12 **仍存在**（实测），补丁非必需但无害
  · **transformers 是 5.15.1，而 GPT-SoVITS 要求 >=4.51,<5** -> 必须降级
  · peft 是 0.20.0，要求 <0.18.0 -> 必须降级
  · 缺 12 个依赖 -> 必须装
  · numpy 2.3.5 可用（不要按 requirements 降到 <2.0）

产出：D:\\A\\zfy-v5\\zfy_v5_fullft.ipynb
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(r"D:\A\zfy-v5\zfy_v5_fullft.ipynb")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").split("\n")}


def code(text: str) -> dict:
    src = text.strip("\n").split("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": [l + "\n" for l in src[:-1]] + [src[-1]]}


cells: list[dict] = []

# ---------------------------------------------------------------- 0
cells.append(md(r"""
# 庄方宜 v5 全量微调（魔搭 Notebook · ROCm）

**目标**：在 GPT-SoVITS v5 上做**全量微调**（而非官方只支持的 LoRA），
以补上音色短板，同时保住 v5 在语气/韵律上的优势。

## 背景（已实测确认）

| 项 | 值 |
|---|---|
| 数据 | 825 条 / 63.8 分钟（官方逐字文本），特征已在本机算好 |
| v5 SoVITS 参数量 | **381.51 M**（v2Pro 仅 42.39 M） |
| 全量微调显存需求 | 约 6.10 GB（优化器状态），**本机 8GB 放不下** |
| 本机 LoRA r32 效果 | SIM 0.8236；v2Pro 全量 0.8888 → 差 0.065 |
| 本机 LoRA r128 效果 | 反而更差（0.809~0.821）→ **秩不是瓶颈** |

## 环境事实（你已实测）

```
Python 3.12.13 | torch 2.12.0+rocm7.2 (gfx942) | 显存 205.8 GB
torch.cuda.amp 仍存在 ✅   transformers 5.15.1 ⚠️(需<5)   peft 0.20.0 ⚠️(需<0.18)
```

## 本 notebook 流程

1. 定位/拉取仓库
2. 检查环境
3. **装依赖**（关键：降级 transformers 与 peft）
4. 自检（能否 import v5 模型）
5. 打 torch2.12 兼容补丁（无害，幂等）
6. **冒烟测试**（20 条，先验证链路）
7. 全量训练
8. 保存权重

> 每个代码格都可独立重跑。建议**按顺序**执行。
"""))

# ---------------------------------------------------------------- 1
cells.append(md("## 1. 配置与定位仓库"))
cells.append(code(r'''
import os, sys, subprocess, json, shutil
from pathlib import Path

# ============ 配置 ============
REPO_DIR = Path("/mnt/workspace/zfy-v5-fullft-pkg")   # clone 后的检出根
DATASET_ID = "yangwenjie1231/zfy-v5-fullft-pkg"       # 私有数据集
TOKEN = ""            # 若需重新 clone，填入你的魔搭 token；已 clone 则留空

# ============ 定位检出根 ============
def find_root():
    """找含 config.py + GPT_SoVITS/ 的那一层（检出根）。"""
    cands = [REPO_DIR, Path("/mnt/workspace/zfy-v5-fullft-pkg"),
             Path("/mnt/workspace/zfy-v5"), Path("/mnt/workspace")]
    here = Path.cwd()
    cands += [here, *here.parents]
    for c in cands:
        if (c / "config.py").exists() and (c / "GPT_SoVITS").is_dir():
            return c.resolve()
    return None

ROOT = find_root()

if ROOT is None:
    print("未找到检出根，开始 clone …")
    if not TOKEN:
        raise SystemExit("请在上方 TOKEN 填入魔搭 token，或先手动 clone")
    REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://oauth2:{TOKEN}@www.modelscope.cn/datasets/{DATASET_ID}.git"
    r = subprocess.run(["git", "lfs", "install"], capture_output=True, text=True)
    print(r.stdout, r.stderr)
    r = subprocess.run(["git", "clone", url, str(REPO_DIR)],
                       capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-2000:])
    ROOT = find_root()
    if ROOT is None:
        raise SystemExit("clone 后仍未找到检出根，请检查路径")

os.chdir(ROOT)
print(f"✓ 检出根: {ROOT}")
print()
print("目录内容:")
for p in sorted(ROOT.iterdir()):
    if p.name.startswith("."):
        continue
    kind = "dir " if p.is_dir() else "file"
    print(f"  [{kind}] {p.name}")
'''))

# ---------------------------------------------------------------- 2
cells.append(md("## 2. 检查环境（torch / ROCm / 关键 API）"))
cells.append(code(r'''
import platform, sys, os
import torch

print(f"Python      : {sys.version.split()[0]}")
print(f"torch       : {torch.__version__}")
print(f"torch.hip   : {getattr(torch.version, 'hip', None)}")
print(f"设备可用    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"设备名      : {torch.cuda.get_device_name(0)}")
    print(f"显存        : {p.total_memory/1e9:.1f} GB")
    print(f"gcnArch     : {getattr(p, 'gcnArchName', '?')}")

print()
print("关键 API:")
for name, expr in [
    ("torch.amp.autocast",        "from torch.amp import autocast"),
    ("torch.cuda.amp.autocast",   "from torch.cuda.amp import autocast"),
    ("torch.amp.GradScaler",      "from torch.amp import GradScaler"),
    ("allow_tf32",                "x=torch.backends.cuda.matmul.allow_tf32"),
    ("cudnn.benchmark",           "x=torch.backends.cudnn.benchmark"),
]:
    try:
        exec(expr, {"torch": torch})
        print(f"  OK   {name}")
    except Exception as e:
        print(f"  FAIL {name}  {type(e).__name__}: {str(e)[:70]}")
'''))

# ---------------------------------------------------------------- 3
cells.append(md(r"""
## 3. 装依赖 ⚠️ 最容易踩坑的一步

### 核心原则：**不要按用途裁剪依赖清单**

我踩过四次，全是同一个根因 —— 按"当前要做什么"裁清单：

| 次 | 漏的包 | 报错 |
|---|---|---|
| 1 | `peft` / `pytorch_lightning` | 推理 `ModuleNotFoundError` |
| 2 | `matplotlib` | **GPT 训练**崩（`AR/modules/lr_schedulers.py:6`） |
| 2 | `gradio` | **SoVITS 重训**崩（`tools/my_utils.py:7`） |
| 3 | `fast_langdetect` / `split_lang` | 推理崩（相对导入链没追踪到） |

所以本格**直接调用仓库里的脚本**，不再在 notebook 里重复一份清单：

- `_tools/install_all_deps.sh` —— 三条链路一次装全
- `_tools/fix_deps_patch.sh` —— 装后验证 + 强制修复

### ⚠️ 一个必须知道的版本互斥

| 包 | 要求 `huggingface_hub` | 不满足时 |
|---|---|---|
| **transformers 5.x** | **>= 1.x** | `cannot import name 'is_offline_mode'` |
| **gradio 4.44** | **< 0.37**（它 `import HfFolder`） | `cannot import name 'HfFolder'` |

**同一个 hf-hub 版本不可能同时满足两者** —— 所以必须把
`transformers` 钉到 `<5`，而不是去升 `huggingface_hub`。

已验证可用的组合（本机三条链路都能 import）：

```
transformers 4.57.6  +  huggingface_hub 0.36.2  +  gradio 4.44.1
```

> 注：**SoVITS 训练本身不需要 transformers**，只有推理和 GPT 训练需要。
> 所以即使 transformers 没修好，也可以先跑 SoVITS 训练。

**下面这格是 Python 代码**（内部用 `subprocess` 调 shell），直接运行。
"""))
cells.append(code(r'''
import os, subprocess, sys
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "GPT_SoVITS").is_dir():
    for c in (Path("/mnt/workspace/zfy-v5-fullft-pkg"), Path("/mnt/workspace")):
        if (c / "GPT_SoVITS").is_dir():
            ROOT = c
            break
os.chdir(ROOT)


def sh(script: Path, tail: int = 60, timeout: int = 3600):
    """跑仓库里的 shell 脚本，过滤 pip 的 root 警告。"""
    if not script.exists():
        print(f"  !! 找不到 {script}（需要先 git pull）")
        return 1
    print(f"\n{'=' * 62}\n$ bash {script.relative_to(ROOT)}\n{'=' * 62}")
    r = subprocess.run(["bash", str(script)], cwd=str(ROOT),
                       capture_output=True, text=True, timeout=timeout)
    out = (r.stdout or "") + (r.stderr or "")
    lines = [l for l in out.split("\n")
             if not l.startswith("WARNING: Running pip as the 'root'")]
    print("\n".join(lines[-tail:]))
    return r.returncode


# ---- 1) 三条链路一次装全 ----
sh(ROOT / "_tools" / "install_all_deps.sh", tail=45)

# ---- 2) 装后验证 + 强制修复（处理 hf-hub 被顶回新版）----
sh(ROOT / "_tools" / "fix_deps_patch.sh", tail=45)

print("""
============================================================
若上面出现 "三条链路全部可 import"，即可继续下一格。
若仍有 FAIL，把那几行贴出来。
============================================================
""")
'''))

cells.append(md(r"""
### 3b. 重启内核

装完依赖后 **必须重启 Notebook 内核**（菜单：Kernel → Restart），
否则当前进程里还是旧模块。

重启后**不用重跑** 3a，直接从下面继续。
"""))


# ---------------------------------------------------------------- 4
cells.append(md("## 4. 自检：能否 import v5 模型"))
cells.append(code(r'''
import os, sys
from pathlib import Path

ROOT = Path.cwd()
# 若上一格 chdir 过，这里重新定位
if not (ROOT / "GPT_SoVITS").is_dir():
    for c in [Path("/mnt/workspace/zfy-v5-fullft-pkg"), Path("/mnt/workspace")]:
        if (c / "GPT_SoVITS").is_dir():
            ROOT = c
            break
os.chdir(ROOT)

# sys.path 布局（顺序重要：检出根必须在最前，否则 f5_tts 命名空间包会解析错）
NEED = [str(ROOT), str(ROOT / "GPT_SoVITS"),
        str(ROOT / "GPT_SoVITS" / "BigVGAN"),
        str(ROOT / "tools"), str(ROOT / "tools" / "asr"),
        str(ROOT / "tools" / "uvr5")]
for p in reversed(NEED):
    while p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

print(f"检出根: {ROOT}\n")

# 1) f5_tts.DiT 是否解析到 v5 版本
import inspect
try:
    from f5_tts.model import DiT
    sig = list(inspect.signature(DiT.__init__).parameters)
    ok = "use_step_embedding" in sig
    print(f"{'✓' if ok else '✗'} f5_tts.DiT")
    print(f"    {inspect.getfile(DiT)}")
    print(f"    含 use_step_embedding: {ok}")
    if not ok:
        print("    !! 解析到旧版 DiT -> 检查 sys.path 顺序")
except Exception as e:
    print(f"✗ f5_tts 导入失败: {type(e).__name__}: {e}")

# 2) models 能否导入
try:
    from module.models import SynthesizerTrnV3
    print(f"✓ module.models.SynthesizerTrnV3")
except Exception as e:
    print(f"✗ models 导入失败: {type(e).__name__}: {str(e)[:200]}")

# 3) 能否构造 v5 模型（最关键的集成测试）
try:
    import json, torch
    from module.models import SynthesizerTrnV3 as S
    cfg = json.load(open(ROOT / "GPT_SoVITS" / "configs" / "s2.json",
                         encoding="utf-8"))
    kw = dict(cfg["model"]); kw["version"] = "v5dev"
    net = S(cfg["data"]["filter_length"] // 2 + 1,
            cfg["train"]["segment_size"] // cfg["data"]["hop_length"],
            n_speakers=cfg["data"]["n_speakers"], **kw)
    n = sum(p.numel() for p in net.parameters())
    from module.models_v5 import CFMV5
    print(f"✓ 构造 v5 模型成功: {n/1e6:.2f} M 参数  "
          f"(cfm={type(net.cfm).__name__}, 是CFMV5={isinstance(net.cfm, CFMV5)})")
    del net
    import gc; gc.collect()
    torch.cuda.empty_cache()
except Exception as e:
    print(f"✗ v5 模型构造失败: {type(e).__name__}: {str(e)[:300]}")
'''))

# ---------------------------------------------------------------- 5
cells.append(md(r"""
## 5. 数据检查

训练读取的是 `logs/zfy_v5/`（**特征已在本机算好，无需重算**）：

| 目录/文件 | 内容 |
|---|---|
| `2-name2text.txt` | 音素序列（825 行） |
| `3-bert/` | BERT 特征（825 个） |
| `4-cnhubert/` | HuBERT 特征（825 个） |
| `5-wav32k/` | 32kHz wav（825 个，**训练直接读它**） |
| `6-name2semantic.tsv` | 语义 token（825 行） |

> ⚠️ 若这些文件只有几百字节，说明 **LFS 没拉取**，需要 `git lfs pull`。
"""))
cells.append(code(r'''
from pathlib import Path
import os

ROOT = Path.cwd()
if not (ROOT / "GPT_SoVITS").is_dir():
    ROOT = Path("/mnt/workspace/zfy-v5-fullft-pkg")
OPT = ROOT / "logs" / "zfy_v5"

print(f"特征目录: {OPT}\n")

if not OPT.exists():
    print("✗ 目录不存在")
else:
    checks = [
        ("2-name2text.txt", "file"),
        ("3-bert", "dir"),
        ("4-cnhubert", "dir"),
        ("5-wav32k", "dir"),
        ("6-name2semantic.tsv", "file"),
    ]
    bad = []
    for name, kind in checks:
        p = OPT / name
        if not p.exists():
            print(f"  ✗ {name} 不存在")
            bad.append(name)
            continue
        if kind == "dir":
            fs = list(p.iterdir())
            size = sum(f.stat().st_size for f in fs if f.is_file())
            print(f"  ✓ {name:24s} {len(fs):4d} 个  {size/1e6:8.1f} MB")
            if size < 1e6 and len(fs) > 0:
                print(f"      !! 体积异常小，可能是 LFS 指针未拉取")
                bad.append(name)
        else:
            sz = p.stat().st_size
            print(f"  ✓ {name:24s} {sz/1024:8.1f} KB")
            if sz < 1024:
                print(f"      !! 体积异常小，可能是 LFS 指针未拉取")
                bad.append(name)

    print()
    if bad:
        print(f"有问题的项: {bad}")
        print("解决: 在检出根执行 `git lfs install --local && git lfs pull`")
    else:
        print("✓ 数据完整")

    # list 文件
    lst = ROOT / "data" / "zfy_v5" / "zfy_v5.list"
    print()
    if lst.exists():
        lines = lst.read_text(encoding="utf-8").strip().split("\n")
        print(f"✓ list: {lst}  {len(lines)} 行")
        print(f"  首行: {lines[0][:100]}")
    else:
        print(f"✗ 缺 list: {lst}")
'''))

# ---------------------------------------------------------------- 6
cells.append(md(r"""
## 6. 打 torch2.12 兼容补丁（可选，幂等）

你实测 `torch.cuda.amp` 在 2.12 **仍存在**，所以补丁**非必需**。
但它无害（只是优先走新 API），且能让代码对未来版本更稳。

补丁对象：`GPT_SoVITS/module/models.py:23` 的硬 import
"""))
cells.append(code(r'''
import subprocess, sys, os
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "GPT_SoVITS").is_dir():
    ROOT = Path("/mnt/workspace/zfy-v5-fullft-pkg")
os.chdir(ROOT)

script = ROOT / "_tools" / "patch_models_for_torch212.py"
if script.exists():
    r = subprocess.run([sys.executable, str(script), "--root", str(ROOT)],
                       capture_output=True, text=True)
    print(r.stdout[-3000:])
    if r.returncode != 0:
        print(r.stderr[-2000:])
else:
    # 仓库里没有就内联打补丁
    print("未找到补丁脚本，内联处理…")
    p = ROOT / "GPT_SoVITS" / "module" / "models.py"
    src = p.read_text(encoding="utf-8")
    OLD = "from torch.cuda.amp import autocast"
    MARK = "# [torch212-patch]"
    if MARK in src:
        print("  已打过补丁，跳过")
    elif OLD in src:
        # 用列表拼装，避免嵌套三引号
        lines = [
            MARK + " torch 2.12+ 兼容",
            "try:",
            "    from torch.amp import autocast as _ac_new",
            "",
            "    def autocast(enabled=True, **kw):",
            '        return _ac_new(device_type="cuda", enabled=enabled, **kw)',
            "except ImportError:",
            "    from torch.cuda.amp import autocast",
        ]
        NEW = chr(10).join(lines)
        (p.parent / (p.name + ".bak")).write_text(src, encoding="utf-8")
        p.write_text(src.replace(OLD, NEW, 1), encoding="utf-8")
        print("  已打补丁，备份 -> " + p.name + ".bak")
    else:
        print("  未找到目标 import，可能版本不同")
'''))

# ---------------------------------------------------------------- 7
cells.append(md(r"""
## 7. 冒烟测试（20 条 · 先验证链路）

**不要一上来就跑全量。** 先用 20 条确认链路通畅。

### 关键简化：不必重算特征

我查了数据加载器 `TextAudioSpeakerLoaderV4.__init__`，
训练**只读 3 个东西**：

| 文件 | 用途 |
|---|---|
| `logs/zfy_v5/2-name2text.txt` | 音素序列 |
| `logs/zfy_v5/4-cnhubert/` | HuBERT 特征（825 个 .pt） |
| `logs/zfy_v5/5-wav32k/` | 32kHz 音频（825 个 wav） |

**不读** `3-bert/` 和 `6-name2semantic.tsv`（那是 `s1_train` 用的）。

所以冒烟测试可以直接**从全量数据里抽样**，只需：
1. 建一个只含 20 行的 `2-name2text.txt`
2. 把对应的 20 个 `.pt` 和 `.wav` 软链过去

**省掉重算特征（原本要跑 1-get-text / 2-get-hubert / 3-get-semantic 三步）**。

### 要确认的三件事

| 检查 | 期望 | 若失败 |
|---|---|---|
| 不报 `ncclInternalError` | 用 gloo | 脚本已自动检测 ROCm |
| 不报 `hipFFT` 相关错 | — | 把 `fp16_run` 设 `False` |
| 能存出权重 | `SoVITS_weights_smoke/*.pth` | 看日志定位 |
"""))

cells.append(code(r'''
import os, json, random, shutil
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "GPT_SoVITS").is_dir():
    ROOT = Path("/mnt/workspace/zfy-v5-fullft-pkg")
os.chdir(ROOT)

FULL = ROOT / "logs" / "zfy_v5"
SMOKE = ROOT / "logs" / "smoke"
N = 20

# 清理旧的冒烟产物
if SMOKE.exists():
    shutil.rmtree(SMOKE)
SMOKE.mkdir(parents=True)

# ---- 读全量 name2text，抽 20 条 ----
src_txt = FULL / "2-name2text.txt"
lines = [l for l in src_txt.read_text(encoding="utf-8").strip().split("\n")
         if l.strip()]
print(f"全量 name2text: {len(lines)} 行")

random.seed(0)
picked = random.sample(lines, min(N, len(lines)))

# ---- 只保留同时有 .pt 和 .wav 的条目 ----
hubert = FULL / "4-cnhubert"
wavs = FULL / "5-wav32k"
valid = []
for l in picked:
    parts = l.split("\t")
    if len(parts) != 4:
        continue
    name = parts[0]
    if (hubert / f"{name}.pt").exists() and (wavs / name).exists():
        valid.append(l)
print(f"抽中且特征齐全: {len(valid)} 条")

if len(valid) < 2:
    raise SystemExit("有效样本太少，检查 logs/zfy_v5/")

# ---- 写 smoke 的 name2text ----
(SMOKE / "2-name2text.txt").write_text("\n".join(valid) + "\n",
                                       encoding="utf-8")

# ---- 链接 .pt 和 .wav（优先软链，失败则回退硬链/复制）----
# 为什么优先软链：825 条特征有 700MB+，软链不占空间、瞬间完成。
# 为什么要回退：Windows 上创建软链需要管理员权限（WinError 1314），
#               云上 Linux 无此限制，但加回退更稳。
def link_or_copy(src: Path, dst: Path):
    try:
        dst.symlink_to(src)
        return "symlink"
    except (OSError, NotImplementedError):
        pass
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


for sub, ext in (("4-cnhubert", ".pt"), ("5-wav32k", "")):
    d = SMOKE / sub
    d.mkdir()
    kinds = {}
    for l in valid:
        name = l.split("\t")[0]
        src = FULL / sub / f"{name}{ext}"
        if src.exists():
            k = link_or_copy(src, d / f"{name}{ext}")
            kinds[k] = kinds.get(k, 0) + 1
    print(f"  {sub}: {sum(kinds.values())} 个 {kinds}")

print(f"\n✓ 冒烟数据就绪: {SMOKE}")
print(f"  样本数: {len(valid)}")
'''))

cells.append(code(r'''
import os, sys, subprocess, json
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "GPT_SoVITS").is_dir():
    ROOT = Path("/mnt/workspace/zfy-v5-fullft-pkg")
os.chdir(ROOT)

GSV = ROOT / "GPT_SoVITS"
PRE = GSV / "pretrained_models"

# ---- 冒烟配置 ----
cfg = json.loads((GSV / "configs" / "s2.json").read_text(encoding="utf-8"))
cfg["train"].update({
    "batch_size": 2,
    "epochs": 2,
    "pretrained_s2G": str(PRE / "gsv-v5-pretrained" / "s2Gv5dev.pth"),
    "pretrained_s2D": "",
    "if_save_latest": True,
    "if_save_every_weights": True,
    "save_every_epoch": 1,
    "gpu_numbers": "0",
    "grad_ckpt": True,
    "lora_rank": 0,
    "fp16_run": True,       # 若报 hipFFT 错，改 False 重跑本格
})
cfg["model"]["version"] = "v5dev"
cfg["data"]["exp_dir"] = cfg["s2_ckpt_dir"] = str(ROOT / "logs" / "smoke")
cfg["save_weight_dir"] = "SoVITS_weights_smoke"
cfg["name"] = "smoke"
cfg["version"] = "v5dev"

# ⚠️ 必须清掉上一轮的 checkpoint，否则会「resume 后立刻退出」
#    if_save_latest 会在 save_root 留 G_*.pth，训练脚本优先 resume，
#    于是出现 "start training from epoch 3 / training done" —— 一步都没训。
import shutil as _sh
for _d in (ROOT / "logs" / "smoke" / "logs_s2_v5dev_full",
           ROOT / "SoVITS_weights_smoke"):
    if _d.exists():
        _sh.rmtree(_d)
        print(f"  已清除旧目录 {_d.relative_to(ROOT)}（避免 resume 后空跑）")
(ROOT / "logs" / "smoke" / "logs_s2_v5dev_full").mkdir(parents=True, exist_ok=True)
(ROOT / "SoVITS_weights_smoke").mkdir(parents=True, exist_ok=True)
cfg_path = ROOT / "tmp_smoke.json"
cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                    encoding="utf-8")
print(f"✓ 配置: {cfg_path}")
print(f"  batch=2  epochs=2  grad_ckpt=True  fp16=True")
print(f"  exp_dir={cfg['data']['exp_dir']}")

# ---- 启动训练（严格模式：数据加载失败立刻中断）----
env = os.environ.copy()
env["PYTHONPATH"] = str(ROOT)       # 关键：否则 f5_tts 命名空间包解析错
env["ZFY_STRICT_AUDIO"] = "1"       # 数据坏就抛异常，绝不喂全零张量
print("\n=== 冒烟训练（2 epoch，预计几分钟）===")
r = subprocess.run(
    [sys.executable, "-s", "GPT_SoVITS/s2_train_v5_full.py",
     "--config", "tmp_smoke.json"],
    cwd=str(ROOT), env=env, capture_output=True, text=True)

out = (r.stdout or "") + "\n" + (r.stderr or "")
print(out[-3500:])

# ---------------- 验收判定（这一步才是"冒烟通过"的依据）----------------
print("\n" + "=" * 62)
print("冒烟验收")
print("=" * 62)

fails = []
AUDIO_ERR = "load audio or ssl error"
if AUDIO_ERR in out:
    fails.append(f"出现 {out.count(AUDIO_ERR)} 次 '{AUDIO_ERR}' "
                 f"-> 数据加载失败，已被替换成全零张量（训练无效）")
if "ZFY_AUDIO_GUARD" in out:
    fails.append("触发了数据加载防护，见上方 [ZFY_AUDIO_GUARD] 行")
if r.returncode != 0:
    fails.append(f"训练进程退出码 {r.returncode}")

# ⚠️ "start training from epoch N" 是**无条件打印**的（s2_train_v5_full.py:266），
#    不能拿它判断是否 resume —— 必须解析 N，N>1 才是真 resume。
#    （我第一版就误判了，把正常启动也报成 "resume: True"）
import re as _re
_m = _re.search(r"start training from epoch (\d+)", out)
start_epoch = int(_m.group(1)) if _m else None
resumed = start_epoch is not None and start_epoch > 1
did_train = ("it/s]" in out) or ("s/it]" in out)   # 出现了进度条 = 真跑了

w = sorted((ROOT / "SoVITS_weights_smoke").glob("*.pth"))
if len(w) < 2:
    fails.append(f"权重产出不足：{len(w)} 个（应有 2 个）")
if resumed:
    fails.append(f"从第 {start_epoch} 轮 resume（不是从第 1 轮开始）—— "
                 f"本次结果不可信，请确认上一格已清除 logs/smoke")
if not did_train:
    fails.append("日志里没有训练进度条 —— 训练没有真正执行")

for kw in ("nccl", "hipFFT", "out of memory", "undefined symbol"):
    hits = [l for l in out.split("\n") if kw.lower() in l.lower()]
    if hits:
        fails.append(f"日志含 '{kw}': {hits[0].strip()[:120]}")

print(f"  真实训练发生   : {did_train}")
print(f"  起始轮次       : {start_epoch}（应为 1）")
print(f"  产出权重       : {[p.name for p in w]}")
print(f"  数据加载报错   : {AUDIO_ERR in out}")

if fails:
    print("\n  ✗ 冒烟未通过：")
    for f in fails:
        print(f"    - {f}")
    print("""
  常见处理：
    · nccl 相关        -> 确认 PYTHONPATH 已设（脚本按 ROCm 自动切 gloo）
    · hipFFT 相关      -> 把 tmp_smoke.json 的 fp16_run 改 False，重跑本格
    · load audio error -> 数据仍是坏的：
                          python _tools/sdk_repair_lfs.py --check
""")
else:
    print("\n  ✓ 冒烟通过：数据加载正常，训练真实发生")
    for p in w:
        print(f"    {p.name}  {p.stat().st_size/1e6:.1f} MB")
    print("\n  下一步：跑第 8 节的全量训练")
'''))

# ---------------------------------------------------------------- 8
cells.append(md(r"""
## 8. 全量训练

冒烟通过后执行。

**`EPOCHS` 默认已从 8 提到 32** —— 理由见格子开头的"训练量诊断"：

| | 参数量 | 优化步数 | 步数/参数 | 相对 v2Pro |
|---|---|---|---|---|
| v2Pro | 42.39 M | ~180 | 4.25e-6 | 1.00x |
| v5 **e8**（已跑） | 383.08 M | 432（实测 54 步/epoch） | 1.13e-6 | **0.27x** |
| v5 e24 | 383.08 M | 1296 | 3.38e-6 | 0.80x |
| v5 **e32** | 383.08 M | 1728 | 4.51e-6 | **1.06x** |

学习率两者相同（`1e-4`），`lr_decay 0.999875` 意味着 e8 跑完 LR 几乎没降
（1e-4 → 9.5e-5，只降 5%）。**383M 全量微调只跑 432 步，很可能没训够。**

**每轮都存权重**（`save_every_epoch: 1`），所以 e1..e32 全都在 ——
**别默认"轮数越大越好"，靠盲听挑**。e32 一共约 24 GB。

**输出目录带轮数**（`logs/v5_full_e32/`、`SoVITS_weights_v5_full_e32/`），
所以改 `EPOCHS` 重跑**不会覆盖**已有的 e8 结果，可以并存对比。

> 追加实验只需改 `EPOCHS`，其他都不用动 —— 启动格会自动找最新的
> `tmp_full_e*.json`，监控格会同时列出所有实验。
"""))
cells.append(code(r'''
import os, sys, subprocess, json
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "GPT_SoVITS").is_dir():
    ROOT = Path("/mnt/workspace/zfy-v5-fullft-pkg")
os.chdir(ROOT)

GSV = ROOT / "GPT_SoVITS"
PRE = GSV / "pretrained_models"

# ============================== 关键旋钮 ==============================
BATCH = 16        # 192GB 显存可开大；OOM 就降
EPOCHS = 32       # ★ 从 8 提到 32（理由见下方"训练量诊断"）
FP16 = True       # 报 hipFFT 错就改 False
# ====================================================================

# 输出目录**带轮数** -> 多次实验互不覆盖，便于盲听对比
# （否则改 EPOCHS 重跑会写进同一个目录，和 e8 的结果混在一起）
import re as _re
TAG = f"v5_full_e{EPOCHS}"
EXP_DIR = ROOT / "logs" / TAG
WEIGHT_DIR = ROOT / f"SoVITS_weights_{TAG}"
cfg_path = ROOT / f"tmp_full_e{EPOCHS}.json"

# ------------------------- 训练量诊断 -------------------------
# 为什么 e8 可能不够：
#   v2Pro   42.39 M 参数 / 约 180 步   = 4.25e-6 步/参数
#   v5 e8  383.08 M 参数 / 实测 432 步 = 1.13e-6 步/参数   <- 少约 4 倍
# 学习率两者相同（1e-4），lr_decay 0.999875 -> e8 跑完 LR 几乎没降
# （1e-4 -> 9.5e-5）。所以 383M 全量微调很可能"没训够"。
FULL_FEAT = ROOT / "logs" / "zfy_v5"
_n = len([l for l in (FULL_FEAT / "6-name2semantic.tsv").read_text(
    encoding="utf-8", errors="replace").strip().split("\n") if l.strip()])
_n = max(1, _n - 1)                    # 去掉表头

# 每 epoch 步数：优先用**实测值**（从已有权重名 e{N}_s{step} 反推），
# 否则估算。分桶采样器按长度分桶，实际批数会比 n/batch 略多
# —— 实测 e8 是 432 步 / 8 = 54，而 825/16 = 51。
_spe, _spe_src = None, ""
for _d in sorted(ROOT.glob("SoVITS_weights_v5_full*")):
    for _w in _d.glob("*_e*_s*_l0.pth"):
        _m = _re.match(r".*_e(\d+)_s(\d+)_l0\.pth$", _w.name)
        if _m and int(_m.group(1)) > 0:
            _spe = int(_m.group(2)) // int(_m.group(1))
            _spe_src = f"实测（来自 {_w.name}）"
            break
    if _spe:
        break
if not _spe:
    _spe = max(1, round(_n / BATCH * 1.06))   # 估算，+6% 补分桶余量
    _spe_src = "估算（尚未跑过，按 n/batch×1.06）"

_total = _spe * EPOCHS
V5_PARAMS, V2PRO_PARAMS, V2PRO_STEPS = 383.08e6, 42.39e6, 180
_base_ratio = V2PRO_STEPS / V2PRO_PARAMS
_ratio = (_total / V5_PARAMS) / _base_ratio

print("=" * 62)
print("训练量诊断")
print("=" * 62)
print(f"  样本            : {_n} 条")
print(f"  每 epoch 步数   : {_spe}   [{_spe_src}]")
print(f"  总步数          : {_total}")
print(f"  模型参数量      : {V5_PARAMS/1e6:.2f} M")
print(f"  步数 / 参数     : {_total/V5_PARAMS:.2e}")
print()
print("  对照基准（步数/参数越大，相对训练量越足）:")
print(f"    v2Pro          {V2PRO_PARAMS/1e6:6.2f} M /"
      f" {V2PRO_STEPS:4d} 步 = {_base_ratio:.2e}")
print(f"    v5 e8（已跑）  {V5_PARAMS/1e6:6.2f} M /"
      f" {_spe*8:4d} 步 = {_spe*8/V5_PARAMS:.2e}   <- 少约 4 倍")
print(f"    本次 e{EPOCHS:<3d}      {V5_PARAMS/1e6:6.2f} M /"
      f" {_total:4d} 步 = {_total/V5_PARAMS:.2e}")
print(f"    相对 v2Pro     {_ratio:.2f}x "
      f"{'（达到/超过）' if _ratio >= 1 else '（仍偏低，可再加大 EPOCHS）'}")
print()
_lr0, _dec = 1e-4, 0.999875
print(f"  LR: {_lr0:.0e} x {_dec}^{_total} = {_lr0*_dec**_total:.2e}"
      f"  (初始 {_lr0:.0e}，衰减 {_dec})")
print(f"  权重产出: e1..e{EPOCHS} 共 {EPOCHS} 个 x ~736 MB = "
      f"约 {EPOCHS*0.736:.1f} GB（每轮都存，便于盲听挑）")
print("=" * 62)
print()

cfg = json.loads((GSV / "configs" / "s2.json").read_text(encoding="utf-8"))
cfg["train"].update({
    "batch_size": BATCH,
    "epochs": EPOCHS,
    "pretrained_s2G": str(PRE / "gsv-v5-pretrained" / "s2Gv5dev.pth"),
    "pretrained_s2D": "",
    "if_save_latest": True,
    "if_save_every_weights": True,
    "save_every_epoch": 1,
    "gpu_numbers": "0",
    "grad_ckpt": True,
    "lora_rank": 0,
    "fp16_run": FP16,
})
cfg["model"]["version"] = "v5dev"
cfg["data"]["exp_dir"] = cfg["s2_ckpt_dir"] = str(EXP_DIR)
cfg["save_weight_dir"] = WEIGHT_DIR.name
cfg["name"] = TAG
cfg["version"] = "v5dev"

EXP_DIR.mkdir(parents=True, exist_ok=True)

# ⚠️ 清掉上一轮的 checkpoint —— 否则会「resume 后立刻退出」
#    （和冒烟那次同一个坑：if_save_latest 留下 G_*.pth，脚本优先 resume，
#      发现 epoch 已超过总轮数就直接结束，一步都没训）
import shutil as _sh
_ckpt = EXP_DIR / "logs_s2_v5dev_full"
if _ckpt.exists():
    stale = list(_ckpt.glob("G_*.pth")) + list(_ckpt.glob("D_*.pth"))
    if stale:
        _sh.rmtree(_ckpt)
        print(f"  已清除 {len(stale)} 个旧 checkpoint（避免 resume 空跑）")
        for f in stale[:3]:
            print(f"    {f.name}")
_ckpt.mkdir(parents=True, exist_ok=True)
WEIGHT_DIR.mkdir(parents=True, exist_ok=True)

# 全量训练用的是**本机已算好的全量特征**（logs/zfy_v5），
# 所以把特征链接到本次的 exp_dir（软链，不复制）
for name in ["2-name2text.txt", "3-bert", "4-cnhubert", "5-wav32k",
             "6-name2semantic.tsv"]:
    src, dst = FULL_FEAT / name, EXP_DIR / name
    if src.exists() and not dst.exists():
        dst.symlink_to(src)
        print(f"  链接 {name}")

cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"✓ 配置: {cfg_path}")
print(f"  batch={BATCH}  epochs={EPOCHS}  fp16={FP16}  grad_ckpt=True")
print(f"  权重输出: {WEIGHT_DIR.name}/")
'''))

cells.append(md("### 启动全量训练（nohup 后台，防会话超时）"))
cells.append(code(r'''
import os, sys, subprocess, json
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "GPT_SoVITS").is_dir():
    ROOT = Path("/mnt/workspace/zfy-v5-fullft-pkg")
os.chdir(ROOT)

# 自动找"最近一次生成"的 tmp_full_e*.json
# -> 改了 EPOCHS 重跑上一格后，这里不用改任何东西
cands = sorted(ROOT.glob("tmp_full_e*.json"),
               key=lambda p: p.stat().st_mtime, reverse=True)
if not cands:
    raise SystemExit("找不到 tmp_full_e*.json，请先跑上一格生成配置")
cfg_path = cands[0]
cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
tag = cfg["name"]
epochs = cfg["train"]["epochs"]

LOG = ROOT / f"train_{tag}.log"
cmd = (f'cd {ROOT} && PYTHONPATH={ROOT} nohup {sys.executable} -s '
       f'GPT_SoVITS/s2_train_v5_full.py --config {cfg_path.name} '
       f'> {LOG} 2>&1 &')
print(f"配置   : {cfg_path.name}  (exp={tag}, epochs={epochs})")
print(f"日志   : {LOG.name}")
print(f"启动: {cmd}\n")
os.system(cmd)
import time; time.sleep(20)
print("--- 日志前 30 行 ---")
print(LOG.read_text(encoding="utf-8", errors="replace")[:3000] if LOG.exists() else "(无日志)")
'''))

cells.append(md("### 监控训练进度（可反复运行）"))
cells.append(code(r'''
from pathlib import Path
import re, os

ROOT = Path.cwd()
if not (ROOT / "GPT_SoVITS").is_dir():
    ROOT = Path("/mnt/workspace/zfy-v5-fullft-pkg")

# 同时监控所有实验（train_v5_full_e*.log），这样 e8 和 e24 都能看到
logs = sorted(ROOT.glob("train_v5_full_e*.log"),
              key=lambda p: p.stat().st_mtime, reverse=True)
if not logs:
    logs = sorted(ROOT.glob("train_*.log"),
                  key=lambda p: p.stat().st_mtime, reverse=True)

if not logs:
    print("没有找到训练日志（train_*.log）")
for LOG in logs[:3]:
    txt = LOG.read_text(encoding="utf-8", errors="replace")
    exp = LOG.stem.replace("train_", "")
    eps = re.findall(r"saving ckpt %s_e(\d+):Success" % re.escape(exp), txt)
    done = "training done" in txt
    print("=" * 62)
    print(f"{LOG.name}   exp={exp}   已完成 epoch: {len(eps)}"
          f"{'  [training done]' if done else '  [运行中]'}")
    print(f"  epochs: {eps if eps else '(尚未存权重)'}")
    for kw in ("out of memory", "CUDA error", "hipFFT", "nccl",
               "load audio or ssl error"):
        n = txt.lower().count(kw.lower())
        if n:
            print(f"  !! '{kw}' x{n}")
    tail = [l for l in txt.split("\n") if l.strip()][-3:]
    for l in tail:
        print(f"    {l[:120]}")
    print()

# 列出所有实验的权重
import glob as _g
print("=" * 62)
print("权重产出（所有实验）")
print("=" * 62)
for d in sorted(ROOT.glob("SoVITS_weights_v5_full*")):
    if not d.is_dir():
        continue
    w = sorted(d.glob("*.pth"))
    tot = sum(p.stat().st_size for p in w) / 1e9
    print(f"  {d.name:34s} {len(w):3d} 个  {tot:5.2f} GB")
    if w:
        print(f"      {w[0].name}  ..  {w[-1].name}")
'''))

# ---------------------------------------------------------------- 9
cells.append(md(r"""
## 9. 训练完成后

### 9.1 权重位置

`SoVITS_weights_v5_full_e{N}/v5_full_e{N}_e{epoch}_s{step}_l0.pth`，
每个约 **1.5 GB**（全量 383M 参数）。

**多个实验并存**：`SoVITS_weights_v5_full_e8/`（旧）与 `..._e24/`（新）
互不覆盖，便于盲听对比"训练量够不够"。

### 9.2 传回本机评测

**方式 A**：传回魔搭数据集（复用同一个私有仓）

```bash
cd /mnt/workspace/zfy-v5-fullft-pkg
# 只传最好的那个 epoch，省流量
cp SoVITS_weights_v5_full/zfy_v5_full_e8_*.pth ./weights_v5_full_e8.pth
git add weights_v5_full_e8.pth && git commit -m "v5 fullft e8" && git push
```

**方式 B**：本机直接下载

```python
# 本机执行
from modelscope import snapshot_download
snapshot_download("yangwenjie1231/zfy-v5-fullft-pkg", repo_type="dataset",
                  allow_patterns=["weights_v5_full_*.pth"],
                  local_dir=r"D:\A\zfy-v5\dist\from_cloud")
```

### 9.3 本机评测（口径已修正）

```powershell
$PY = "C:\Users\A\miniconda3\envs\voice\python.exe"
cd D:\A\zfy-v5
& $PY eval_retrain.py --rank full --n 20
& $PY compare_r32_r128.py --n 20
& $PY make_listen_rank.py
```

### 9.4 成功判据

| 指标 | 现状（LoRA r32） | 目标（全量） |
|---|---|---|
| SIM | 0.8236 | **> 0.86**（v2Pro 是 0.8888） |
| 语气 | 你已认为优于 v2Pro | **必须保持** |
| 带宽 | 22.8 kHz | 保持 |

> 评测时务必用**生产参考音频** `zfh-models-v2pro/ref/default.wav`，
> 不要用数据集里那个错配的 `ref_audio/ref.wav`。
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"已生成: {OUT}")
print(f"  单元格数: {len(cells)}  "
      f"(markdown {sum(1 for c in cells if c['cell_type']=='markdown')}, "
      f"code {sum(1 for c in cells if c['cell_type']=='code')})")
print(f"  体积: {OUT.stat().st_size/1024:.1f} KB")
