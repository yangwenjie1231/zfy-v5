# 魔搭 Notebook 跑 v5 全量微调 · 操作手册

> 目标环境：**ubuntu22.04 / rocm7.2.3 / py3.12 / torch2.12.0**
> 8 核 / 200GB 内存 / **192GB 显存（AMD ROCm）**
> 本机负责：打包 + 上传 + 评测。云上负责：训练。

---

## 〇、**先跑自检**（下载 2.77GB 之前）

环境与我本机差异很大（本机 py3.13 + torch2.7.1 + CUDA），
跨了 5 个 torch 大版本，**部分 API 可能已移除**。

自检脚本在仓库的 `_tools/` 下（clone 后即可跑）：

```bash
cd /mnt/workspace/zfy-v5-fullft-pkg
python _tools/notebook_preflight.py
```

它会逐项报告：torch/ROCm 版本、**关键 API 是否存在**、依赖是否齐备、
f5_tts 解析是否正确、能否构造 v5 模型。
**从哪个目录跑都行** —— 它会自动定位检出根。

### 已知的两个风险点

| 风险 | 代码位置 | 状态 |
| --- | --- | --- |
| **`torch.cuda.amp` 可能被移除** | `module/models.py:23`（无保护硬 import） | ✅ **已打补丁** |
| `numpy<2.0` 约束可能过时 | `requirements.txt` | ⚠️ 见下 |

**`models.py` 的补丁**（`patch_models_for_torch212.py` 生成）：
优先用 `torch.amp`，回退 `torch.cuda.amp`，再回退 `nullcontext`。
已在本机 torch 2.7.1 验证**不破坏**原环境（构造 381.51M 模型成功）。

> 若自检显示 `torch.cuda.amp` 仍存在，补丁也无害（只是走了新 API）。

**numpy 建议**：**不要盲目按 `requirements.txt` 降级到 <2.0**。
本机经验：numpy 1.26.4 曾报 `OverflowError: cannot convert longdouble
infinity to integer`，必须用 2.x。优先保留环境预装的 numpy。

---

## 一、前置状态（本机已完成）

| 项 | 状态 |
| --- | --- |
| 私有数据集 | `yangwenjie1231/zfy-v5-fullft-pkg`（**visibility=1 = 私有，已匿名验证**） |
| 仓库结构 | **目录树**（2681 文件 / 2.9 GB，git + LFS） |
| 训练脚本 | `GPT_SoVITS/s2_train_v5_full.py`（已打 ROCm 补丁，`py_compile` 通过） |
| 辅助脚本 | `_tools/`（自检 + torch2.12 补丁 + 本手册） |
| 魔搭 CLI | 已登录（`yangwenjie1231`） |

### 为什么用 git 直传而不是 zip

最初我打包成 zip 再上传，是**多余的一层**：
- 魔搭数据集本身就是 git 仓库（带 LFS），`ms upload` 底层也走 git
- 压缩只省 **10%**（2.77 GB vs 3.09 GB）—— 因为大头是已压缩的模型权重
- zip 无法增量同步：改一个文件要重传整个包
- notebook 里还得多一步解压

现已改为 **git 直传目录树**，notebook 里 clone 完即可用。

### 关于 visibility 的坑（重要）

**魔搭的 `visibility` 取值与直觉相反**，已用「上传纯文本 + 匿名下载」定论测试：

| 值 | 匿名访问 | 实际含义 |
| --- | --- | --- |
| **1** | HTTP 404 | **私有** |
| **5** | HTTP 200 且取到内容 | **公开** |

我最初假设「5=私有」是**错的**，已修正为 `1`。
验证脚本：`probe_visibility_definitive.py`、`fix_dataset_private.py --verify`

---

## 二、Notebook 内操作

### 2.1 取代码（git clone，**无需解压**）

魔搭官方给两种下载方式，**本仓库只能用 git**：

| 方式 | 命令 | 本仓库可用？ |
| --- | --- | --- |
| **GIT** | `git lfs install` + `git clone` | ✅ **可用**（已实测） |
| SDK | `MsDataset.load(...)` | ❌ **不可用** |

**为什么 SDK 不行**（已实测）：`MsDataset.load()` 报
`DataFilesNotFoundError`。它期望标准 ML 数据集结构
（可解析的 `data/*.csv` / `*.jsonl` 等），而本仓库是
「**代码 + 模型权重 + 特征**」的目录树，没有可解析的数据文件。

#### 正式用法

```bash
cd /mnt/workspace
git lfs install

git clone https://oauth2:<你的token>@www.modelscope.cn/datasets/yangwenjie1231/zfy-v5-fullft-pkg.git

cd zfy-v5-fullft-pkg
ls
```

**clone 后这个目录就是 GPT-SoVITS 检出根**（含 `config.py`、`GPT_SoVITS/`、`tools/`），
不要在它下面再找 `GPT-SoVITS/` 子目录 —— `GPT_SoVITS/`（下划线）才是 Python 包。

```
zfy-v5-fullft-pkg/          <- 检出根，在这里跑训练脚本
├─ GPT_SoVITS/              <- Python 包（module/ f5_tts/ configs/）
├─ _tools/                  <- 自检 + 补丁 + 本手册
├─ config.py
├─ requirements.txt
├─ tools/
├─ logs/                    <- 已算好的特征
└─ data/
```

token 在 [魔搭访问令牌](https://modelscope.cn/my/myaccesstoken) 获取。

> ⚠️ **不要把带 token 的 URL 提交进任何仓库，也不要粘贴到公开场合。**

#### 想先看结构、暂不拉 3GB 权重

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone \
  https://oauth2:<token>@www.modelscope.cn/datasets/yangwenjie1231/zfy-v5-fullft-pkg.git
cd zfy-v5-fullft-pkg
git lfs pull        # 确认无误后再拉大文件
```

#### 用本地凭据（免手填 token）

```bash
# 魔搭 CLI 登录后，token 存在 ~/.modelscope/credentials/git_token
git clone https://oauth2:$(cat ~/.modelscope/credentials/git_token)@www.modelscope.cn/datasets/yangwenjie1231/zfy-v5-fullft-pkg.git
```

#### 校验下载完整性

```bash
cd zfy-v5-fullft-pkg
git lfs ls-files | wc -l          # LFS 文件数（应为 2523）
find . -type f -not -path './.git/*' | wc -l   # 总文件数（应为 2681）
du -sh .                           # 约 2.9 GB
```

#### ⚠️ 两个 `wav` 目录，别搞混

仓库里有两个都叫"音频"的目录，**只有一个是训练需要的**：

| 目录 | 内容 | 训练需要？ |
| --- | --- | --- |
| `logs/zfy_v5/5-wav32k/` | 825 个 32kHz wav（**已切片、已重采样**） | ✅ **必需** —— `data_utils.py:28` 直接读它 |
| `data/zfy_v5/wav32k/` | 原始 32kHz wav | ❌ 不需要 —— 只是**数据准备**阶段的输入 |

我打包时**没有包含** `data/zfy_v5/wav32k/`（省 245MB），因为训练不读它。
只有你要**重算特征**（比如做小样本冒烟测试）时才需要它 —— 那时从
`logs/zfy_v5/5-wav32k/` 复制即可，内容等价：

```bash
cd zfy-v5-fullft-pkg
mkdir -p data/zfy_v5/wav32k
cp logs/zfy_v5/5-wav32k/*.wav data/zfy_v5/wav32k/
```

若 LFS 文件显示为几百字节的指针文本，说明没走 LFS：

```bash
git lfs install --local
git lfs pull
```

---

魔搭 Notebook 通常**已预装 ROCm 版 torch**，先确认：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.hip)"
# 期望：2.12.0+rocm7.2  True  7.2.3xxxx
```

**若已预装 ROCm torch，不要再装**（避免把 ROCm 版覆盖成 CUDA/CPU 版）：

```bash
cd /mnt/workspace/zfy-v5-fullft-pkg        # 检出根

# 先跑自检（确认 torch API / 依赖 / f5_tts 解析）
python _tools/notebook_preflight.py

# 给 models.py 打 torch2.12 兼容补丁（幂等，可重复跑）
python _tools/patch_models_for_torch212.py --root $PWD

# 装依赖 —— 注意：不要用 requirements.txt 整份装
# 它钉了 numpy<2.0，与 py3.12 环境可能冲突
pip install librosa==0.10.2 "transformers>=4.51,<5" "peft<0.18.0" \
            pytorch-lightning "torchmetrics<=1.5" \
            x_transformers rotary_embedding_torch einops \
            cn2an pypinyin jieba_fast chardet psutil tensorboard \
            ffmpeg-python soundfile
# onnxruntime-gpu 在 ROCm 上无效，训练也不需要，跳过
```

装完再跑一次自检确认：

```bash
python _tools/notebook_preflight.py
```

> `notebook_preflight.py` 会**自动定位检出根**（找含 `config.py` +
> `GPT_SoVITS/` 的那一层），所以从哪个目录跑都行。
> 也可显式指定：`export GSV_ROOT=/mnt/workspace/zfy-v5-fullft-pkg`


### 2.3 冒烟测试（**必须先做**）

社区经验：**不要一上来就跑全量**。先用 1–2 分钟数据验证链路。

```python
# 生成小样本 list
import os, random
ROOT = "/mnt/workspace/zfy-v5-fullft-pkg"      # 检出根
src = f"{ROOT}/data/zfy_v5/zfy_v5.list"
lines = open(src, encoding="utf-8").read().strip().split("\n")
random.seed(0)
small = random.sample(lines, 20)               # 20 条 ≈ 1-2 分钟
os.makedirs(f"{ROOT}/data/smoke", exist_ok=True)
open(f"{ROOT}/data/smoke/smoke.list", "w",
     encoding="utf-8").write("\n".join(small) + "\n")
print("小样本 list:", len(small), "条")
```

小样本的数据准备（特征要重算，因为 `logs/zfy_v5` 是全量的）：

```bash
cd /mnt/workspace/zfy-v5-fullft-pkg
export PYTHONPATH=$PWD          # 关键！否则 f5_tts 命名空间包解析错
export inp_text=$PWD/data/smoke/smoke.list
export inp_wav_dir=$PWD/data/zfy_v5/wav32k
export exp_name=smoke
export opt_dir=$PWD/logs/smoke
export i_part=0
export all_parts=1
export _CUDA_VISIBLE_DEVICES=0
export is_half=True
export version=v5dev

python -s GPT_SoVITS/prepare_datasets/1-get-text.py
python -s GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py
python -s GPT_SoVITS/prepare_datasets/3-get-semantic.py   # 需 s2config_path / pretrained_s2G
```

> 简化做法：直接用 `train_v5.py` 的 `stage_data` 逻辑，把 `LIST`/`WAV_DIR`
> 指到小样本即可（脚本已在包里）。

**冒烟训练**（2 epoch 足够暴露问题）：

```bash
python - <<'PY'
import json, os
cfg = json.load(open("GPT_SoVITS/configs/s2.json"))
cfg["train"].update({
    "batch_size": 2, "epochs": 2,
    "pretrained_s2G": os.path.abspath(
        "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth"),
    "pretrained_s2D": "", "if_save_latest": True,
    "if_save_every_weights": True, "save_every_epoch": 1,
    "gpu_numbers": "0", "grad_ckpt": True, "lora_rank": 0,
})
cfg["model"]["version"] = "v5dev"
cfg["data"]["exp_dir"] = cfg["s2_ckpt_dir"] = os.path.abspath("logs/smoke")
cfg["save_weight_dir"] = "SoVITS_weights_smoke"
cfg["name"] = "smoke"; cfg["version"] = "v5dev"
os.makedirs("logs/smoke/logs_s2_v5dev_full", exist_ok=True)
os.makedirs("SoVITS_weights_smoke", exist_ok=True)
json.dump(cfg, open("tmp_smoke.json", "w"), ensure_ascii=False, indent=2)
print("ok")
PY

export PYTHONPATH=$PWD
python -s GPT_SoVITS/s2_train_v5_full.py --config tmp_smoke.json
```

**冒烟要确认的三件事**：

| 检查 | 期望 | 若失败 |
| --- | --- | --- |
| 不报 `ncclInternalError` | 用 gloo | 脚本已自动检测 ROCm 并切 gloo |
| 不报 `hipFFT` 相关错 | — | 把 `fp16_run` 设为 `False`（见下） |
| 能存出权重 | `SoVITS_weights_smoke/*.pth` | 看日志定位 |

### 2.4 全量训练

冒烟通过后：

```bash
cd /mnt/workspace/zfy-v5-fullft-pkg
python - <<'PY'
import json, os
cfg = json.load(open("GPT_SoVITS/configs/s2.json"))
cfg["train"].update({
    # 192GB 显存可以开大；全量微调 + grad_ckpt 下先试 16
    "batch_size": 16,
    "epochs": 24,
    "pretrained_s2G": os.path.abspath(
        "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth"),
    "pretrained_s2D": "",
    "if_save_latest": True, "if_save_every_weights": True,
    "save_every_epoch": 1,
    "gpu_numbers": "0", "grad_ckpt": True, "lora_rank": 0,
    # 若遇 hipFFT/半精度问题，改成 False（192GB 显存扛得住 FP32）
    "fp16_run": True,
})
cfg["model"]["version"] = "v5dev"
cfg["data"]["exp_dir"] = cfg["s2_ckpt_dir"] = os.path.abspath("logs/zfy_v5")
cfg["save_weight_dir"] = "SoVITS_weights_v5_full"
cfg["name"] = "zfy_v5_full"; cfg["version"] = "v5dev"
os.makedirs("logs/zfy_v5/logs_s2_v5dev_full", exist_ok=True)
os.makedirs("SoVITS_weights_v5_full", exist_ok=True)
json.dump(cfg, open("tmp_full.json", "w"), ensure_ascii=False, indent=2)
print("ok")
PY

export PYTHONPATH=$PWD
nohup python -s GPT_SoVITS/s2_train_v5_full.py --config tmp_full.json \
  > train_full.log 2>&1 &
tail -f train_full.log
```

---

## 三、ROCm 三个已知障碍与对策

社区反馈 + 我在代码里逐条核实：

| 障碍 | 代码证据 | 对策 | 状态 |
| --- | --- | --- | --- |
| **NCCL 不兼容** | `s2_train_v3_lora.py:81` `backend="gloo" if os.name=="nt" else "nccl"` | 改成检测 `torch.version.hip` 就用 gloo | ✅ 已打补丁 |
| **hipFFT 不支持 FP16** | `fp16_run` 控制 `autocast` | 报错就把 `fp16_run` 设 `False` | ⚠️ 运行时按需 |
| **cudnn / allow_tf32 不存在** | `torch.backends.cudnn.*` | try/except 包裹 | ✅ 已打补丁 |
| **torch.cuda.amp 弃用** | `from torch.cuda.amp import ...` | 改 `torch.amp` | ✅ 已打补丁 |
| **版本匹配** | 官方 `install.sh` 用 rocm6.2 | 优先用 Notebook 自带；不行再降级 | ⚠️ 环境相关 |

**补丁清单**（`s2_train_v5_full.py` 里搜 `[v5-fullft]`）：

```
L16   torch.amp（CUDA/ROCm 通用）
L47   cudnn benchmark/deterministic 加保护
L54   allow_tf32 加保护
L60   set_float32_matmul_precision 加保护
L98   backend: 检测 ROCm -> gloo
L170  目录名去 lora_rank
L174  LoraConfig 整块注释
L209/240  去掉 get_peft_model（全量微调）
```

---

## 四、训练完成后

### 4.1 把权重传回本机

权重在 `SoVITS_weights_v5_full/*.pth`，每个约 **1.5 GB**（全量 381M 参数 fp32）。

```bash
# 方式 A：传到魔搭（再建一个私有数据集，或复用同一个）
ms upload yangwenjie1231/zfy-v5-fullft-pkg \
  SoVITS_weights_v5_full --repo-type dataset \
  --commit-message "v5 fullft weights"
```

```bash
# 方式 B：只传最好的那个 epoch，省流量
ls -la SoVITS_weights_v5_full/
```

### 4.2 本机评测

```powershell
$PY = "C:\Users\A\miniconda3\envs\voice\python.exe"
cd D:\A\zfy-v5
# 权重放到 GPT-SoVITS\SoVITS_weights_v5_full\ 后：
& $PY eval_retrain.py --rank full --n 20
& $PY compare_r32_r128.py --n 20
& $PY make_listen_rank.py
```

评测口径已修正（前几轮的教训）：
- 参考音频用生产 `default.wav`（不是数据集里错配的 `ref.wav`）
- GPT 统一用 `zfh-e4.ckpt`（生产同款）
- 20 句配对检验

---

## 五、预期与判据

| 项 | v5 LoRA r32（现状） | v5 全量（目标） |
| --- | ---: | ---: |
| SIM | 0.8236 | 期望接近/超过 v2Pro 的 0.8888 |
| 语气 | **用户认为已优于 v2Pro** | 需保持 |
| 带宽 | 22.8 kHz | 保持 |

**成功判据**：SIM 明显提升（>0.86）**且**语气优势未丢失。

---

## 六、注意事项

1. **`export PYTHONPATH=$PWD` 不能省** —— `f5_tts` 是无 `__init__.py` 的
   命名空间包，v5 根目录不在 `sys.path` 最前时会解析到旧版 `DiT`
   （无 `use_step_embedding`）→ `TypeError`
2. **Notebook 会话可能超时** —— 长训练建议用 `nohup`，并定期把权重
   传到数据集（持久化），避免会话中断丢失
3. **数据集是私有的** —— 别人访问会 404，这是**正确**的
4. **不要再上传到公开仓库** —— 语料版权归鹰角网络
