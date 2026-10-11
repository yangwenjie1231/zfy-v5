# 云算力（ROCm）跑 v5 全量微调 · 操作指南

> 目标机：ubuntu22.04 / rocm7.2.3 / py3.12 / torch2.12.0
> 8 核 / 200GB 内存 / **192GB 显存（AMD ROCm）**

---

## 一、结论：需要上云，且可行

### 1.1 本机为什么不行

| 模型 | SoVITS 参数量 | 全量微调优化器+权重 | 本机 8GB |
| --- | ---: | ---: | --- |
| v2Pro（已跑通） | 42.39 M | 0.68 GB | 可行 |
| **v5 全量** | **381.51 M** | **6.10 GB** | **不可行** |
| v5 LoRA r128（已跑通） | 23.07 M 可训练 | 1.13 GB + 冻结权重 | 已跑通(7.7GB) |

关键：6.10 GB **只是优化器状态+权重**，还没算激活。
v5 的 CFM 是 22 层 DiT（335.79 M），激活占用远大于 v2Pro；
8GB 卡连 LoRA（冻结全部 381 M 权重）都已占到 7.7 GB。
→ **v5 全量微调在本机放不下。**

### 1.2 为什么全量微调值得做

已核实（`check_method_final.py`）：

| 模型 | 训练脚本 | 方式 | 权重键数 | is_lora |
| --- | --- | --- | ---: | --- |
| v2 | `s2_train.py` | **全量** | 673 | False |
| v2Pro | `s2_train.py` | **全量** | 678 | False |
| v5dev/v5turbo | `s2_train_v3_lora.py` | **LoRA** | 258 | True |

**v2Pro 音色好是因为全量微调。** v5 官方只给 LoRA，
LoRA r128 只训练 23.07 M（**6.05%**）—— 这就是 v5 音色不如 v2Pro 的根因。

已确认 v5 **可以做全量**：
`models.py` L1306-1308 里 `version in {v5dev,v5turbo}` → `self.cfm = CFMV5(...)`，
即 `SynthesizerTrnV3` 就是 v5 的完整模型（构造成功，381.51 M）。
只是官方训练脚本把 `get_peft_model` 写死了。

### 1.3 云机够不够

```
v5 全量微调基础占用 : 6.10 GB
192GB 显存余量      : 185.90 GB
→ 完全可行，显存不是瓶颈
```

---

## 二、ROCm 兼容性（已逐条核查）

| 风险项 | 等级 | 说明与处理 |
| --- | --- | --- |
| `torch.backends.cuda.matmul.allow_tf32` | **中** | ROCm 无此属性会 AttributeError → **已加 try/except** |
| `torch.cuda.amp` (GradScaler/autocast) | 低 | deprecated 但仍存在；**已改为 `torch.amp`** |
| `flash_attn` | 低 | 训练路径**不需要**（仅推理可选加速，11 个文件提到但都有 fallback） |
| `bitsandbytes` / `apex` / `deepspeed` | 无 | 代码里 **0 处**引用 |
| `CUDA_VISIBLE_DEVICES` | 低 | ROCm 下被忽略，单卡不影响 |
| `DistributedBucketSampler` | 中 | 单卡时不启用 DDP，风险低 |
| `cuda_graph`（v5 新特性） | 中 | 11 个文件提到；**训练不用**，推理若开需关掉 |

**已生成了打好这些补丁的训练脚本**：`GPT_SoVITS/s2_train_v5_full.py`
（由 `make_v5_fullft_script.py` 从 `s2_train_v3_lora.py` 生成，已 `py_compile` 通过）

---

## 三、操作步骤

### 3.1 本机打包（已完成，3.09 GB）

```powershell
cd D:\A\zfy-v5
& "C:\Users\A\miniconda3\envs\voice\python.exe" pack_for_cloud.py
# 产出 dist\zfy_v5_fullft_cloud.zip
```

包内容（已校验，无多余大文件）：

| 内容 | 体积 |
| --- | ---: |
| `GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth` | 766 MB |
| `chinese-roberta-wwm-ext-large/pytorch_model.bin` | 651 MB |
| `text/G2PWModel/g2pW.onnx` | 635 MB |
| `logs/zfy_v5/`（**特征已算好，云上不用重算**） | ~700 MB |
| `chinese-hubert-base/pytorch_model.bin` | 189 MB |
| `s1v3.ckpt`（GPT） | 155 MB |
| `vocoder.pth` | 58 MB |
| 代码（含 `s2_train_v5_full.py`） | ~30 MB |

**已排除**：`G_*.pth`（训练中间状态，每个 2GB）、`SoVITS_weights`、
`G2PWModel_1.1.zip`(589MB)、`ja_userdic`(17MB)、`s2Gv5turbo.pth`(766MB)

### 3.2 上传并安装

```bash
unzip zfy_v5_fullft_cloud.zip -d ~/zfy-v5
cd ~/zfy-v5/GPT-SoVITS

# ROCm 版 torch（关键：不要装 CUDA 版）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements.txt

# 校验 ROCm 可用
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望：2.12.0+rocm7.2 True   （ROCm 下 is_available() 也返回 True）
```

> 注意：`requirements.txt` 里有 `onnxruntime-gpu`，ROCm 机器上装不上或无效，
> 但**训练不需要它**，可以忽略或删掉该行。

### 3.3 启动全量微调

先生成配置（把 `lora_rank` 留着不影响，脚本已不读它）：

```bash
cd ~/zfy-v5/GPT-SoVITS
python - <<'PY'
import json, os
cfg = json.load(open("GPT_SoVITS/configs/s2.json"))
cfg["train"].update({
    "batch_size": 16,          # 192GB 显存可以开大；显存不够就降
    "epochs": 24,
    "pretrained_s2G": os.path.abspath(
        "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth"),
    "pretrained_s2D": "",
    "if_save_latest": True,
    "if_save_every_weights": True,
    "save_every_epoch": 1,
    "gpu_numbers": "0",
    "grad_ckpt": True,         # 全量微调建议保留
    "lora_rank": 0,            # 脚本已不读
})
cfg["model"]["version"] = "v5dev"
cfg["data"]["exp_dir"] = cfg["s2_ckpt_dir"] = os.path.abspath("logs/zfy_v5")
cfg["save_weight_dir"] = "SoVITS_weights_v5_full"
cfg["name"] = "zfy_v5_full"
cfg["version"] = "v5dev"
os.makedirs("logs/zfy_v5/logs_s2_v5dev_full", exist_ok=True)
os.makedirs("SoVITS_weights_v5_full", exist_ok=True)
json.dump(cfg, open("tmp_s2_full.json","w"), ensure_ascii=False, indent=2)
print("配置已写入 tmp_s2_full.json")
PY

# 启动（注意 PYTHONPATH，避免 f5_tts 命名空间包解析错）
export PYTHONPATH=$PWD
python -s GPT_SoVITS/s2_train_v5_full.py --config tmp_s2_full.json
```

**关键：`export PYTHONPATH=$PWD` 不能省。**
`f5_tts` 是无 `__init__.py` 的命名空间包，若 v5 根目录不在
`sys.path` 最前，会解析到别的检出里的旧版 `DiT`
（无 `use_step_embedding`）→ `TypeError`。

### 3.4 权重回传与评测

```bash
# 云上产出：SoVITS_weights_v5_full/zfy_v5_full_e{N}_*.pth
# 每个约 1.5 GB（全量，381M 参数 fp32）
# 传回本机 D:\A\zfy-v5\GPT-SoVITS\SoVITS_weights_v5_full\
```

本机评测（脚本已就绪，参考音频与 GPT 口径都已修正）：

```powershell
& $PY eval_retrain.py --rank full --n 20
& $PY make_listen_retrain.py --rank full --epochs 4,8,12
```

---

## 四、预计时间与资源

| 项 | 估计 |
| --- | --- |
| 本机 LoRA r128 | 407 step/epoch，约 11 分钟/epoch |
| 全量微调每 step | 更慢（要更新 381 M 参数），但云机 GPU 通常更强 |
| 建议先跑 | **4-8 epoch** 看效果，再决定是否加长 |
| batch_size | 192GB 可开到 16-32；显存不够就降 |

**显存不是瓶颈，时间才是。**

---

## 五、注意事项

1. **数据准备不用上云** —— `logs/zfy_v5/` 里 825 条特征已在本机算好
2. **GPT 不用重训** —— v5 与 v2Pro 共用 `s1v3.ckpt`
3. **`s2_train_v5_full.py` 是自动生成的**，改前先看 `make_v5_fullft_script.py`
4. **ROCm 上 `torch.cuda.is_available()` 返回 True**（HIP 伪装），
   所以 `.cuda()` 调用可用，无需改代码
5. **若遇到 ROCm 专属报错**，优先检查：
   - `torch.backends.cudnn` 相关（已加保护）
   - `flash_attn` 是否被意外 import（训练路径不该用）
6. **对比时务必用生产参考音频** —— 数据集里的 `ref_audio/ref.wav` 是错配的
   （ref_text 与实念内容前缀一致率 0%）

---

## 六、附：本机已完成的 r128 LoRA 结果

在等云算力的同时，本机 r128/24ep 已在跑（预计 4 小时）。
它的作用是：**验证"提高秩"是否有收益**。
若 r128 相比 r32 有明显提升，说明 LoRA 容量是瓶颈；
若提升有限，则更说明必须上全量微调。
