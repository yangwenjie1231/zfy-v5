# 庄方宜 · GPT-SoVITS v5 微调工作区

> 为《明日方舟：终末地》角色 **庄方宜**（zhuangfy）训练中文专属音色模型的完整工作区。
> 含脚本、结论、**三种可切换的语气参考音频**。

本仓库是 **v5 工作区**，与旧的 v2Pro 生产仓库区分开。

---

## 快速开始

### 1. 切换语气风格

```powershell
python switch_style.py --list      # 列出可用风格
python switch_style.py --show      # 看当前是哪个
python switch_style.py normal      # 正常说话（默认）
python switch_style.py firm        # 语气强硬
python switch_style.py gentle      # 语音娇
```

切换会同步更新 `ref/default.wav` 与 `ref/ref.txt`
（**参考文本必须与音频严格对应**，否则整个音色提示失效）。

### 2. 三种语气风格

| 风格 | 时长 | 采样率 | 听感 | 客观特征 |
|---|---|---|---|---|
| `normal` | 7.95s | 48kHz | **正常说话** | F0 尾部 +1.89，500-800Hz 8.94% |
| `firm` | 8.66s | 32kHz | **语气强硬** | F0 尾部 −4.13，说话人相似度最高 0.8589 |
| `gentle` | 5.26s | 32kHz | **语音娇** | F0 尾部 −4.86，500-800Hz 最干净 4.25% |

> 三种风格均经**盲听确认可用**，区别在语气而非质量。

---

## 生产配置

```
SoVITS : zfh_v2pro_e12_s384.pth    （v2Pro，LoRA 微调）
GPT    : zfh-e4.ckpt               （s1，v2 底模微调 4 epoch）
参考   : ref/default.wav           （见上表，可切换）
推理   : top_k 15 / temperature 1.0 / repetition_penalty 1.35
         text_split_method cut1 / seed 42
```

**评估结果**：说话人相似度 **0.8488**（天然上限 0.8893，**达成率 95.4%**），语速 3.13 字/秒。

---

## 目录结构

```
.
├── switch_style.py              一键切换语气风格
├── make_style_variants.py       固化三种参考音频
├── compare_ref_swap.py          换参考 A/B 合成对比
├── compare_gpt.py               GPT 微调 A/B 对比
├── train_gpt.py                 GPT（s1）微调
│
├── patch_*.py                   环境/代码补丁（共 12 个）
├── verify_*.py                  验证脚本（共 20+ 个）
├── diag_*.py                    诊断脚本
│
├── GPT-SoVITS/ref/
│   ├── ref.txt                  当前参考文本
│   └── styles/                  **三种语气参考音频**（核心交付物）
│       ├── normal.wav
│       ├── firm.wav
│       ├── gentle.wav
│       ├── styles.json
│       └── report.txt
│
└── dist/                        （未跟踪）产物与评估报告
```

---

## 关键结论（本工作区的主要发现）

### 1. 参考音频决定语气风格

同一模型 + 不同参考音频 = **不同语气**。参考音频不只提供音色，
还提供**韵律风格**（F0 走向、语速、停顿）。

| 参考的 F0 尾部斜率 | 输出的 F0 尾部斜率 | 听感 |
|---|---|---|
| +16.71（问句「好吗？」） | +1.89 | 正常 |
| −11.20（陈述句） | −4.13 | 强硬 |
| −18.07（陈述句） | −4.86 | 娇 |

### 2. 说话人相似度**不测语调自然度**

`firm` 的相似度（0.8589）高于 `normal`（0.8488），但盲听显示
`firm` 的语调"怪怪的"。**相似度只测音色，不测韵律。**

### 3. v5 全量微调**没有超过** v2Pro

32 epoch 全量微调（解冻 `ssl_proj`/`quantizer`/`enc_p`）与 v2Pro 听感等价
（相似度差 0.0048，在噪声内）。

### 4. `freeze_quantizer` 是"全有全无"，存在设计缺陷

`models.py` 的 `freeze_quantizer` 只能三组一起冻结/解冻。
**只解冻 `enc_p`+`ssl_proj`、保留 `quantizer` 冻结**才能避免码本 EMA 漂移：

```
三组全解冻 -> 码本按 EMA 漂移（实测 delta 7.28）
           -> GPT 的旧 token 不再对应 -> **提前 EOS**（e32 有 3 句异常）
只解冻 enc_p+ssl_proj -> 码本 delta 0.00 -> 正常
```

用 `patch_unfreeze_fine.py` + `ZFY_UNFREEZE=enc_p,ssl_proj`。

### 5. `lr_schedulers.py:58` 的 LR 是死代码

```python
self.lr = lr = self.end_lr = 0.002  ###锁定用线性###不听话，直接锁定！
```

**无论 YAML 写什么，GPT 实际 LR 永远是 2e-3。**
用 `patch_s1_lr.py` + `ZFY_S1_LR` 环境变量可控。

---

## 环境补丁清单

在云端/新实例上按顺序执行：

```bash
python patch_core_vq_dist.py     --root $PWD   # 分布式守卫（单进程崩溃根因）
python patch_unfreeze_fine.py    --root $PWD   # 细粒度解冻（保护码本）
python patch_infer_env.py        --root $PWD   # 推理环境（torchaudio 回退等）
python patch_v5_fullft_load.py   --root $PWD   # 全量/LoRA 权重加载对齐
python patch_no_ffmpeg.py        --root $PWD   # ffmpeg 可选
python patch_ckpt_atomic.py      --root $PWD   # checkpoint 原子保存
python patch_s1_lr.py            --root $PWD   # GPT LR 可控
python patch_win_encoding.py     --root $PWD   # Windows 编码（本地训练用）
```

**每个补丁都有配套的 `verify_*.py`**，改动前会备份 `.bak_zfy_*`。

---

## 已知问题

| 问题 | 状态 |
|---|---|
| 尾音「浊」（500-800Hz 比真人多一倍） | 已量化定位；后处理 5 版均有取舍，**未采用** |
| 尾音「往上高」 | 根因 = 参考音频以问句结尾；换 `firm`/`gentle` 可消除 |
| 说话人相似度不测韵律 | 需人工盲听配合 |
| 长文本（>40 字）未做压力测试 | 待做 |

---

## 语料说明

- 训练语料 **825 条 / 63.5 分钟**（游戏解包语音 + 角色台词）
- 语料**不包含在本仓库**（体积大 + 版权）
- 评估用留出集 131 条（训练集外）

---

## 许可

脚本与文档：MIT
参考音频：来自游戏《明日方舟：终末地》角色语音，版权归鹰角网络所有，
**仅供个人研究与学习使用**。
