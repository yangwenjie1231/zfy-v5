#!/usr/bin/env bash
# ============================================================
# SoVITS 全量微调 —— 启动脚本（可在依赖未完全修好时先跑）
#
# ## 为什么单独做这个
#
# 依赖修复卡在 transformers/huggingface_hub/gradio 的三方冲突上：
#   · transformers 5.x  -> 需要 hf-hub >= 1.x -> 报 is_offline_mode 缺失
#   · gradio 4.44       -> 需要 hf-hub < 0.37 -> 报 HfFolder 缺失
# 两者互斥，而**推理链路才需要 transformers**。
#
# 但 **SoVITS 训练链路不依赖 transformers** —— 实测已 OK：
#     OK   SoVITS 训练链路（TextAudioSpeakerLoaderV4）
#
# 所以不必等依赖全修好：先把 60~90 分钟的 SoVITS 训练跑起来，
# 边跑边修 GPT/推理的依赖。
#
# ## 用法
#   cd /mnt/workspace/zfy-v5-fullft-pkg
#   bash _tools/run_sovits_e32.sh              # 默认后台跑 e32
#   bash _tools/run_sovits_e32.sh --fg         # 前台（看实时输出）
#   bash _tools/run_sovits_e32.sh --epochs 16  # 换轮数
# ============================================================
set -u

ROOT=/mnt/workspace/zfy-v5-fullft-pkg
cd "$ROOT" || exit 1

EPOCHS=32
FG=0
RESUME=0
GRAD_CKPT=""          # 留空 = 用默认（True）；--no-grad-ckpt 关掉
BATCH=""
while [ $# -gt 0 ]; do
    case "$1" in
        --fg) FG=1 ;;
        --epochs) EPOCHS="$2"; shift ;;
        --resume) RESUME=1 ;;
        --resume-from) RESUME_FROM="$2"; shift ;;
        --no-grad-ckpt) GRAD_CKPT="false" ;;
        --grad-ckpt) GRAD_CKPT="true" ;;
        --batch) BATCH="$2"; shift ;;
        *) echo "未知参数: $1" ;;
    esac
    shift
done

echo "============================================================"
echo "SoVITS 全量微调启动（e${EPOCHS}）"
echo "  模式: $([ "$RESUME" = "1" ] && echo '续训（继承优化器状态）' || echo '全新')"
[ -n "$GRAD_CKPT" ] && echo "  grad_ckpt: $GRAD_CKPT"
[ -n "$BATCH" ] && echo "  batch_size: $BATCH"
echo "============================================================"

# ---------- 续训模式：把 exp_dir 指回旧目录 ----------
# 说明：e8 的 SoVITS_weights_*.pth 只是**权重快照**，不含优化器状态。
# 真正能续训的是 exp_dir/logs_s2_v5dev_full/G_*.pth（含 AdamW 动量）。
# 所以续训的做法是让 exp_dir 指向旧实验目录，让 latest_checkpoint_path 找到它。
if [ "$RESUME" = "1" ]; then
    RESUME_FROM="${RESUME_FROM:-logs/zfy_v5_full}"
    echo
    echo "--- 续训来源: $RESUME_FROM ---"
    CK="$RESUME_FROM/logs_s2_v5dev_full"
    if ! ls "$CK"/G_*.pth >/dev/null 2>&1; then
        echo "  !! 在 $CK 里找不到 G_*.pth"
        echo "     续训需要的是**含优化器状态**的 checkpoint，不是权重快照。"
        echo "     可用的候选："
        for d in logs/*/logs_s2_v5dev_full; do
            [ -d "$d" ] || continue
            n=$(ls -1 "$d"/G_*.pth 2>/dev/null | wc -l)
            echo "       $d  ($n 个 G_*.pth)"
        done
        echo "     用法: --resume-from logs/<实验名>"
        exit 1
    fi
    latest=$(ls -1 "$CK"/G_*.pth | sort -V | tail -1)
    echo "  找到: $(basename "$latest")  ($(( $(stat -c%s "$latest") / 1000000 )) MB)"
fi

# ---------- 0) 前置检查 ----------
echo
echo "--- 0) 前置检查 ---"
missing=0
for f in GPT_SoVITS/s2_train_v5_full.py \
         GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth \
         logs/zfy_v5/2-name2text.txt \
         logs/zfy_v5/6-name2semantic.tsv \
         logs/zfy_v5/4-cnhubert \
         logs/zfy_v5/5-wav32k; do
    if [ -e "$f" ]; then
        echo "  OK   $f"
    else
        echo "  缺失 $f"
        missing=1
    fi
done

# LFS 指针检查（.pth 只有几百字节 = 没拉下来）
p=GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth
if [ -f "$p" ]; then
    sz=$(stat -c%s "$p")
    if [ "$sz" -lt 1000000 ]; then
        echo "  !! $p 只有 $sz 字节 —— 疑似 LFS 指针未拉取"
        echo "     修：git lfs install --local && git lfs pull"
        missing=1
    fi
fi

if [ "$missing" = "1" ]; then
    echo
    echo "!! 前置检查未通过，中止"
    exit 1
fi

# 生成配置（若还没有）
CFG="tmp_full_e${EPOCHS}.json"
if [ "$RESUME" = "1" ]; then
    CFG="tmp_resume_e${EPOCHS}.json"
fi
if [ ! -f "$CFG" ]; then
    echo
    echo "--- 1) 生成配置 $CFG ---"
    python - "$EPOCHS" "$RESUME" "${RESUME_FROM:-logs/zfy_v5_full}" \
             "${GRAD_CKPT:-}" "${BATCH:-}" <<'PYEOF'
import json, os, shutil, sys
from pathlib import Path
root = Path(os.getcwd())
epochs = int(sys.argv[1])
resume = sys.argv[2] == "1"
resume_from = sys.argv[3]
grad_ckpt_arg = sys.argv[4] if len(sys.argv) > 4 else ""
batch_arg = sys.argv[5] if len(sys.argv) > 5 else ""

gsv = root / "GPT_SoVITS"
pre = gsv / "pretrained_models"
tag = f"v5_full_e{epochs}"
exp = root / "logs" / tag
wdir = root / f"SoVITS_weights_{tag}"

# grad_ckpt：本机 8GB 才需要；云上 205GB 关掉可快 1.3~1.5 倍
grad_ckpt = True
if grad_ckpt_arg == "false":
    grad_ckpt = False
batch = int(batch_arg) if batch_arg else 16

cfg = json.loads((gsv / "configs" / "s2.json").read_text(encoding="utf-8"))
cfg["train"].update({
    "batch_size": batch, "epochs": epochs,
    "pretrained_s2G": str(pre / "gsv-v5-pretrained" / "s2Gv5dev.pth"),
    "pretrained_s2D": "",
    "if_save_latest": True, "if_save_every_weights": True,
    "save_every_epoch": 1, "gpu_numbers": "0",
    "grad_ckpt": grad_ckpt, "lora_rank": 0, "fp16_run": True,
})
cfg["model"]["version"] = "v5dev"
cfg["save_weight_dir"] = wdir.name
cfg["name"] = tag
cfg["version"] = "v5dev"

if grad_ckpt is False:
    print("  grad_ckpt=False（省掉一次前向重算，通常快 1.3~1.5 倍）")
    print("  显存估算：优化器 ~6.1 GB + 激活 ~15-25 GB ≈ 21-31 GB << 205 GB")
if batch != 16:
    print(f"  batch_size={batch}")

if resume:
    # 续训：exp_dir 指向**旧实验目录**，让 latest_checkpoint_path 找到
    # 含优化器状态的 G_*.pth。注意不能指向新目录，否则找不到只能从头来。
    src = root / resume_from
    cfg["data"]["exp_dir"] = cfg["s2_ckpt_dir"] = str(src)
    print(f"  续训模式：exp_dir -> {src}")
    print(f"  （该目录下的 logs_s2_v5dev_full/G_*.pth 含 AdamW 状态）")
    # 权重仍写到**新**目录，避免覆盖旧的 e8 快照
    wdir.mkdir(parents=True, exist_ok=True)
else:
    exp.mkdir(parents=True, exist_ok=True)
    # 全新模式：清旧 checkpoint（否则脚本会 resume 后立刻退出）
    ck = exp / "logs_s2_v5dev_full"
    if ck.exists():
        stale = list(ck.glob("G_*.pth")) + list(ck.glob("D_*.pth"))
        if stale:
            shutil.rmtree(ck)
            print(f"  已清除 {len(stale)} 个旧 checkpoint（避免 resume 空跑）")
    ck.mkdir(parents=True, exist_ok=True)
    wdir.mkdir(parents=True, exist_ok=True)

# 软链特征（续训时旧目录已有，跳过）
feat = root / "logs" / "zfy_v5"
target = (root / resume_from) if resume else exp
for n in ("2-name2text.txt", "3-bert", "4-cnhubert", "5-wav32k",
          "6-name2semantic.tsv"):
    s, d = feat / n, target / n
    if s.exists() and not d.exists():
        d.symlink_to(s)
        print(f"  链接 {n}")

cfg_path = root / (f"tmp_resume_e{epochs}.json" if resume
                   else f"tmp_full_e{epochs}.json")
cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                    encoding="utf-8")
print(f"  ✓ {cfg_path}")
print(f"  权重输出: SoVITS_weights_{tag}/")
PYEOF
fi

echo
echo "--- 2) 训练量诊断 ---"
python - "$EPOCHS" "$RESUME" <<'PYEOF'
import os, re, sys
from pathlib import Path
root = Path(os.getcwd())
epochs = int(sys.argv[1])
resume = sys.argv[2] == "1"
n = len([l for l in (root/"logs"/"zfy_v5"/"6-name2semantic.tsv")
         .read_text(encoding="utf-8", errors="replace").strip().split("\n")
         if l.strip()]) - 1
spe = None
for d in sorted(root.glob("SoVITS_weights_v5_full*")):
    for w in d.glob("*_e*_s*_l0.pth"):
        m = re.match(r".*_e(\d+)_s(\d+)_l\d+\.pth$", w.name)
        if m and int(m.group(1)) > 0:
            spe = int(m.group(2)) // int(m.group(1))
            break
    if spe:
        break
if not spe:
    spe = round(n / 16 * 1.06)

V5, V2, V2S = 383.08e6, 42.39e6, 180
br = V2S / V2

if resume:
    # 续训：起点已是第 N 轮，实际新增步数 = (epochs - N) * spe
    start = 8          # 旧实验已跑到 e8
    new_steps = max(0, epochs - start) * spe
    total = epochs * spe
    print(f"  续训模式：从第 {start + 1} 轮开始，新增 {new_steps} 步")
    print(f"  总步数（含已跑的 {start*spe} 步） {total}")
else:
    new_steps = epochs * spe
    total = new_steps
    print(f"  全新模式：{epochs} 轮，共 {total} 步")

print(f"  样本 {n} 条 / 每 epoch 约 {spe} 步")
print(f"  步数·参数比 {total/V5:.2e}   相对 v2Pro {(total/V5)/br:.2f}x")
print(f"  末步 LR {1e-4 * 0.999875 ** total:.2e}")
print(f"  本次新增权重 {new_steps//spe if spe else 0} 个 x ~736 MB")
PYEOF

echo
echo "--- 3) 启动 ---"
if [ "$RESUME" = "1" ]; then
    LOG="$ROOT/train_resume_e${EPOCHS}.log"
    CFG="tmp_resume_e${EPOCHS}.json"
else
    LOG="$ROOT/train_v5_full_e${EPOCHS}.log"
    CFG="tmp_full_e${EPOCHS}.json"
fi
export PYTHONPATH="$ROOT"
# ZFY_STRICT_AUDIO=1 -> 数据加载失败立刻中断，而不是静默喂全零张量
export ZFY_STRICT_AUDIO=1

if [ "$FG" = "1" ]; then
    echo "  前台运行，日志同时写入 $LOG"
    python -s GPT_SoVITS/s2_train_v5_full.py --config "$CFG" 2>&1 | tee "$LOG"
else
    nohup python -s GPT_SoVITS/s2_train_v5_full.py \
        --config "$CFG" > "$LOG" 2>&1 &
    PID=$!
    echo "  已后台启动 PID=$PID"
    echo "  配置: $CFG"
    echo "  日志: $LOG"
    echo
    echo "  等待 30 秒看开头…"
    sleep 30
    echo "  --- 日志前 40 行 ---"
    head -40 "$LOG" 2>/dev/null || echo "  (暂无日志)"
    echo
    if [ "$RESUME" = "1" ]; then
        echo "  ⚠️ 续训验收（关键）："
        echo "     必须看到 'Loaded checkpoint ... (iteration N)' 且 N > 0"
        echo "     以及 'start training from epoch N' 且 N > 1"
        echo "     如果看到 'loaded pretrained ...s2Gv5dev.pth' 而**没有**"
        echo "     'Loaded checkpoint'，说明 resume 失败、静默从头开始了 ——"
        echo "     因为那段代码是裸 except，会吞掉一切错误。"
        echo
        grep -E "Loaded checkpoint|start training from epoch|loaded pretrained" "$LOG" 2>/dev/null | head -5
    fi
fi

echo
echo "============================================================"
echo "注意：本脚本用 ZFY_STRICT_AUDIO=1"
echo "      数据加载失败会立刻中断（不会静默喂全零张量）"
echo "      需要先跑过 patch_strict_audio.py 才生效："
echo "        python _tools/patch_strict_audio.py --root \$PWD"
echo "============================================================"
