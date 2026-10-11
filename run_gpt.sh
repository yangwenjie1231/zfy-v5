#!/usr/bin/env bash
# ============================================================
# GPT（s1）微调一键启动 —— 带**完整前置检查**
#
# ## 为什么需要这个脚本
# GPT 训练已经失败过 3 次，每次都是不同原因：
#   1. 缺 matplotlib（AR/modules/lr_schedulers.py 模块级 import）
#   2. YAML 科学计数法 -> lr_init 被读成字符串 -> TypeError: float - str
#   3. 修复没拉到云端（云端 HEAD 落后）
# 所以本脚本把这三类都变成**显式前置检查**，任一项不过就不启动训练。
#
# ## 用法
#   cd /mnt/workspace/zfy-v5-fullft-pkg
#   bash _tools/run_gpt.sh                 # 默认 LR=2e-3（与生产一致）
#   bash _tools/run_gpt.sh --lr 1e-4       # 恒定 1e-4
#   bash _tools/run_gpt.sh --lr schedule   # 真正的 warmup+cosine
#   bash _tools/run_gpt.sh --check-only    # 只检查不训练
# ============================================================
set -u

ROOT=/mnt/workspace/zfy-v5-fullft-pkg
cd "$ROOT" || { echo "!! 找不到 $ROOT"; exit 1; }

LR_MODE=""
CHECK_ONLY=0
EXTRA=""
while [ $# -gt 0 ]; do
    case "$1" in
        --lr) LR_MODE="$2"; shift ;;
        --check-only) CHECK_ONLY=1 ;;
        *) EXTRA="$EXTRA $1" ;;
    esac
    shift
done

echo "============================================================"
echo "GPT（s1）微调启动"
echo "============================================================"

fail=0

# ---------- 检查 1：脚本修复是否到位 ----------
echo
echo "--- 1) 脚本修复是否到位 ---"
n1=$(grep -c "YAML 回读校验通过" _tools/train_gpt.py 2>/dev/null || echo 0)
n2=$(grep -c "ZFY_S1_LR" _tools/patch_s1_lr.py 2>/dev/null || echo 0)
echo "  train_gpt.py   YAML 回读校验: $n1  (需 >=1)"
echo "  patch_s1_lr.py ZFY_S1_LR    : $n2  (需 >=1)"
if [ "$n1" -lt 1 ]; then
    echo "  !! train_gpt.py 缺 YAML 修复 —— 会报 TypeError: float - str"
    echo "     修：git fetch origin master && git checkout FETCH_HEAD -- _tools/train_gpt.py"
    fail=1
fi
if [ "$n2" -lt 1 ]; then
    echo "  !! patch_s1_lr.py 缺失 —— 无法控制 LR"
    fail=1
fi

# ---------- 检查 2：依赖 ----------
echo
echo "--- 2) GPT 训练依赖（三条链路里最容易漏的一条）---"
python - <<'PYEOF'
import importlib, sys
# GPT 训练链路的第三方依赖（由 trace_deps_v2.py 追踪得出）
NEED = ["torch", "numpy", "pandas", "matplotlib", "pytorch_lightning",
        "torchmetrics", "yaml", "tqdm"]
bad = []
for m in NEED:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "?")
        print(f"  OK   {m:20s} {v}")
    except Exception as e:
        print(f"  FAIL {m:20s} {type(e).__name__}: {str(e)[:50]}")
        bad.append(m)
if bad:
    print(f"\n  缺: {bad}")
    print("  修：bash _tools/install_all_deps.sh")
    sys.exit(1)
print("  ✓ 依赖齐全")
PYEOF
[ $? -ne 0 ] && fail=1

# ---------- 检查 3：语料 ----------
echo
echo "--- 3) 语料 ---"
for f in logs/zfy_v5/2-name2text.txt logs/zfy_v5/6-name2semantic.tsv; do
    if [ -f "$f" ]; then
        n=$(wc -l < "$f")
        echo "  OK   $f  $n 行"
    else
        echo "  缺   $f"
        fail=1
    fi
done
if [ -d logs/zfy_v5/3-bert ]; then
    echo "  OK   logs/zfy_v5/3-bert  $(ls -1 logs/zfy_v5/3-bert | wc -l) 个"
else
    echo "   缺 logs/zfy_v5/3-bert"; fail=1
fi

# ---------- 检查 4：底模 ----------
echo
echo "--- 4) 两条底模 ---"
for b in \
  "GPT_SoVITS/pretrained_models/s1v3.ckpt" \
  "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
do
    if [ -f "$b" ]; then
        sz=$(stat -c%s "$b")
        if [ "$sz" -lt 1000000 ]; then
            echo "  !! $b 只有 $sz 字节（LFS 指针？）"; fail=1
        else
            echo "  OK   $(basename "$b")  $((sz/1000000)) MB"
        fi
    else
        echo "  缺   $b"; fail=1
    fi
done

if [ "$fail" = "1" ]; then
    echo
    echo "============================================================"
    echo "!! 前置检查未通过，**不启动训练**"
    echo "============================================================"
    exit 1
fi

if [ "$CHECK_ONLY" = "1" ]; then
    echo
    echo "（--check-only，检查全过，未训练）"
    exit 0
fi

# ---------- 打补丁 ----------
echo
echo "--- 5) 打补丁 ---"
python _tools/patch_s1_train_rocm.py --root "$PWD" 2>&1 | tail -4
python _tools/patch_s1_lr.py --root "$PWD" 2>&1 | grep -E "已|备份|跳过" | head -5

# ---------- 启动训练 ----------
echo
echo "--- 6) 启动训练 ---"
if [ -n "$LR_MODE" ]; then
    echo "  ZFY_S1_LR=$LR_MODE"
    export ZFY_S1_LR="$LR_MODE"
else
    echo "  LR 模式：默认（恒 2e-3，与生产 zfh-e4 一致）"
fi
echo

# shellcheck disable=SC2086
python _tools/train_gpt.py --base both $EXTRA

echo
echo "============================================================"
echo "完成。产出在 GPT_weights_gptA/ 与 GPT_weights_gptB/"
echo "============================================================"
ls -la GPT_weights_gptA/ GPT_weights_gptB/ 2>/dev/null | head -20
