#!/usr/bin/env bash
# ============================================================
# 新实例恢复：一条命令把环境恢复到可训练/可推理
#
# ## 背景
#
# 魔搭 Notebook 的存储分两层：
#   · /mnt/workspace          **持久化** —— 仓库、权重、logs 都在
#   · pip 包 / apt 包 / 二进制  容器文件系统，**新实例就没了**
#
# 所以新开实例后：仓库和 checkpoint 还在，但 ffmpeg 和所有 pip 依赖都没了。
#
# ## 本脚本依次做
#   1. 确认仓库状态（并提示 git pull）
#   2. 装 ffmpeg 可执行文件        （apt 装的，不持久）
#   3. 装三条链路的 pip 依赖        （pip 装的，不持久）
#   4. 逐条 import 验证
#   5. 盘点**幸存**的训练产物（权重 / logs / 特征）
#
# ## 用法
#   cd /mnt/workspace/zfy-v5-fullft-pkg
#   git pull                       # 先拿最新脚本
#   bash _tools/recover_instance.sh
# ============================================================
set -u

ROOT=/mnt/workspace/zfy-v5-fullft-pkg

echo "############################################################"
echo "# 新实例恢复"
echo "# $(date '+%Y-%m-%d %H:%M:%S')"
echo "############################################################"

if [ ! -d "$ROOT" ]; then
    echo "!! 找不到 $ROOT"
    echo "   如果 /mnt/workspace 也丢了，需要先重新 clone："
    echo "     cd /mnt/workspace"
    echo "     git lfs install"
    echo "     git clone https://oauth2:<token>@www.modelscope.cn/datasets/\\"
    echo "yangwenjie1231/zfy-v5-fullft-pkg.git zfy-v5-fullft-pkg"
    exit 1
fi

cd "$ROOT"

echo
echo "============================================================"
echo "1) 仓库状态"
echo "============================================================"
echo "  路径: $ROOT"
if [ -d .git ]; then
    echo "  提交: $(git log -1 --format='%h %s' 2>/dev/null | cut -c1-60)"
    echo "  分支: $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
    n=$(git ls-files 2>/dev/null | wc -l)
    echo "  跟踪文件: $n"
    echo
    echo "  检查是否有更新（不影响已有权重）…"
    if git fetch --depth 1 origin master >/dev/null 2>&1; then
        behind=$(git rev-list --count HEAD..FETCH_HEAD 2>/dev/null || echo "?")
        echo "  落后 origin/master: $behind 个提交"
        if [ "$behind" != "0" ] && [ "$behind" != "?" ]; then
            echo "  -> 执行 git pull"
            git pull --ff-only origin master 2>&1 | tail -5
        fi
    else
        echo "  （fetch 失败，跳过；网络问题时不影响后续安装）"
    fi
else
    echo "  !! 不是 git 仓库"
fi

echo
echo "============================================================"
echo "2) ffmpeg 可执行文件（apt 装的，新实例会丢）"
echo "============================================================"
if command -v ffmpeg >/dev/null 2>&1; then
    echo "  已有: $(command -v ffmpeg)"
    ffmpeg -version 2>&1 | head -1
else
    echo "  缺失 -> 调用 install_ffmpeg.sh"
    if [ -f _tools/install_ffmpeg.sh ]; then
        bash _tools/install_ffmpeg.sh 2>&1 | tail -25
    else
        echo "  找不到 _tools/install_ffmpeg.sh，直接 apt 装"
        apt-get update -qq 2>&1 | tail -2 || true
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg 2>&1 | tail -3 || true
    fi
fi
echo "  结果: $(command -v ffmpeg || echo '**仍缺失**')"

echo
echo "============================================================"
echo "3) pip 依赖（三条链路）"
echo "============================================================"
if [ -f _tools/install_all_deps.sh ]; then
    # 注意：不要把输出截得太短，否则看不出哪一步失败
    bash _tools/install_all_deps.sh 2>&1 | grep -v "^WARNING: Running pip" | tail -70
else
    echo "  !! 找不到 _tools/install_all_deps.sh"
    echo "     先 git pull；若仍没有，说明仓库是旧版"
    exit 1
fi

echo
echo "============================================================"
echo "3b) 依赖修复补丁"
echo "============================================================"
echo "  处理两件 install_all_deps.sh 覆盖不到的事："
echo "    · huggingface_hub 被后续包顶回新版 -> gradio 缺 HfFolder"
echo "    · 相对导入链漏掉的包（fast_langdetect / split_lang）"
if [ -f _tools/fix_deps_patch.sh ]; then
    bash _tools/fix_deps_patch.sh 2>&1 | grep -v "^WARNING: Running pip" | tail -40
else
    echo "  !! 找不到 _tools/fix_deps_patch.sh（需 git pull）"
fi

echo
echo "============================================================"
echo "4) 盘点幸存的训练产物"
echo "============================================================"
echo "  （/mnt/workspace 是持久的，这些应该都还在）"
echo
echo "  --- SoVITS 全量微调权重 ---"
found_w=0
for d in SoVITS_weights_v5_full* SoVITS_weights_v5 SoVITS_weights_v5_r128; do
    [ -d "$d" ] || continue
    n=$(ls -1 "$d"/*.pth 2>/dev/null | wc -l)
    sz=$(du -sh "$d" 2>/dev/null | cut -f1)
    echo "    $d: $n 个  $sz"
    found_w=1
done
[ "$found_w" = "0" ] && echo "    （无）"

echo
echo "  --- GPT 权重 ---"
for d in GPT_weights*; do
    [ -d "$d" ] || continue
    n=$(ls -1 "$d"/*.ckpt 2>/dev/null | wc -l)
    echo "    $d: $n 个"
done

echo
echo "  --- 训练特征（logs/）---"
for d in logs/zfy_v5 logs/v5_full_e* logs/smoke logs/gpt*; do
    [ -d "$d" ] || continue
    n=$(find "$d" -type f 2>/dev/null | wc -l)
    echo "    $d: $n 个文件"
done

echo
echo "  --- 训练日志 ---"
for f in train_v5_full_e*.log train_*.log; do
    [ -f "$f" ] || continue
    echo "    $f  ($(wc -l < "$f") 行)"
done

echo
echo "  --- 参考/生产资产 ---"
for f in ref/default.wav ref/ref.txt ref/zfh-e4.ckpt ref/v2pro_sovits.pth; do
    if [ -f "$f" ]; then
        sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
        if [ "$sz" -lt 2000 ]; then
            echo "    !! $f 只有 $sz 字节 —— 疑似 LFS 指针未拉取"
        else
            echo "    OK $f  $((sz/1000000)) MB"
        fi
    else
        echo "    缺失 $f"
    fi
done

echo
echo "============================================================"
echo "完成。可做的事："
echo "============================================================"
cat <<'EOF'
  SoVITS 全量微调（e32）:
      python -s GPT_SoVITS/s2_train_v5_full.py --config tmp_full_e32.json

  GPT 微调（A/B 两条底模）:
      python _tools/patch_s1_train_rocm.py --root $PWD
      python _tools/train_gpt.py --base both --dry-run
      python _tools/train_gpt.py --base both

  云端合成（含 v2Pro 基准）:
      python _tools/cloud_synth_sweep.py --epochs 8,16,24,32

  权重回传数据集:
      python _tools/cloud_push_weights.py --list
EOF
echo "============================================================"
