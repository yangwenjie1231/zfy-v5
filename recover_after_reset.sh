#!/usr/bin/env bash
# ============================================================
# 实例重置后的一键恢复（2026-10-11 版）
#
# ## 这次比上次多一个坑
# 上次云端 `_tools/` 下累积了**被暂存(staged)**的脚本
# （来自 `git checkout FETCH_HEAD -- <文件>`），它们会挡住 `git pull`：
#     Your local changes to the following files would be overwritten by merge:
#       _tools/fix_push_stuck.py
# 而 `git checkout -- <file>` **清不掉它们** —— 因为那是"从暂存区恢复到
# 工作区"，暂存区里就是改过的版本。
#
# 正确命令：`git restore --source=HEAD --staged --worktree <path>`
#          （同时重置暂存区 + 工作区）
#
# ## 本脚本顺序
#   0) 定位仓库
#   1) 重置 _tools/ 到 HEAD（**只动脚本，不碰 logs/权重/音频**）
#   2) git pull
#   3) 校验关键修复是否到位（YAML 修复 + LR 开关）
#   4) 装 ffmpeg
#   5) 装依赖（三条链路）
#   6) 打补丁（ROCm + LR）
#   7) 盘点幸存产物
#
# ## 用法
#   cd /mnt/workspace/zfy-v5-fullft-pkg
#   bash _tools/recover_after_reset.sh
# ============================================================
set -u

ROOT=/mnt/workspace/zfy-v5-fullft-pkg
cd "$ROOT" 2>/dev/null || { echo "!! 找不到 $ROOT"; exit 1; }

echo "############################################################"
echo "# 实例重置后恢复"
echo "# $(date '+%Y-%m-%d %H:%M:%S')"
echo "############################################################"

# ---------------- 1) 先装 git-lfs（**必须在任何 reset 之前**）----------------
echo
echo "============================================================"
echo "1) 装 git-lfs（**必须在任何 reset 之前**）"
echo "============================================================"
echo "  ⚠️ 为什么必须先装："
echo "     仓库 .gitattributes 把 *.pt / *.txt / *.json / *.pickle 都划进 LFS。"
echo "     没有 git-lfs 时，git 把**真实文件**与索引里的**指针**相比 -> 永远不同"
echo "     -> logs/ 下 2488 个特征全被报成 M。"
echo "     此时若执行 git restore --worktree，会把真实 .pt **换成 130 字节指针**！"
echo "     （实测：logs/zfy_v5/2-name2text.txt 的 filter=lfs）"

if command -v git-lfs >/dev/null 2>&1; then
    echo "  已有: $(git lfs version 2>&1 | head -1)"
else
    echo "  缺失 -> 安装"
    apt-get update -qq 2>&1 | tail -1 || true
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git-lfs 2>&1 | tail -3 || true
fi
if command -v git-lfs >/dev/null 2>&1; then
    git lfs install --skip-smudge 2>&1 | tail -2
    echo "  结果: $(git lfs version 2>&1 | head -1)"
else
    echo "  !! git-lfs 仍缺失 —— 后续**不会**做批量 reset（保护 logs/）"
fi

echo
echo "  装完后 status 变化（LFS 假象应大幅减少）："
n_lfs=$(git status --porcelain 2>/dev/null | wc -l)
echo "    改动数: $n_lfs"

# ---------------- 1b) 只重置**明确挡路的**文件 ----------------
echo
echo "============================================================"
echo "1b) 解开挡 pull 的文件（**只动 _tools/ 与显式白名单**）"
echo "============================================================"
echo "  说明：批量 reset 在 LFS 仓库里太危险（见上）。"
echo "        这里只重置 _tools/（纯脚本，非 LFS）+ 白名单里的补丁文件。"
echo "        logs/ SoVITS_weights*/ out_sweep/ ref/ **完全不碰**"

before=$(git status --porcelain 2>/dev/null | wc -l)
echo "  重置前改动数: $before"

# 只重置 _tools/（全是脚本，git check-attr 确认 filter=unspecified）
git restore --source=HEAD --staged --worktree _tools/ 2>&1 | grep -v "git-lfs" | head -3
echo "  _tools/ 已重置"

# 显式白名单：补丁会就地修改的**非 LFS** 文件
WHITELIST="
GPT_SoVITS/s1_train.py
GPT_SoVITS/module/data_utils.py
GPT_SoVITS/module/models.py
GPT_SoVITS/TTS_infer_pack/TTS.py
GPT_SoVITS/text/LangSegmenter/langsegmenter.py
GPT_SoVITS/text/chinese2.py
GPT_SoVITS/s2_train_v5_full.py
"
for f in $WHITELIST; do
    [ -f "$f" ] || continue
    # 二次确认：只重置 filter=unspecified（非 LFS）的
    attr=$(git check-attr filter -- "$f" 2>/dev/null | awk -F': ' '{print $NF}')
    if [ "$attr" = "lfs" ]; then
        echo "    !! 跳过 $f（是 LFS 文件，重置会毁数据）"
        continue
    fi
    git restore --source=HEAD --staged --worktree "$f" 2>/dev/null
done
echo "  白名单文件已重置"

after=$(git status --porcelain 2>/dev/null | wc -l)
echo "  重置后改动数: $after"
echo "  （剩下的多为 LFS 假象，**不影响 pull 之外的操作**）"

# ---------------- 2) pull ----------------
echo
echo "============================================================"
echo "2) git pull"
echo "============================================================"
git fetch origin master 2>&1 | grep -v "git-lfs" | tail -2
git pull --no-rebase origin master 2>&1 | grep -v "git-lfs" | tail -8
echo "  当前 HEAD: $(git log -1 --format='%h %s' | cut -c1-70)"

# ---------------- 3) 校验关键修复 ----------------
echo
echo "============================================================"
echo "3) 校验关键修复是否到位（**必看**）"
echo "============================================================"
ok=1
if [ -f _tools/train_gpt.py ]; then
    n1=$(grep -c "YAML 回读校验通过" _tools/train_gpt.py 2>/dev/null || echo 0)
    n2=$(grep -c 'f"{v:.12f}"' _tools/train_gpt.py 2>/dev/null || echo 0)
    echo "  train_gpt.py  YAML校验=$n1  科学计数法修复=$n2"
    [ "$n1" -ge 1 ] && [ "$n2" -ge 1 ] || { echo "  !! YAML 修复**未到位**"; ok=0; }
else
    echo "  !! 缺 _tools/train_gpt.py"; ok=0
fi
if [ -f _tools/patch_s1_lr.py ]; then
    n3=$(grep -c "ZFY_S1_LR" _tools/patch_s1_lr.py 2>/dev/null || echo 0)
    echo "  patch_s1_lr.py  ZFY_S1_LR=$n3"
    [ "$n3" -ge 1 ] || { echo "  !! LR 开关**未到位**"; ok=0; }
else
    echo "  !! 缺 _tools/patch_s1_lr.py"; ok=0
fi
[ "$ok" = "1" ] && echo "  ✓ 关键修复全部到位" || echo "  ✗ 有修复未到位，后面的训练会失败"

# ---------------- 4) ffmpeg ----------------
echo
echo "============================================================"
echo "4) ffmpeg 可执行文件"
echo "============================================================"
if command -v ffmpeg >/dev/null 2>&1; then
    echo "  已有: $(command -v ffmpeg)"
else
    echo "  缺失 -> 安装"
    if [ -f _tools/install_ffmpeg.sh ]; then
        bash _tools/install_ffmpeg.sh 2>&1 | tail -12
    else
        apt-get update -qq 2>&1 | tail -1 || true
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg 2>&1 | tail -2 || true
    fi
fi
echo "  结果: $(command -v ffmpeg || echo '**仍缺失**')"

# ---------------- 5) 依赖 ----------------
echo
echo "============================================================"
echo "5) 依赖（三条链路）"
echo "============================================================"
if [ -f _tools/install_all_deps.sh ]; then
    bash _tools/install_all_deps.sh 2>&1 \
        | grep -v "^WARNING: Running pip" | tail -35
else
    echo "  !! 缺 _tools/install_all_deps.sh"
fi
if [ -f _tools/fix_deps_patch.sh ]; then
    bash _tools/fix_deps_patch.sh 2>&1 \
        | grep -v "^WARNING: Running pip" | tail -25
fi

# ---------------- 6) 补丁 ----------------
echo
echo "============================================================"
echo "6) 打补丁（ROCm + LR 开关）"
echo "============================================================"
if [ -f _tools/patch_s1_train_rocm.py ]; then
    python _tools/patch_s1_train_rocm.py --root "$PWD" 2>&1 | tail -6
fi
if [ -f _tools/patch_s1_lr.py ]; then
    python _tools/patch_s1_lr.py --root "$PWD" 2>&1 | tail -8
fi

# ---------------- 7) 盘点幸存产物 ----------------
echo
echo "============================================================"
echo "7) 盘点幸存产物（/mnt/workspace 是持久的）"
echo "============================================================"
echo "  --- SoVITS 权重 ---"
for d in SoVITS_weights_v5_full_e32 SoVITS_weights_v5_full; do
    [ -d "$d" ] || continue
    n=$(ls -1 "$d"/*.pth 2>/dev/null | wc -l)
    sz=$(du -sh "$d" 2>/dev/null | cut -f1)
    echo "    $d: $n 个  $sz"
done
echo "  --- GPT 权重 ---"
for d in GPT_weights_gptA GPT_weights_gptB; do
    [ -d "$d" ] && echo "    $d: $(ls -1 "$d"/*.ckpt 2>/dev/null | wc -l) 个"
done
echo "  --- 合成音频 ---"
[ -d out_sweep ] && echo "    out_sweep: $(find out_sweep -name '*.wav' | wc -l) 个 wav，$(ls -d out_sweep/*/ 2>/dev/null | wc -l) 个目录"
echo "  --- 训练特征 ---"
[ -d logs/zfy_v5 ] && echo "    logs/zfy_v5: $(find logs/zfy_v5 -type f | wc -l) 个文件"
echo "  --- 参考资产 ---"
for f in ref/default.wav ref/ref.txt ref/zfh-e4.ckpt ref/v2pro_sovits.pth; do
    if [ -f "$f" ]; then
        sz=$(stat -c%s "$f")
        if [ "$sz" -lt 1000000 ] && [ "${f##*.}" != "txt" ]; then
            echo "    !! $f 只有 $sz 字节 —— 疑似 LFS 指针"
        else
            echo "    OK $f  $((sz/1000000)) MB"
        fi
    else
        echo "    缺失 $f"
    fi
done

echo
echo "############################################################"
echo "# 完成。下一步："
echo "#   python _tools/train_gpt.py --base both"
echo "#"
echo "# 成功标志（开头两行）："
echo "#   ✓ YAML 回读校验通过（0 处不一致）"
echo "#   [ZFY] s1 LR 模式=locked  恒定 0.002"
echo "############################################################"
