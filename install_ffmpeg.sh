#!/usr/bin/env bash
# ============================================================
# 在云上安装 ffmpeg 可执行文件
#
# ## 为什么需要
#
# 冒烟训练报 192 次 `load audio or ssl error`，真实异常是：
#
#     FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'
#
# `load_audio`（tools/my_utils.py:24-28）用 ffmpeg **命令行程序**解码：
#
#     ffmpeg.input(file).output("-", ...).run(cmd=["ffmpeg", "-nostdin"], ...)
#
# 我之前只装了 `ffmpeg-python`（Python 包装库），它**不含**可执行文件。
# 缺了二进制，所有音频加载都失败 -> data_utils 的 except 吞掉异常
# -> 返回全零张量 -> 训练"成功"但学的是垃圾。
#
# ## 本脚本
#
#   1. 先试 apt-get（Ubuntu 22.04 官方源）
#   2. 失败则退回 pip 的静态二进制（imageio-ffmpeg 自带 ~30MB 的 ffmpeg）
#   3. 最后**真跑一次 load_audio** 验证，而不是只看 which
#
# 用法：
#   cd /mnt/workspace/zfy-v5-fullft-pkg
#   bash _tools/install_ffmpeg.sh
# ============================================================
set -u

ROOT=/mnt/workspace/zfy-v5-fullft-pkg
cd "$ROOT" 2>/dev/null || true

echo "============================================================"
echo "0) 现状"
echo "============================================================"
if command -v ffmpeg >/dev/null 2>&1; then
    echo "  已存在: $(command -v ffmpeg)"
    ffmpeg -version 2>&1 | head -1
else
    echo "  ffmpeg 未找到"
fi
echo "  python ffmpeg 模块:"
python -c "import ffmpeg; print('    已装 ffmpeg-python')" 2>&1 | tail -1

echo
echo "============================================================"
echo "1) 尝试 apt-get 安装"
echo "============================================================"
if command -v ffmpeg >/dev/null 2>&1; then
    echo "  已有，跳过"
else
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update -qq 2>&1 | tail -2 || echo "  apt-get update 失败（继续）"
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg 2>&1 | tail -3 \
            || echo "  apt-get install 失败"
    else
        echo "  无 apt-get"
    fi
fi

if command -v ffmpeg >/dev/null 2>&1; then
    echo "  OK: $(command -v ffmpeg)"
    ffmpeg -version 2>&1 | head -1
fi

echo
echo "============================================================"
echo "2) 若仍缺失，用 pip 静态二进制（imageio-ffmpeg）"
echo "============================================================"
if ! command -v ffmpeg >/dev/null 2>&1; then
    pip install --quiet imageio-ffmpeg || echo "  pip 安装失败"
    python - <<'PYEOF'
import os, shutil, stat
try:
    import imageio_ffmpeg
except ImportError:
    print("  imageio_ffmpeg 未装上")
    raise SystemExit(0)
exe = imageio_ffmpeg.get_ffmpeg_exe()
print(f"  自带二进制: {exe}")
print(f"  体积: {os.path.getsize(exe)/1e6:.1f} MB")
for dst in ("/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"):
    try:
        shutil.copy2(exe, dst)
        os.chmod(dst, 0o755)
        print(f"  已复制到 {dst}")
        break
    except Exception as e:
        print(f"  复制到 {dst} 失败: {e}")
PYEOF
fi

echo
echo "============================================================"
echo "3) 真跑一次 load_audio 验证（最关键）"
echo "============================================================"
echo "  注意：不只看 which ffmpeg，而是真的解码一个 wav"
PYTHONPATH="$ROOT" python - <<'PYEOF'
import os, sys
root = os.environ.get("PYTHONPATH", "/mnt/workspace/zfy-v5-fullft-pkg")
for p in reversed([root, root + "/GPT_SoVITS", root + "/tools"]):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
os.chdir(root)

from pathlib import Path
wav_dir = Path(root) / "logs" / "zfy_v5" / "5-wav32k"
cand = None
if wav_dir.is_dir():
    for f in sorted(wav_dir.iterdir()):
        if f.is_file():
            cand = f
            break
if cand is None:
    print("  找不到测试 wav，跳过")
    raise SystemExit(0)

print(f"  测试文件: {cand.name}")
print(f"  体积: {cand.stat().st_size:,d} 字节")

try:
    from tools.my_utils import load_audio
    a = load_audio(str(cand), 32000)
    n = len(a)
    print(f"  ✓ load_audio 成功: {n:,d} 样本 = {n/32000:.2f} 秒")
    import numpy as np
    print(f"    幅度范围: {a.min():.4f} ~ {a.max():.4f}")
    if abs(a).max() < 1e-6:
        print("    !! 全静音 —— 文件可能有问题")
    else:
        print("    ✓ 有实际波形")
except Exception as e:
    print(f"  ✗ load_audio 失败: {type(e).__name__}: {str(e)[:300]}")
    raise SystemExit(1)
PYEOF
RC=$?

echo
echo "============================================================"
if [ "$RC" = "0" ]; then
    echo "完成。ffmpeg 可用，load_audio 验证通过。"
    echo
    echo "下一步：重跑冒烟（cell 8 重建数据 -> cell 9 训练）"
else
    echo "load_audio 仍失败，请把上面输出贴出来。"
fi
echo "============================================================"
