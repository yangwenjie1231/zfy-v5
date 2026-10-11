# -*- coding: utf-8 -*-
"""让 `tools/my_utils.py` 不再硬依赖 ffmpeg 与 gradio。

## 为什么需要
训练时 `data_utils.py:635 get_audio()` 会调 `load_audio(filename, 32000)`。
上游实现只走 ffmpeg：

    ffmpeg.input(file, threads=0).output("-", format="f32le", ..., ar=sr) \\
        .run(cmd=["ffmpeg", "-nostdin"], ...)

于是 ffmpeg 缺失就抛 `FileNotFoundError: [Errno 2] ... 'ffmpeg'`。
而 `data_utils.py:624` 的 `except` 会**吞掉异常**并返回全零张量：

    except:
        traceback.print_exc()
        mel = torch.zeros(100, 192)
        spec = torch.zeros(1025, 96)
        ssl  = torch.zeros(1, 768, 96)
        print("load audio or ssl error!!!!!!", audiopath)

**这是最坏的一种失败** —— 训练**不会崩**，而是拿一堆全零样本
训出一个废模型。日志里只有一行 `load audio or ssl error!!!!!!` 和一串
traceback，很容易被当成无害警告忽略。

## 关键观察：其实不需要 ffmpeg
`get_audio` 读的是 `self.path5`（`5-wav32k/` 目录），
而**那些文件本来就是 32kHz 的 wav** —— `ar=self.sampling_rate` 等于
原采样率，ffmpeg 在这里只起"解码 wav"的作用。
`soundfile` 能直接读 wav，且能给出准确采样率。

## 本补丁做三件事
  1. **load_audio 多级回退**：ffmpeg -> soundfile -> torchaudio
     并在采样率不符时用 scipy.signal.resample_poly 重采样
  2. **gradio 变成可选导入**：`my_utils.py:7` 是模块级 `import gradio as gr`，
     而整个文件只在个别函数里用 `gr.Warning`。
     gradio 出任何问题（如 `HfFolder` 缺失）都会连带**训练一起挂**。
     改为 try/except，缺失时用桩函数。
  3. **失败时抛明确异常**（而不是让 data_utils 静默返回零张量）

## 用法
    python _tools/patch_no_ffmpeg.py --root $PWD           # 打补丁
    python _tools/patch_no_ffmpeg.py --root $PWD --check   # 只检查
    python _tools/patch_no_ffmpeg.py --root $PWD --revert  # 还原
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MARK = "ZFY_NOFFMPEG_PATCH"
REL = "tools/my_utils.py"

OLD_IMPORTS = """import ctypes
import os
import sys
from pathlib import Path

import ffmpeg
import gradio as gr
import numpy as np
import pandas as pd

from tools.i18n.i18n import I18nAuto
"""

NEW_IMPORTS = """import ctypes
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---- ZFY_NOFFMPEG_PATCH ----
# ffmpeg（python 包）与 ffmpeg（可执行文件）都变成**可选**。
# 原因：训练读的是 5-wav32k/ 里已经是 32kHz 的 wav，
#       完全可以用 soundfile 直接读，不需要 ffmpeg 子进程。
#       而 ffmpeg 缺失会让 data_utils.py:624 的 except 吞掉异常、
#       静默返回全零张量 -> 训出废模型且不报错。
try:
    import ffmpeg as _zfy_ffmpeg
except Exception:  # noqa: BLE001
    _zfy_ffmpeg = None
ffmpeg = _zfy_ffmpeg

# gradio 也是可选的：上游 my_utils.py 在**模块级** import gradio，
# 而 gradio 只在个别函数里用到 gr.Warning。
# 一旦 gradio 的依赖出问题（如 huggingface_hub 缺 HfFolder），
# 整个训练会连带挂掉 —— 一个 WebUI 依赖不该阻塞训练。
try:
    import gradio as gr
except Exception as _e:  # noqa: BLE001
    print(f"[ZFY] gradio 不可用（{type(_e).__name__}: {_e}）"
          f"-> 用桩函数代替，训练不受影响")

    class _ZfyGrStub:
        \"\"\"gradio 的最小桩：只实现 my_utils 里用到的那几个入口。\"\"\"

        @staticmethod
        def Warning(msg, *a, **k):  # noqa: N802
            print(f"[ZFY][warn] {msg}")

        @staticmethod
        def Info(msg, *a, **k):  # noqa: N802
            print(f"[ZFY][info] {msg}")

        @staticmethod
        def Error(msg, *a, **k):  # noqa: N802
            print(f"[ZFY][error] {msg}")

        def __getattr__(self, name):
            def _noop(*a, **k):
                return None
            return _noop

    gr = _ZfyGrStub()

from tools.i18n.i18n import I18nAuto
"""

OLD_LOAD = '''def load_audio(file, sr):
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = clean_path(file)  # 防止小白拷路径头尾带了空格和"和回车
        if os.path.exists(file) is False:
            raise RuntimeError("You input a wrong audio path that does not exists, please fix it!")
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception:
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True)
        )  # Expose the Error
        raise RuntimeError(i18n("音频加载失败"))

    return np.frombuffer(out, np.float32).flatten()
'''

NEW_LOAD = '''def _zfy_resample(x, sr_in, sr_out):
    """多相 FIR 重采样（优先 scipy；退化时用线性插值）。

    ⚠️ 不要用线性插值当首选：其频响是 sinc^2(f/sr_out)，
       16kHz 处衰减 -3.3 dB（实测），会削掉高频气音。
    """
    if sr_in == sr_out:
        return x
    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(sr_in), int(sr_out))
        return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)
    except Exception:  # noqa: BLE001
        n = int(round(len(x) * sr_out / sr_in))
        return np.interp(np.linspace(0, len(x) - 1, n),
                         np.arange(len(x)), x).astype(np.float32)


def _zfy_load_soundfile(file, sr):
    """用 soundfile 直接读（wav/flac/ogg 原生支持）。"""
    import soundfile as sf

    x, sr_in = sf.read(str(file), always_2d=True, dtype="float32")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if sr_in != sr:
        print(f"[ZFY] load_audio: soundfile 读到 {sr_in}Hz，"
              f"重采样到 {sr}Hz  ({os.path.basename(str(file))})")
        x = _zfy_resample(x, sr_in, sr)
    return x


def _zfy_load_torchaudio(file, sr):
    """用 torchaudio 读（soundfile 失败时）。"""
    import torchaudio

    x, sr_in = torchaudio.load(str(file))
    x = x.mean(dim=0) if x.shape[0] > 1 else x[0]
    x = x.numpy().astype(np.float32)
    if sr_in != sr:
        print(f"[ZFY] load_audio: torchaudio 读到 {sr_in}Hz，"
              f"重采样到 {sr}Hz  ({os.path.basename(str(file))})")
        x = _zfy_resample(x, sr_in, sr)
    return x


def load_audio(file, sr):
    # ---- ZFY_NOFFMPEG_PATCH ----
    # 多级回退：ffmpeg -> soundfile -> torchaudio
    #
    # 为什么改：训练读的是 5-wav32k/ 里**已经是 32kHz** 的 wav，
    # ffmpeg 只被用来解码，soundfile 完全够。
    # 而 ffmpeg 缺失时上游会抛异常，被 data_utils.py:624 的
    # `except:` 吞掉 -> 静默返回全零张量 -> 训出废模型。
    file = clean_path(file)  # 防止小白拷路径头尾带了空格和"和回车
    if os.path.exists(file) is False:
        raise RuntimeError(
            f"[ZFY] 音频文件不存在: {file!r}（原样传入: {file}）")

    # 1) ffmpeg（上游原路径）
    if ffmpeg is not None:
        try:
            out, _ = (
                ffmpeg.input(file, threads=0)
                .output("-", format="f32le", acodec="pcm_f32le",
                        ac=1, ar=sr)
                .run(cmd=["ffmpeg", "-nostdin"],
                     capture_stdout=True, capture_stderr=True)
            )
            return np.frombuffer(out, np.float32).flatten()
        except Exception as e:  # noqa: BLE001
            print(f"[ZFY] load_audio: ffmpeg 失败"
                  f"（{type(e).__name__}）-> 回退 soundfile")

    # 2) soundfile
    try:
        return _zfy_load_soundfile(file, sr)
    except Exception as e:  # noqa: BLE001
        print(f"[ZFY] load_audio: soundfile 失败"
              f"（{type(e).__name__}: {e}）-> 回退 torchaudio")

    # 3) torchaudio
    try:
        return _zfy_load_torchaudio(file, sr)
    except Exception as e:  # noqa: BLE001
        # ⚠️ 这里必须**抛异常**，不能返回空数组。
        #    否则 data_utils 的 except 会静默用全零样本训练。
        raise RuntimeError(
            f"[ZFY] 三种解码方式全部失败: {file}\\n"
            f"  ffmpeg 可执行文件: "
            f"{shutil_which('ffmpeg') or '**未安装**'}\\n"
            f"  最后错误: {type(e).__name__}: {e}\\n"
            f"  解决: apt-get install -y ffmpeg   "
            f"或 pip install soundfile") from e


def shutil_which(name):
    import shutil as _sh
    return _sh.which(name)
'''


def patch(root: Path, check: bool, revert: bool) -> int:
    p = root / REL
    if not p.exists():
        print(f"  !! 找不到 {p}")
        return 1
    src = p.read_text(encoding="utf-8", errors="replace")
    bak = p.with_suffix(".py.bak_zfy_noffmpeg")

    if revert:
        if bak.exists():
            shutil.copy2(bak, p)
            print(f"  已从 {bak.name} 还原")
            return 0
        print("  未打过补丁（或备份已丢）")
        return 0

    if MARK in src:
        print(f"  已打过补丁: {p.name}")
        return 0

    ok = True
    if OLD_IMPORTS not in src:
        print("  !! 找不到 import 段（文件可能已被其他补丁改过）")
        ok = False
    if OLD_LOAD not in src:
        print("  !! 找不到 load_audio 函数体")
        ok = False
    if not ok:
        return 1

    if check:
        print("  [1/2] 将把 ffmpeg / gradio 改为可选导入")
        print("  [2/2] 将把 load_audio 改为三级回退")
        return 0

    src = src.replace(OLD_IMPORTS, NEW_IMPORTS, 1)
    src = src.replace(OLD_LOAD, NEW_LOAD, 1)
    if not bak.exists():
        shutil.copy2(p, bak)
        print(f"  备份: {bak.name}")
    p.write_text(src, encoding="utf-8")
    print("  [1/2] ffmpeg / gradio 已改为可选导入")
    print("  [2/2] load_audio 已改为三级回退（ffmpeg -> soundfile -> torchaudio）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    print("=" * 74)
    print(f"去除 ffmpeg/gradio 硬依赖: {root}")
    print("=" * 74)
    rc = patch(root, args.check, args.revert)
    print()
    if rc == 0 and not args.check and not args.revert:
        print("""
  补丁完成。**现在没有 ffmpeg 也能训练**（用 soundfile 读 5-wav32k 的 wav）。

  验证：
      python -c "import sys; sys.path.insert(0,'.'); \\
from tools.my_utils import load_audio; \\
import numpy as np; \\
x = load_audio('logs/zfy_v5/5-wav32k/<某个>.wav', 32000); \\
print('shape', x.shape, 'rms', float(np.sqrt((x**2).mean())))"

  预期：打印出非零的 shape 与 rms（不是全零）。

  ⚠️ 注意：本补丁只解决"读 wav"。若日志里出现
     `load audio or ssl error!!!!!!`
     说明 data_utils 的 except 吞掉了异常 —— 那是**上游的坏设计**，
     它会让训练拿全零样本继续跑。本补丁让 load_audio 抛明确异常，
     但 data_utils 仍会 catch。要彻底暴露需另打 patch_strict_audio.py。
""")
    elif rc != 0:
        print("  有改动未应用，见上面的 !! 行")
    print("=" * 74)
    return rc


if __name__ == "__main__":
    sys.exit(main())
