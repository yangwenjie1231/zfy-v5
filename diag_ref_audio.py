# -*- coding: utf-8 -*-
"""诊断参考音频是否与它的 ref_text 匹配 —— 这可能是"语气差/串词"的真正根因。

线索
----
合成输出里反复出现参考音频**自己的文本尾部**：
    "我扫回更一些物资" / "不想回做一些物资" / "我扫回赠一些木资"
它们都是 ref_text 结尾 "我想回赠一些物资" 的变形。
这是典型的 **prompt 泄漏**：模型没在念目标文本，而是在"补完"参考文本。

可疑点
------
数据集 README 说主参考音频时长 **6.95s**，但实测文件只有 **4.32s**。
而 ref_text 是「多亏你及时制止那些寨民袭击武陵城，为了感谢你们，
我想回赠一些物资。」共 30 字 —— 4.32s 念 30 字 = 6.9 字/秒，
远超正常语速（3-4.5 字/秒）。
=> 怀疑**参考音频被截断**，与 ref_text 不匹配。

本脚本
------
1. 用 ASR 转写参考音频，看它实际念了什么
2. 逐字对比 ref_text，定位不匹配的位置
3. 测语速，判断是否异常
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATASET = Path(r"D:\Users\A\Downloads\仿通行证\新素材\音频\庄方宜_TTS训练集")
PROD_REF = Path(r"D:\A\voiceclone\zfh-models-v2pro\ref\default.wav")

REF_TEXT = "多亏你及时制止那些寨民袭击武陵城，为了感谢你们，我想回赠一些物资。"
PROD_REF_TEXT = "再好的武器，也得朝夕相处磨合上几天。等用惯了，我再和你说说感受，好吗？"


def norm(s: str) -> str:
    s = re.sub(r"<\|[^|]*\|>", "", s or "")
    return re.sub(r"[^\u4e00-\u9fff0-9a-z]", "", s.lower())


def main() -> int:
    sys.path.insert(0, r"D:\A\zfy-v5\GPT-SoVITS")
    from funasr import AutoModel
    import librosa

    asr = AutoModel(model="iic/SenseVoiceSmall", device="cuda:0",
                    disable_update=True, cache_dir=r"D:\A\voiceclone\models")

    targets = [
        ("数据集 ref.wav (我在用)", DATASET / "ref_audio" / "ref.wav", REF_TEXT),
        ("生产 default.wav", PROD_REF, PROD_REF_TEXT),
    ]

    for label, p, expect in targets:
        if not p.exists():
            print(f"[跳过] {label}: 不存在")
            continue
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        dur = len(x) / sr
        w16 = librosa.load(str(p), sr=16000, mono=True)[0]
        r = asr.generate(input=w16, cache={}, language="zh", use_itn=False,
                         batch_size_s=60, merge_vad=True, merge_length_s=15)
        hyp = re.sub(r"<\|[^|]*\|>", "", r[0]["text"] if r else "").strip()

        ne, nh = norm(expect), norm(hyp)
        cps = len(ne) / dur if dur else 0

        print("=" * 78)
        print(f"{label}")
        print("=" * 78)
        print(f"  路径    : {p}")
        print(f"  采样率  : {sr}   时长: {dur:.2f}s")
        print(f"  ref_text: {expect}")
        print(f"  实念(ASR): {hyp}")
        print(f"  字数    : 期望 {len(ne)} 字 / 实念 {len(nh)} 字")
        print(f"  语速    : {cps:.2f} 字/秒   "
              f"({'**异常快，疑似音频被截断**' if cps > 5 else '正常范围(3-4.5)'})")

        # 前缀匹配长度：实念覆盖了期望文本的多少
        k = 0
        while k < min(len(ne), len(nh)) and ne[k] == nh[k]:
            k += 1
        print(f"  前缀一致: {k} 字 "
              f"({k/max(len(ne),1)*100:.0f}%)")
        if k < len(ne):
            print(f"  期望从第 {k+1} 字起: ...{ne[max(0,k-6):k+14]}...")
            print(f"  实念从第 {k+1} 字起: ...{nh[max(0,k-6):k+14]}...")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
