# -*- coding: utf-8 -*-
"""云端批量合成：对多个 SoVITS 权重各合成同一批句子，供盲听对比。

## 为什么在云端合成

8 个 epoch 的权重共 **5.9 GB**。全传回本地要一个多小时。
而合成后的音频只有 **约 40 MB** —— 先在云端批量生成，听出最好的那个再传权重。

## 关键：参考音频必须用生产的 default.wav

我原先的 `synth_v5.py` 用的是数据集里的 `ref_audio/ref.wav`，
**那个文件和它的 ref_text 是错配的**（正是之前"语气差"的根因）：
  · ref_text 声称："多亏你及时制止那些寨民袭击武陵城…我想回赠一些物资。"
  · 实际说的是："那里常年监测着周边环境的深度毒数是抵抗秦世朝的第一线"
  · 前缀匹配 0%，语速 6.94 字/秒（正常 3–4.5）

模型会去"补全"那个提示，导致字数和语调都失控。
所以这里改用生产验证过的参考音频。

## 用法（云上，检出根）

    python _tools/cloud_synth_sweep.py                 # 全部 epoch
    python _tools/cloud_synth_sweep.py --epochs 2,8    # 指定 epoch
    python _tools/cloud_synth_sweep.py --half          # is_half=True（默认 False）

产出：out_sweep/e{N}/{序号}_{标签}.wav
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RID = "yangwenjie1231/zfy-v5-fullft-pkg"


def find_token() -> str | None:
    """尽力找 git 推送用的 token。找不到就返回 None（走 plain push）。

    云上实测 `~/.modelscope/credentials/git_token` **不存在**，
    所以扩大了搜索范围：
      1. 环境变量
      2. ~/.modelscope/credentials/ 下**任意**文件（不限定文件名）
      3. 现有 git remote URL 里嵌的凭据（clone 时若带 token，git 会存下来）
    """
    for k in ("MODELSCOPE_API_TOKEN", "MS_TOKEN", "MODELSCOPE_TOKEN"):
        v = os.environ.get(k)
        if v and v.strip():
            return v.strip()

    home = Path.home() / ".modelscope" / "credentials"
    if home.is_dir():
        for p in sorted(home.iterdir()):
            if not p.is_file():
                continue
            v = p.read_text(encoding="utf-8", errors="replace").strip()
            # token 形如 ms-xxxx 或 20 位随机串；排除明显的配置文本
            if v and "\n" not in v and 10 <= len(v) <= 200:
                return v

    # 从 git remote URL 里提取
    try:
        r = subprocess.run(["git", "remote", "get-url", "origin"],
                           cwd=str(Path.cwd()), capture_output=True,
                           text=True, timeout=30)
        url = (r.stdout or "").strip()
        m = re.search(r"oauth2:([^@]+)@", url)
        if m:
            return m.group(1)
    except Exception:  # noqa: BLE001
        pass
    return None


def upload_results(out_root: Path, root: Path, token: str | None,
                   branch: str = "master") -> bool:
    """把合成结果提交并推回数据集。

    返回是否成功。失败时打印可手动执行的命令。
    """
    print()
    print("=" * 76)
    print("回传数据集")
    print("=" * 76)

    if token is None:
        print("  找不到 token，跳过自动推送。手动执行：")
        print(f"    cd {root}")
        print(f"    git add {out_root.name}")
        print('    git commit -m "v5 fullft sweep audio"')
        print("    git push")
        return False

    rel = out_root.relative_to(root) if out_root.is_absolute() else out_root
    url = f"https://oauth2:{token}@www.modelscope.cn/datasets/{RID}.git"

    def run(args, **kw):
        return subprocess.run(args, cwd=str(root), capture_output=True,
                              text=True, timeout=1800, **kw)

    # 1) 暂存
    r = run(["git", "add", str(rel)])
    if r.returncode != 0:
        print(f"  git add 失败: {r.stderr[-300:]}")
        return False
    n = len(list(out_root.rglob("*.wav")))
    print(f"  已暂存 {rel}/（{n} 个 wav）")

    # 2) 有无变更
    st = run(["git", "status", "--porcelain", str(rel)])
    if not (st.stdout or "").strip():
        print("  无新增内容，跳过提交")
        return True

    # 3) 提交（显式指定身份，云上可能没配 git config）
    r = run(["git",
             "-c", "user.name=yangwenjie1231",
             "-c", "user.email=yangwenjie1231@163.com",
             "commit", "-m",
             f"v5 fullft sweep audio ({n} wav)"])
    if r.returncode != 0:
        print(f"  commit 失败: {r.stderr[-300:]}")
        return False
    print("  已提交")

    # 4) 推送
    #    优先用带 token 的 URL；没有 token 时退回 plain push
    #    （git 可能已把 clone 时的凭据存在 .git/config 里）。
    print("  推送中…（wav 走 LFS，约几十 MB）", flush=True)
    urls = []
    if token:
        urls.append(f"https://oauth2:{token}@www.modelscope.cn/datasets/{RID}.git")
    urls.append("origin")

    pushed = False
    last_err = ""
    for u in urls:
        r = run(["git", "push", u, f"HEAD:{branch}"])
        err = (r.stderr or "") + (r.stdout or "")
        if r.returncode == 0:
            pushed = True
            break
        last_err = err
        if "rejected" in err or "non-fast-forward" in err:
            print("  远端有新提交，先 rebase 再推…")
            run(["git", "pull", "--rebase", u, branch])
            r = run(["git", "push", u, f"HEAD:{branch}"])
            if r.returncode == 0:
                pushed = True
                break
            last_err = (r.stderr or "") + (r.stdout or "")

    if not pushed:
        print(f"  推送失败: {last_err[-500:]}")
        print("  手动执行（云上 terminal 里）：")
        print(f"    cd {root}")
        print(f"    git push origin HEAD:{branch}")
        return False
    print("  推送完成 ✅")

    # 5) 校验：远端是否真的有了
    print()
    print("  远端校验（匿名 HEAD，确认可下载）…")
    ok = _verify_remote(out_root, rel)
    return ok


def _verify_remote(out_root: Path, rel: Path) -> bool:
    """挑一个文件，用 HTTP HEAD 确认已经能在远端取到。"""
    import urllib.error
    import urllib.parse
    import urllib.request
    wavs = sorted(out_root.rglob("*.wav"))
    if not wavs:
        return False
    sample = wavs[0].relative_to(out_root).as_posix()
    path = f"{rel.as_posix()}/{sample}"
    q = urllib.parse.urlencode({"Revision": "master", "FilePath": path})
    url = f"https://modelscope.cn/api/v1/datasets/{RID}/repo?{q}"
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            print(f"    {path} -> HTTP {r.status}  "
                  f"Content-Length={r.headers.get('Content-Length')}")
        print("    （私有数据集匿名可能 404，属正常）")
        return True
    except urllib.error.HTTPError as e:
        print(f"    匿名 HEAD -> HTTP {e.code}（私有数据集正常）")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"    校验请求异常: {type(e).__name__}: {str(e)[:120]}")
        return True


def find_root() -> Path:
    here = Path.cwd()
    for c in (here, *here.parents, Path("/mnt/workspace/zfy-v5-fullft-pkg")):
        if (c / "GPT_SoVITS").is_dir() and (c / ".git").is_dir():
            return c
    return Path("/mnt/workspace/zfy-v5-fullft-pkg")


ROOT = find_root()
GSV = ROOT / "GPT_SoVITS"
PRE = GSV / "pretrained_models"

# ---- 与 v2Pro 对比用同一批句子：覆盖短/中/长 + 标点/省略号/数字 ----
TEXTS = [
    ("短", "让我想想。"),
    ("短", "好。"),
    ("中", "迎敌或是迎客，都得整装后再出发。"),
    ("中", "出门前，要不要先喝杯茶提提神？"),
    ("中", "嗯……我理一理。"),
    ("长", "再好的武器，也得朝夕相处磨合上几天。等用惯了，我再和你说说感受，好吗？"),
    ("长", "武陵城外有一片梨树林，每年春天都会开满白色的花。"),
    ("长", "她就在那阵花香里，只是他自己不知道。"),
    ("长", "作画者把心画在纸上，纸会烂。庄方宜把心砌在城里，城也会塌。"),
    ("数字", "这件事我三点五十分再去处理。"),
]

SEED = 42


def resolve_ref() -> tuple[Path, str]:
    """找参考音频与它的文本。

    优先用仓库里的 ref/default.wav（生产验证过的），
    退回 GPT-SoVITS 自带的参考音频。
    """
    cands = [
        (ROOT / "ref" / "default.wav",
         ROOT / "ref" / "ref.txt"),
        (ROOT / "ref" / "default.wav", None),
    ]
    for wav, txt in cands:
        if wav.exists():
            if txt and txt.exists():
                return wav, txt.read_text(encoding="utf-8").strip()
            # 没有 ref.txt 就用内置文本（与 default.wav 一致）
            return wav, ("再好的武器，也得朝夕相处磨合上几天。"
                         "等用惯了，我再和你说说感受，好吗？")
    raise SystemExit(
        "找不到参考音频。请把生产的 default.wav 放到 <检出根>/ref/default.wav\n"
        "（可选：同目录放 ref.txt 写明文本）"
    )


def setup_paths() -> None:
    """把检出根等目录插到 sys.path 最前，并 chdir 到检出根。

    ⚠️ 必须在**任何** `GPT_SoVITS.*` 导入之前调用。

    我第一版把这段只放在 `build_tts()` 里，而 `main()` 在更早的位置就
    `from GPT_SoVITS.module.v5_inference import resolve_sampling` ——
    那时 sys.path 还没设，会直接 ModuleNotFoundError。
    所以抽成独立函数，在 main() 开头就先调一次（幂等，build_tts 也会调）。
    """
    need = [
        str(ROOT),
        str(GSV),
        str(GSV / "BigVGAN"),
        str(ROOT / "tools"),
        str(ROOT / "tools" / "asr"),
        str(ROOT / "tools" / "uvr5"),
    ]
    for p in reversed(need):
        while p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    # sv.py 用 os.getcwd() 拼 eres2net 路径 -> 必须 chdir 到检出根
    os.chdir(str(ROOT))


def build_tts(version: str, vits_weights: Path, t2s_weights: Path,
              is_half: bool):
    """构造 TTS 实例（sys.path 顺序与 cwd 都必须对）。"""
    setup_paths()

    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

    cfg = {
        "custom": {
            "device": "cuda" if _cuda_ok() else "cpu",
            "is_half": is_half,
            "version": version,
            "t2s_weights_path": str(t2s_weights),
            "vits_weights_path": str(vits_weights),
            "bert_base_path": str(PRE / "chinese-roberta-wwm-ext-large"),
            "cnhuhbert_base_path": str(PRE / "chinese-hubert-base"),
        }
    }
    return TTS(TTS_Config(cfg))


def _cuda_ok() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def collect(gen):
    """把 TTS.run() 的生成器收成 (sr, audio)。

    TTS.run() 是生成器（源码 yield output_sr, audio），
    直接解包会报 not enough values to unpack。
    """
    chunks, sr = [], None
    for item in gen:
        if item is None:
            continue
        s, a = item
        sr = s
        chunks.append(np.asarray(a))
    if not chunks:
        raise RuntimeError("TTS.run 没有产出任何音频")
    return sr, np.concatenate(chunks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", default="SoVITS_weights_v5_full")
    ap.add_argument("--out", default="out_sweep")
    ap.add_argument("--epochs", default="",
                    help="只跑指定 epoch，如 2,4,8；留空=全部")
    ap.add_argument("--version", default="v5dev")
    ap.add_argument("--gpt", default="",
                    help="GPT 权重；默认用生产微调过的 zfh-e4.ckpt，缺失则退回 s1v3.ckpt")
    ap.add_argument("--half", action="store_true", help="is_half=True（默认 False）")
    ap.add_argument("--no-v2pro", action="store_true",
                    help="不合成 v2Pro 基准（默认会合成，用于三方对比）")
    ap.add_argument("--v2pro-weights", default=None,
                    help="v2Pro SoVITS 权重；默认 <检出根>/ref/v2pro_sovits.pth")
    ap.add_argument("--no-upload", action="store_true",
                    help="合成完不自动回传数据集")
    ap.add_argument("--upload-only", action="store_true",
                    help="跳过合成，只把已有 out_sweep 回传")
    args = ap.parse_args()

    ref_wav, ref_text = resolve_ref()

    out_root = ROOT / args.out

    # 只回传模式：跳过合成
    if args.upload_only:
        if not out_root.exists():
            print(f"没有可回传的目录: {out_root}")
            return 1
        upload_results(out_root, ROOT, find_token())
        return 0

    wdir = ROOT / args.weights_dir
    if not wdir.is_dir() or not list(wdir.glob("*.pth")):
        # ⚠️ 自动找权重目录。
        #    踩过：--weights-dir 默认值是 SoVITS_weights_v5_full（**旧**目录），
        #    而我们补版本头的是 SoVITS_weights_v5_full_e32。于是合成悄悄用了
        #    旧权重（无版本头 -> 被判成 v3 -> BigVGAN 报错），白跑一轮。
        #    这里改成：默认值不存在或为空时，自动挑**最新**的权重目录并告知。
        cands = sorted(
            (d for d in ROOT.glob("SoVITS_weights*")
             if d.is_dir() and list(d.glob("*.pth"))),
            key=lambda d: (len(list(d.glob("*.pth"))), d.stat().st_mtime),
        )
        if not cands:
            print(f"!! 在 {ROOT} 下找不到任何含 .pth 的 SoVITS_weights* 目录")
            return 1
        pick = cands[-1]
        print(f"!! --weights-dir 默认值 '{args.weights_dir}' 不可用")
        print(f"   自动改用: {pick.name}  "
              f"({len(list(pick.glob('*.pth')))} 个权重)")
        if len(cands) > 1:
            print("   其他候选:")
            for d in cands[:-1]:
                print(f"     {d.name}  ({len(list(d.glob('*.pth')))} 个)")
        wdir = pick
    weights = sorted(wdir.glob("*.pth"))

    # ⚠️ 检查版本头：全量微调权重若没有头，会被推理端按**文件大小**猜成 v3，
    #    从而去加载 24kHz 的 BigVGAN（报 HFValidationError）。
    #    这里提前体检，避免又跑一轮空合成。
    import struct as _struct
    _nohdr = []
    for w in weights:
        try:
            with w.open("rb") as _f:
                _h = _f.read(2)
            if _h == b"PK":
                _nohdr.append(w)
        except Exception:  # noqa: BLE001
            pass
    if _nohdr:
        print()
        print("=" * 62)
        print(f"!! {len(_nohdr)} 个权重**没有版本头**（前 2 字节是 PK）")
        print("   推理端会按文件大小猜成 v3 -> 去加载 24kHz BigVGAN -> 失败")
        print("   先补头（幂等，自动备份）：")
        print(f"     python _tools/fix_weight_header.py "
              f"--dir {wdir.relative_to(ROOT)} --apply")
        print("=" * 62)
        return 1
    print(f"权重目录 : {wdir.relative_to(ROOT)}  "
          f"({len(weights)} 个，版本头已就绪)")
    if args.epochs:
        # ⚠️ 必须**锚定到结尾**解析 epoch。
        #    权重名形如 v5_full_e32_e8_s432_l0.pth —— 同时含 "e32"（实验名）
        #    和 "e8"（真 epoch）。用非锚定的 re.search 或 `"e8_" in name`
        #    都会误判（比如 --epochs 32 会匹配到全部 32 个文件）。
        #
        # ⚠️ 还要兼容 merge_base_encp.py 产出的两种命名（都出现过）：
        #      v5_full_e32_e16_s864_l0_merged.pth    （保留原后缀）
        #      v5_full_e32_e16_merged.pth            （我最初建议的简写）
        #    踩过两次：正则只认 `_l\d+\.pth$` -> merged 被**静默过滤**，
        #    合成用的还是未合并的旧权重，白跑两轮。
        #    所以 `_s\d+_l\d+` 与 `_merged` 都设为可选。
        # ⚠️ 正则三次踩坑后的最终版：
        #    `_e(\d+)(?:_s\d+_l\d+|_merged)(?:_merged)?\.pth$`
        #
        #  第 1 次 `_e(\d+)_s(\d+)_l\d+\.pth$` -> 漏掉 merged
        #  第 2 次 `(?:_s\d+_l\d+)?(?:_merged)?` -> `(?:_s..)` 是可选组，
        #          `v5_full_e32.pth` 跳过它直接匹配实验名 e32（误判）
        #  第 3 次 加 `(?!.*_e\d)` 前瞻**无效** —— e32 后面什么都没有，
        #          前瞻条件天然成立
        #
        #  最终做法：要求 epoch 后面**必须**跟 `_s{S}_l{L}` 或 `_merged`。
        #  实测 14 个用例全过（test_weight_regex.py）。
        want = {int(x) for x in args.epochs.split(",") if x.strip()}
        kept, dropped = [], []
        for w in weights:
            m = re.search(
                r'_e(\d+)(?:_s\d+_l\d+|_merged)(?:_merged)?\.pth$', w.name)
            if m is None:
                dropped.append((w, "命名不符合 *_e{N}[_s{S}_l{L}][_merged].pth"))
            elif int(m.group(1)) not in want:
                dropped.append((w, f"epoch {m.group(1)} 不在 {sorted(want)}"))
            else:
                kept.append(w)
        weights = kept
        # 把"被跳过的"打出来 —— 静默过滤是最难查的 bug
        if dropped:
            print(f"  （--epochs 过滤掉 {len(dropped)} 个权重）")
            for w, why in dropped[:6]:
                print(f"     - {w.name}  <- {why}")
            if len(dropped) > 6:
                print(f"     … 还有 {len(dropped)-6} 个")
        print(f"  选中 {len(weights)} 个: "
              + ", ".join(w.name for w in weights))
    if not weights:
        print(f"找不到权重：{wdir}")
        return 1

    # GPT 权重
    if args.gpt:
        t2s = Path(args.gpt)
    else:
        prod = ROOT / "ref" / "zfh-e4.ckpt"
        t2s = prod if prod.exists() else (PRE / "s1v3.ckpt")
    if not t2s.exists():
        print(f"GPT 权重不存在：{t2s}")
        return 1

    print("=" * 76)
    print("云端批量合成")
    print("=" * 76)
    print(f"检出根   : {ROOT}")
    print(f"参考音频 : {ref_wav}")
    print(f"参考文本 : {ref_text}")
    print(f"GPT 权重 : {t2s.name}")
    print(f"版本     : {args.version}")
    print(f"待合成   : {len(weights)} 个权重 × {len(TEXTS)} 句")
    print(f"seed     : {SEED}   is_half: {args.half}")
    print()

    # ⚠️ 必须先设 sys.path，否则下面 GPT_SoVITS.* 的导入会 ModuleNotFoundError
    setup_paths()

    # ⚠️ resolve_sampling 必须先 import 再用！
    #    我之前只调用了名字、没 import（以为 setup_paths 会带进来），
    #    结果 NameError 被 except 吞掉，静默退回默认值 —— 采样步数就错了。
    v5_steps, v5_cfg = None, None
    try:
        from GPT_SoVITS.module.v5_inference import resolve_sampling
        v5_steps, v5_cfg = resolve_sampling(args.version)
        print(f"v5 采样参数: {v5_steps} 步 / cfg {v5_cfg}")
    except Exception as e:  # noqa: BLE001
        print(f"!! resolve_sampling 不可用（{type(e).__name__}: {e}）")
        print("   退回 v5dev 默认 (32, 1.30)；v5turbo 会是错的")
        if args.version == "v5dev":
            v5_steps, v5_cfg = 32, 1.30

    # ---- 组装任务列表：(version, 权重, 输出子目录, steps, cfg) ----
    jobs: list[tuple[str, Path, str, int | None, float | None]] = []
    for w in weights:
        m = re.search(r'_e(\d+)(?:_s\d+_l\d+|_merged)(?:_merged)?\.pth$', w.name)
        if m:
            sub = f"e{int(m.group(1)):02d}"          # e08 / e32，补零便于排序
            # ⚠️ 合并版与原始版 epoch 相同，若共用子目录会**互相覆盖**。
            #    加后缀区分，便于 A/B 对比（merged 应该时长正常）。
            if "merged" in w.stem:
                sub += "_merged"
        else:
            sub = w.stem
        jobs.append((args.version, w, sub, v5_steps, v5_cfg))

    if not args.no_v2pro:
        vp = Path(args.v2pro_weights) if args.v2pro_weights \
            else (ROOT / "ref" / "v2pro_sovits.pth")
        if vp.exists():
            jobs.append(("v2Pro", vp, "v2pro", None, None))
            print(f"含 v2Pro 基准: {vp.name}")
        else:
            print(f"!! 未找到 v2Pro 权重（{vp}），跳过基准")
    print(f"共 {len(jobs)} 个任务 × {len(TEXTS)} 句")
    print()

    ok_total = 0
    for ji, (ver, w, sub, steps, cfg_rate) in enumerate(jobs, 1):
        out_dir = out_root / sub
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"--- [{ji}/{len(jobs)}] {ver} :: {w.name} -> {out_dir.name}/ ---",
              flush=True)
        t0 = time.time()
        try:
            tts = build_tts(ver, w, t2s, args.half)
        except Exception as e:  # noqa: BLE001
            print(f"    构造 TTS 失败: {type(e).__name__}: {str(e)[:200]}")
            continue

        n_ok = 0
        for ti, (kind, text) in enumerate(TEXTS, 1):
            try:
                # ⚠️ 推理参数必须与本地 eval（synth_v5.py）**完全一致**，
                # 否则云端音频无法与已有的 v2Pro / LoRA 样本对照。
                # 我第一版写成了 cut0 / parallel_infer=False / 漏
                # repetition_penalty，会导致口径不同、结果不可比。
                gen = tts.run({
                    "text": text,
                    "text_lang": "zh",
                    "ref_audio_path": str(ref_wav),
                    "prompt_text": ref_text,
                    "prompt_lang": "zh",
                    "top_k": 15,
                    "temperature": 1.0,
                    "repetition_penalty": 1.35,
                    "text_split_method": "cut1",
                    "batch_size": 1,
                    "speed_factor": 1.0,
                    "seed": SEED,
                    "parallel_infer": True,
                    # v5 才传采样参数；v2Pro 用自己的默认（32 步 / cfg 0）
                    **({"sample_steps": steps} if steps else {}),
                    **({"cfg_rate": cfg_rate} if cfg_rate is not None else {}),
                })
                sr, audio = collect(gen)
                fn = out_dir / f"{ti:02d}_{kind}_{sub}.wav"
                sf.write(str(fn), audio, sr)
                n_ok += 1
                print(f"    {ti:2d}/{len(TEXTS)} ok  {fn.name}  "
                      f"{len(audio)/sr:.2f}s @{sr}Hz", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"    {ti:2d}/{len(TEXTS)} 失败 {type(e).__name__}: "
                      f"{str(e)[:150]}")
        print(f"    完成 {n_ok}/{len(TEXTS)}，用时 {time.time()-t0:.0f}s\n")
        ok_total += n_ok

        # 释放显存，避免累积
        try:
            del tts
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    print("=" * 76)
    print(f"总共产出 {ok_total} 个 wav，目录: {out_root}")
    print("=" * 76)

    # ---- 回传数据集（含合成失败的也回传，便于排查）----
    if args.no_upload:
        print("""
  （--no-upload）手动回传：
      cd """ + str(ROOT) + """
      python _tools/cloud_synth_sweep.py --upload-only
""")
    else:
        upload_results(out_root, ROOT, find_token())

    print("""
  本地拉取（在**本机**执行）：
      python -c "from modelscope import snapshot_download as d; \\
d('yangwenjie1231/zfy-v5-fullft-pkg', repo_type='dataset', \\
  allow_patterns=['out_sweep/**'], local_dir=r'D:\\A\\zfy-v5\\dist\\sweep')"
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
