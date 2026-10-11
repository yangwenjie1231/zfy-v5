# -*- coding: utf-8 -*-
"""把 notebook 的 section 3（装依赖）整体替换成"自验证版"。

## 为什么要重写

原 section 3 是**按用途裁剪**的（只为 SoVITS 训练），导致连环踩坑：

  1. 漏 peft / pytorch_lightning      -> 推理报 ModuleNotFoundError
  2. 漏 matplotlib / gradio           -> GPT 训练 / SoVITS 重训崩
  3. 漏 fast_langdetect / split_lang  -> 推理崩（相对导入链没追踪到）
  4. transformers 与 gradio 对 hf-hub 的要求**互斥**，没识别出来：
       transformers 5.x -> 要 hf-hub >= 1.x   -> is_offline_mode 缺失
       gradio 4.44      -> 要 hf-hub < 0.37   -> HfFolder 缺失

新 section 3 的策略：
  · **直接调仓库里的 install_all_deps.sh + fix_deps_patch.sh**
    （单一事实来源，不再在 notebook 里重复一份清单）
  · 装完**真实 import 三条链路**验证（GPT 训练 / SoVITS 训练 / 推理）
  · 明确说明三方版本互斥关系与正确组合

用法：
    python rewrite_notebook_section3.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = Path(r"D:\A\zfy-v5\make_notebook.py")
src = P.read_text(encoding="utf-8")

START = "# ---------------------------------------------------------------- 3\n"
END = "# ---------------------------------------------------------------- 4\n"

i = src.find(START)
j = src.find(END)
if i < 0 or j < 0 or j <= i:
    print(f"找不到 section 3 边界: i={i} j={j}")
    sys.exit(1)

NEW = '''# ---------------------------------------------------------------- 3
cells.append(md(r"""
## 3. 装依赖 ⚠️ 最容易踩坑的一步

### 核心原则：**不要按用途裁剪依赖清单**

我踩过四次，全是同一个根因 —— 按"当前要做什么"裁清单：

| 次 | 漏的包 | 报错 |
|---|---|---|
| 1 | `peft` / `pytorch_lightning` | 推理 `ModuleNotFoundError` |
| 2 | `matplotlib` | **GPT 训练**崩（`AR/modules/lr_schedulers.py:6`） |
| 2 | `gradio` | **SoVITS 重训**崩（`tools/my_utils.py:7`） |
| 3 | `fast_langdetect` / `split_lang` | 推理崩（相对导入链没追踪到） |

所以本格**直接调用仓库里的脚本**，不再在 notebook 里重复一份清单：

- `_tools/install_all_deps.sh` —— 三条链路一次装全
- `_tools/fix_deps_patch.sh` —— 装后验证 + 强制修复

### ⚠️ 一个必须知道的版本互斥

| 包 | 要求 `huggingface_hub` | 不满足时 |
|---|---|---|
| **transformers 5.x** | **>= 1.x** | `cannot import name 'is_offline_mode'` |
| **gradio 4.44** | **< 0.37**（它 `import HfFolder`） | `cannot import name 'HfFolder'` |

**同一个 hf-hub 版本不可能同时满足两者** —— 所以必须把
`transformers` 钉到 `<5`，而不是去升 `huggingface_hub`。

已验证可用的组合（本机三条链路都能 import）：

```
transformers 4.57.6  +  huggingface_hub 0.36.2  +  gradio 4.44.1
```

> 注：**SoVITS 训练本身不需要 transformers**，只有推理和 GPT 训练需要。
> 所以即使 transformers 没修好，也可以先跑 SoVITS 训练。

**下面这格是 Python 代码**（内部用 `subprocess` 调 shell），直接运行。
"""))
cells.append(code(r\'\'\'
import os, subprocess, sys
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "GPT_SoVITS").is_dir():
    for c in (Path("/mnt/workspace/zfy-v5-fullft-pkg"), Path("/mnt/workspace")):
        if (c / "GPT_SoVITS").is_dir():
            ROOT = c
            break
os.chdir(ROOT)


def sh(script: Path, tail: int = 60, timeout: int = 3600):
    """跑仓库里的 shell 脚本，过滤 pip 的 root 警告。"""
    if not script.exists():
        print(f"  !! 找不到 {script}（需要先 git pull）")
        return 1
    print(f"\\n{'=' * 62}\\n$ bash {script.relative_to(ROOT)}\\n{'=' * 62}")
    r = subprocess.run(["bash", str(script)], cwd=str(ROOT),
                       capture_output=True, text=True, timeout=timeout)
    out = (r.stdout or "") + (r.stderr or "")
    lines = [l for l in out.split("\\n")
             if not l.startswith("WARNING: Running pip as the 'root'")]
    print("\\n".join(lines[-tail:]))
    return r.returncode


# ---- 1) 三条链路一次装全 ----
sh(ROOT / "_tools" / "install_all_deps.sh", tail=45)

# ---- 2) 装后验证 + 强制修复（处理 hf-hub 被顶回新版）----
sh(ROOT / "_tools" / "fix_deps_patch.sh", tail=45)

print("""
============================================================
若上面出现 "三条链路全部可 import"，即可继续下一格。
若仍有 FAIL，把那几行贴出来。
============================================================
""")
\'\'\'))

cells.append(md(r"""
### 3b. 重启内核

装完依赖后 **必须重启 Notebook 内核**（菜单：Kernel → Restart），
否则当前进程里还是旧模块。

重启后**不用重跑** 3a，直接从下面继续。
"""))


'''

src = src[:i] + NEW + src[j:]
P.write_text(src, encoding="utf-8")
print(f"已替换 section 3（{j - i} 字符 -> {len(NEW)} 字符）")
