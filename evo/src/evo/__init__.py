"""智训引擎 — 工业模型自进化系统.

两层自进化:
  * 模型层自迭代: 漂移 -> 重训/微调 -> 候选模型
  * 策略层自改进: 更新决策参数 (触发阈值/增益门限/窗口) 在受控 bandit 下自适应
"""

from __future__ import annotations

import sys

__version__ = "0.1.0"


def _harden_stdout_encoding() -> None:
    """让标准输出/错误能安全打印非 GBK 字符 (emoji、✓、→ 等)。

    为什么需要这个:
        Windows 控制台默认编码是 GBK。SwanLab SDK 在 `finish()` 与
        `atexit` 处理器里会打印 emoji (📁 / 🚀 / 💾), 触发
        `UnicodeEncodeError: 'gbk' codec can't encode character ...`。
        那个异常发生在 **atexit 阶段**, 绕过了我们自己的 try/except,
        表现为"实验跑完了但进程以异常堆栈退出" —— 极易被误判成
        实验失败, 也可能让 runner 的退出码变成非 0。

    处理方式:
        把 stdout/stderr 重配为 UTF-8 + `errors="replace"`。
        不可编码字符退化成 '?' 而不是抛异常, 保证流程能跑完。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # 某些被重定向/包装过的流不支持重配 —— 忽略即可, 不阻断流程。
            pass


_harden_stdout_encoding()
