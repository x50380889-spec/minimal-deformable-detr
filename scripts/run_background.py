"""在后台启动一条命令并重定向输出到日志文件（供 make/脚本编排使用）。

用法::

    python scripts/run_background.py python scripts/train_teacher.py --config configs/defect.json

环境变量：
    BG_LOG: stdout 日志路径（默认 bg.out.log）
    BG_ERR: stderr 日志路径（默认 bg.err.log）
    BG_CWD: 工作目录（默认当前目录）
"""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("用法: python run_background.py <command> [args...]")
    cmd = sys.argv[1:]
    log = open(os.environ.get("BG_LOG", "bg.out.log"), "a", encoding="utf-8")
    err = open(os.environ.get("BG_ERR", "bg.err.log"), "a", encoding="utf-8")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.Popen(
        cmd,
        cwd=os.environ.get("BG_CWD") or None,
        stdout=log,
        stderr=err,
        creationflags=flags,
    )
    print(proc.pid)


if __name__ == "__main__":
    main()
