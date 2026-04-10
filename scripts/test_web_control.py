#!/usr/bin/env python3
"""验证 Web 键盘控制 Demo 是否能正常启动.

这个脚本就是 web_control_demo.py 的入口，加了 --mock 选项说明。
真机上不带 --mock 启动即可。

Usage:
    python scripts/web_control_demo.py               # 真机模式
    python scripts/web_control_demo.py --mock         # 无硬件调试
"""

# 直接复用已有的 web_control_demo.py
# 这个文件只是为了 README 中步骤清晰，实际入口就是 web_control_demo.py

import subprocess
import sys

if __name__ == "__main__":
    cmd = [sys.executable, "scripts/web_control_demo.py"] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))
