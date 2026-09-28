#!/usr/bin/env bash
# 在终端运行全部自动化测试并显示结果。
# 用法: ./run_tests.sh            （等价于 python -m pytest tests/ -v）
set -euo pipefail
cd "$(dirname "$0")"
python -m pytest tests/ -v "$@"
