#!/bin/bash

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "未找到项目虚拟环境。请先在项目目录执行：python3.11 -m venv .venv && .venv/bin/pip install -e '.[dev]'"
  exit 1
fi

cd "$PROJECT_DIR" || exit 1

if [ "${1:-}" = "--check" ]; then
  "$PYTHON_BIN" -c "import streamlit; import probstat_tutor; print('macOS 启动检查通过')"
  exit $?
fi

exec "$PYTHON_BIN" -m streamlit run app.py --server.address 127.0.0.1
