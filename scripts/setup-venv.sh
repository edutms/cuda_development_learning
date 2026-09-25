#!/usr/bin/env bash
# Local Python env, used only for authoring/previewing notebooks.
# Nothing GPU-related goes in here -- there is no CUDA device on this machine.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv .venv
./.venv/bin/pip install --quiet --upgrade pip
./.venv/bin/pip install -r requirements.txt

echo
echo "Done. Launch JupyterLab with:"
echo "  ./.venv/bin/jupyter lab notebooks/00_colab_setup.ipynb"
