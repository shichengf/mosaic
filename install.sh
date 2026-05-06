#!/usr/bin/env bash
# install.sh — one-click environment setup for MOSAIC.
#
# Creates (or activates) a conda env named "mosaic", installs PyTorch with
# CUDA support if a GPU is detected, and installs MOSAIC + all extras in
# editable mode. Falls back to CPU-only PyTorch if no NVIDIA GPU is present.
#
# Usage:
#   bash install.sh                # default env name `mosaic`, CUDA 12.1
#   ENV_NAME=foo CUDA=cu118 bash install.sh
#   bash install.sh --no-extras    # install core only, no data-prep / analysis extras

set -euo pipefail

ENV_NAME="${ENV_NAME:-mosaic}"
CUDA="${CUDA:-cu121}"
PY="${PY:-3.10}"
EXTRAS="${EXTRAS:-full}"

if [[ "${1:-}" == "--no-extras" ]]; then
    EXTRAS=""
fi

# ---------------------------------------------------------------
# 1. Choose package manager (conda preferred, fallback: venv)
# ---------------------------------------------------------------
if command -v conda >/dev/null 2>&1; then
    HAS_CONDA=1
else
    HAS_CONDA=0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------
# 2. GPU detection
# ---------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    HAS_GPU=1
    echo "[install] NVIDIA GPU detected — installing CUDA ($CUDA) PyTorch."
else
    HAS_GPU=0
    echo "[install] No GPU found — installing CPU-only PyTorch."
fi

# ---------------------------------------------------------------
# 3. Create / activate env
# ---------------------------------------------------------------
if [[ $HAS_CONDA -eq 1 ]]; then
    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        echo "[install] Reusing existing conda env: $ENV_NAME"
    else
        echo "[install] Creating conda env: $ENV_NAME (python=$PY)"
        conda create -y -n "$ENV_NAME" "python=$PY" pip
    fi
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
else
    VENV_DIR=".venv-$ENV_NAME"
    if [[ ! -d "$VENV_DIR" ]]; then
        echo "[install] No conda; creating venv at $VENV_DIR"
        python3 -m venv "$VENV_DIR"
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

python -m pip install --upgrade pip wheel setuptools

# ---------------------------------------------------------------
# 4. PyTorch (must match CUDA before pip install -e .)
# ---------------------------------------------------------------
if [[ $HAS_GPU -eq 1 ]]; then
    pip install --index-url "https://download.pytorch.org/whl/$CUDA" torch torchvision
else
    pip install --index-url "https://download.pytorch.org/whl/cpu" torch torchvision
fi

# ---------------------------------------------------------------
# 5. MOSAIC + extras
# ---------------------------------------------------------------
if [[ -n "$EXTRAS" ]]; then
    echo "[install] pip install -e .[$EXTRAS]"
    pip install -e ".[$EXTRAS]"
else
    echo "[install] pip install -e ."
    pip install -e .
fi

# ---------------------------------------------------------------
# 6. Smoke test
# ---------------------------------------------------------------
python - <<'PY'
import importlib, sys
mods = ["mosaic", "mosaic.models", "mosaic.components", "mosaic.data",
        "mosaic.metrics"]
for m in mods:
    importlib.import_module(m)
print("[install] OK — mosaic package imports cleanly.")
import torch
print(f"[install] torch={torch.__version__}, cuda={torch.cuda.is_available()}")
PY

echo
echo "Done. Activate the env later with:"
if [[ $HAS_CONDA -eq 1 ]]; then
    echo "  conda activate $ENV_NAME"
else
    echo "  source .venv-$ENV_NAME/bin/activate"
fi
echo "Quickstart:"
echo "  bash examples/quickstart_synthetic.sh"
