#!/usr/bin/env bash
set -euo pipefail

# Run from the project root after activating your conda/venv environment.
# Choose a PyTorch CUDA wheel compatible with your driver. The example below uses CUDA 12.1.
# Change cu121 to cu124/cu126 if your cluster uses a different supported wheel.

python -m pip install --upgrade pip setuptools wheel
python -m pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt

rm -rf external/mace
mkdir -p external
git clone https://github.com/ACEsuit/mace.git external/mace
cd external/mace
git checkout main
python -m pip install .
cd ../..

python -m pip install git+https://github.com/WillBaldwin0/graph_electrostatics.git
python -m pip install -e .

python install/smoke_test_polarmace.py --device cuda --dtype float32
