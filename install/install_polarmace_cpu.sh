#!/usr/bin/env bash
set -euo pipefail

# Run from the project root after activating your conda/venv environment.
# This installs PyTorch CPU, MACE main branch, graph_electrostatics, then this MC package.

python -m pip install --upgrade pip setuptools wheel
python -m pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt

# PolarMACE currently needs MACE main branch rather than the PyPI mace-torch release.
rm -rf external/mace
mkdir -p external
git clone https://github.com/ACEsuit/mace.git external/mace
cd external/mace
git checkout main
python -m pip install .
cd ../..

python -m pip install git+https://github.com/WillBaldwin0/graph_electrostatics.git
python -m pip install -e .

python install/smoke_test_polarmace.py --device cpu --dtype float64
