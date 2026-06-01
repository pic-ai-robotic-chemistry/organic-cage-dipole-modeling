# Installation: PolarMACE + fixed-N MC toolkit

This project assumes a non-periodic cluster: cage + fixed number of anions. The MC scripts call PolarMACE through ASE.

## 1. Create an environment

```bash
conda env create -f environment.yml
conda activate polarmace-mc
```

or with venv:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## 2. Install PolarMACE

PolarMACE currently requires MACE from the latest `main` branch plus `graph_electrostatics`. The convenience scripts below install those pieces.

CPU:

```bash
bash install/install_polarmace_cpu.sh
```

CUDA example:

```bash
bash install/install_polarmace_cuda.sh
```

If your cluster uses another CUDA version, edit the PyTorch wheel line in `install/install_polarmace_cuda.sh` before running it.

## 3. Smoke test

```bash
python install/smoke_test_polarmace.py --device cpu --dtype float64
```

Expected output includes an energy, a total dipole, and `density_coefficients shape`.

## 4. Package install

The install scripts run:

```bash
python -m pip install -e .
```

If you installed dependencies manually, run this command from the project root.

## Notes

- Set `polarmace.device: cuda` and usually `default_dtype: float32` in config files for faster exploratory MC on GPU.
- For final checks, repeat low-energy frames with `float64`.
- For neutral 12-anion shells here, use `charge: 0`.
- For 8 anions around a +12 cage, use `charge: 4` unless your chemical charge model differs.
- `spin: 1` assumes closed-shell singlet multiplicity.
