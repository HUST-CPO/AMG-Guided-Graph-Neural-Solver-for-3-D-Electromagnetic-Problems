# Environment and Hardware Setup

This repository provides separate reproducible environments for CPU-only and
NVIDIA CUDA workflows.

## Conda (recommended)

The Conda files include a complex-scalar PETSc build required by the
`DiagScale` preprocessing step.

```bash
# CUDA training and evaluation
conda env create -f environment-gpu.yml
conda activate femgrl-gpu

# Or: preprocessing, plotting, tests, and CPU AMG benchmarks
conda env create -f environment-cpu.yml
conda activate femgrl-cpu
```

## pip

The pip files are convenient for workflows that do not require PETSc:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-gpu.txt  # NVIDIA CUDA 11.8
# python -m pip install -r requirements-cpu.txt  # CPU-only alternative
```

Do not use a default real-scalar PETSc build for `DiagScale`. Use one of the
Conda environments above, or provide a separately built complex PETSc and a
matching `petsc4py`.

## Verification

```bash
python -c "import torch, torch_geometric, numpy, scipy, pyamg; print(torch.__version__, torch.cuda.is_available())"
python -c "from petsc4py import PETSc; print(PETSc.ScalarType)"
```

The second command must report a complex scalar type. For resource sizing and
the hardware used by the reference runs, see `hardware-config.yml`. CPU-only
mode supports preprocessing and classical AMG workflows; the DDP launcher
requires CUDA GPUs.

