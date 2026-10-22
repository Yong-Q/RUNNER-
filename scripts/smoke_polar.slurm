#!/bin/bash
#SBATCH --job-name=al_polar_smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --mem=12G
#SBATCH --output=/home/qiuyong/bza/MLLP/RUNNER-/logs/polar_smoke-%j.out
#SBATCH --error=/home/qiuyong/bza/MLLP/RUNNER-/logs/polar_smoke-%j.err
set -euo pipefail
cd /home/qiuyong/bza/MLLP/RUNNER-
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/local/cache/matplotlib"
PYTHON=/home/qiuyong/bza/MLLP/committee_probe_20261022/env/bin/python
"$PYTHON" scripts/run_cli.py sample --config local/polar.json --input examples/gas_inputs.extxyz --index 5 --steps 12 --out "runs/polar_smoke_${SLURM_JOB_ID}"
"$PYTHON" scripts/run_cli.py benchmark --config local/polar.json --input examples/gas_inputs.extxyz --index 5 --warmup 2 --repeats 5 --out "runs/polar_bench_${SLURM_JOB_ID}"
