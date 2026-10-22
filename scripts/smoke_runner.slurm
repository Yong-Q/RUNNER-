#!/bin/bash
#SBATCH --job-name=al_runner_smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --output=/home/qiuyong/bza/MLLP/RUNNER-/logs/runner_smoke-%j.out
#SBATCH --error=/home/qiuyong/bza/MLLP/RUNNER-/logs/runner_smoke-%j.err
set -euo pipefail
cd /home/qiuyong/bza/MLLP/RUNNER-
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$PWD/build/runner_v4":/opt/gcc/11.3.0/lib64:/opt/blas/openblas/0.3.23-gcc11.3.0/lib:/opt/fftw/3.3.10-gcc11.3.0/lib:${LD_LIBRARY_PATH:-}
.venv/bin/python scripts/run_cli.py benchmark --config local/runner_ag.json --input local/runner_ag_inputs_v2.extxyz --out "runs/runner_bench_${SLURM_JOB_ID}"
.venv/bin/python scripts/run_cli.py sample --config local/runner_ag.json --input local/runner_ag_inputs_v2.extxyz --steps 3 --out "runs/runner_smoke_${SLURM_JOB_ID}"
