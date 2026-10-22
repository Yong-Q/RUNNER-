#!/bin/bash
#SBATCH --job-name=build_runner
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --output=/home/qiuyong/bza/MLLP/RUNNER-/logs/build_runner_v5.out
#SBATCH --error=/home/qiuyong/bza/MLLP/RUNNER-/logs/build_runner_v5.err
set -euo pipefail
cd /home/qiuyong/bza/MLLP/RUNNER-
export PATH="$PWD/local/bin":/opt/cmake/3.27.4/bin:/opt/gcc/11.3.0/bin:$PATH
export LD_LIBRARY_PATH=/opt/gcc/auxiliary/mpc/1.3.1/lib:/opt/gcc/auxiliary/mpfr/4.2.1/lib:/opt/gcc/auxiliary/gmp/6.3.0/lib:/opt/gcc/11.3.0/lib64:/opt/blas/openblas/0.3.23-gcc11.3.0/lib:/opt/fftw/3.3.10-gcc11.3.0/lib:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=/home/qiuyong/.conda/pkgs/sysroot_linux-64-2.17-h0157908_18/x86_64-conda-linux-gnu/sysroot/usr/lib64:/home/qiuyong/.conda/pkgs/sysroot_linux-64-2.17-h0157908_18/x86_64-conda-linux-gnu/sysroot/lib64
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cmake -DCMAKE_SYSROOT=/home/qiuyong/.conda/pkgs/sysroot_linux-64-2.17-h0157908_18/x86_64-conda-linux-gnu/sysroot -S external/runner2 -B build/runner_v4 -DUSE_MPI=OFF -DUSE_MKL=OFF -DBUILD_SHARED_LIB=ON -DENABLE_TESTS=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_Fortran_FLAGS_RELWITHDEBINFO='-O2 -g' -DBLAS_LIBRARIES=/opt/blas/openblas/0.3.23-gcc11.3.0/lib/libopenblas.so -DLAPACK_LIBRARIES=/opt/blas/openblas/0.3.23-gcc11.3.0/lib/libopenblas.so -DFFTW_ROOT=/opt/fftw/3.3.10-gcc11.3.0
cmake --build build/runner_v4 -j 4
test -s build/runner_v4/libRuNNer.so
test -x build/runner_v4/RuNNer.x
