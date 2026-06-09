#!/bin/bash
# Run the multi_inspiral_fir unit tests on a GPU node (CUDA scheme).
# Used for Phase 2 CPU-vs-CUDA parity (test suite A).
#
# Submit:  sbatch test/multi_inspiral_fir/run_gpu_tests.sbatch
#
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH -J mifir_gpu
#SBATCH -A phy220048p
#SBATCH -p TWIG-GPU
#SBATCH -o /hildafs/projects/phy220048p/xhall/scratch/mifir/gpu_%j.log
#SBATCH -e /hildafs/projects/phy220048p/xhall/scratch/mifir/gpu_%j.log

set -uo pipefail
module load cuda/12.4.0

export MAMBA_ROOT_PREFIX=/hildafs/projects/phy220048p/share/micromamba
set +u
eval "$(/hildafs/projects/phy220048p/share/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /hildafs/projects/phy220048p/share/envs/ssm_pipeline
set -u
export PYTHONPATH=/hildafs/home/xhall/GitHub/pycbc:${PYTHONPATH:-}
export PYCUDA_DEFAULT_NVCC_FLAGS="-ccbin /usr/bin/gcc"

REPO=/hildafs/home/xhall/GitHub/pycbc
cd ${REPO}

echo "================ node: $(hostname)  date: $(date) ================"
nvidia-smi -L || true

echo ">> CPU unit tests"
python3 test/test_multi_inspiral_fir.py --scheme cpu

echo ">> CUDA unit tests"
python3 test/test_multi_inspiral_fir.py --scheme cuda

echo "================ done: $(date) ================"
