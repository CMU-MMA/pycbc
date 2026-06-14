#!/bin/bash
# Phase-3 (batched engine) correctness gates on a GPU node, one allocation:
#   Gate A  unit parity   : test/test_multi_inspiral_fir.py (cpu + cuda),
#                           incl. test_batched_matches_single (tiled vs single).
#   Gate B  end-to-end     : brute pycbc_multi_inspiral (cpu),
#                            FIR pycbc_multi_inspiral_fir (cpu) and (cuda),
#                            then compare FIR-CUDA vs brute and FIR-CUDA vs FIR-CPU.
# The batched CUDA path must reproduce the per-template path and the brute
# search (results must not move).
#
# Submit: sbatch test/multi_inspiral_fir/run_gate_ab.sbatch
#
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH -J mifir_gate_ab
#SBATCH -A phy220048p
#SBATCH -p TWIG-GPU
#SBATCH -o /hildafs/projects/phy220048p/xhall/scratch/mifir/gate_ab_%j.log
#SBATCH -e /hildafs/projects/phy220048p/xhall/scratch/mifir/gate_ab_%j.log

set -uo pipefail
module load cuda/12.4.0

export MAMBA_ROOT_PREFIX=/hildafs/projects/phy220048p/share/micromamba
set +u
eval "$(/hildafs/projects/phy220048p/share/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /hildafs/projects/phy220048p/share/envs/ssm_pipeline
set -u
export PYTHONPATH=/hildafs/home/xhall/GitHub/pycbc:${PYTHONPATH:-}
export PYCUDA_DEFAULT_NVCC_FLAGS="-ccbin /usr/bin/gcc"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

REPO=/hildafs/home/xhall/GitHub/pycbc
HERE=${REPO}/test/multi_inspiral_fir
D=${MIFIR_DIR:-/hildafs/projects/phy220048p/xhall/scratch/mifir}
cd ${REPO}

echo "================ node: $(hostname)  date: $(date) ================"
nvidia-smi -L || true
rc=0

echo ">> [$(date +%T)] threshold_cuda unit test (device-buffer threshold_only)"
python3 test/test_threshold.py -s cuda || { echo "THRESHOLD UNIT TEST FAILED"; rc=1; }

echo ">> [$(date +%T)] Gate A: CPU unit tests"
python3 test/test_multi_inspiral_fir.py --scheme cpu || { echo "GATE A (cpu) FAILED"; rc=1; }

echo ">> [$(date +%T)] Gate A: CUDA unit tests (incl. batched-vs-single)"
python3 test/test_multi_inspiral_fir.py --scheme cuda || { echo "GATE A (cuda) FAILED"; rc=1; }

# ---- Gate B: end-to-end (same injection + data for all three runs) ----
[ -f "${D}/ratio_bank.hdf" ] || { echo "ERROR: build ratio_bank.hdf first (run_validation.sbatch)"; exit 1; }

EVENT=1187008882
SR=2048; FLOW=45
RA="3.44527994344"; DEC="-0.408407044967"
PSD_MODEL=aLIGOZeroDetHighPower
GPS_START=$((EVENT - 512)); GPS_END=$((EVENT + 512))
TRIG_START=$((GPS_START + 256)); TRIG_END=$((GPS_END - 256))

echo ">> [$(date +%T)] Creating injection (seeded, reproducible)"
python3 ${REPO}/bin/pycbc_create_injections \
    --config-files ${HERE}/injection.ini --ninjections 1 --seed 1234 \
    --output-file ${D}/injection.hdf --force || { echo "ERROR: injection failed"; rc=1; }

COMMON_ARGS="--instruments H1 L1 --projection standard \
  --trigger-time ${EVENT} --ra ${RA} --dec ${DEC} \
  --gps-start-time ${GPS_START} --gps-end-time ${GPS_END} \
  --trig-start-time ${TRIG_START} --trig-end-time ${TRIG_END} \
  --fake-strain H1:zeroNoise L1:zeroNoise --channel-name H1:FAKE L1:FAKE \
  --injection-file ${D}/injection.hdf \
  --approximant TaylorF2 --order -1 --low-frequency-cutoff ${FLOW} --sample-rate ${SR} \
  --segment-length 256 --segment-start-pad 111 --segment-end-pad 17 \
  --psd-model ${PSD_MODEL} --strain-high-pass 25 --pad-data 8 \
  --sngl-snr-threshold 3.0 --nifo-sngl-snr-threshold 2 --coinc-threshold 0.0 \
  --cluster-method window --cluster-window 0.1"

FIR_ARGS="--bank-file ${D}/ratio_bank.hdf --fir-length 4096 --batch-size 64 \
  --template-normalization-method precalculated_sigma"

echo ">> [$(date +%T)] Gate B: brute (cpu)"
python3 ${REPO}/bin/pycbc_multi_inspiral $COMMON_ARGS --processing-scheme cpu \
    --bank-file ${D}/fine_bank.hdf --chisq-bins 16 --output ${D}/brute_triggers.hdf \
    || { echo "ERROR: brute run failed"; rc=1; }

echo ">> [$(date +%T)] Gate B: FIR (cpu)"
python3 ${REPO}/bin/pycbc_multi_inspiral_fir $COMMON_ARGS --processing-scheme cpu \
    $FIR_ARGS --output ${D}/fir_cpu_triggers.hdf \
    || { echo "ERROR: FIR-cpu run failed"; rc=1; }

echo ">> [$(date +%T)] Gate B: FIR (cuda, batched)"
python3 ${REPO}/bin/pycbc_multi_inspiral_fir $COMMON_ARGS --processing-scheme cuda \
    $FIR_ARGS --output ${D}/fir_cuda_triggers.hdf \
    || { echo "ERROR: FIR-cuda run failed"; rc=1; }

echo ">> [$(date +%T)] ===== GATE: FIR-CUDA vs brute ====="
python3 ${HERE}/compare_triggers.py --brute ${D}/brute_triggers.hdf \
    --fir ${D}/fir_cuda_triggers.hdf --tolerance 0.02 --report-snr 15 \
    || { echo "GATE FAILED (FIR-CUDA vs brute)"; rc=1; }

echo ">> [$(date +%T)] ===== PARITY: FIR-CUDA vs FIR-CPU (tight) ====="
python3 ${HERE}/compare_triggers.py --brute ${D}/fir_cpu_triggers.hdf \
    --fir ${D}/fir_cuda_triggers.hdf --tolerance 0.001 --report-snr 15 \
    || { echo "PARITY FAILED (FIR-CUDA vs FIR-CPU)"; rc=1; }

echo "================ done: $(date)  (exit ${rc}) ================"
exit ${rc}
