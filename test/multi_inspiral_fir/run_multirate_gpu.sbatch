#!/bin/bash
# Targeted correctness check: build FIR taps at 4096 Hz, search at 2048 Hz.
# This exercises the multi-rate filter FFT/decimation path and the reference
# SNR decimation normalization in pycbc_multi_inspiral_fir.
#
# Submit: sbatch test/multi_inspiral_fir/run_multirate_gpu.sbatch
#
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 16
#SBATCH -t 03:00:00
#SBATCH -J mifir_mrate
#SBATCH -A phy220048p
#SBATCH -p TWIG-GPU
#SBATCH --mem 32G
#SBATCH -o /hildafs/projects/phy220048p/xhall/scratch/mifir/mrate_%j.log
#SBATCH -e /hildafs/projects/phy220048p/xhall/scratch/mifir/mrate_%j.log

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
BASE=/hildafs/projects/phy220048p/xhall/scratch/mifir
D=${MIFIR_MRATE_DIR:-${BASE}/multirate}
mkdir -p "$D"

EVENT=1187008882
ENGINE_SR=2048
TAP_SR=4096
FLOW=45
RA="3.44527994344"
DEC="-0.408407044967"
PSD_MODEL=aLIGOZeroDetHighPower
GPS_START=$((EVENT - 512))
GPS_END=$((EVENT + 512))
TRIG_START=$((GPS_START + 256))
TRIG_END=$((GPS_END - 256))

echo "================ node: $(hostname)  date: $(date) ================"
nvidia-smi -L || true

[ -f "${BASE}/coarse_bank.hdf" ] || { echo "ERROR: missing ${BASE}/coarse_bank.hdf"; exit 1; }
[ -f "${BASE}/fine_bank.hdf" ] || { echo "ERROR: missing ${BASE}/fine_bank.hdf"; exit 1; }

if [ ! -f "${D}/ratio_bank_4096.hdf" ]; then
  echo ">> [$(date +%T)] Building 4096 Hz ratio bank"
  python3 ${REPO}/bin/pycbc_fir_bank \
      --coarse-bank ${BASE}/coarse_bank.hdf --fine-bank ${BASE}/fine_bank.hdf \
      --output-file ${D}/ratio_bank_4096.hdf \
      --f-low ${FLOW} --sample-rate ${TAP_SR} --delta-f 0.01 \
      --n-taps 501 --max-taps 1023 --tap-step 200 --min-match 0.99 \
      --search-depth 2 --decimation-factor 64 \
      --psd-model ${PSD_MODEL} --n-processes 16 \
      || { echo "ERROR: 4096 Hz ratio bank build failed"; exit 1; }
  echo ">> [$(date +%T)] 4096 Hz ratio bank built"
else
  echo ">> 4096 Hz ratio bank already present, skipping build"
fi

echo ">> [$(date +%T)] Creating injection"
python3 ${REPO}/bin/pycbc_create_injections \
    --config-files ${HERE}/injection.ini --ninjections 1 --seed 1234 \
    --output-file ${D}/injection.hdf --force

COMMON_ARGS="--instruments H1 L1 --projection standard \
  --trigger-time ${EVENT} --ra ${RA} --dec ${DEC} \
  --gps-start-time ${GPS_START} --gps-end-time ${GPS_END} \
  --trig-start-time ${TRIG_START} --trig-end-time ${TRIG_END} \
  --fake-strain H1:zeroNoise L1:zeroNoise --channel-name H1:FAKE L1:FAKE \
  --injection-file ${D}/injection.hdf \
  --approximant TaylorF2 --order -1 --low-frequency-cutoff ${FLOW} --sample-rate ${ENGINE_SR} \
  --segment-length 256 --segment-start-pad 111 --segment-end-pad 17 \
  --psd-model ${PSD_MODEL} --strain-high-pass 25 --pad-data 8 \
  --sngl-snr-threshold 3.0 --nifo-sngl-snr-threshold 2 --coinc-threshold 0.0 \
  --cluster-method window --cluster-window 0.1"

FIR_ARGS="--bank-file ${D}/ratio_bank_4096.hdf --fir-length 4096 --batch-size 64 \
  --template-normalization-method precalculated_sigma"

rc=0

echo ">> [$(date +%T)] Brute (cpu, 2048 Hz)"
python3 ${REPO}/bin/pycbc_multi_inspiral $COMMON_ARGS --processing-scheme cpu \
    --bank-file ${BASE}/fine_bank.hdf --chisq-bins 16 \
    --output ${D}/brute_mrate.hdf \
    || { echo "ERROR: brute run failed"; rc=1; }

echo ">> [$(date +%T)] FIR (cpu, 4096 Hz taps -> 2048 Hz engine)"
python3 ${REPO}/bin/pycbc_multi_inspiral_fir $COMMON_ARGS --processing-scheme cpu \
    $FIR_ARGS --output ${D}/fir_cpu_mrate.hdf \
    || { echo "ERROR: FIR-cpu run failed"; rc=1; }

echo ">> [$(date +%T)] FIR (cuda, 4096 Hz taps -> 2048 Hz engine)"
python3 ${REPO}/bin/pycbc_multi_inspiral_fir $COMMON_ARGS --processing-scheme cuda \
    $FIR_ARGS --output ${D}/fir_cuda_mrate.hdf \
    || { echo "ERROR: FIR-cuda run failed"; rc=1; }

echo ">> [$(date +%T)] ===== GATE: FIR-CUDA vs brute ====="
python3 ${HERE}/compare_triggers.py --brute ${D}/brute_mrate.hdf \
    --fir ${D}/fir_cuda_mrate.hdf --tolerance 0.02 --report-snr 15 \
    || { echo "GATE FAILED (FIR-CUDA vs brute)"; rc=1; }

echo ">> [$(date +%T)] ===== PARITY: FIR-CUDA vs FIR-CPU (tight) ====="
python3 ${HERE}/compare_triggers.py --brute ${D}/fir_cpu_mrate.hdf \
    --fir ${D}/fir_cuda_mrate.hdf --tolerance 0.001 --report-snr 15 \
    || { echo "PARITY FAILED (FIR-CUDA vs FIR-CPU)"; rc=1; }

echo "================ done: $(date)  (exit ${rc}) ================"
exit ${rc}
