#!/bin/bash
# End-to-end Phase 0/1 validation on a CPU compute node (NEVER the login node).
#   1. Build the hierarchical ratio bank (pycbc_fir_bank) [skipped if present]
#   2. Create the deterministic injection
#   3. Brute-force coherent baseline (pycbc_multi_inspiral)
#   4. FIR-de-chirped coherent search (pycbc_multi_inspiral_fir)
#   5. Compare coherent SNR (FIR-vs-brute gate, test suite B)
#
# Submit:  sbatch test/multi_inspiral_fir/run_validation.sbatch
#
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 01:00:00
#SBATCH -J mifir_val
#SBATCH -A phy220048p
#SBATCH -p RM
#SBATCH --mem 32G
#SBATCH -o /hildafs/projects/phy220048p/xhall/scratch/mifir/val_%j.log
#SBATCH -e /hildafs/projects/phy220048p/xhall/scratch/mifir/val_%j.log

set -uo pipefail

export MAMBA_ROOT_PREFIX=/hildafs/projects/phy220048p/share/micromamba
set +u
eval "$(/hildafs/projects/phy220048p/share/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /hildafs/projects/phy220048p/share/envs/ssm_pipeline
set -u
export PYTHONPATH=/hildafs/home/xhall/GitHub/pycbc:${PYTHONPATH:-}
# Pin BLAS to one thread per worker process to avoid oversubscription.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

REPO=/hildafs/home/xhall/GitHub/pycbc
HERE=${REPO}/test/multi_inspiral_fir
D=${MIFIR_DIR:-/hildafs/projects/phy220048p/xhall/scratch/mifir}
mkdir -p "$D"

EVENT=1187008882
SR=2048
FLOW=45
# Bare radians (no "rad" suffix) so they stay single tokens inside COMMON_ARGS.
RA="3.44527994344"
DEC="-0.408407044967"
PSD_MODEL=aLIGOZeroDetHighPower
GPS_START=$((EVENT - 512))
GPS_END=$((EVENT + 512))
TRIG_START=$((GPS_START + 256))
TRIG_END=$((GPS_END - 256))

echo "================ node: $(hostname)  date: $(date) ================"

# --- 1. Build the ratio bank (skip if already present) ---
if [ ! -f "${D}/ratio_bank.hdf" ]; then
  echo ">> [$(date +%T)] Building ratio bank"
  python3 ${REPO}/bin/pycbc_fir_bank \
      --coarse-bank ${D}/coarse_bank.hdf --fine-bank ${D}/fine_bank.hdf \
      --output-file ${D}/ratio_bank.hdf \
      --f-low ${FLOW} --sample-rate ${SR} --delta-f 0.01 \
      --n-taps 251 --max-taps 511 --tap-step 100 --min-match 0.99 \
      --search-depth 2 --decimation-factor 64 \
      --psd-model ${PSD_MODEL} --n-processes 16
  echo ">> [$(date +%T)] ratio bank built"
else
  echo ">> ratio bank already present, skipping build"
fi

# --- 2. Injection ---
echo ">> [$(date +%T)] Creating injection"
python3 ${REPO}/bin/pycbc_create_injections \
    --config-files ${HERE}/injection.ini --ninjections 1 \
    --output-file ${D}/injection.hdf --force

COMMON_ARGS="--instruments H1 L1 --projection standard \
  --trigger-time ${EVENT} --ra ${RA} --dec ${DEC} \
  --gps-start-time ${GPS_START} \
  --gps-end-time ${GPS_END} \
  --trig-start-time ${TRIG_START} --trig-end-time ${TRIG_END} \
  --fake-strain H1:zeroNoise L1:zeroNoise --channel-name H1:FAKE L1:FAKE \
  --injection-file ${D}/injection.hdf \
  --approximant TaylorF2 --order -1 --low-frequency-cutoff ${FLOW} --sample-rate ${SR} \
  --segment-length 256 --segment-start-pad 111 --segment-end-pad 17 \
  --psd-model ${PSD_MODEL} --strain-high-pass 25 --pad-data 8 \
  --sngl-snr-threshold 3.0 --nifo-sngl-snr-threshold 2 --coinc-threshold 0.0 \
  --cluster-method window --cluster-window 0.1 --processing-scheme cpu"

# Track failures across all steps; the job exits nonzero (Slurm FAILED) if any
# executable crashes or any gate fails. We do NOT use `set -e` so that all
# results are printed even when an early step fails, but `rc` is the verdict.
rc=0

# --- 3. Brute-force baseline ---
echo ">> [$(date +%T)] Brute pycbc_multi_inspiral"
python3 ${REPO}/bin/pycbc_multi_inspiral $COMMON_ARGS \
    --bank-file ${D}/fine_bank.hdf --chisq-bins 16 \
    --output ${D}/brute_triggers.hdf || { echo "ERROR: brute run failed"; rc=1; }

# --- 4. FIR coherent search ---
echo ">> [$(date +%T)] FIR pycbc_multi_inspiral_fir"
python3 ${REPO}/bin/pycbc_multi_inspiral_fir $COMMON_ARGS \
    --bank-file ${D}/ratio_bank.hdf \
    --fir-length 4096 --batch-size 64 \
    --output ${D}/fir_triggers.hdf || { echo "ERROR: FIR run failed"; rc=1; }

# --- 5. Compare (gate) ---
echo ">> [$(date +%T)] Comparing FIR vs brute"
python3 ${HERE}/compare_triggers.py \
    --brute ${D}/brute_triggers.hdf --fir ${D}/fir_triggers.hdf \
    --tolerance 0.02 || { echo "GATE FAILED"; rc=1; }

echo "================ done: $(date)  (exit ${rc}) ================"
exit ${rc}
