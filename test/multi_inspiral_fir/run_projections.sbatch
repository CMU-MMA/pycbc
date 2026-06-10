#!/bin/bash
# Targeted correctness check: non-standard coherent projections.
# Runs left, right, and left+right against brute force.
#
# Submit: sbatch test/multi_inspiral_fir/run_projections.sbatch
#
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH -J mifir_proj
#SBATCH -A phy220048p
#SBATCH -p RM
#SBATCH --mem 32G
#SBATCH -o /hildafs/projects/phy220048p/xhall/scratch/mifir/proj_%j.log
#SBATCH -e /hildafs/projects/phy220048p/xhall/scratch/mifir/proj_%j.log

set -uo pipefail

export MAMBA_ROOT_PREFIX=/hildafs/projects/phy220048p/share/micromamba
set +u
eval "$(/hildafs/projects/phy220048p/share/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /hildafs/projects/phy220048p/share/envs/ssm_pipeline
set -u
export PYTHONPATH=/hildafs/home/xhall/GitHub/pycbc:${PYTHONPATH:-}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

REPO=/hildafs/home/xhall/GitHub/pycbc
HERE=${REPO}/test/multi_inspiral_fir
BASE=/hildafs/projects/phy220048p/xhall/scratch/mifir
D=${MIFIR_PROJ_DIR:-${BASE}/projections}
mkdir -p "$D"

EVENT=1187008882
SR=2048
FLOW=45
RA="3.44527994344"
DEC="-0.408407044967"
PSD_MODEL=aLIGOZeroDetHighPower
GPS_START=$((EVENT - 512))
GPS_END=$((EVENT + 512))
TRIG_START=$((GPS_START + 256))
TRIG_END=$((GPS_END - 256))

echo "================ node: $(hostname)  date: $(date) ================"

[ -f "${BASE}/ratio_bank.hdf" ] || { echo "ERROR: build ${BASE}/ratio_bank.hdf first"; exit 1; }
[ -f "${BASE}/fine_bank.hdf" ] || { echo "ERROR: build ${BASE}/fine_bank.hdf first"; exit 1; }

echo ">> [$(date +%T)] Creating injection"
python3 ${REPO}/bin/pycbc_create_injections \
    --config-files ${HERE}/injection.ini --ninjections 1 --seed 1234 \
    --output-file ${D}/injection.hdf --force

COMMON_TAIL="--trigger-time ${EVENT} --ra ${RA} --dec ${DEC} \
  --gps-start-time ${GPS_START} --gps-end-time ${GPS_END} \
  --trig-start-time ${TRIG_START} --trig-end-time ${TRIG_END} \
  --fake-strain H1:zeroNoise L1:zeroNoise --channel-name H1:FAKE L1:FAKE \
  --injection-file ${D}/injection.hdf \
  --approximant TaylorF2 --order -1 --low-frequency-cutoff ${FLOW} --sample-rate ${SR} \
  --segment-length 256 --segment-start-pad 111 --segment-end-pad 17 \
  --psd-model ${PSD_MODEL} --strain-high-pass 25 --pad-data 8 \
  --sngl-snr-threshold 3.0 --nifo-sngl-snr-threshold 2 --coinc-threshold 0.0 \
  --cluster-method window --cluster-window 0.1 --processing-scheme cpu"

FIR_ARGS="--bank-file ${BASE}/ratio_bank.hdf --fir-length 4096 --batch-size 64 \
  --template-normalization-method precalculated_sigma"

rc=0

for PROJ in left right left+right; do
  TAG=${PROJ//+/_plus_}
  echo ">> [$(date +%T)] ===== Projection ${PROJ} ====="

  COMMON_ARGS="--instruments H1 L1 --projection ${PROJ} ${COMMON_TAIL}"

  echo ">> [$(date +%T)] Brute pycbc_multi_inspiral (${PROJ})"
  python3 ${REPO}/bin/pycbc_multi_inspiral $COMMON_ARGS \
      --bank-file ${BASE}/fine_bank.hdf --chisq-bins 16 \
      --output ${D}/brute_${TAG}.hdf \
      || { echo "ERROR: brute run failed for ${PROJ}"; rc=1; continue; }

  echo ">> [$(date +%T)] FIR pycbc_multi_inspiral_fir (${PROJ})"
  python3 ${REPO}/bin/pycbc_multi_inspiral_fir $COMMON_ARGS \
      $FIR_ARGS --output ${D}/fir_${TAG}.hdf \
      || { echo "ERROR: FIR run failed for ${PROJ}"; rc=1; continue; }

  echo ">> [$(date +%T)] Comparing FIR vs brute (${PROJ})"
  python3 ${HERE}/compare_triggers.py \
      --brute ${D}/brute_${TAG}.hdf --fir ${D}/fir_${TAG}.hdf \
      --tolerance 0.02 --report-snr 15 \
      || { echo "GATE FAILED for ${PROJ}"; rc=1; }
done

echo "================ done: $(date)  (exit ${rc}) ================"
exit ${rc}
