#!/bin/bash
# Targeted correctness check: wider sky grid + short slides, with both SNR and
# metadata-aware trigger recovery gates.
#
# Submit: sbatch test/multi_inspiral_fir/run_slides_skygrid_wide.sbatch
#
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH -J mifir_skywide
#SBATCH -A phy220048p
#SBATCH -p RM
#SBATCH --mem 32G
#SBATCH -o /hildafs/projects/phy220048p/xhall/scratch/mifir/skywide_%j.log
#SBATCH -e /hildafs/projects/phy220048p/xhall/scratch/mifir/skywide_%j.log

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
D=${MIFIR_SKYWIDE_DIR:-${BASE}/slides_skygrid_wide}
mkdir -p "$D"

EVENT=1187008882
SR=2048
FLOW=45
RA="3.44527994344"
DEC="-0.408407044967"
SKY_OFFSET=${MIFIR_SKY_OFFSET:-0.05}
SLIDE_SHIFT=${MIFIR_SLIDE_SHIFT:-16}
PSD_MODEL=aLIGOZeroDetHighPower
GPS_START=$((EVENT - 512))
GPS_END=$((EVENT + 512))
TRIG_START=$((GPS_START + 256))
TRIG_END=$((GPS_END - 256))

echo "================ node: $(hostname)  date: $(date) ================"

[ -f "${BASE}/ratio_bank.hdf" ] || { echo "ERROR: build ${BASE}/ratio_bank.hdf first"; exit 1; }
[ -f "${BASE}/fine_bank.hdf" ] || { echo "ERROR: build ${BASE}/fine_bank.hdf first"; exit 1; }

echo ">> [$(date +%T)] Creating 3-point sky grid (offset ${SKY_OFFSET} rad)"
python3 -c "from pycbc.tmpltbank.sky_grid import SkyGrid; ra=float('${RA}'); dec=float('${DEC}'); off=float('${SKY_OFFSET}'); SkyGrid([ra, ra+off, ra-off], [dec, dec+off, dec-off], ['H1','L1'], ${EVENT}).write_to_file('${D}/sky_grid.hdf')"

echo ">> [$(date +%T)] Creating injection"
python3 ${REPO}/bin/pycbc_create_injections \
    --config-files ${HERE}/injection.ini --ninjections 1 --seed 1234 \
    --output-file ${D}/injection.hdf --force

COMMON_ARGS="--instruments H1 L1 --projection standard \
  --trigger-time ${EVENT} --sky-grid ${D}/sky_grid.hdf \
  --gps-start-time ${GPS_START} --gps-end-time ${GPS_END} \
  --trig-start-time ${TRIG_START} --trig-end-time ${TRIG_END} \
  --fake-strain H1:zeroNoise L1:zeroNoise --channel-name H1:FAKE L1:FAKE \
  --injection-file ${D}/injection.hdf \
  --approximant TaylorF2 --order -1 --low-frequency-cutoff ${FLOW} --sample-rate ${SR} \
  --segment-length 256 --segment-start-pad 111 --segment-end-pad 17 \
  --psd-model ${PSD_MODEL} --strain-high-pass 25 --pad-data 8 \
  --sngl-snr-threshold 3.0 --nifo-sngl-snr-threshold 2 --coinc-threshold 0.0 \
  --cluster-method window --cluster-window 0.1 \
  --do-shortslides --slide-shift ${SLIDE_SHIFT} --processing-scheme cpu"

FIR_ARGS="--bank-file ${BASE}/ratio_bank.hdf --fir-length 4096 --batch-size 64 \
  --template-normalization-method precalculated_sigma"

rc=0

echo ">> [$(date +%T)] Brute pycbc_multi_inspiral"
python3 ${REPO}/bin/pycbc_multi_inspiral $COMMON_ARGS \
    --bank-file ${BASE}/fine_bank.hdf --chisq-bins 16 \
    --output ${D}/brute_slides_sky.hdf \
    || { echo "ERROR: brute run failed"; rc=1; }

echo ">> [$(date +%T)] FIR pycbc_multi_inspiral_fir"
python3 ${REPO}/bin/pycbc_multi_inspiral_fir $COMMON_ARGS \
    $FIR_ARGS --output ${D}/fir_slides_sky.hdf \
    || { echo "ERROR: FIR run failed"; rc=1; }

echo ">> [$(date +%T)] Comparing SNR"
python3 ${HERE}/compare_triggers.py \
    --brute ${D}/brute_slides_sky.hdf --fir ${D}/fir_slides_sky.hdf \
    --tolerance 0.02 --report-snr 15 \
    || { echo "SNR GATE FAILED"; rc=1; }

echo ">> [$(date +%T)] Comparing slide/sky metadata"
python3 ${HERE}/compare_trigger_metadata.py \
    --brute ${D}/brute_slides_sky.hdf --fir ${D}/fir_slides_sky.hdf \
    --report-snr 15 --sky-tolerance 1e-6 \
    || { echo "METADATA GATE FAILED"; rc=1; }

echo "================ done: $(date)  (exit ${rc}) ================"
exit ${rc}
