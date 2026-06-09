#!/bin/bash
# Brute-force coherent baseline: run pycbc_multi_inspiral (H1+L1) on the small
# fine bank, over deterministic zeroNoise data containing a single sub-solar
# injection at a fixed sky position. This is the reference for the FIR<->brute
# correctness gate (Phase 1 / test suite B).
set -euo pipefail

export MAMBA_ROOT_PREFIX=/hildafs/projects/phy220048p/share/micromamba
set +u
eval "$(/hildafs/projects/phy220048p/share/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /hildafs/projects/phy220048p/share/envs/ssm_pipeline
set -u
export PYTHONPATH=/hildafs/home/xhall/GitHub/pycbc:${PYTHONPATH}

REPO=/hildafs/home/xhall/GitHub/pycbc
D=${MIFIR_DIR:-/hildafs/projects/phy220048p/xhall/scratch/mifir}
HERE=${REPO}/test/multi_inspiral_fir

EVENT=1187008882
SR=2048
FLOW=45
RA="3.44527994344 rad"
DEC="-0.408407044967 rad"
PSD_MODEL=aLIGOZeroDetHighPower

GPS_START=$((EVENT - 512))
GPS_END=$((EVENT + 512))
TRIG_START=$((GPS_START + 256))
TRIG_END=$((GPS_END - 256))

# --- 1. Create the injection (deterministic, single signal) ---
echo ">> Creating injection"
python3 ${REPO}/bin/pycbc_create_injections \
    --config-files ${HERE}/injection.ini \
    --ninjections 1 \
    --output-file ${D}/injection.hdf \
    --force

# --- 2. Brute-force coherent search on the FINE bank ---
echo ">> Running brute pycbc_multi_inspiral"
python3 ${REPO}/bin/pycbc_multi_inspiral \
    --verbose \
    --processing-scheme cpu \
    --instruments H1 L1 \
    --projection standard \
    --trigger-time ${EVENT} \
    --ra "${RA}" --dec "${DEC}" \
    --gps-start-time H1:${GPS_START} L1:${GPS_START} \
    --gps-end-time H1:${GPS_END} L1:${GPS_END} \
    --trig-start-time ${TRIG_START} \
    --trig-end-time ${TRIG_END} \
    --fake-strain H1:zeroNoise L1:zeroNoise \
    --channel-name H1:FAKE L1:FAKE \
    --injection-file ${D}/injection.hdf \
    --bank-file ${D}/fine_bank.hdf \
    --approximant TaylorF2 \
    --low-frequency-cutoff ${FLOW} \
    --sample-rate ${SR} \
    --segment-length 256 \
    --segment-start-pad 111 \
    --segment-end-pad 17 \
    --psd-model ${PSD_MODEL} \
    --strain-high-pass 25 \
    --pad-data 8 \
    --sngl-snr-threshold 3.0 \
    --nifo-sngl-snr-threshold 2 \
    --coinc-threshold 0.0 \
    --chisq-bins 16 \
    --cluster-method window \
    --cluster-window 0.1 \
    --output ${D}/brute_triggers.hdf

echo ">> Brute output: ${D}/brute_triggers.hdf"
