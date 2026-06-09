#!/bin/bash
# FIR-de-chirped coherent search: run pycbc_multi_inspiral_fir (H1+L1) on the
# hierarchical ratio bank, over the SAME deterministic zeroNoise data + single
# injection as run_brute.sh. Output is compared against the brute baseline
# (compare_triggers.py) for the Phase 1 correctness gate (test suite B).
#
# Assumes run_brute.sh has already created ${D}/injection.hdf (identical
# injection). Pass MIFIR_SCHEME=cuda (default cpu) to exercise the GPU engine.
set -euo pipefail

export MAMBA_ROOT_PREFIX=/hildafs/projects/phy220048p/share/micromamba
set +u
eval "$(/hildafs/projects/phy220048p/share/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /hildafs/projects/phy220048p/share/envs/ssm_pipeline
set -u
export PYTHONPATH=/hildafs/home/xhall/GitHub/pycbc:${PYTHONPATH}

REPO=/hildafs/home/xhall/GitHub/pycbc
D=${MIFIR_DIR:-/hildafs/projects/phy220048p/xhall/scratch/mifir}
SCHEME=${MIFIR_SCHEME:-cpu}

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

echo ">> Running FIR pycbc_multi_inspiral_fir (scheme=${SCHEME})"
python3 ${REPO}/bin/pycbc_multi_inspiral_fir \
    --verbose \
    --processing-scheme ${SCHEME} \
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
    --bank-file ${D}/ratio_bank.hdf \
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
    --fir-length 4096 \
    --batch-size 64 \
    --template-normalization-method mchirp \
    --cluster-method window \
    --cluster-window 0.1 \
    --output ${D}/fir_triggers.hdf

echo ">> FIR output: ${D}/fir_triggers.hdf"
