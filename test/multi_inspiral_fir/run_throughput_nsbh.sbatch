#!/bin/bash
# Gate C: throughput + GPU saturation on the real nsbh shard-0 search, 2 hr
# benchmark segment (the configuration measured in the handoff: ~36 min, A100
# starved at 79-102 W / 3-13% mem-bandwidth). Runs the exact production
# pycbc_multi_inspiral_fir invocation (from jobs/submit_pygrb.sh) with the
# batched engine, while sampling nvidia-smi power/util in the background, and
# prints the per-stage wall profile (MIFIR_PROFILE) at the end.
#
# Env knobs:
#   MIFIR_MAX_COARSE  bound the number of coarse groups (0/unset = full shard;
#                     set e.g. 40 for a fast crash/OOM/saturation probe).
#   MIFIR_TILE        --fir-tile (batch size); default 32.
#   MIFIR_FORCE_HOST  =1 to run the legacy per-template host path (A/B "before").
#
# Submit (full):  sbatch test/multi_inspiral_fir/run_throughput_nsbh.sbatch
# Submit (probe): sbatch --export=ALL,MIFIR_MAX_COARSE=40 -p HENON-GPU \
#                        test/multi_inspiral_fir/run_throughput_nsbh.sbatch
#
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 02:00:00
#SBATCH -J mifir_thru
#SBATCH -A phy220048p
#SBATCH -p TWIG-GPU
#SBATCH -o /hildafs/projects/phy220048p/xhall/scratch/mifir/thru_%j.log
#SBATCH -e /hildafs/projects/phy220048p/xhall/scratch/mifir/thru_%j.log

set -uo pipefail
module load cuda/12.4.0

export MAMBA_ROOT_PREFIX=/hildafs/projects/phy220048p/share/micromamba
set +u
eval "$(/hildafs/projects/phy220048p/share/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /hildafs/projects/phy220048p/share/envs/ssm_pipeline
set -u
export PYTHONPATH=/hildafs/home/xhall/GitHub/pycbc:${PYTHONPATH:-}
export PYCUDA_DEFAULT_NVCC_FLAGS="-ccbin /usr/bin/gcc"
export LAL_DATA_PATH=/hildafs/projects/phy220048p/xhall/GitHub/ssm-bilby-maps/lalsuite-waveform-data/waveform_data:${LAL_DATA_PATH:-}

REPO=/hildafs/home/xhall/GitHub/pycbc
SND=/hildafs/projects/phy220048p/xhall/scratch/pygrb/SN2024gqd/nsbh
BANK=/hildafs/projects/phy220048p/xhall/scratch/banks_tidal/nsbh/fir/ratio_shard_0000.hdf
OUT=/hildafs/projects/phy220048p/xhall/scratch/mifir/thru_triggers_${SLURM_JOB_ID}.hdf
SMP=/hildafs/projects/phy220048p/xhall/scratch/mifir/thru_smi_${SLURM_JOB_ID}.csv

export MIFIR_PROFILE=1
TILE=${MIFIR_TILE:-32}
CW=${MIFIR_COARSE_WORKERS:-0}   # --fir-coarse-workers (0 = serial coarse gen)

echo "================ node: $(hostname)  date: $(date) ================"
echo "MIFIR_MAX_COARSE=${MIFIR_MAX_COARSE:-0} (0=full)  --fir-tile=${TILE}  --fir-coarse-workers=${CW}  MIFIR_FORCE_HOST=${MIFIR_FORCE_HOST:-0}"
nvidia-smi -L || true

# Background nvidia-smi sampler (power, gpu/mem util, mem used) every 2 s.
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,memory.used \
    --format=csv,noheader,nounits -l 2 > "${SMP}" &
SMI_PID=$!

T0=$(date +%s)
python3 ${REPO}/bin/pycbc_multi_inspiral_fir \
    --instruments H1 L1 \
    --trigger-time 1397038736 \
    --ra  "4.470401131521157 rad" \
    --dec "0.02347861687174437 rad" \
    --bank-file "${BANK}" \
    --gps-start-time 1396433944 \
    --gps-end-time   1396441080 \
    --channel-name H1:GWOSC-16KHZ_R1_STRAIN L1:GWOSC-16KHZ_R1_STRAIN \
    --frame-cache H1:${SND}/cache_seg1_H1.lcf L1:${SND}/cache_seg1_L1.lcf \
    --sample-rate 2048 --strain-high-pass 10 --pad-data 8 \
    --segment-length 4096 --segment-start-pad 560 --segment-end-pad 17 \
    --low-frequency-cutoff 20 \
    --autogating-threshold 100 --autogating-cluster 0.5 \
    --autogating-width 0.25 --autogating-taper 0.25 --autogating-pad 0 \
    --psd-estimation median --psd-segment-length 32 --psd-segment-stride 8 \
    --psd-num-segments 29 \
    --sngl-snr-threshold 4.0 --nifo-sngl-snr-threshold 2 \
    --cluster-method window --cluster-window 0.1 \
    --approximant TaylorF2 --order 7 \
    --fir-length 4096 --batch-size 64 --fir-tile ${TILE} \
    --fir-coarse-workers ${CW} \
    --template-normalization-method precalculated_sigma \
    --processing-scheme cuda \
    --output "${OUT}" --verbose
EC=$?
T1=$(date +%s)
kill ${SMI_PID} 2>/dev/null || true

echo "================ pycbc exit ${EC}  wall $((T1-T0)) s ================"

# Summarise GPU saturation. Drop the first ~30 samples (data read / PSD setup,
# GPU idle) so the numbers reflect the filtering phase.
python3 - "${SMP}" <<'PY'
import sys, numpy as np
rows=[l.strip() for l in open(sys.argv[1]) if l.strip()]
def parse(r):
    p=[x.strip() for x in r.split(',')]
    # gpu_util, mem_util, power, mem_used
    return float(p[1]), float(p[2]), float(p[3]), float(p[4])
vals=[]
for r in rows:
    try: vals.append(parse(r))
    except Exception: pass
if not vals:
    print("no nvidia-smi samples"); sys.exit(0)
a=np.array(vals)
skip=min(30, max(0,len(a)//5))
f=a[skip:]
print(f"nvidia-smi: {len(a)} samples ({skip} setup samples dropped)")
for i,name in enumerate(['gpu_util(%)','mem_util(%)','power(W)','mem_used(MiB)']):
    print(f"  {name:16s} mean={f[:,i].mean():8.1f}  median={np.median(f[:,i]):8.1f}  max={f[:,i].max():8.1f}")
PY

echo "trigger output: ${OUT}"
echo "================ done: $(date) ================"
exit ${EC}
