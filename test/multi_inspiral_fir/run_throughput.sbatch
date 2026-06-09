#!/bin/bash
# Throughput (test suite D) + GPU re-validation on a LARGER bank.
# Builds ~150 fine / ~15 coarse (10:1), then times:
#   brute pycbc_multi_inspiral (cpu), FIR (cpu), FIR (cuda)
# reports templates/s and speedups, and re-checks correctness:
#   FIR-CUDA vs brute (gate), FIR-CUDA vs FIR-CPU (parity).
#
# Submit: sbatch test/multi_inspiral_fir/run_throughput.sbatch
#
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH -J mifir_tput
#SBATCH -A phy220048p
#SBATCH -p TWIG-GPU
#SBATCH -o /hildafs/projects/phy220048p/xhall/scratch/mifir/tput_%j.log
#SBATCH -e /hildafs/projects/phy220048p/xhall/scratch/mifir/tput_%j.log

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
D=/hildafs/projects/phy220048p/xhall/scratch/mifir/big
mkdir -p "$D"

EVENT=1187008882; SR=2048; FLOW=45
RA="3.44527994344"; DEC="-0.408407044967"; PSD_MODEL=aLIGOZeroDetHighPower
GPS_START=$((EVENT - 512)); GPS_END=$((EVENT + 512))
TRIG_START=$((GPS_START + 256)); TRIG_END=$((GPS_END - 256))

echo "================ node: $(hostname)  date: $(date) ================"
nvidia-smi -L || true

if [ ! -f "${D}/ratio_bank.hdf" ]; then
  echo ">> [$(date +%T)] Building larger bank (~150 fine / ~15 coarse)"
  python3 ${HERE}/make_validation_banks.py \
    --fine-output ${D}/fine_bank.hdf --coarse-output ${D}/coarse_bank.hdf \
    --f-lower ${FLOW} --tau0-min 10 --tau0-max 13 \
    --fine-tau0-step 0.02 --coarse-tau0-step 0.20 --q-values 1.0 --approximant TaylorF2
  python3 ${REPO}/bin/pycbc_fir_bank \
    --coarse-bank ${D}/coarse_bank.hdf --fine-bank ${D}/fine_bank.hdf \
    --output-file ${D}/ratio_bank.hdf --f-low ${FLOW} --sample-rate ${SR} --delta-f 0.01 \
    --n-taps 251 --max-taps 511 --tap-step 100 --min-match 0.99 \
    --search-depth 2 --decimation-factor 64 --psd-model ${PSD_MODEL} --n-processes 16
fi
NFINE=$(python3 -c "import h5py;print(len(h5py.File('${D}/fine_bank.hdf')['mass1']))")
echo ">> fine bank size: ${NFINE}"

python3 ${REPO}/bin/pycbc_create_injections --config-files ${HERE}/injection.ini \
    --ninjections 1 --seed 1234 --output-file ${D}/injection.hdf --force

COMMON="--instruments H1 L1 --projection standard --trigger-time ${EVENT} \
  --ra ${RA} --dec ${DEC} --gps-start-time ${GPS_START} --gps-end-time ${GPS_END} \
  --trig-start-time ${TRIG_START} --trig-end-time ${TRIG_END} \
  --fake-strain H1:zeroNoise L1:zeroNoise --channel-name H1:FAKE L1:FAKE \
  --injection-file ${D}/injection.hdf --approximant TaylorF2 --order -1 \
  --low-frequency-cutoff ${FLOW} --sample-rate ${SR} --segment-length 256 \
  --segment-start-pad 111 --segment-end-pad 17 --psd-model ${PSD_MODEL} \
  --strain-high-pass 25 --pad-data 8 --sngl-snr-threshold 3.0 \
  --nifo-sngl-snr-threshold 2 --coinc-threshold 0.0 --cluster-method window --cluster-window 0.1"
FIR="--bank-file ${D}/ratio_bank.hdf --fir-length 4096 --batch-size 64 \
  --template-normalization-method precalculated_sigma"

run() { /usr/bin/time -v python3 "$@" 2>${D}/time.txt; grep -E "Elapsed|wall" ${D}/time.txt | head -1; }

echo ">> [$(date +%T)] BRUTE (cpu)"
T0=$SECONDS
python3 ${REPO}/bin/pycbc_multi_inspiral $COMMON --processing-scheme cpu \
    --bank-file ${D}/fine_bank.hdf --chisq-bins 16 --output ${D}/brute.hdf 2>${D}/brute_run.log
TB=$((SECONDS-T0)); echo "   brute wall: ${TB}s"

echo ">> [$(date +%T)] FIR (cpu)"
T0=$SECONDS
python3 ${REPO}/bin/pycbc_multi_inspiral_fir $COMMON --processing-scheme cpu \
    $FIR --output ${D}/fir_cpu.hdf 2>${D}/fircpu_run.log
TFC=$((SECONDS-T0)); echo "   FIR-cpu wall: ${TFC}s"

echo ">> [$(date +%T)] FIR (cuda)"
T0=$SECONDS
python3 ${REPO}/bin/pycbc_multi_inspiral_fir $COMMON --processing-scheme cuda \
    $FIR --output ${D}/fir_cuda.hdf 2>${D}/fircuda_run.log
TFG=$((SECONDS-T0)); echo "   FIR-cuda wall: ${TFG}s"

echo ">> ===== THROUGHPUT (${NFINE} fine templates, wall-clock incl. setup) ====="
python3 - <<PY
nf=${NFINE}; tb=${TB}; tfc=${TFC}; tfg=${TFG}
for name,t in [("brute",tb),("FIR-cpu",tfc),("FIR-cuda",tfg)]:
    print(f"  {name:9s}: {t:5d}s  -> {nf/max(t,1):.2f} templates/s")
print(f"  speedup FIR-cpu  vs brute: {tb/max(tfc,1):.2f}x")
print(f"  speedup FIR-cuda vs brute: {tb/max(tfg,1):.2f}x")
print(f"  speedup FIR-cuda vs FIR-cpu: {tfc/max(tfg,1):.2f}x")
PY

echo ">> ===== GATE: FIR-CUDA vs brute ====="
python3 ${HERE}/compare_triggers.py --brute ${D}/brute.hdf --fir ${D}/fir_cuda.hdf \
    --tolerance 0.02 --report-snr 15 || echo "(gate nonzero)"
echo ">> ===== PARITY: FIR-CUDA vs FIR-CPU ====="
python3 ${HERE}/compare_triggers.py --brute ${D}/fir_cpu.hdf --fir ${D}/fir_cuda.hdf \
    --tolerance 0.001 --report-snr 15 || echo "(parity nonzero)"
echo "================ done: $(date) ================"
