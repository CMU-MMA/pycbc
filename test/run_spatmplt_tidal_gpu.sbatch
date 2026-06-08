#!/bin/bash
#SBATCH -N 1 -G 1 -c 8 -t 00:30:00 -A phy220048p -p TWIG-GPU
#SBATCH -o spatmplt_tidal_gpu_%j.log -e spatmplt_tidal_gpu_%j.err
#SBATCH -J spatmplt_tidal_gpu
module load cuda/12.4.0
export MAMBA_ROOT_PREFIX=/hildafs/projects/phy220048p/share/micromamba
eval "$(/hildafs/projects/phy220048p/share/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /hildafs/projects/phy220048p/share/envs/ssm_pipeline
export PYTHONPATH=/hildafs/home/xhall/GitHub/pycbc:${PYTHONPATH}
export PYCUDA_DEFAULT_NVCC_FLAGS="-ccbin /usr/bin/gcc"
cd /hildafs/home/xhall/GitHub/pycbc/test

echo "=== GPU / scheme availability ==="
python3 -c "import pycbc; print('HAVE_CUDA', pycbc.HAVE_CUDA)"
nvidia-smi -L

echo
echo "=== CUDA-backend test suite (one scheme per process: --scheme cuda) ==="
python3 test_spatmplt_tidal.py --scheme cuda

echo
echo "=== throughput: SPAtmpltTidal vs SPAtmplt on the CUDA backend ==="
python3 - <<'PY'
import time
from pycbc.scheme import CUDAScheme
from pycbc.types import zeros, complex64
from pycbc.waveform import get_waveform_filter
import pycuda.driver as drv

DF = 1.0 / 256
FLEN = int(1024 / DF) + 1
COMMON = dict(delta_f=DF, f_lower=20., f_final=1024., distance=1.,
              phase_order=-1, spin_order=-1, amplitude_order=0,
              mass1=1.4, mass2=1.4, spin1z=0., spin2z=0.)

def bench(approx, out, n=2000, **extra):
    get_waveform_filter(out, approximant=approx, **COMMON, **extra)  # JIT warm-up
    drv.Context.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        get_waveform_filter(out, approximant=approx, **COMMON, **extra)
    drv.Context.synchronize()
    return (time.perf_counter() - t0) / n * 1e3   # ms / template

# A single CUDA scheme context for the whole benchmark (one scheme per process).
with CUDAScheme():
    out = zeros(FLEN, dtype=complex64)
    t_pp = bench('SPAtmplt', out)
    t_tid = bench('SPAtmpltTidal', out, lambda1=400., lambda2=400.)

print("SPAtmplt      : %.4f ms/template" % t_pp)
print("SPAtmpltTidal : %.4f ms/template" % t_tid)
print("overhead      : %+.2f %%" % (100 * (t_tid / t_pp - 1)))
PY
echo "=== done ==="
