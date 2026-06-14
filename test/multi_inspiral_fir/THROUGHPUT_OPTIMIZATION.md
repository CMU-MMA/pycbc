# `pycbc_multi_inspiral_fir` GPU throughput optimization

**Branch:** `fir-batch` (off `fir-ondevice`) — 6 commits, not pushed.
**Goal:** stop starving the A100 in the FIR coherent search, *without moving results*.
**Result:** **~11.8× on the filtering loop** (nsbh shard 0, 2 hr segment, A100‑40: ~36 min → ~3.0 min), all correctness gates bit‑identical, GPU memory‑bandwidth utilization 3–13% → mean 29% / max 60%. The search is now **GPU‑compute‑bound** instead of launch/transfer/host‑bound.

---

## 1. The problem

`pycbc_multi_inspiral_fir` ran correctly on GPU but the A100 was idle ~75–80% of the time (power 79–102 W of 400 W, mem‑bandwidth util 3–13%, 2.1 GB used). The work was hundreds of thousands of tiny, serial, host‑orchestrated GPU ops, each gated by host overhead (kernel launch, per‑call allocation, host↔device transfers, CPU waveform gen). Throughput was launch/transfer/host‑bound, not compute‑bound.

Test case (the handoff benchmark, unchanged production config): nsbh shard 0, segment 1 (2 hr), H1+L1, `--segment-length 4096`, `--fir-length 4096`, `--sngl-snr-threshold 4.0`, TaylorF2. **960 coarse groups, 58,296 fine templates, n_seg = 2.**

---

## 2. Approach — profile first, re‑profile after every change

The per‑group wall‑clock was fit to `a + b·n_fine` (a = fixed cost per coarse group, b = marginal cost per fine template). The existing verbose log already timestamps every `Coarse N/960` line, so the *before* split came for free; an env‑gated per‑stage profiler (`MIFIR_PROFILE`, one INFO line) gave the *after* splits.

| Stage | per‑coarse‑group wall | per‑fine‑template |
|---|---|---|
| **Original** | `0.418 + 0.0301·n_fine` | 30.1 ms |
| **Final** | `0.029 + 0.0046·n_fine` | **4.6 ms** |
| | **fixed cost 14×** | **per‑template 6.6×** |

The bottleneck moved three times. Each fix targeted the *measured* dominant term:

| After… | dominant term | filtering loop |
|---|---|---|
| (baseline) | per‑fine GPU loop 81% | 36 min |
| batched SNR convolutions | **threshold 59%**, coarse‑gen 34%, batch_snr 5% | 24 min |
| + parallel coarse‑gen | **threshold ~90%**, coarse‑gen 1% | — |
| + device‑memory threshold | threshold 60% (153 s), batch_snr 26% | ~4.2 min |
| + GPU scan‑compaction threshold | threshold 43% (84 s), **batch_snr 34% (GPU floor)** | **~3.0 min** |

Profiling knobs (all env‑gated, off/cheap by default):
`MIFIR_PROFILE_SYNC=1` device‑syncs after the batched SNR pre‑pass so `batch_snr` is attributed to the GPU compute instead of bleeding into the next synchronising stage; `MIFIR_THR_PROFILE=1` splits `threshold_only` into kernel/copy/sort; `MIFIR_MAX_COARSE=N` bounds the run for fast probes.

---

## 3. The four optimizations

### 3.1 Batch the fine‑template convolutions (the headline structural change)
`pycbc/filter/matched_ratio.py::RatioMatchedFilterControl.fine_snr_timeseries_batch`

Instead of one (multiply → ifft → assemble) per fine template, a **tile of K=32 fine filters** is processed in one pass:
- one tiled half‑spectrum multiply (broadcast the single cached forward FFT of the reference SNR across all K filters),
- **one** batched inverse cuFFT over all `K·n_blocks` transforms,
- one tiled assemble+rescale that writes the full output row (so the reusable output pool is safe to reuse across tiles even as the overlap‑save valid region shifts).

Supporting changes: a **reusable on‑device scratch buffer** (no per‑tile `cudaMalloc`/`cudaFree` — every free synchronises the device and starves it); the reference SNR is **kept on the device** (a gather kernel builds the block‑FFT input device→device, removing a host round trip per (coarse, segment, detector)).

The per‑element math is identical to the single‑template kernels → bit‑for‑bit the per‑template result (unit test `test_batched_matches_single` ≤ 7e‑8). This collapsed the GPU SNR compute to ~5% of the wall.

### 3.2 Device‑memory `threshold_only`
`pycbc/events/threshold_cuda.py`

After batching, the per‑template `threshold_only` dominated. It wrote every above‑threshold crossing to **device‑mapped (zero‑copy) pinned host memory** via `atomicAdd` — so each crossing was a serialized PCIe atomic. At `--sngl-snr-threshold 4.0` the search produces ~5×10⁸ crossings across the shard, so the PCIe atomics alone were ~10 min. Switched to ordinary **device global memory** for the counter + outputs, with a single bulk copy of the compacted result. `threshold_and_cluster` / `CUDAThresholdCluster` keep the mapped buffers (unchanged). ~4× on the threshold.

### 3.3 GPU ordered stream‑compaction in `threshold_only`
`pycbc/events/threshold_cuda.py` (default; `MIFIR_THR_SCAN=0` falls back)

Even on device, batching the *calls* didn't help — the residual was the **host numpy radix `argsort`** (~85% of the threshold; `sort=12.5 s` vs `kernel=1.6 s` in a probe). The 2‑detector coincidence (`get_coinc_indexes_cython_twodet_twocoinc`) is a two‑pointer merge that *requires sorted* per‑detector indices, so the sort can't be skipped — but it can move to the GPU:

```
mask[i] = |series[i]| > thr ? 1 : 0
pos     = ExclusiveScanKernel(mask)         # pos[i] = #crossings before i
if mask[i]:  out_loc[pos[i]] = i; out_val[pos[i]] = series[i]
```

The scatter writes each crossing at its prefix‑sum position, so the output is **already sorted by location** (= input index) — bit‑identical to the sorted output, with **no host sort**, and the work stays on the GPU (better duty cycle). (pycuda here has no `GenericScanKernel`, so this is `ExclusiveScanKernel` + mask/scatter `ElementwiseKernel`s.) threshold 153 → 84 s, `sort=0`, and bandwidth utilization rose.

### 3.4 Parallel coarse‑template generation
`bin/pycbc_multi_inspiral_fir` (`--fir-coarse-workers N`, default 0 = serial) + `matched_ratio.coarse_gen_init/coarse_gen_one/coarse_reconstruct`

The CPU TaylorF2 coarse references (~0.43 s each × 960 ≈ 7 min, GPU idle) are generated in background CPU workers and prefetched ahead of the main loop, overlapping with GPU filtering: 407 → 9 s.

Implementation gotcha (took one wrong turn): the worker pool **must** be a `fork` `multiprocessing.Pool` created **before** `scheme.from_cli` (the CUDA context).
- `spawn` re‑executes the driver script in every worker (no `__main__` guard) → workers re‑read frames instead of generating templates (diagnostic: "Reading Frames" appears 34× instead of 2×).
- `fork` *after* CUDA init can't safely duplicate the CUDA context.
- An eager `Pool` forks all workers at construction, so they provably predate CUDA init; `flen`/`delta_f` are available pre‑CUDA from `strain_segments_dict`.

Workers return FD samples + the scalar attributes `FilterBank.__getitem__` sets; `coarse_reconstruct` rebuilds an identical device `FrequencySeries` (reattaching the cached `sigmasq`). CPU LAL generation is deterministic → bit‑identical (`test_parallel_coarse_matches_serial`).

---

## 4. Driver structure (semantics unchanged)

The fine‑template loop is now tiled. On CUDA a **batched pre‑pass** reconstructs the K SNR series for every (segment, detector) into a reusable per‑(segment,detector) output pool, then runs **one batched threshold** over the K contiguous rows per (segment, detector); flat crossings decode to (template = `loc//L`, time = `loc%L`) via `np.searchsorted` (rows come back sorted = row‑major). The per‑template coherent processing — `new_template` → segment loop → `finalize_template_events`, the full coherent/null/coincidence statistic — is **unchanged**, so EventManager semantics and results are identical. The CPU path computes each series per template as before (the bit‑exact parity reference).

---

## 5. Auto‑tuning to the GPU type

The driver logs the detected device and sizes the batch tile from **measured free memory**, so the same code adapts across GPUs without edits:

```
INFO : GPU: NVIDIA A100-SXM4-40GB (compute 8.0), 38.2 GB free / 39.4 GB total
INFO : FIR batch tile K=32 (~13.93 GB batch buffers; free 38.2 GB of 39.4 GB)
```

`K = min(--fir-tile, 0.5·free / bytes_per_k)`, where `bytes_per_k` accounts for the retained SNR rows (per segment×detector), the per‑detector multiply/ifft scratch, and the device threshold + scan buffers (20 B/sample with scan, 12 without). On an A100‑40/80 this takes the full K=32 (default `--fir-tile`); on a V100‑16/32 GB it automatically takes a smaller K. The 0.5×free budget leaves headroom for the cuFFT work areas (only 6 distinct tap‑counts in the bank → a bounded set of plans). The threshold/scan device buffers grow on demand to the largest series seen.

---

## 6. Correctness gates (results did NOT move)

Run via Slurm GPU (`test/multi_inspiral_fir/run_gate_ab.sbatch`). Validated after **every** stage above.

- **Gate A — unit parity:** `test/test_multi_inspiral_fir.py` (cpu + cuda) 12/12 OK + `test/test_threshold.py -s cuda` 6/6 OK. Includes `test_batched_matches_single` (tiled vs single ≤ 7e‑8), `test_parallel_coarse_matches_serial` (samples + sigmasq bit‑identical), CPU↔CUDA engine parity ≤ 2.4e‑7.
- **Gate B — FIR vs brute (end‑to‑end):** FIR‑CUDA vs `pycbc_multi_inspiral` brute = **0.2057%** on the loudest coherent SNR — *identical to the pre‑optimization number*; FIR‑CUDA vs FIR‑CPU = **1.79e‑5%** (bit‑identical to single precision); 0 strong missed/extra.
- **Gate C — throughput + saturation:** §7.

---

## 7. Throughput + GPU saturation (Gate C)

nsbh shard 0, segment 1 (2 hr), A100‑40, `test/multi_inspiral_fir/run_throughput_nsbh.sbatch`.

| Metric | Before (original) | After (this branch) |
|---|---|---|
| **Filtering loop** | **~36 min** (2160 s) | **~3.0 min** (183 s) → **~11.8×** |
| Total job wall | — | 7.2 min (incl. ~4 min unchanged frame I/O + PSD) |
| GPU power (filtering) | 79–102 W | mean 121 W, median 94 W, **max 274 W** |
| GPU mem‑bandwidth util | 3–13% | mean 29%, median 41%, **max 60%** |
| GPU util ("kernel scheduled") | 100% (misleading) | mean 35%, median 53%, max 84% |
| Peak GPU memory | 2.1 GB | ~20 GB (K=32 batch buffers) |

**Clean per‑stage profile** (`MIFIR_PROFILE_SYNC=1`, so `batch_snr` reflects true GPU time):

```
coarse_gen=12.8(6%)  ref_setup=21.5(11%)  batch_snr=67.7(34%)  threshold=84.5(43%)  coherent=8.3(4%)  cluster=3.5(2%)
  threshold internals:  kernel=56.2s  copy=5.7s  sort=0.0s   (host argsort eliminated)
```

---

## 8. New dominant term & remaining levers

The search is now **GPU‑compute‑bound**, which is the desired end state ("feed the GPU"):
- `batch_snr` ≈ 67 s (34%) — the FFT‑based SNR reconstruction; this is the genuine compute floor.
- threshold kernel ≈ 56 s — the mask + scan + scatter compaction (now on the GPU).
- `thr_decode` ≈ 21 s — the only remaining notable **host** term (per‑crossing flat‑index → (template,time) decode + bucketing).

Levers not pursued (diminishing returns past the ≥5× target / ~10× hope; would chase the last bit and further raise the duty cycle):
1. Fuse the threshold mask into the assemble kernel (save one full read of the SNR per tile).
2. Vectorize `thr_decode` / reduce its per‑call numpy overhead.
3. Pipeline (double‑buffer): overlap GPU SNR compute of tile N+1 with host post‑processing of tile N.

---

## 9. How to reproduce

```bash
# env (see also gpu-test-env-setup memory)
export MAMBA_ROOT_PREFIX=/hildafs/projects/phy220048p/share/micromamba
eval "$(/hildafs/projects/phy220048p/share/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /hildafs/projects/phy220048p/share/envs/ssm_pipeline
export PYTHONPATH=/hildafs/home/xhall/GitHub/pycbc:${PYTHONPATH}
export PYCUDA_DEFAULT_NVCC_FLAGS="-ccbin /usr/bin/gcc"; module load cuda/12.4.0

# correctness (Gate A + B)
sbatch test/multi_inspiral_fir/run_gate_ab.sbatch

# throughput (Gate C) — full nsbh shard 0, both optimizations
sbatch -c 32 --export=ALL,MIFIR_TILE=32,MIFIR_COARSE_WORKERS=24 \
       test/multi_inspiral_fir/run_throughput_nsbh.sbatch
# add MIFIR_PROFILE_SYNC=1 for a clean per-stage profile;
# MIFIR_MAX_COARSE=N for a fast bounded probe; MIFIR_THR_SCAN=0 to A/B the threshold path.
```

Production usage: pass `--fir-coarse-workers <≈cores>` (e.g. 24) to `pycbc_multi_inspiral_fir`; `--fir-tile` defaults to 32 and is auto‑capped to GPU memory. The threshold scan‑compaction is on by default.

---

## 10. Files changed

```
bin/pycbc_multi_inspiral_fir                       (+506/-...)  tiling, fork prefetch, profiler, auto-tune
pycbc/filter/matched_ratio.py                      (+~440)      fine_snr_timeseries_batch, device ref-SNR, coarse_gen_*
pycbc/events/threshold_cuda.py                     (+~200)      device-memory threshold_only + GPU scan-compaction
test/test_multi_inspiral_fir.py                    (+104)       test_batched_matches_single, test_parallel_coarse_matches_serial
test/multi_inspiral_fir/run_gate_ab.sbatch         (new)        Gate A+B in one allocation
test/multi_inspiral_fir/run_throughput_nsbh.sbatch (new)        Gate C + nvidia-smi sampler
```

Commits (`git log fir-ondevice..fir-batch`):
```
FIR engine: batch fine-template convolutions on GPU (Phase 1)
mifir: add MIFIR_PROFILE_SYNC for accurate batched-GPU profiling attribution
mifir: batch the per-template threshold + parallel coarse-template gen
mifir: fix parallel coarse-gen — fork pool created pre-CUDA (was broken spawn)
threshold_cuda: device-memory threshold_only (kill per-crossing PCIe atomics)
threshold_cuda: ordered GPU stream-compaction (no host sort) + GPU auto-tune
```
