# Testing handoff — `pycbc_multi_inspiral_fir`

You are picking up a **bug-hunting / verification** task on a new, partially
validated executable: `bin/pycbc_multi_inspiral_fir` — a coherent, FIR-de-chirped,
GPU-accelerated directed (known-sky) multi-detector matched-filter search. Your
job is to **find correctness bugs and untested failure modes**, not to add
features. Two review passes have already happened (see §7–8 — do not re-report
those). This document tells you the environment, what's validated, what's *not*,
and the highest-yield places to look.

Branch: `multi-inspiral-fir` (off `master`). Working dir / checkout:
`/hildafs/home/xhall/GitHub/pycbc` (this is a symlink to
`/hildafs/projects/phy220048p/xhall/GitHub/pycbc`; same files). Changes are
**uncommitted** in the working tree — do not commit/push without the human's OK.

---

## 1. HARD CONSTRAINTS (read first — violating these is the #1 mistake)

1. **NEVER run more than a few SECONDS of CPU compute on the login node.** It's a
   shared HPC head node. Submit Slurm jobs to **RM** or **HENON** (CPU) /
   **TWIG-GPU** / **HENON-GPU** / **RITA-GPU** (GPU), account `phy220048p`.
   Quick things OK inline: imports, `h5py` header reads, `<~few-second` unit
   checks. Bank builds, brute/FIR runs, anything multiprocess+BLAS → Slurm.
2. **NEVER modify the shared env** `/hildafs/projects/phy220048p/share/envs/ssm_pipeline`.
3. **NEVER use `pip`.** If a package seems missing, stop and ask the human.
4. **Out of scope to modify**: the original `bin/pycbc_multi_inspiral`,
   `SPAtmplt`. (You MAY read them as the ground-truth reference.)
5. In multiprocess jobs always `export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`
   (BLAS oversubscription otherwise — it bit us hard).

## 2. Environment (copy exactly)

```bash
export MAMBA_ROOT_PREFIX=/hildafs/projects/phy220048p/share/micromamba
eval "$(/hildafs/projects/phy220048p/share/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /hildafs/projects/phy220048p/share/envs/ssm_pipeline
export PYTHONPATH=/hildafs/home/xhall/GitHub/pycbc:${PYTHONPATH:-}
module load cuda/12.4.0                                    # GPU jobs only
export PYCUDA_DEFAULT_NVCC_FLAGS="-ccbin /usr/bin/gcc"     # GPU jobs only
```
Verify overlay: `python3 -c "import pycbc.filter.matched_ratio as m; print(m.__file__)"`
must resolve under the checkout. **Env facts that surprised us**: this env has
**no `mkl_fft` and no `cupy`**; GPU = **PyCUDA + scikit-cuda** (the `cuda`
scheme), NOT cupy. (A separate `/opt/packages/anaconda3` python has mkl_fft but
is broken/no-GPU — do not use it.)

## 3. The code (what to scrutinize)

- `bin/pycbc_multi_inspiral_fir` — the driver. Starts from `pycbc_multi_inspiral`
  and keeps the coherent statistic **verbatim**, replacing the per-IFO brute
  matched filter with the ratio engine. Loop is **coarse → (reference SNR per
  detector, cached per segment) → fine template → segment → slides → sky**. The
  fine template is the `EventManagerCoherent` "template" unit.
- `pycbc/filter/matched_ratio.py` — `RatioMatchedFilterControl`. Key methods:
  `compute_reference_snr` (per-detector coarse-template SNR), `fine_snr_timeseries`
  (CPU: scipy.fft + cython kernels), `_fine_snr_timeseries_cuda` (GPU: scikit-cuda
  batched cufft + a PyCUDA `ElementwiseKernel` half-spectrum multiply), plus the
  older `process_segment`/`_execute_blocked_kernel` peak-finder (used by the
  single-detector `pycbc_inspiral_fir`, not by the coherent driver).
- `pycbc/waveform/bank.py::RatioFilterBank` — coarse+fine+taps bank reader,
  `snr_rescale`/`sigma_rescale` (mchirp vs precalculated_sigma).
- Reference (ground truth, do not modify): `bin/pycbc_multi_inspiral`,
  `pycbc/events/coherent.py`, `pycbc/events/eventmgr.py` (`EventManagerCoherent`).
- Tests: `test/test_multi_inspiral_fir.py` (unittest; `python3 ... [--scheme cuda]`).
- Harness: `test/multi_inspiral_fir/` — `make_validation_banks.py` (tau0-grid
  banks), `injection.ini`, `compare_triggers.py` (template_hash+time matching),
  and Slurm scripts (see §5).

## 4. What is validated — and the NARROW box it was validated in

Passing gates (see `RESULTS.md` for numbers):
- FIR vs brute coherent SNR: **~0.2%** (CPU and GPU); 0 strong triggers missed.
- CPU↔CUDA parity: engine **1.5e-7**, full search **3.6e-5%**.
- Tap fidelity ≥0.99; throughput FIR-CPU ~2×, FIR-CUDA ~3× (filtering, 16 s templates).

**Everything above was measured in ONE narrow scenario.** Treat anything outside
this box as UNTESTED and likely buggy:
- **2 detectors only** (H1+L1).
- **Single sky point** (no `--sky-grid`).
- **Zero-lag only** (no `--do-shortslides`).
- **`standard` projection only** (no left/right/left+right).
- **`zeroNoise` fake data** (no Gaussian noise, no real frames).
- **Static analytic PSD** (`aLIGOZeroDetHighPower`) — same for every segment.
- **One injection**, point-particle **TaylorF2**, masses ~1.7–2.0 M☉, τ₀≈10–13 s.
- **Multi-rate OFF**: tap rate == engine rate (2048/2048, decimate=1).
- **Odd tap counts** (251) only.
- Tiny banks (20–150 templates), short (~16 s) templates.

## 5. How to run things

CPU unit suite (login-node OK, ~1 s): `python3 test/test_multi_inspiral_fir.py --scheme cpu`

Slurm (submit, then poll `squeue -j <id>` / read the `*_%j.log` in
`/hildafs/projects/phy220048p/xhall/scratch/mifir`):
```bash
sbatch test/multi_inspiral_fir/run_validation.sbatch      # RM: build bank + CPU gate
sbatch test/multi_inspiral_fir/run_validation_gpu.sbatch  # TWIG-GPU: gate + CPU/CUDA parity
sbatch test/multi_inspiral_fir/run_throughput.sbatch      # TWIG-GPU: suite D + larger bank
sbatch test/multi_inspiral_fir/run_gpu_tests.sbatch       # TWIG-GPU: unit suite (cpu+cuda)
```
Gate scripts now hard-fail (exit nonzero → Slurm `FAILED`) on any executable
crash or gate failure. The comparator (`compare_triggers.py`) exits 2 on a
failed gate, 0 on pass. **Always check `sacct -j <id> --format=State,ExitCode`,
not just that the log printed** — a green-looking log with `FAILED`/nonzero exit
is a real failure. Assets persist in `…/scratch/mifir/` (small bank) and
`…/scratch/mifir/big/` (throughput bank).

## 6. HIGHEST-YIELD places to hunt for bugs (ranked)

These are real, currently-untested paths. Each is a concrete experiment.

1. **Segment-dependent PSDs (multi-segment, real/Gaussian noise).** The
   per-segment `sigma` fix is correct-by-construction but UNEXERCISED (static
   PSD). Run with `--fake-strain H1:aLIGOZeroDetHighPower …` + a fixed
   `--fake-strain-seed`, OR `--psd-estimation median` over multi-segment data so
   each segment gets a different estimated PSD. Confirm FIR still matches brute
   when σ genuinely varies per segment. **This is the most important gap.**
2. **Multi-rate (decimate ≠ 1).** Build the tap bank at `--sample-rate 4096` and
   run the search at `--sample-rate 2048` (engine_sr). The `_fft_all_filters`
   high-res decimation path and the `decimate` factor in `compute_reference_snr`
   are completely untested in this work. High bug probability.
3. **Even tap counts, end-to-end.** Build a bank with even `--n-taps` (e.g. 250)
   and confirm the gate vs brute. The `bad_start` fix changed even-tap behavior;
   it's only been argued by algebra + shown a no-op for odd. NOTE: the per-block
   half-spectrum approximation confounds *cross-blocking* unit tests for this —
   compare against **brute** (full segment) instead, at the detection level.
4. **>2 detectors and single detector.** H1+L1+V1 (`--instruments H1 L1 V1`) and
   single-IFO. The `nifo>1` vs `nifo==1` branches in the driver are copied from
   brute but untested here; `network_out_vals['null_snr']` is not set on the
   nifo==1 path (inherited from brute — check it doesn't write garbage).
5. **Projections** `left`, `right`, `left+right` (the `left+right` max-over-
   polarization block is intricate and copied verbatim — verify it).
6. **Time slides** (`--do-shortslides --slide-shift 1`) and **sky grids**
   (`--sky-grid file.hdf` with several points). Confirms the slide/position
   loops and `time_delay_idx` indexing match brute.
7. **Off-peak / sub-threshold accuracy.** The per-block half-spectrum (analytic)
   makes the FIR SNR ~1% different from the true full-segment SNR OFF the peak
   (we only validated the peak to 0.2%). Quantify FIR-vs-brute over the whole
   SNR time series (not just at triggers) — relevant for background/false-alarm
   estimation. Is ~1% acceptable? Does it create/destroy sub-threshold triggers?
8. **Chi-square reality.** χ² is stubbed to 0 (so `reweighted_snr == coherent_snr`
   and clustering ranks on un-downweighted SNR; `chisq_dof` is written as the
   nonphysical 1.5). Compare which triggers survive clustering vs brute (brute
   downweights by χ²). Quantify the divergence.
9. **Normalization far from the coarse reference.** `mchirp` vs
   `precalculated_sigma` diverge for fine templates whose best coarse match is
   poor. Build a bank with sparse coarse coverage and compare both methods vs
   brute. (`--high-frequency-cutoff` vs `h_norm` was a real bug — FIXED in
   round 3, gated by unit suite E; the fine-template rescale remains a
   full-band ratio, documented in the CLI help.)
10. **Scale / memory.** Only ≤150 templates tested. The driver caches
    `ref_cache[(s_num, ifo)]` (ref SNR + block-FFT cache) for a whole coarse
    group across all segments — memory grows with n_segments × n_ifo. Try many
    segments / a larger coarse group and watch RSS / GPU memory.
11. **Injection parameter recovery**, not just SNR ratio: does the FIR trigger
    recover the correct geocentric time, sky bin, and template (mass) as brute?
12. **Tidal templates** (`SPAtmpltTidal` / TaylorF2 with lambdas) — the eventual
    science target; never run through this pipeline.

## 7. Already FIXED (do NOT re-report as new)

Round 1 (human review): per-segment σ in projection; Slurm scripts hard-fail;
comparator enforces 0 strong missed/extra; default normalization →
`precalculated_sigma`; `chmod +x`. Round 2 (adversarial workflow, 16/18
confirmed): `bad_start` `n//2`→`(n-1)//2` (even-tap overlap-save); CPU
`loop_start` clamp; GPU `n_blocks==0` guard; `RatioFilterBank.template_thinning`
empty→`np.array([])`; hard-error on `--injection-filter-rejector-trigger-window`
(FIR doesn't implement brute's per-trigger injection-window trim). Also: removed
the hard `mkl_fft` dependency (→ scipy.fft); fixed a pre-existing crash in
`InjFilterRejector` for disabled rejectors. All re-validated (gate 0.2057%,
parity 3.6e-5%, jobs `COMPLETED 0:0`).

Round 3 (Slurm campaign 962176–962189 + this pass; see RESULTS.md "Round 3
fixes" for details): `--high-frequency-cutoff` now folded into the reference
`h_norm` (was a 5.5%-class SNR deflation when set; unit suite E gates it);
even-tap-count filter centering aligned to the `pycbc_fir_bank` design
convention — roll `(K-1)//2`, `bad_start = K//2` (was the 1.5% multi-rate
even-tap deficit in 962189; no-op for odd counts; unit suite F gates it;
`get_fd_fir` aligned too); `compare_triggers.py --check-metadata` strict mode
(slide + sky recovery), with the near-degenerate-sky-grid caveat from 962187
documented in the comparator and the sbatch scripts.

## 8. Known/documented limitations (real, but already disclosed in RESULTS.md)

χ² vetoes stubbed; speedup modest at short template lengths (grows in the
sub-solar long-template regime, which can't be brute-compared); two-σ
`rec`/`target` normalization in `precalculated_sigma` (bounded sub-percent);
fine-template rescale is a full-band sigma ratio even under
`--high-frequency-cutoff` (second-order; the first-order `h_norm` bug is
fixed); single-scenario validation (widened by the round-3 campaign).

## 9. REFUTED in round 2 (verified NOT bugs — don't waste time)

- GPU `_gpu_batched` cache "missing n_taps guard": safe — the driver passes a
  fixed per-coarse-group tap count and a fresh `block_cache` per (segment,
  detector), so the cached grid is never reused with a different `n_taps`.
- "Fine template could belong to >1 coarse group → double count": each fine
  template maps to exactly one coarse group via `fine_coarse_map`.

## 10. Gotchas that will waste your time if you don't know them

- **Cross-blocking comparisons are confounded.** The per-block half-spectrum
  (analytic) on COMPLEX `ref_snr` makes the engine output depend on the block
  grid (`fir_fft_len`, `n_taps`) in BOTH real and imaginary parts (~1% off-peak).
  So you CANNOT validate the overlap-save valid-region offset by comparing two
  blockings (different `n_taps`, full-FFT vs blocked, even-vs-odd) — they differ
  for reasons unrelated to bugs. Compare against **brute** (full-segment matched
  filter) at the detection level, or compare CPU-vs-CUDA at IDENTICAL blocking.
- `pycbc.fft`/cufft **ifft is unnormalized** (no 1/N); the GPU kernel folds 1/N
  in. If you touch the FFT path, keep that.
- Multi-IFO CLI: pass `--gps-start-time`/`--gps-end-time` and `--ra`/`--dec`
  **without** an `IFO:`/`rad` token issue — bare values become the all-IFO
  default; `H1:`-prefixed keys break `EventManagerCoherent.write_to_hdf`'s
  lowercased lookup, and a `" rad"` suffix splits into a stray arg.
- `--order -1` is required (TaylorF2/SPAtmplt length function does `int(phase_order)`).
- Match triggers between two outputs by **`template_hash` + time**, never time
  alone (a loud signal rings up many templates at the same time).

## 11. Suggested first move

Reproduce the green baseline (`sbatch run_validation_gpu.sbatch`, check
`sacct` ExitCode == 0:0 and the gate text), then attack §6 item 1
(segment-dependent PSD) and item 2 (multi-rate) first — those are the most
likely to surface a real bug and the least exercised.
