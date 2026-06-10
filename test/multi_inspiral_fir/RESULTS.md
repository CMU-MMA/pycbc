# `pycbc_multi_inspiral_fir` — results note

Coherent, FIR-de-chirped, GPU-accelerated directed (known-sky) multi-detector
matched-filter search. Built on branch `multi-inspiral-fir`. This note records
the test-suite outcomes (A–E from the handoff), what was built, how to
reproduce, and the honest caveats.

Method: Ratio-Filter de-chirping (Nitz, Kacanja & Soni 2026, arXiv:2601.18835).
Coherent statistic: Harry & Fairhurst 2011; Williamson et al. 2014.

## Status summary

| Suite | What | Result | Verdict |
|---|---|---|---|
| **B** | FIR pipeline vs brute `pycbc_multi_inspiral`, coherent SNR (CPU) | signal **0.22%**, matched ≤0.7%, 0 missed | **PASS** (gate ~1–2%) |
| **B** | FIR pipeline vs brute, coherent SNR (GPU) | signal **0.21%** | **PASS** |
| **B (unit)** | new FIR engine vs existing `matched_filter_core`, single template | **0.001%** | **PASS** |
| **A** | CPU↔CUDA engine parity (`fine_snr_timeseries`) | **1.5e-7** rel | **PASS** (tight) |
| **A** | CPU↔CUDA full-search parity (coherent SNR) | **3.6e-7** rel, all triggers | **PASS** (tight) |
| **C** | FIR tap fidelity (reconstructed vs target) | min **0.998**, 100% ≥0.99 | **PASS** |
| **D** | Throughput, FIR vs brute (filtering only, 16 s templates) | FIR-CPU **2.0×**, FIR-CUDA **3.0×** | measured |
| **E** | Distance-constrained statistic | not implemented | **deferred** (Phase 3, experimental) |

All numbers are on H1+L1, point-particle TaylorF2, a single zeroNoise +
injection segment at a fixed sky position.

## Deliverables (branch `multi-inspiral-fir`, uncommitted)

- `bin/pycbc_multi_inspiral_fir` — the coherent FIR search executable. Starts
  from `pycbc_multi_inspiral`, keeps the full coherent statistic
  (`get_coinc_indexes`, `coherent_snr`, `null_snr`, polarization projection,
  `EventManagerCoherent`) **unchanged**, and replaces the per-IFO brute matched
  filter with the ratio engine. Loop is restructured coarse → (reference SNR
  once per detector) → fine → segment; fine template is the EventManager unit.
- `pycbc/filter/matched_ratio.py` — extended `RatioMatchedFilterControl`:
  - `compute_reference_snr` — per-detector reference (coarse-template) SNR.
  - `fine_snr_timeseries` — full per-IFO complex SNR(t) for a fine template
    (the coherent-stage input), CPU (scipy + cython) **and** CUDA
    (`_fine_snr_timeseries_cuda`).
  - GPU path: batched on-device cufft (scikit-cuda) + a PyCUDA ElementwiseKernel
    `fir_analytic_mult_batched` for the half-spectrum multiply; the
    fine-template-independent block FFTs are computed once per (segment,
    detector) and reused. Removed the hard `mkl_fft` dependency (→ scipy.fft;
    `ssm_pipeline` has no mkl_fft).
- `test/test_multi_inspiral_fir.py` — unittest suite (`parse_args_all_schemes`
  idiom): engine CPU↔CUDA parity (A), new-vs-existing matched-filter (B unit),
  block-cache consistency, tap fidelity (C). Run:
  `python3 test/test_multi_inspiral_fir.py [--scheme cuda]`.
- `test/multi_inspiral_fir/` harness:
  - `make_validation_banks.py` — τ₀-grid coarse+fine banks (controls the
    coarse↔fine chirp-time gap so FIR filters stay short).
  - `injection.ini`, `compare_triggers.py` (template-hash + time matching).
  - `run_validation.sbatch` (CPU gate, RM), `run_validation_gpu.sbatch`
    (GPU gate + CPU/CUDA parity, TWIG-GPU), `run_throughput.sbatch` (suite D),
    `run_gpu_tests.sbatch` (unit tests on GPU).
- `pycbc/inject/injfilterrejector.py` — one-line robustness fix: a *disabled*
  `InjFilterRejector` now exposes `match_threshold`/`chirp_time_window`/
  `inj_trigger_window` as `None` instead of crashing
  `InjectionSet.apply` — this bug broke **any** `--injection-file` run without
  IFR options, brute included.

## Detailed results

### B — FIR vs brute (the main correctness gate)
Small bank (30 fine, 15 coarse, τ₀ 10–13 s), moderate injection (170 Mpc,
coherent SNR ~25), `--template-normalization-method precalculated_sigma`:
- CPU: brute loudest coherent SNR 25.382 vs FIR 25.325 → **0.224%**; all matched
  triggers ≤0.67%; **0 missed**, 2 extra weak (~SNR 6) sidelobe triggers.
- GPU (149-template bank): brute 25.4297 vs FIR-CUDA 25.3774 → **0.206%**.

`mchirp` normalization gave a 3% outlier on a neighbour template;
`precalculated_sigma` (the bank's stored σ ratios) brought it to 0.2%. The few
weak extra/missed triggers are FIR-approximation differences on sidelobes well
below the signal — expected, and the handoff explicitly allows it.

### B (unit) — new vs existing, single template
`test_fir_reconstructs_brute_matched_filter`: inject one fine template, recover
its complex SNR two ways — existing `matched_filter_core(fine)` vs new
`compute_reference_snr(coarse)` + `fine_snr_timeseries(fine taps)`. Peak ratio
**0.99999**. No shared code path. (Non-zero residual = FIR tap mismatch — a real
cross-check, not a tautology.)

### A — CPU↔CUDA parity
- Engine (`fine_snr_timeseries`, synthetic): **1.5e-7** relative. The test
  asserts the difference is both `> 0` (a 0.0 would mean a silent CPU fallback)
  and `< 1e-4` (single-precision cufft-vs-scipy).
- Full coherent search (149 templates): all 36 triggers match, max |ratio−1| =
  **3.6e-5%** (~3.6e-7), 0 missed/extra. The batched-GPU optimization is a pure
  performance change — it did not move the results.

### C — tap fidelity
`pycbc_fir_bank` QA `filter_match`: min 0.998, median 0.999, **100% ≥ 0.99**
(both banks). Achieved at the lowest tap count (251) by keeping the coarse↔fine
τ₀ gap ≲0.1 s.

### D — throughput (149 fine templates, 16 s templates, H1+L1)
| Engine | Total wall | Filtering-only* | vs brute |
|---|---|---|---|
| brute | 37 s | 21.6 s | 1× |
| FIR-CPU | 28 s | 10.8 s | **2.0×** |
| FIR-CUDA | 25 s | 7.3 s | **3.0×** (1.5× over FIR-CPU) |

\*filtering-only = total × (1 − `setup_time_fraction`). Setup (waveform
generation, PSD, bank read, CUDA init) is shared overhead that dilutes the
wall-clock and amortizes as the bank grows.

### E — distance-constrained statistic
Not implemented. Phase 3 is flagged experimental/optional in the handoff; Phases
0–2 + suite D are the shippable deliverable.

## Review fixes (two verification passes)

A reviewer pass + a 7-reviewer/4-verifier adversarial workflow (16/18 candidate
findings confirmed) drove these fixes:

Correctness / robustness (fixed):
- **Per-segment sigma** in the coherent projection (was hoisted to segment 0;
  wrong for time-varying PSDs). Matches brute.
- **Overlap-save valid-region offset** `bad_start`: `n//2` is correct only for
  ODD tap counts; for EVEN counts the linear-convolution-clean region starts at
  `ceil(n/2)-1`, so `n//2` dropped one clean sample and committed one
  wraparound-contaminated sample per block. Fixed to `(n-1)//2` (identical for
  odd, so a no-op for the odd 251-tap production banks; corrects even-tap
  templates). Applied to the CPU, CUDA, and peak-finder paths. NOTE: the effect
  is one sparse boundary sample per block and sits *below* the engine's inherent
  per-block half-spectrum (analytic) approximation, so it is not isolable by a
  cross-blocking unit test and does not move the detection-level results; the
  fix rests on the direct wraparound algebra (clean region = [R-1, N-1-L],
  R=n-n//2, L=n//2) and is a no-op on all currently-tested (odd) banks.
- **CPU `loop_start` clamp** to 0 (the CUDA path already clamped) -- guards a
  negative `ref_snr` index when `analyze.start < bad_start`.
- **GPU empty-grid guard**: return zeros for `n_blocks==0` (was an IndexError;
  CPU returned zeros).
- **`--injection-filter-rejector-trigger-window` now hard-errors** in the FIR
  driver: brute trims per-detector triggers to injection windows
  (`find_indices_in_injection_intervals`); the FIR path doesn't implement that,
  so rather than silently diverge it refuses the (off-by-default, out-of-scope)
  option.
- **`RatioFilterBank.template_thinning`** empty result is now `np.array([])`
  (was a Python list that broke a subsequent per-detector call).
- **Slurm gate scripts hard-fail**: `rc`-tracked `exit $rc` (were `set -uo
  pipefail` with `|| echo`, so a failed gate still reported `COMPLETED 0:0`).
  Verified the comparator returns exit 2 on failure.
- **Comparator enforces zero *strong* missed/extra** (`--max-strong-mismatch`,
  default 0); previously "0 missed" was printed but not gated. Weak sidelobe
  differences are reported but tolerated. Also: missing/empty trigger groups
  now produce a clean gate FAIL instead of a KeyError.
- Default `--template-normalization-method` -> `precalculated_sigma` (the
  validated-accurate path; `mchirp` gave a ~3% neighbour-template outlier).
- `chmod +x bin/pycbc_multi_inspiral_fir`.

Documented (intentional / opt-in / out-of-scope, not changed):
- Two-sigma normalization (`rec_sigma` for SNR amplitude vs `target_sigma` for
  the projection weight) -- intentional in `precalculated_sigma`; bounded by
  (1 - tap_match), sub-percent.
- `--high-frequency-cutoff` is not folded into the reference `h_norm` (opt-in;
  default None->Nyquist is consistent; unexercised by the validation).
  **Superseded: FIXED in round 3 (below).**
- chi-square stub -> `chisq_dof` output is the nonphysical 1.5 and the
  clustering rank (`reweighted_snr`) lacks the chi-square downweight (Phase-1
  stub; documented).

Refuted by verification (not bugs as used): the GPU `_gpu_batched` cache lacks
an `n_taps` guard (safe -- the driver uses a fixed per-coarse-group tap count
and a fresh cache per (segment, detector)); the fine-template double-count
concern (each fine template maps to exactly one coarse group).

## Round 3 fixes (bug-hunt pass, 2026-06)

Background: the round-3 Slurm campaign (jobs 962176--962189, all
`COMPLETED 0:0`) extended validation to PSD variation, multi-rate, even taps,
single/three IFO, projections, and slides+sky grids. Two real defects came out
of it, both fixed here:

1. **`--high-frequency-cutoff` deflated every SNR.**
   `RatioMatchedFilterControl.compute_reference_snr` used the cached
   *full-band* `ref_template.sigmasq(psd)` as `h_norm` while passing
   `high_frequency_cutoff=self.f_high` to `matched_filter_core`, so with a
   cutoff set the filter integrated `[f_lower, f_high]` but the normalization
   integrated the full band: every SNR low by
   `sqrt(sigmasq_full/sigmasq_band)` (measured 5.5% on the validation bank at
   256 Hz). Fixed: `h_norm` is now the band-limited sigmasq over exactly the
   filter band. Affects `pycbc_multi_inspiral_fir` AND `pycbc_inspiral_fir`
   (shared engine). The fine-template rescale still uses the bank's stored
   full-band sigma ratios -- a second-order (ratio-of-ratios) approximation of
   the same class as the generation-PSD approximation, documented in the CLI
   help. No validated configuration used the option (all ran f_high=None), so
   all green results stand. The brute `pycbc_multi_inspiral` has no such
   option, so cutoff runs cannot be gated against it; unit suite E gates this
   instead (peak SNR == band sigma for a template-as-data segment).

2. **Even tap counts: engine centering was one tap off the bank's design
   convention** -- the real cause of the 1.501% multi-rate+even-taps deficit
   (job 962189). `pycbc_fir_bank` designs (and QA-verifies) tap `j` at time
   `j - (K-1)//2` for BOTH parities (even K spans `-K/2+1 .. K/2`); the
   engine's `_fft_all_filters` rolled by `K//2`, i.e. one tap-rate sample off
   for even K. Tap *design* quality was ruled out directly: stored
   `filter_match` min is 0.99841 on the multi-rate even bank, same as odd. At
   decimation 1 the offset is a whole engine sample -- a pure translation that
   preserves peaks (hence the benign 0.19% in job 962179, plus a silent
   1-sample trigger-time bias). At decimation 2 it is HALF an engine sample
   (244 us), so the reconstructed series samples the SNR envelope off-peak:
   the 1.5%. Fixed: roll by `(K-1)//2` and `bad_start = K//2` (CPU, CUDA, and
   peak-finder paths), and `RatioFilterBank.get_fd_fir` aligned to the same
   convention. Both are identical for odd K, so every odd-tap result is
   bit-unchanged; this *supersedes* the round-2 `bad_start` algebra, which was
   internally consistent with the old (mis-centered) roll. Unit suite F gates
   the convention (a delta tap at `(K-1)//2` must be the identity filter,
   single-rate and decimated).

Round-3 verification (all `COMPLETED 0:0`):
- 962192 unit suite: 10/10 pass on CPU AND CUDA, including new suites E
  (cutoff normalization) and F (even-tap centering identity).
- 962193 multi-rate + even taps rerun: loudest SNR error **1.501% ->
  0.4462%**; CUDA/CPU parity 3.6e-05%. The residual ~0.45% (vs 0.135%
  multi-rate odd) is consistent with a half-tap-rate-sample (quarter
  engine-sample) effect inherent to even-count designs -- an even-span FIR
  has no exact center tap -- plus the ~0.1% tap mismatch; it is a design-side
  property, not an engine indexing error. Practical guidance: prefer ODD
  `--n-taps` (an exact center exists); even counts pass comfortably but carry
  the extra fraction of a percent under decimation.
- 962194 single-rate even taps rerun: 0.1932% -- bit-identical to the
  pre-fix 962179, confirming the decimate=1 offset was a pure (peak-
  preserving) integer-sample translation; trigger times now align without
  the silent 1-sample bias.
- 962195 baseline GPU validation rerun: 0.2057% / parity 1.788e-05% --
  bit-identical to pre-fix 962176, confirming odd-tap paths are unchanged.
- 962196 FULL `--slide-shift 1` (129 slides) + 3-point sky grid
  (`MIFIR_SLIDE_SHIFT=1`, now a parameter of `run_slides_skygrid.sbatch`):
  SNR gate PASS (loudest 0.2057%, 0 strong missed/extra). The un-gated
  metadata report shows the same known near-degenerate-grid bin flip as
  962187 (same trigger, same slide -- including slide 128 -- neighboring
  0.01 rad bin); the wide-grid job gates this and passes.

Also in round 3:
- `compare_triggers.py --check-metadata` strict mode: matching additionally
  requires the same `slide_id`, and strong matched pairs must recover the same
  sky point (`--sky-tolerance`). Standalone `compare_trigger_metadata.py` does
  the same as a separate gate. CAUTION: on near-degenerate grids (points
  closer than the network sky resolution, e.g. the 0.01 rad 3-point grid) the
  recovered bin can legitimately flip between near-equal maxima -- job 962187
  showed exactly that (same loud trigger/slide, neighboring bin) while the
  wider grid (962188) passed. Gate metadata only on non-degenerate grids;
  `run_slides_skygrid.sbatch` reports it un-gated, `run_slides_skygrid_wide.sbatch`
  gates it.

## Honest caveats / future work
- **Speedups are modest here because the templates are short (16 s).** FIR
  de-chirping attacks the memory wall of long-template FFTs, so its advantage
  grows with template length and with the fine:coarse ratio. The tool's actual
  target — sub-solar templates of hundreds–thousands of seconds — is exactly
  where brute force is infeasible (so it can't be directly compared), and where
  the paper reports order-of-magnitude+ gains. The 16 s result confirms the
  machinery is correct and already net-faster; it does not capture the full win.
- **Chi-square vetoes are stubbed to zero** in `pycbc_multi_inspiral_fir`
  (Phase 1): the FIR engine does not yet produce the per-detector correlation
  vector the power χ² needs. The coherent/coincident/null SNR (the gate) is
  exact; reweighted SNR therefore reduces to coherent SNR. Real χ² is future
  work.
- **GPU at scale**: the current GPU path keeps the coherent combination on the
  host and returns each SNR series to host; for the full sub-solar bank, keeping
  the reference SNR on-device and batching across fine templates would reduce
  host transfers further.
- Validation is single-segment, single sky point, zeroNoise + one injection.
  Broader validation (Gaussian noise, multiple injections, sky grid, time
  slides, tidal `SPAtmpltTidal` templates) is straightforward to add with this
  harness but not yet run.

## Reproduce
```
# CPU gate (RM):           sbatch test/multi_inspiral_fir/run_validation.sbatch
# GPU gate + parity:       sbatch test/multi_inspiral_fir/run_validation_gpu.sbatch
# throughput (suite D):    sbatch test/multi_inspiral_fir/run_throughput.sbatch
# unit tests on GPU:       sbatch test/multi_inspiral_fir/run_gpu_tests.sbatch
# unit tests (CPU, local): python3 test/test_multi_inspiral_fir.py --scheme cpu
```
Env: `ssm_pipeline` micromamba env + this checkout on `PYTHONPATH`; GPU =
PyCUDA + scikit-cuda (`--processing-scheme cuda`), not CuPy. NEVER run the
compute on the login node — use Slurm (RM/HENON for CPU, TWIG-GPU for GPU).
