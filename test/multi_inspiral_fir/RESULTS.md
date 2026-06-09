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
