# `SPAtmpltTidal` — results & validation note

GPU-native tidal TaylorF2 SPA filter approximant added to PyCBC. Clones
`SPAtmplt` (point particle) and adds the TaylorF2 tidal phase terms; reduces
bit-for-bit to `SPAtmplt` when `lambda1 = lambda2 = 0`.

## Files changed (branch `gpu-tidal-spatmplt`)
- `pycbc/waveform/spa_tmplt.py` — new `spa_tmplt_tidal()`; engine stub gains
  `pfa10, pfa12, pfa13, pfa14` as trailing keyword args defaulting to 0 (so
  the original `spa_tmplt()` source is **untouched**).
- `pycbc/waveform/spa_tmplt_cuda.py` — tidal terms in `taylorf2_text`, kernel
  arg string, and engine (the production CUDA / pycuda+scikit-cuda path).
- `pycbc/waveform/spa_tmplt_cupy.py` — same change (CuPy backend; not exercised
  — CuPy is not installed / not used by the pipeline).
- `pycbc/waveform/spa_tmplt_cpu.pyx` — tidal terms in both Horner loops.
- `pycbc/waveform/waveform.py` — registers `SPAtmpltTidal` in the filter dicts
  (`_cuda/_cupy/_inspiral_fd_filters`, `_filter_norms`, `_filter_preconditions`,
  `_filter_ends`, `_template_amplitude_norms`, `_filter_time_lengths`).
- `test/test_spatmplt_tidal.py`, `test/run_spatmplt_tidal_gpu.sbatch` — tests.

## Key correction to the handoff spec
The handoff specified only the 5PN (`v^10`) and 6PN (`v^12`) tidal terms. LAL's
`SimInspiralTaylorF2AlignedPhasing` actually populates **four** tidal orders:
`v^10` (5PN), `v^12` (6PN), `v^13` (6.5PN), `v^14` (7PN), all with zero log
terms. Carrying only `v^10/v^12` leaves a detrended phase residual vs LAL that
**grows linearly with lambda** (e.g. ~8.5 rad at lambda=5e5, m=1.4+1.4). All
four orders are implemented; the residual then drops to ~0.

## Tests (run once per backend: `python3 test_spatmplt_tidal.py [--scheme cuda]`)
Band 20–1024 Hz, `delta_f = 1/256`, `f_lower = 20`, aLIGOZeroDetHighPower PSD.

| Test | What | CPU | CUDA |
|---|---|---|---|
| T1 | reduction `SPAtmpltTidal(λ=0)` vs `SPAtmplt` | **bit-identical** | **bit-identical** |
| T2a | non-spinning vs LAL TaylorF2, all λ≤7e5, m∈{0.5,1,1.4,2} | min match **0.999992** | min match 0.99999 |
| T2c | point-particle + aligned spin vs LAL (λ=0) | ≥0.999 | ≥0.999 |
| T2b | backend vs double-precision reference of same formula (incl. spin×tidal, λ≤7e5, Mtot≥1) | min **0.999992** | min ≈0.9999 |
| T2 phase | detrended `arg(h)−arg(h_LAL)`, m=1.4+1.4, λ≤7e5 | < 1e-4 rad | (host) |
| T3 | mismatch(λ vs 0) tracks LAL & monotonic in λ | ratio 1.002, monotonic | same |
| T4 | `phasing.v[10]/v[12]` vs Vines–Flanagan–Hinderer/Wade2014 closed form | rel ≤ 4e-16 | (host) |
| T6 | float32 stress, m=0.1+0.1, λ=7e5 | pass | pass |

All 8 test groups pass on both CPU and the CUDA backend (A100, CUDA 12.4).

### T2 match distribution (vs LAL, non-spinning)
For m1,m2 ∈ {0.5,1.0,1.4,2.0}, λ ∈ {0,1e3,1e4,1e5,7e5}: every match ≥ 0.999992.
Agreement is limited only by the float32 phase-wrapping floor, not by tides.

### T6 float32 verdict — **no double-precision accumulator needed**
At the worst case (m=0.1+0.1, λ=7e5), match vs the double-precision reference:

| backend | point-particle `SPAtmplt` | tidal `SPAtmpltTidal` | Δ (tidal cost) |
|---|---|---|---|
| CPU (`.pyx`, libm) | 0.998564 | 0.998537 | 2.7e-5 |
| CUDA (fast-math intrinsics) | 0.987290 | 0.987154 | 1.4e-4 |

The float32 floor at sub-solar mass is set by the **point-particle** template
(long inspiral → large wrapped phase), and is lower on CUDA because the kernel
uses `__sincosf`/`__powf`/`__logf` fast-math intrinsics. Adding the tidal terms
costs ≤1.4e-4 in match — i.e. the tidal phase is **not** lost in float32. A
double-precision accumulator is therefore unnecessary (and would require
modifying `SPAtmplt`, which is out of scope).

### Throughput (CUDA, A100, 1.4+1.4, per template incl. host overhead)
- `SPAtmplt`      : 0.131 ms
- `SPAtmpltTidal` : 0.133 ms  (**+1.4 %**)

## Known limitation — spin × tidal vs the LAL TaylorF2 *generator*
With **non-zero aligned spin AND appreciable lambda**, `SPAtmpltTidal` diverges
from `get_fd_waveform(approximant='TaylorF2', ...)` (match can fall to
~0.95–0.98). This is **not** a bug in this filter or a float32 issue:
- a **double-precision** evaluation of the identical AlignedPhasing-based
  formula diverges from the LAL generator by the *same* amount (T2b passes,
  proving the kernel is correct), and
- the point-particle (λ=0) spinning case agrees with LAL (T2c), as does the
  non-spinning tidal case at all λ (T2a).

The LAL TaylorF2 *generator* treats the spin×tidal coupling differently from
the `SimInspiralTaylorF2AlignedPhasing` coefficients that **both** `SPAtmplt`
and `SPAtmpltTidal` are built from. `SPAtmplt` inherits the same relationship.
The spin×tidal sector is validated against the double-precision reference of
the same formula — the construction the downstream filter actually uses.

**Action for the human:** if the downstream search must agree bit-for-bit with
LAL's TaylorF2 generator for *spinning* tidal templates, this divergence needs
a decision (e.g. whether banks/injections use the AlignedPhasing convention or
the generator). For non-spinning or small-λ sub-solar templates it is a
non-issue.
