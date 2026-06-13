# Copyright (C) 2026
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.

"""Unit tests for the FIR-de-chirped coherent search (pycbc_multi_inspiral_fir)
and its engine extensions in pycbc.filter.matched_ratio.

Run as:
    python3 test/test_multi_inspiral_fir.py            # CPU
    python3 test/test_multi_inspiral_fir.py --scheme cuda   # GPU (Slurm)

Suites:
  A. Engine SNR(t) self-consistency: the full per-template SNR time series
     reconstructed by RatioMatchedFilterControl.fine_snr_timeseries reproduces,
     at every above-threshold time index, the values found by the validated
     peak-finding kernel (_execute_blocked_kernel). When run under the CUDA
     scheme this becomes a CPU-vs-CUDA parity check of the engine (Phase 2).
  C. Tap fidelity: if a ratio bank is available (env MIFIR_DIR), every fine
     template's stored filter_match is >= 0.99.
  E. --high-frequency-cutoff consistency: the reference SNR normalization
     integrates the same band as the filter integral (regression test).
  F. Even-tap centering: a delta tap at the pycbc_fir_bank design center
     (K-1)//2 is the identity filter, single-rate AND under decimation
     (regression test for the K//2 roll that shifted even-count filters by
     one tap-rate sample -- a half engine sample at decimation 2).

Suites B (FIR-vs-brute correctness) and D (throughput) are driven end-to-end
on a compute node by test/multi_inspiral_fir/run_validation.sbatch, which runs
both executables on a shared injection and compares coherent SNR.
"""

import os
import unittest
import numpy as np

from pycbc.filter.matched_ratio import RatioMatchedFilterControl
from utils import parse_args_all_schemes, simple_exit

_scheme, _context = parse_args_all_schemes("MultiInspiralFIR")


class TestRatioEngineSNRSeries(unittest.TestCase):
    """Suite A: the reconstructed SNR(t) matches the peak-finder kernel."""

    def setUp(self):
        self.context = _context
        self.scheme = _scheme
        np.random.seed(2026)
        self.eng = RatioMatchedFilterControl(
            snr_threshold=3.0, delta_f=1.0 / 256, fir_fft_length=4096,
            batch_size=8, tap_sample_rate=2048, engine_sample_rate=2048)
        self.n_samples = 120000
        ref = (np.random.randn(self.n_samples)
               + 1j * np.random.randn(self.n_samples)).astype(np.complex64)
        for t in (30000, 60000, 90000):
            ref[t] += 50.0  # strong spikes -> clear-threshold peaks
        self.ref = ref
        ntap = 151
        taps = (np.random.randn(3, ntap) * 0.1).astype(np.float32)
        taps[:, ntap // 2] += 1.0  # near-delta -> tracks reference
        self.counts = np.array([ntap, ntap, ntap], dtype=np.int32)
        self.filters_f, _ = self.eng.prepare_filters(taps, self.counts)
        self.valid = slice(2000, self.n_samples - 2000)

    def test_series_matches_kernel(self):
        eng = self.eng
        eng.ref_snr = self.ref.copy()
        eng._block_fft_cache = {}
        f_idx, t_idx, snr_vals, _ = eng._execute_blocked_kernel(
            self.ref, self.filters_f, self.counts, self.valid)
        series = [
            eng.fine_snr_timeseries(self.filters_f[j], self.counts[j],
                                    self.valid, ref_snr=self.ref,
                                    block_cache={})
            for j in range(3)
        ]
        self.assertGreater(len(f_idx), 0, "kernel found no peaks")
        maxerr = 0.0
        for fi, ti, sv in zip(f_idx, t_idx, snr_vals):
            maxerr = max(maxerr, abs(series[fi][ti] - sv))
        self.assertLess(maxerr, 1e-2,
                        f"series vs kernel max error {maxerr:.3e}")

    def test_cpu_cuda_parity(self):
        """Suite A (Phase 2): the GPU-reconstructed SNR(t) matches the CPU
        reference to single precision.

        Only meaningful under ``--scheme cuda``: on CPU both branches run the
        identical deterministic CPU code path, so the comparison is a tautology
        (exactly 0.0) and is skipped. On CUDA the reference runs the scipy+cython
        CPU path while the active branch runs the cufft+PyCUDA path; we require
        the difference to be (a) > 0 -- a 0.0 would mean the GPU path silently
        fell back to CPU -- and (b) < 1e-4 relative (single-precision FFT
        round-off between two different FFT libraries)."""
        if self.scheme != 'cuda':
            self.skipTest("CPU-vs-CUDA parity is only meaningful under "
                          "--scheme cuda (on CPU it compares identical code).")
        from pycbc.scheme import CPUScheme
        with CPUScheme():
            ref_cpu = [
                self.eng.fine_snr_timeseries(self.filters_f[j], self.counts[j],
                                             self.valid, ref_snr=self.ref,
                                             block_cache={})
                for j in range(3)]
        with self.context:
            ref_sch = [
                self.eng.fine_snr_timeseries(self.filters_f[j], self.counts[j],
                                             self.valid, ref_snr=self.ref,
                                             block_cache={})
                for j in range(3)]
        for j in range(3):
            denom = np.abs(ref_cpu[j]).max()
            relerr = np.abs(ref_sch[j] - ref_cpu[j]).max() / denom
            print(f"  filter {j}: CUDA-vs-CPU rel err = {relerr:.3e}")
            # > 0: guards against a silent CPU fallback (identical paths -> 0.0)
            self.assertGreater(relerr, 0.0,
                               f"filter {j}: CUDA result is bit-identical to "
                               "CPU -- the GPU path did not actually run.")
            self.assertLess(relerr, 1e-4,
                            f"filter {j} CUDA-vs-CPU rel err {relerr:.2e}")

    def test_block_cache_reuse_is_consistent(self):
        """Reusing one block-FFT cache across filters gives identical output."""
        eng = self.eng
        shared = {}
        a = eng.fine_snr_timeseries(self.filters_f[0], self.counts[0],
                                    self.valid, ref_snr=self.ref,
                                    block_cache=shared)
        # Filter 1 then re-run filter 0 against the now-populated shared cache.
        eng.fine_snr_timeseries(self.filters_f[1], self.counts[1],
                                self.valid, ref_snr=self.ref,
                                block_cache=shared)
        a2 = eng.fine_snr_timeseries(self.filters_f[0], self.counts[0],
                                     self.valid, ref_snr=self.ref,
                                     block_cache=shared)
        self.assertTrue(np.array_equal(a, a2),
                        "block-FFT cache reuse changed the result")

    def test_batched_matches_single(self):
        """Suite A (Phase 3): the tiled GPU path (fine_snr_timeseries_batch)
        reproduces the validated single-template GPU path to single precision.

        This is the guard that batching K fine templates into one
        multiply+ifft+assemble did not change the per-template math: the only
        differences from the single-filter kernels are the batch dimension (a
        larger cufft batch -> single-precision round-off) and the fine-rescale
        division folded into the assemble. CUDA-only (the batched method is the
        GPU throughput core; on CPU the driver uses the per-template path)."""
        if self.scheme != 'cuda':
            self.skipTest("fine_snr_timeseries_batch is the CUDA throughput "
                          "path; nothing to compare on CPU.")
        from pycbc.types import zeros, complex64, Array
        eng = self.eng
        K = 3
        n_taps = int(self.counts.max())
        valid = self.valid
        L = valid.stop - valid.start
        # Non-trivial per-template rescales: batched divides them out, so we
        # multiply them back before comparing to the (un-rescaled) single path.
        rescales = np.array([1.0, 1.7, 0.6], dtype=np.float32)
        with self.context:
            singles = [
                eng.fine_snr_timeseries(self.filters_f[j], n_taps, valid,
                                        ref_snr=self.ref, block_cache={},
                                        return_device=False)
                for j in range(K)]
            filt_dev = Array(np.ascontiguousarray(
                self.filters_f[:K].reshape(-1), dtype=np.complex64))
            ref_dev = Array(np.ascontiguousarray(self.ref, dtype=np.complex64))
            rescales_dev = Array(rescales)
            out_buf = zeros(K * L, dtype=complex64)
            rows = eng.fine_snr_timeseries_batch(
                filt_dev, K, n_taps, valid, ref_dev, {}, rescales_dev, out_buf)
            rows_host = [r.numpy() for r in rows]
        for j in range(K):
            got = rows_host[j] * rescales[j]          # undo folded /rescale
            want = singles[j][valid]
            denom = np.abs(want).max()
            # The batched path must actually produce the (non-trivial) series,
            # not zeros -- ``want`` has clear SNR peaks from the seeded spikes.
            self.assertGreater(np.abs(got).max(), 0.0,
                               f"filter {j}: batched output is all zeros")
            relerr = np.abs(got - want).max() / denom
            print(f"  batched filter {j}: vs single-template rel err "
                  f"= {relerr:.3e}")
            # Bit-identical (relerr == 0) is acceptable here: both run on the
            # GPU, so unlike the CPU/CUDA parity test there is no silent-fallback
            # to catch -- the only guard is that batching did not change the math.
            self.assertLess(
                relerr, 1e-4, f"filter {j}: batched-vs-single rel err "
                f"{relerr:.2e} (batching changed the per-template math)")


class TestEvenTapCentering(unittest.TestCase):
    """Suite F: even tap counts must use the pycbc_fir_bank centering.

    pycbc_fir_bank designs tap j at time j - (K-1)//2 for BOTH parities
    (even K spans -K/2+1 .. K/2). So a tap vector that is 1.0 at index
    (K-1)//2 and zero elsewhere is exactly the identity filter: the
    reconstructed series must equal the reference series. Regression test
    for the engine rolling by K//2, which shifted every even-count filter
    by one tap-rate sample (an envelope-sampling half engine sample under
    decimation 2 -- the 1.5% loss seen in the multi-rate even-tap
    validation).

    The reference is built block-analytic (upper half spectrum zero) so the
    engine's analytic multiply is transparent to the comparison.
    """

    N_FFT = 4096

    def setUp(self):
        np.random.seed(2027)
        spec = np.zeros(self.N_FFT, dtype=np.complex64)
        nhalf = self.N_FFT // 2
        spec[:nhalf] = (np.random.randn(nhalf)
                        + 1j * np.random.randn(nhalf))
        self.ref = np.fft.ifft(spec).astype(np.complex64)
        self.valid = slice(400, 3600)

    def _delta_series(self, n_taps, tap_sr, engine_sr):
        eng = RatioMatchedFilterControl(
            snr_threshold=3.0, delta_f=1.0 / 256,
            fir_fft_length=self.N_FFT, batch_size=8,
            tap_sample_rate=tap_sr, engine_sample_rate=engine_sr)
        taps = np.zeros((1, n_taps), dtype=np.float32)
        taps[0, (n_taps - 1) // 2] = 1.0   # the design-center tap (t = 0)
        counts = np.array([n_taps], dtype=np.int32)
        filters_f, _ = eng.prepare_filters(taps, counts)
        return eng.fine_snr_timeseries(filters_f[0], n_taps, self.valid,
                                       ref_snr=self.ref, block_cache={})

    def _check_identity(self, out, label):
        scale = np.abs(self.ref).max()
        err = np.abs(out[self.valid] - self.ref[self.valid]).max() / scale
        self.assertLess(err, 1e-4,
                        f"{label}: delta tap at (K-1)//2 is not the identity "
                        f"filter (max rel err {err:.3e}) -- even-tap "
                        "centering is off")

    def test_even_taps_single_rate(self):
        self._check_identity(self._delta_series(150, 2048, 2048),
                             "K=150, decimate=1")

    def test_even_taps_multirate(self):
        self._check_identity(self._delta_series(300, 4096, 2048),
                             "K=300, decimate=2")

    def test_odd_taps_unchanged(self):
        self._check_identity(self._delta_series(151, 2048, 2048),
                             "K=151, decimate=1")
        self._check_identity(self._delta_series(301, 4096, 2048),
                             "K=301, decimate=2")


class TestFIRvsBruteMatchedFilter(unittest.TestCase):
    """The new FIR reconstruction of a fine template's complex SNR(t) must
    reproduce pycbc's existing brute-force matched filter of that same fine
    template -- the unit-level, code-independent check (the FIR engine and
    matched_filter_core share no code path).

    Setup: build a single data segment that is a (time-shifted) copy of one
    fine template, then compute that template's SNR two ways:
      (existing) matched_filter_core(fine_template, data)
      (new)      compute reference SNR on the *coarse* template, then
                 fine_snr_timeseries() with the fine template's FIR taps,
                 divided by the bank's snr_rescale.
    They should agree to roughly the FIR tap mismatch (sub-percent).
    """

    def _build(self):
        import os
        import h5py
        from pycbc.filter.matchedfilter import matched_filter_core, sigmasq
        from pycbc.types import FrequencySeries
        from pycbc.waveform.bank import RatioFilterBank
        import pycbc.psd
        d = os.environ.get('MIFIR_DIR',
                           '/hildafs/projects/phy220048p/xhall/scratch/mifir')
        path = os.path.join(d, 'ratio_bank.hdf')
        if not os.path.exists(path):
            return None
        flow = 45.0
        sample_rate = 2048
        tlen = 32768            # 16 s at 2048 Hz (> the ~10-13 s templates)
        delta_f = sample_rate / float(tlen)
        flen = tlen // 2 + 1
        bank = RatioFilterBank(path, flen, delta_f, np.complex64,
                               low_frequency_cutoff=flow, phase_order='-1',
                               approximant='TaylorF2')
        # A flat (unit) PSD, used identically by BOTH the brute and FIR matched
        # filters here. Rationale: (1) conditioning -- the bank templates are
        # O(1) with no DYN_RANGE scaling, so a realistic ~1e-47 PSD overflows
        # float32 sigmasq; (2) validity -- for whatever PSD both sides use, the
        # FIR reconstruction equals matched_filter(fine) because the coarse
        # template cancels analytically (fft(MF(coarse))*conj(taps) ~
        # conj(h_fine)*data/psd) and the taps approximate h_fine/h_coarse. This
        # does NOT claim the SNR is PSD-independent; it claims a flat PSD is a
        # legitimate common choice for the FIR-vs-brute comparison. The residual
        # is the tap mismatch (~1 - filter_match), which is why we pick the
        # best-matched fine template below to keep the bound tight.
        psd = FrequencySeries(np.ones(flen, dtype=np.float32), delta_f=delta_f)
        coarse_idx = int(bank.coarse_indices[0])
        coarse = bank.get_coarse_template(coarse_idx)
        taps, counts, fine_indices = bank.get_firs(coarse_idx)
        # Pick the fine template best represented by this coarse reference.
        with h5py.File(path, 'r') as f:
            fm = f['fir_data'][str(coarse_idx)]['filter_match'][:]
        j = int(np.argmax(fm))
        fine_gid = int(fine_indices[j])
        fine = bank[fine_gid]
        # Data = fine template shifted to mid-segment (peak in the interior).
        k = np.arange(flen)
        shift = (-1.0) ** k                       # time shift by tlen/2 samples
        stilde = FrequencySeries(
            (fine.numpy() * shift).astype(np.complex64), delta_f=delta_f,
            dtype=np.complex64)
        return dict(bank=bank, psd=psd, coarse=coarse, fine=fine,
                    fine_gid=fine_gid, taps=taps, counts=counts, j=j,
                    stilde=stilde, flow=flow, sample_rate=sample_rate,
                    tlen=tlen)

    def test_fir_reconstructs_brute_matched_filter(self):
        from pycbc.filter.matchedfilter import matched_filter_core, sigmasq
        b = self._build()
        if b is None:
            self.skipTest("no ratio bank for the FIR-vs-brute unit test")
        psd, stilde, flow = b['psd'], b['stilde'], b['flow']

        # (existing) brute matched filter of the FINE template
        hf = sigmasq(b['fine'], psd=psd, low_frequency_cutoff=flow)
        snr_f, _, norm_f = matched_filter_core(
            b['fine'], stilde, psd=psd, low_frequency_cutoff=flow, h_norm=hf)
        brute = (snr_f.numpy() * norm_f)

        # (new) reference SNR on the COARSE template, then FIR-filter
        eng = RatioMatchedFilterControl(
            snr_threshold=4.0, delta_f=stilde.delta_f, fir_fft_length=4096,
            batch_size=8, tap_sample_rate=b['sample_rate'],
            engine_sample_rate=b['sample_rate'])
        hc = sigmasq(b['coarse'], psd=psd, low_frequency_cutoff=flow)
        snr_c, _, norm_c = matched_filter_core(
            b['coarse'], stilde, psd=psd, low_frequency_cutoff=flow, h_norm=hc)
        delta_t = 1.0 / b['sample_rate']
        ref_snr = (snr_c.numpy() * (norm_c * delta_t)).astype(np.complex64)
        filters_f, _ = eng.prepare_filters(b['taps'], b['counts'])
        valid = slice(b['tlen'] // 4, 3 * b['tlen'] // 4)
        series = eng.fine_snr_timeseries(filters_f[b['j']],
                                         int(b['counts'][b['j']]), valid,
                                         ref_snr=ref_snr, block_cache={})
        rescale = float(b['bank'].snr_rescale(b['fine_gid'],
                                              method='precalculated_sigma'))
        fir = series / rescale

        # Compare peak (recovered SNR) of new vs existing.
        bpk = np.abs(brute).max()
        fpk = np.abs(fir[valid]).max()
        ratio = fpk / bpk
        print(f"  brute fine SNR peak = {bpk:.4f}, FIR peak = {fpk:.4f}, "
              f"ratio = {ratio:.5f}")
        self.assertLess(abs(ratio - 1.0), 0.02,
                        f"FIR vs brute matched-filter peak off by "
                        f"{abs(ratio-1)*100:.2f}%")
        # Complex (magnitude AND phase) agreement at the peak: the coherent
        # combination uses the complex SNR, so phase must match too. Compare
        # the complex residual, and the phase explicitly.
        bt = int(np.argmax(np.abs(brute)))
        complex_relerr = abs(fir[bt] - brute[bt]) / abs(brute[bt])
        phase_diff = abs(np.angle(fir[bt] / brute[bt]))
        print(f"  at brute peak t={bt}: complex rel err = {complex_relerr:.5f}, "
              f"phase diff = {phase_diff:.2e} rad")
        self.assertLess(complex_relerr, 0.02,
                        "FIR complex SNR at the brute peak disagrees (mag/phase)")
        self.assertLess(phase_diff, 0.05,
                        f"FIR SNR phase at the brute peak off by {phase_diff:.3f} rad")


class TestReferenceCutoffNormalization(unittest.TestCase):
    """Suite E: --high-frequency-cutoff consistency of the reference SNR.

    Regression test: compute_reference_snr used the cached *full-band*
    template sigmasq as h_norm while handing matched_filter_core
    high_frequency_cutoff=f_high, so every SNR came out low by
    sqrt(sigmasq_full / sigmasq_band) (~5.5% on the validation bank at
    f_high=256 Hz). The normalization must integrate exactly the band the
    filter integrates.

    Self-contained (no ratio bank): a synthetic f^{-7/6} template on a unit
    PSD, with the data segment equal to the template, has matched-filter
    peak |SNR| == sigma over the filtered band, for ANY cutoff.
    """

    def setUp(self):
        from pycbc.types import FrequencySeries
        self.sample_rate = 2048
        tlen = 8192
        self.delta_t = 1.0 / self.sample_rate
        self.delta_f = self.sample_rate / float(tlen)
        flen = tlen // 2 + 1
        self.flow = 30.0
        f = np.arange(flen) * self.delta_f
        amp = np.zeros(flen)
        band = (f >= self.flow) & (f <= 900.0)
        amp[band] = (f[band] / 100.0) ** (-7.0 / 6.0)
        htilde = FrequencySeries(amp.astype(np.complex64),
                                 delta_f=self.delta_f)
        htilde.f_lower = self.flow  # engine reads this for the low cutoff
        self.htilde = htilde
        # data = the template itself -> peak SNR = sigma of the filtered band
        self.stilde = FrequencySeries(amp.astype(np.complex64),
                                      delta_f=self.delta_f)
        self.psd = FrequencySeries(np.ones(flen, dtype=np.float32),
                                   delta_f=self.delta_f)

    def _engine(self, f_high):
        return RatioMatchedFilterControl(
            snr_threshold=4.0, delta_f=self.delta_f, fir_fft_length=4096,
            batch_size=8, tap_sample_rate=self.sample_rate,
            engine_sample_rate=self.sample_rate,
            high_frequency_cutoff=f_high)

    def test_h_norm_honors_cutoff(self):
        from pycbc.filter.matchedfilter import sigmasq
        f_cut = 128.0
        expected = float(sigmasq(self.htilde, psd=self.psd,
                                 low_frequency_cutoff=self.flow,
                                 high_frequency_cutoff=f_cut))
        full = float(sigmasq(self.htilde, psd=self.psd,
                             low_frequency_cutoff=self.flow))
        # The test is vacuous unless the cutoff removes real in-band power.
        self.assertLess(expected, 0.95 * full)
        eng = self._engine(f_cut)
        h_norm = eng.compute_reference_snr(self.stilde, self.psd, self.htilde)
        self.assertAlmostEqual(
            h_norm / expected, 1.0, places=5,
            msg="h_norm does not integrate [f_lower, f_high]")

    def test_reference_snr_peak_is_band_sigma(self):
        """The engine's ref_snr carries an extra delta_t (and 1/decimate,
        = 1 here) factor; after removing it the peak must equal the band
        sigma. With the old bug the peak is low by sqrt(band/full): 7.0%
        at f_high=128 Hz and 2.4% at 256 Hz for this template -- both far
        above the 0.1% gate."""
        from pycbc.filter.matchedfilter import sigmasq
        for f_cut in (128.0, 256.0):
            eng = self._engine(f_cut)
            eng.compute_reference_snr(self.stilde, self.psd, self.htilde)
            sigma_band = np.sqrt(float(sigmasq(
                self.htilde, psd=self.psd, low_frequency_cutoff=self.flow,
                high_frequency_cutoff=f_cut)))
            peak = np.abs(eng.ref_snr).max() / self.delta_t
            self.assertLess(
                abs(peak / sigma_band - 1.0), 1e-3,
                f"f_high={f_cut}: peak SNR {peak:.3f} != band sigma "
                f"{sigma_band:.3f} (filter/normalization band mismatch)")


class TestTapFidelity(unittest.TestCase):
    """Suite C: stored FIR tap match >= 0.99 across the bank (if present)."""

    def test_tap_match(self):
        import h5py
        d = os.environ.get('MIFIR_DIR',
                           '/hildafs/projects/phy220048p/xhall/scratch/mifir')
        path = os.path.join(d, 'ratio_bank.hdf')
        if not os.path.exists(path):
            self.skipTest(f"no ratio bank at {path}")
        with h5py.File(path, 'r') as f:
            fg = f['fir_data']
            ck = [k for k in fg.keys() if k.isdigit()]
            matches = np.concatenate([fg[k]['filter_match'][:] for k in ck])
        self.assertGreaterEqual(
            matches.min(), 0.99,
            f"min tap match {matches.min():.4f} < 0.99")


if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    suite.addTest(loader.loadTestsFromTestCase(TestRatioEngineSNRSeries))
    suite.addTest(loader.loadTestsFromTestCase(TestEvenTapCentering))
    suite.addTest(loader.loadTestsFromTestCase(TestFIRvsBruteMatchedFilter))
    suite.addTest(loader.loadTestsFromTestCase(TestReferenceCutoffNormalization))
    suite.addTest(loader.loadTestsFromTestCase(TestTapFidelity))
    results = unittest.TextTestRunner(verbosity=2).run(suite)
    simple_exit(results)
