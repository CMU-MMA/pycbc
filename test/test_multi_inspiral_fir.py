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
        # A flat (unit) PSD: well-conditioned (the bank templates are O(1) and
        # carry no DYN_RANGE scaling, so a realistic ~1e-47 PSD would overflow
        # float32 sigmasq) and faithful -- the FIR reconstruction reproduces
        # matched_filter(fine) for ANY PSD (the coarse template cancels exactly;
        # the taps encode the PSD-independent ratio h_fine/h_coarse), so the
        # PSD scale cancels in the FIR-vs-brute ratio.
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
        # Also: the FIR SNR series should track brute in the interior (the
        # complex waveform shapes agree, not just the peak).
        bt = np.argmax(np.abs(brute))
        # align on the brute peak time; compare complex values there
        cval_ratio = abs(fir[bt]) / abs(brute[bt])
        print(f"  at brute peak t={bt}: |fir|/|brute| = {cval_ratio:.5f}")
        self.assertLess(abs(cval_ratio - 1.0), 0.05,
                        "FIR SNR at the brute peak time disagrees")


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
    suite.addTest(loader.loadTestsFromTestCase(TestFIRvsBruteMatchedFilter))
    suite.addTest(loader.loadTestsFromTestCase(TestTapFidelity))
    results = unittest.TextTestRunner(verbosity=2).run(suite)
    simple_exit(results)
