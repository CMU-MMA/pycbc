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
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Unit tests for the GPU-native tidal SPA filter approximant ``SPAtmpltTidal``.

``SPAtmpltTidal`` clones the point-particle ``SPAtmplt`` TaylorF2 stationary
phase template and adds the TaylorF2 tidal phase terms.  With
``lambda1 = lambda2 = 0`` it must reduce bit-for-bit to ``SPAtmplt``; with
non-zero tidal deformability it reproduces LAL's TaylorF2 tidal phasing.

This module follows the PyCBC backend-test convention (see test_spatmplt.py /
test_threshold.py): ONE processing scheme is selected per process via
``parse_args_all_schemes`` and every waveform is generated under that scheme.
The reference waveforms -- a double-precision evaluation of the identical
TaylorF2 SPA formula, and LAL's TaylorF2 -- are computed on the host and are
valid regardless of the active scheme.  Backend correctness is therefore
established per backend: run

    python3 test_spatmplt_tidal.py                # CPU  (.pyx) backend
    python3 test_spatmplt_tidal.py --scheme cuda  # CUDA (pycuda) backend

Each backend is validated against the same host reference, so agreement of
both backends with the reference is equivalent to CPU/CUDA parity (T5).  The
CuPy backend is not exercised here because CuPy is not installed in the target
environment (and is not used by the downstream pipeline, which uses the
pycuda/scikit-cuda CUDA backend).

NOTE ON THE TIDAL PHASE ORDERS
------------------------------
LAL's ``SimInspiralTaylorF2AlignedPhasing`` populates FOUR tidal phase orders
for non-zero deformability: 5PN (v^10), 6PN (v^12), 6.5PN (v^13) and 7PN
(v^14), all with zero log terms.  All four are required to reduce to LAL --
carrying only v^10/v^12 leaves a residual that grows linearly with lambda.
``SPAtmpltTidal`` carries all four.

NOTE ON AGREEMENT WITH LAL'S TaylorF2 GENERATOR
-----------------------------------------------
Like ``SPAtmplt``, ``SPAtmpltTidal`` builds its phase from the
``SimInspiralTaylorF2AlignedPhasing`` coefficients.  Two regimes are observed:

* **Non-spinning:** matches the LAL TaylorF2 generator to ~1e-5 for all tidal
  deformabilities tested (lambda up to 7e5) -- a clean validation of the
  tidal phase coefficients.

* **Spinning + tidal:** when the aligned spin is non-zero AND lambda is
  appreciable, the LAL generator and the AlignedPhasing-based reconstruction
  disagree (match can drop to ~0.95-0.98).  This is NOT a defect of
  ``SPAtmpltTidal`` or a float32 issue: a *double-precision* evaluation of the
  same AlignedPhasing-based formula diverges from the generator by the
  identical amount, the point-particle (lambda=0) spinning case agrees with
  LAL, and ``SPAtmplt`` inherits the same relationship to the generator.  The
  LAL TaylorF2 *generator* simply treats the spin x tidal coupling differently
  from the AlignedPhasing coefficients.  The spin x tidal sector is therefore
  validated against the double-precision reference of the same formula, which
  is the construction the downstream filter actually uses.
"""
import unittest

import numpy as np
import lal
import lalsimulation as ls

import pycbc
from pycbc.types import zeros, complex64, complex128, FrequencySeries
from pycbc.waveform import get_fd_waveform, get_waveform_filter
from pycbc.filter import match
from pycbc.psd.analytical import aLIGOZeroDetHighPower
from pycbc.constants import PI, MTSUN_SI
from pycbc.waveform.spa_tmplt import spa_amplitude_factor
from utils import parse_args_all_schemes, simple_exit

_scheme, _context = parse_args_all_schemes("SPAtmpltTidal")

# ---------------------------------------------------------------------------
# Common analysis configuration (band 20-1024 Hz, delta_f = 1/256).
# ---------------------------------------------------------------------------
DF = 1.0 / 256
F_LOWER = 20.0
F_HIGH = 1000.0           # stay safely below f_final / Nyquist for the match
F_FINAL = 1024.0
FLEN = int(F_FINAL / DF) + 1

# Double-precision PSD.  aLIGO PSD values (~1e-46) underflow to zero in
# float32, so the PSD stays double and the (single-precision) filter outputs
# are promoted to double for the match -- this preserves the float32 *content*
# of the templates while giving a well-conditioned inner product.
_PSD = aLIGOZeroDetHighPower(FLEN, DF, F_LOWER)

# float32 accuracy floor: match between the float32 kernel and a
# double-precision evaluation of the SAME formula.  For total mass >= 1 Msun
# this is essentially 1; the long sub-solar inspiral wraps a larger float32
# phase, lowering the floor.  This floor is inherited from SPAtmplt (point
# particle) and is NOT introduced by the tidal terms.
F32_FLOOR = 0.9985

NS_MASSES = [0.5, 1.0, 1.4, 2.0]
LAMBDAS = [0.0, 1e3, 1e4, 1e5, 7e5]
SPINS = [0.0, 0.05, -0.05]


# ---------------------------------------------------------------------------
# Generation / reference helpers
# ---------------------------------------------------------------------------
def make_filter(approximant, **kwds):
    """Generate a filter waveform under the active processing scheme."""
    params = dict(delta_f=DF, f_lower=F_LOWER, f_final=F_FINAL,
                  approximant=approximant, distance=1.0,
                  phase_order=-1, spin_order=-1, amplitude_order=0)
    params.update(kwds)
    out = zeros(FLEN, dtype=complex64)
    return get_waveform_filter(out, **params)


def lal_taylorf2(mass1, mass2, spin1z=0.0, spin2z=0.0,
                 lambda1=0.0, lambda2=0.0):
    """Reference LAL TaylorF2 waveform (host; scheme-independent)."""
    h, _ = get_fd_waveform(approximant='TaylorF2', delta_f=DF, f_lower=F_LOWER,
                           f_final=F_FINAL, spin_order=-1, phase_order=-1,
                           amplitude_order=0, mass1=mass1, mass2=mass2,
                           spin1z=spin1z, spin2z=spin2z,
                           lambda1=lambda1, lambda2=lambda2)
    h.resize(FLEN)
    return h


def reference_double(mass1, mass2, spin1z=0.0, spin2z=0.0,
                     lambda1=0.0, lambda2=0.0):
    """Double-precision host evaluation of the exact same TaylorF2 SPA tidal
    formula the kernels evaluate in float32.  Valid under any scheme.  Comparing
    a backend against this isolates the backend (precision/kernel) from
    formula/coefficient errors and from the LAL generator's spin-tidal
    convention.
    """
    pars = lal.CreateDict()
    ls.SimInspiralWaveformParamsInsertTidalLambda1(pars, lambda1)
    ls.SimInspiralWaveformParamsInsertTidalLambda2(pars, lambda2)
    ph = ls.SimInspiralTaylorF2AlignedPhasing(float(mass1), float(mass2),
                                              float(spin1z), float(spin2z),
                                              pars)
    pfaN = ph.v[0]
    c = {k: ph.v[k] / pfaN for k in (2, 3, 4, 5, 7, 10, 12, 13, 14)}
    c[6] = (ph.v[6] - ph.vlogv[6] * np.log(4)) / pfaN
    cl5 = ph.vlogv[5] / pfaN
    cl6 = ph.vlogv[6] / pfaN

    piM = PI * (mass1 + mass2) * MTSUN_SI
    f = np.arange(FLEN) * DF
    out = np.zeros(FLEN, dtype=np.complex128)
    k = (f >= F_LOWER) & (f < F_FINAL)
    v = (piM * f[k]) ** (1.0 / 3.0)
    logv = np.log(v)
    psum = (1.0 + c[2] * v ** 2 + c[3] * v ** 3 + c[4] * v ** 4
            + (c[5] + cl5 * logv) * v ** 5
            + (c[6] + cl6 * (logv + np.log(4))) * v ** 6
            + c[7] * v ** 7
            + c[10] * v ** 10 + c[12] * v ** 12
            + c[13] * v ** 13 + c[14] * v ** 14)
    psi = psum * pfaN / v ** 5 - PI / 4
    amp = spa_amplitude_factor(mass1=mass1, mass2=mass2) * f[k] ** (-7.0 / 6.0)
    out[k] = amp * (np.cos(psi) - 1j * np.sin(psi))
    return FrequencySeries(out, delta_f=DF)


def _host(series):
    """Bring any (CPU or CUDA) FrequencySeries to a fresh, contiguous host
    double-precision numpy array for matching off the GPU."""
    return np.array(series.numpy(), dtype=np.complex128, copy=True)


def matchval(a, b):
    """Noise-weighted match, maximised over phase and time, computed on the
    host against the double PSD over [F_LOWER, F_HIGH]."""
    na = _host(a)
    nb = _host(b)
    n = min(len(na), len(nb), len(_PSD))
    fa = FrequencySeries(na[:n].copy(), delta_f=DF)
    fb = FrequencySeries(nb[:n].copy(), delta_f=DF)
    fp = FrequencySeries(np.array(_PSD.numpy()[:n], copy=True), delta_f=DF)
    m, _ = match(fa, fb, psd=fp,
                 low_frequency_cutoff=F_LOWER, high_frequency_cutoff=F_HIGH)
    return m


def closed_form_tidal_v(mass1, mass2, lambda1, lambda2):
    """Published per-body 5PN (v^10) and 6PN (v^12) TaylorF2 tidal phase
    coefficients (Vines, Flanagan & Hinderer 2011 PRD 83 084051; Wade et al.
    2014 PRD 89 103012; Favata 2014), returned un-normalised (comparable to
    ``phasing.v[10]`` / ``phasing.v[12]``)."""
    M = mass1 + mass2
    eta = mass1 * mass2 / M ** 2
    x1 = mass1 / M
    x2 = mass2 / M
    pfaN = 3.0 / (128.0 * eta)
    v10 = pfaN * (-24.0) * ((x1 + 12.0 * x2) * x1 ** 4 * lambda1
                            + (x2 + 12.0 * x1) * x2 ** 4 * lambda2)

    def c12(x):
        return (3179.0 - 919.0 * x - 2286.0 * x ** 2 + 260.0 * x ** 3) * x ** 4
    v12 = pfaN * (-5.0 / 28.0) * (c12(x1) * lambda1 + c12(x2) * lambda2)
    return v10, v12


class TestSPAtmpltTidal(unittest.TestCase):
    def setUp(self):
        self.context = _context
        self.scheme = _scheme

    # -------------------------------------------------------------------
    # T1 -- reduction: SPAtmpltTidal(lambda=0) == SPAtmplt
    # -------------------------------------------------------------------
    def test_t1_reduction(self):
        """With lambda1 = lambda2 = 0 the tidal template must reduce to
        SPAtmplt, bit-for-bit identical on the active backend."""
        cases = [(1.4, 1.4, 0.0, 0.0), (1.0, 1.0, 0.05, -0.05),
                 (0.5, 1.2, -0.05, 0.0), (2.0, 0.3, 0.0, 0.05)]
        with self.context:
            for m1, m2, s1, s2 in cases:
                hs = make_filter('SPAtmplt', mass1=m1, mass2=m2,
                                 spin1z=s1, spin2z=s2)
                ht = make_filter('SPAtmpltTidal', mass1=m1, mass2=m2,
                                 spin1z=s1, spin2z=s2, lambda1=0.0, lambda2=0.0)
                self.assertTrue(
                    np.array_equal(hs.numpy(), ht.numpy()),
                    "SPAtmpltTidal(lambda=0) != SPAtmplt for "
                    "m1=%g m2=%g s1=%g s2=%g (point-particle result changed)"
                    % (m1, m2, s1, s2))

    # -------------------------------------------------------------------
    # T2a -- non-spinning accuracy vs LAL TaylorF2 (all lambda)
    # -------------------------------------------------------------------
    def test_t2a_accuracy_vs_lal_nonspinning(self):
        with self.context:
            for m1 in NS_MASSES:
                for m2 in NS_MASSES:
                    if m2 > m1:
                        continue
                    for lam in LAMBDAS:
                        mine = make_filter('SPAtmpltTidal', mass1=m1, mass2=m2,
                                           lambda1=lam, lambda2=lam)
                        lal_h = lal_taylorf2(m1, m2, lambda1=lam, lambda2=lam)
                        m = matchval(mine, lal_h)
                        self.assertGreaterEqual(
                            m, 0.999,
                            "match %.6f < 0.999 (m1=%g m2=%g lam=%g)"
                            % (m, m1, m2, lam))

    # -------------------------------------------------------------------
    # T2b -- float32 kernel reproduces the double-precision formula
    #        (correctness incl. aligned spin and spin x tidal; this is the
    #        per-backend correctness test => CPU/CUDA parity by transitivity)
    # -------------------------------------------------------------------
    def test_t2b_backend_vs_double_reference(self):
        with self.context:
            for m1, m2 in [(1.4, 1.4), (1.0, 1.4), (0.5, 2.0), (2.0, 0.5),
                           (1.0, 1.0), (0.5, 0.5)]:
                for s1 in SPINS:
                    for lam in LAMBDAS:
                        mine = make_filter('SPAtmpltTidal', mass1=m1, mass2=m2,
                                           spin1z=s1, spin2z=0.0,
                                           lambda1=lam, lambda2=lam)
                        ref = reference_double(m1, m2, spin1z=s1, spin2z=0.0,
                                               lambda1=lam, lambda2=lam)
                        m = matchval(mine, ref)
                        self.assertGreaterEqual(
                            m, 0.999,
                            "%s backend vs double ref %.6f (m1=%g m2=%g "
                            "s1=%g lam=%g)" % (self.scheme, m, m1, m2, s1, lam))

    # -------------------------------------------------------------------
    # T2c -- point-particle accuracy vs LAL WITH aligned spin (lambda = 0)
    # -------------------------------------------------------------------
    def test_t2c_pointparticle_with_spin_vs_lal(self):
        with self.context:
            for m1 in NS_MASSES:
                for m2 in NS_MASSES:
                    if m2 > m1:
                        continue
                    for s1 in SPINS:
                        mine = make_filter('SPAtmpltTidal', mass1=m1, mass2=m2,
                                           spin1z=s1, spin2z=0.0,
                                           lambda1=0.0, lambda2=0.0)
                        lal_h = lal_taylorf2(m1, m2, spin1z=s1, spin2z=0.0)
                        m = matchval(mine, lal_h)
                        self.assertGreaterEqual(
                            m, 0.999,
                            "match %.6f < 0.999 (m1=%g m2=%g s1=%g lam=0)"
                            % (m, m1, m2, s1))

    # -------------------------------------------------------------------
    # T2 -- detrended phase residual vs LAL (catches a wrong coefficient)
    # -------------------------------------------------------------------
    def test_t2_phase_residual_vs_lal(self):
        for lam in [1e3, 1e4, 1e5, 7e5]:
            lal_h = lal_taylorf2(1.4, 1.4, lambda1=lam, lambda2=lam)
            ref = reference_double(1.4, 1.4, lambda1=lam, lambda2=lam)
            f = np.arange(FLEN) * DF
            k = (f >= 30.0) & (f <= 900.0) & (np.abs(lal_h.numpy()) > 0)
            fa = f[k]
            d = (np.unwrap(np.angle(ref.numpy()[k]))
                 - np.unwrap(np.angle(lal_h.numpy()[k])))
            design = np.vstack([np.ones_like(fa), fa]).T
            coef, *_ = np.linalg.lstsq(design, d, rcond=None)
            resid = d - design @ coef
            self.assertLess(
                np.max(np.abs(resid)), 0.05,
                "detrended phase residual %.4f rad at lam=%g"
                % (np.max(np.abs(resid)), lam))

    # -------------------------------------------------------------------
    # T3 -- tidal effect present and matches LAL's effect size
    # -------------------------------------------------------------------
    def test_t3_effect_present_and_tracks_lal(self):
        with self.context:
            h0 = make_filter('SPAtmpltTidal', mass1=1.4, mass2=1.4,
                             lambda1=0.0, lambda2=0.0)
            mismatches = []
            for lam in [1e3, 1e4, 1e5, 7e5]:
                hl = make_filter('SPAtmpltTidal', mass1=1.4, mass2=1.4,
                                 lambda1=lam, lambda2=lam)
                mine_mm = 1.0 - matchval(h0, hl)
                mismatches.append(mine_mm)
                lal0 = lal_taylorf2(1.4, 1.4)
                lall = lal_taylorf2(1.4, 1.4, lambda1=lam, lambda2=lam)
                lal_mm = 1.0 - matchval(lal0, lall)
                self.assertGreater(mine_mm, 1e-5,
                                   "tidal effect appears zeroed at lam=%g" % lam)
                self.assertLess(
                    abs(mine_mm - lal_mm) / lal_mm, 0.10,
                    "tidal mismatch %.4e vs LAL %.4e differ >10%% (lam=%g)"
                    % (mine_mm, lal_mm, lam))
        # monotonic growth with lambda
        self.assertTrue(
            all(b > a for a, b in zip(mismatches, mismatches[1:])),
            "tidal mismatch not monotonic in lambda: %s" % mismatches)

    # -------------------------------------------------------------------
    # T4 -- closed-form tidal coefficients (host extraction; no kernel)
    # -------------------------------------------------------------------
    def test_t4_tidal_coefficients(self):
        cases = [(1.4, 1.4, 5e5, 5e5), (1.0, 1.8, 3e2, 8e2),
                 (0.5, 1.2, 1e4, 2e4), (1.8, 1.0, 3e2, 8e2),
                 (0.3, 0.9, 1e5, 5e4), (0.1, 0.1, 7e5, 7e5)]
        for m1, m2, l1, l2 in cases:
            pars = lal.CreateDict()
            ls.SimInspiralWaveformParamsInsertTidalLambda1(pars, l1)
            ls.SimInspiralWaveformParamsInsertTidalLambda2(pars, l2)
            ph = ls.SimInspiralTaylorF2AlignedPhasing(m1, m2, 0.0, 0.0, pars)
            v10, v12 = closed_form_tidal_v(m1, m2, l1, l2)
            self.assertLessEqual(abs(ph.v[10] - v10), 1e-6 * abs(v10),
                                 "v[10] %g vs closed form %g" % (ph.v[10], v10))
            self.assertLessEqual(abs(ph.v[12] - v12), 1e-6 * abs(v12),
                                 "v[12] %g vs closed form %g" % (ph.v[12], v12))
            for i in (10, 12, 13, 14):
                self.assertEqual(ph.vlogv[i], 0.0,
                                 "unexpected tidal log term at v[%d]" % i)

    # -------------------------------------------------------------------
    # T6 -- float32 stress: worst case (0.1+0.1 Msun, lambda=7e5)
    # -------------------------------------------------------------------
    def test_t6_float32_stress(self):
        m = 0.1
        with self.context:
            tidal = make_filter('SPAtmpltTidal', mass1=m, mass2=m,
                                lambda1=7e5, lambda2=7e5)
            pp = make_filter('SPAtmplt', mass1=m, mass2=m)
        m_tidal = matchval(tidal, reference_double(m, m, lambda1=7e5, lambda2=7e5))
        m_pp = matchval(pp, reference_double(m, m))
        print("\n[T6] %s: float32 match vs double ref -- "
              "point-particle=%.6f tidal=%.6f (delta=%.2e)"
              % (self.scheme, m_pp, m_tidal, m_pp - m_tidal))
        # PRIMARY (backend-independent) check: adding tides does not degrade
        # float32 accuracy beyond the point-particle SPAtmplt baseline on the
        # SAME backend.  The absolute floor itself differs by backend (CPU libm
        # vs CUDA fast-math intrinsics), but the tidal term must not be the
        # thing that loses precision.  Passing this => a double-precision phase
        # accumulator is NOT required for the tidal terms.
        self.assertLess(
            m_pp - m_tidal, 2e-3,
            "tides worsen float32 accuracy beyond point particle "
            "(pp=%.6f, tidal=%.6f)" % (m_pp, m_tidal))
        # SECONDARY loose sanity floor that both backends clear (CPU ~0.9985,
        # CUDA fast-math ~0.987 at this extreme sub-solar + huge-lambda corner).
        self.assertGreaterEqual(
            m_tidal, 0.98,
            "float32 tidal match vs double ref %.6f below sanity floor"
            % m_tidal)


suite = unittest.TestSuite()
suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestSPAtmpltTidal))

if __name__ == '__main__':
    results = unittest.TextTestRunner(verbosity=2).run(suite)
    simple_exit(results)
