#!/usr/bin/env python
"""Generate small, controlled coarse + fine template banks for validating
``pycbc_multi_inspiral_fir`` against brute-force ``pycbc_multi_inspiral``.

The production sub-solar banks hold 10^7 templates whose chirp times run to
hours -- impossible to brute-force. For a correctness gate we instead build a
*small* bank in the short sub-solar corner so a handful of templates can be
brute-forced over a single segment.

FIR de-chirping requires that every fine template sit close (in chirp time) to
a coarse reference: the de-chirp FIR filter's length is roughly the
chirp-time gap times the sample rate, so the gap must be well under
fir_fft_len/sample_rate (~2 s) -- ideally a fraction of a second -- for the
taps to stay short and accurate. We therefore lay both banks out on a regular
grid in **chirp time tau0** (and mass ratio q), with the coarse grid a strict
subset of the fine grid (coarse tau0 step an integer multiple of the fine
step). That guarantees small coarse->fine tau0 gaps -> short filters -> fast,
high-fidelity tap design.

Point-particle TaylorF2, aligned (zero) spins. The banks are plain HDF5 files
with the datasets FilterBank reads (mass1, mass2, spin1z, spin2z, f_lower,
approximant); the approximant is also overridable on the command line at
search time.
"""
import argparse
import numpy as np
import h5py
from pycbc.conversions import (
    mchirp_from_tau0, mass1_from_mchirp_q, mass2_from_mchirp_q,
    tau0_from_mass1_mass2,
)


def grid(tau0_vals, q_vals, f_lower, m_lo=0.1, m_hi=2.0):
    """(tau0, q) grid -> component masses, keeping the sub-solar box."""
    m1l, m2l = [], []
    for q in q_vals:
        for t0 in tau0_vals:
            mc = mchirp_from_tau0(t0, f_lower)
            m1 = mass1_from_mchirp_q(mc, q)
            m2 = mass2_from_mchirp_q(mc, q)
            if m_lo <= m2 <= m1 <= m_hi:
                m1l.append(float(m1))
                m2l.append(float(m2))
    return np.array(m1l), np.array(m2l)


def write_bank(path, m1, m2, f_lower, approximant='TaylorF2'):
    with h5py.File(path, 'w') as f:
        f['mass1'] = m1.astype(np.float64)
        f['mass2'] = m2.astype(np.float64)
        f['spin1z'] = np.zeros_like(m1)
        f['spin2z'] = np.zeros_like(m1)
        f['f_lower'] = np.full_like(m1, float(f_lower))
        f.create_dataset('approximant',
                         data=np.array([approximant] * len(m1),
                                       dtype=h5py.string_dtype('utf-8')))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fine-output', required=True)
    ap.add_argument('--coarse-output', required=True)
    ap.add_argument('--f-lower', type=float, default=45.0)
    ap.add_argument('--tau0-min', type=float, default=8.0)
    ap.add_argument('--tau0-max', type=float, default=13.0)
    ap.add_argument('--fine-tau0-step', type=float, default=0.15)
    ap.add_argument('--coarse-tau0-step', type=float, default=0.30)
    ap.add_argument('--q-values', type=float, nargs='+',
                    default=[1.0, 1.4, 2.0])
    ap.add_argument('--approximant', default='TaylorF2')
    args = ap.parse_args()

    fine_tau0 = np.arange(args.tau0_min, args.tau0_max + 1e-9,
                          args.fine_tau0_step)
    coarse_tau0 = np.arange(args.tau0_min, args.tau0_max + 1e-9,
                            args.coarse_tau0_step)

    fm1, fm2 = grid(fine_tau0, args.q_values, args.f_lower)
    cm1, cm2 = grid(coarse_tau0, args.q_values, args.f_lower)

    write_bank(args.fine_output, fm1, fm2, args.f_lower, args.approximant)
    write_bank(args.coarse_output, cm1, cm2, args.f_lower, args.approximant)

    ftau = tau0_from_mass1_mass2(fm1, fm2, args.f_lower)
    print(f"Fine bank:   {len(fm1):4d} templates -> {args.fine_output}")
    print(f"  tau0 (s): min={ftau.min():.2f} max={ftau.max():.2f} "
          f"(step {args.fine_tau0_step})")
    print(f"Coarse bank: {len(cm1):4d} templates -> {args.coarse_output} "
          f"(step {args.coarse_tau0_step})")
    print(f"  mass1 [{fm1.min():.3f}, {fm1.max():.3f}], "
          f"mass2 [{fm2.min():.3f}, {fm2.max():.3f}]")

    # Recommend an injection at the middle of the q=1 line (a fine grid point).
    t0_mid = fine_tau0[len(fine_tau0) // 2]
    mc = mchirp_from_tau0(t0_mid, args.f_lower)
    inj_m = mass1_from_mchirp_q(mc, 1.0)  # equal mass
    print(f"\nRecommended injection (q=1, tau0={t0_mid:.2f}s): "
          f"mass1=mass2={inj_m:.4f}  (tau0={t0_mid:.2f}s)")


if __name__ == '__main__':
    main()
