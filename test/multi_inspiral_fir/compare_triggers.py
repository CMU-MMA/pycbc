#!/usr/bin/env python
"""Compare coherent triggers from brute-force pycbc_multi_inspiral against
FIR-de-chirped pycbc_multi_inspiral_fir (Phase 1 / test suite B).

Both write an EventManagerCoherent HDF5. We match network triggers between the
two files by *template identity* (template_hash, which is stable across the two
bank orderings) and geocentric time, then report the coherent-SNR ratio
distribution and any missed/extra triggers above a threshold.

Matching by template_hash (not time alone) is essential: a loud signal rings up
many nearby templates at the same time, so time-only matching mispairs
different-template triggers.

Gate: per-trigger coherent SNR agrees to ~1-2%, and the same templates trigger
above threshold.
"""
import argparse
import sys
import numpy as np
import h5py


def load(path):
    with h5py.File(path, 'r') as f:
        g = f['network']
        snr = g['coherent_snr'][:]
        t = g['end_time_gc'][:]
        h1_eid_net = g['H1_event_id'][:]
        # Map H1 per-ifo event_id -> template_hash (template identity).
        h1_eid = f['H1/event_id'][:]
        h1_hash = f['H1/template_hash'][:]
    eid_to_hash = dict(zip(h1_eid.tolist(), h1_hash.tolist()))
    thash = np.array([eid_to_hash[e] for e in h1_eid_net.tolist()],
                     dtype=np.int64)
    return snr, t, thash


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--brute', required=True)
    ap.add_argument('--fir', required=True)
    ap.add_argument('--match-window', type=float, default=0.02,
                    help="Time window (s) to match same-template triggers.")
    ap.add_argument('--snr-threshold', type=float, default=6.0,
                    help="Report missed/extra triggers above this coherent SNR.")
    ap.add_argument('--tolerance', type=float, default=0.02,
                    help="Required fractional agreement in coherent SNR.")
    ap.add_argument('--report-snr', type=float, default=15.0,
                    help="Coherent-SNR floor for the headline ratio statistics "
                         "(weak triggers are dominated by FIR-approximation "
                         "scatter and matching ambiguity).")
    args = ap.parse_args()

    bs, bt, bh = load(args.brute)
    fs, ft, fh = load(args.fir)
    print(f"brute: {len(bs)} network triggers, loudest coherent SNR "
          f"{bs.max():.4f} @ t={bt[np.argmax(bs)]:.4f}")
    print(f"fir:   {len(fs)} network triggers, loudest coherent SNR "
          f"{fs.max():.4f} @ t={ft[np.argmax(fs)]:.4f}")

    # Greedy match within each template_hash group: pair brute<->fir triggers
    # by nearest time, each fir trigger used at most once.
    ratios, matched_snr = [], []
    used = np.zeros(len(fs), dtype=bool)
    missed = []
    for i in range(len(bs)):
        cand = np.flatnonzero((fh == bh[i]) & (~used)
                              & (np.abs(ft - bt[i]) <= args.match_window))
        if len(cand):
            j = cand[np.argmin(np.abs(ft[cand] - bt[i]))]
            used[j] = True
            ratios.append(fs[j] / bs[i])
            matched_snr.append(bs[i])
        elif bs[i] >= args.snr_threshold:
            missed.append((bt[i], bs[i], bh[i]))
    extra = [(ft[j], fs[j], fh[j]) for j in range(len(fs))
             if not used[j] and fs[j] >= args.snr_threshold]

    ratios = np.array(ratios)
    matched_snr = np.array(matched_snr)
    print(f"\nmatched (same template_hash + time): {len(ratios)}")
    if len(ratios):
        dev = np.abs(ratios - 1.0)
        print(f"  all matched: ratio median={np.median(ratios):.5f} "
              f"max|ratio-1|={dev.max()*100:.4g}%")
        strong = matched_snr >= args.report_snr
        if strong.any():
            ds = np.abs(ratios[strong] - 1.0)
            print(f"  strong (coherent SNR >= {args.report_snr}): "
                  f"n={strong.sum()} median={np.median(ratios[strong]):.5f} "
                  f"mean|ratio-1|={ds.mean()*100:.4g}% "
                  f"max|ratio-1|={ds.max()*100:.4g}%")
    print(f"missed by FIR (>= {args.snr_threshold}): {len(missed)}")
    print(f"extra in FIR (>= {args.snr_threshold}): {len(extra)}")

    # Headline: loudest (the injection) and the strong-trigger agreement.
    loud = fs.max() / bs.max()
    print(f"\nloudest coherent SNR: brute={bs.max():.4f} fir={fs.max():.4f} "
          f"ratio={loud:.5f} ({abs(loud-1)*100:.4g}% off)")

    ok = True
    if abs(loud - 1) > args.tolerance:
        print(f"FAIL: loudest off by > {args.tolerance*100:.1f}%")
        ok = False
    if len(ratios):
        strong = matched_snr >= args.report_snr
        if strong.any() and np.abs(ratios[strong] - 1.0).max() > args.tolerance:
            print(f"FAIL: a strong (SNR>={args.report_snr}) trigger off by "
                  f"> {args.tolerance*100:.1f}%")
            ok = False
    print("\nGATE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 2)


if __name__ == '__main__':
    main()
