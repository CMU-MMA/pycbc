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

With --check-metadata, matching additionally requires the same slide_id (the
match key becomes template_hash + slide + time) and every strong matched pair
must recover the same sky point (ra/dec within --sky-tolerance). Use this for
sky-grid / short-slide validations; see also the standalone
compare_trigger_metadata.py. Caution: on near-degenerate sky grids (points
closer than the network's sky resolution) the recovered bin can legitimately
flip between two near-equal maxima -- gate metadata only on grids wide enough
to be non-degenerate.
"""
import argparse
import sys
import numpy as np
import h5py


def load(path):
    """Load network triggers + per-trigger template_hash. Raises ValueError
    (a clean gate failure, not a stray KeyError) if the expected datasets are
    absent -- e.g. an engine that produced no network triggers, or an IFO with
    no events so its event_id/template_hash group was never written."""
    try:
        with h5py.File(path, 'r') as f:
            if 'network' not in f or 'coherent_snr' not in f['network']:
                raise ValueError(f"{path}: no network/coherent_snr (no triggers?)")
            g = f['network']
            snr = g['coherent_snr'][:]
            t = g['end_time_gc'][:]
            slide = g['slide_id'][:].astype(np.int64)
            ra = g['ra'][:]
            dec = g['dec'][:]
            h1_eid_net = g['H1_event_id'][:]
            if 'H1' not in f or 'event_id' not in f['H1'] \
                    or 'template_hash' not in f['H1']:
                raise ValueError(f"{path}: H1 has no event_id/template_hash "
                                 "(no H1 events?)")
            h1_eid = f['H1/event_id'][:]
            h1_hash = f['H1/template_hash'][:]
    except (OSError, KeyError) as e:
        raise ValueError(f"{path}: unable to read triggers ({e})")
    eid_to_hash = dict(zip(h1_eid.tolist(), h1_hash.tolist()))
    try:
        thash = np.array([eid_to_hash[e] for e in h1_eid_net.tolist()],
                         dtype=np.int64)
    except KeyError as e:
        raise ValueError(f"{path}: network trigger references missing H1 "
                         f"event_id {e}")
    return dict(snr=snr, t=t, hash=thash, slide=slide, ra=ra, dec=dec)


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
                    help="Coherent-SNR floor defining a 'strong' (significant) "
                         "trigger: strong triggers must agree in SNR and must "
                         "be present in both engines; weak triggers below this "
                         "are dominated by FIR-approximation scatter and "
                         "matching ambiguity (reported, not gated).")
    ap.add_argument('--max-strong-mismatch', type=int, default=0,
                    help="Max allowed strong (>= --report-snr) triggers that "
                         "are missed by FIR or appear only in FIR (default 0).")
    ap.add_argument('--check-metadata', action='store_true',
                    help="Strict mode: match on template_hash + slide + time "
                         "(instead of template_hash + time) and require strong "
                         "matched triggers to recover the same sky point "
                         "(ra/dec within --sky-tolerance). Only gate this on "
                         "sky grids wide enough to be non-degenerate.")
    ap.add_argument('--sky-tolerance', type=float, default=1e-6,
                    help="Max |ra|/|dec| difference (rad) for a strong matched "
                         "pair in --check-metadata mode.")
    args = ap.parse_args()

    try:
        b = load(args.brute)
        f = load(args.fir)
    except ValueError as e:
        print(f"GATE: FAIL (could not load triggers): {e}")
        sys.exit(2)
    bs, bt, bh = b['snr'], b['t'], b['hash']
    fs, ft, fh = f['snr'], f['t'], f['hash']
    print(f"brute: {len(bs)} network triggers, loudest coherent SNR "
          f"{bs.max():.4f} @ t={bt[np.argmax(bs)]:.4f}")
    print(f"fir:   {len(fs)} network triggers, loudest coherent SNR "
          f"{fs.max():.4f} @ t={ft[np.argmax(fs)]:.4f}")

    # Greedy match within each template_hash group: pair brute<->fir triggers
    # by nearest time, each fir trigger used at most once. In --check-metadata
    # mode the pair must also share the slide_id.
    ratios, matched_snr = [], []
    used = np.zeros(len(fs), dtype=bool)
    missed = []
    sky_mismatch = []
    for i in range(len(bs)):
        same = ((fh == bh[i]) & (~used)
                & (np.abs(ft - bt[i]) <= args.match_window))
        if args.check_metadata:
            same &= f['slide'] == b['slide'][i]
        cand = np.flatnonzero(same)
        if len(cand):
            j = cand[np.argmin(np.abs(ft[cand] - bt[i]))]
            used[j] = True
            ratios.append(fs[j] / bs[i])
            matched_snr.append(bs[i])
            if args.check_metadata and bs[i] >= args.report_snr:
                dra = abs(float(f['ra'][j]) - float(b['ra'][i]))
                ddec = abs(float(f['dec'][j]) - float(b['dec'][i]))
                if dra > args.sky_tolerance or ddec > args.sky_tolerance:
                    sky_mismatch.append((bt[i], bs[i], fs[j], dra, ddec))
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
    # Strong (significant) missed/extra are gated; weak sidelobe differences
    # are reported but tolerated (FIR is an approximation on weak triggers).
    missed_strong = [m for m in missed if m[1] >= args.report_snr]
    extra_strong = [e for e in extra if e[1] >= args.report_snr]
    print(f"missed by FIR (>= {args.snr_threshold}): {len(missed)} "
          f"(strong >= {args.report_snr}: {len(missed_strong)})")
    print(f"extra in FIR (>= {args.snr_threshold}): {len(extra)} "
          f"(strong >= {args.report_snr}: {len(extra_strong)})")

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
    # Enforce "same significant triggers above threshold": no strong trigger may
    # be missed by FIR or appear only in FIR (beyond --max-strong-mismatch).
    if len(missed_strong) > args.max_strong_mismatch:
        print(f"FAIL: {len(missed_strong)} strong trigger(s) missed by FIR "
              f"(> {args.max_strong_mismatch})")
        ok = False
    if len(extra_strong) > args.max_strong_mismatch:
        print(f"FAIL: {len(extra_strong)} strong trigger(s) only in FIR "
              f"(> {args.max_strong_mismatch})")
        ok = False
    if args.check_metadata:
        print(f"strong sky mismatches (> {args.sky_tolerance} rad): "
              f"{len(sky_mismatch)}")
        for bt_i, bs_i, fs_j, dra, ddec in sky_mismatch[:10]:
            print(f"  t={bt_i:.4f} snr brute={bs_i:.4f} fir={fs_j:.4f} "
                  f"dsky=({dra:.3g},{ddec:.3g})")
        if len(sky_mismatch) > args.max_strong_mismatch:
            print(f"FAIL: {len(sky_mismatch)} strong matched trigger(s) "
                  f"recovered a different sky point")
            ok = False
    print("\nGATE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 2)


if __name__ == '__main__':
    main()
