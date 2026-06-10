#!/usr/bin/env python
"""Compare trigger recovery metadata between brute and FIR coherent outputs.

This complements compare_triggers.py. The existing gate checks SNR agreement
after matching by template_hash + geocentric time. This script checks whether
matched loud triggers also recover the same slide and sky-grid point.
"""
import argparse
import sys

import h5py
import numpy as np


def load(path):
    try:
        with h5py.File(path, 'r') as f:
            if 'network' not in f or 'coherent_snr' not in f['network']:
                raise ValueError(f"{path}: no network/coherent_snr")
            g = f['network']
            h1_eid_net = g['H1_event_id'][:]
            if 'H1' not in f or 'event_id' not in f['H1'] \
                    or 'template_hash' not in f['H1']:
                raise ValueError(f"{path}: H1 event_id/template_hash missing")
            eid_to_hash = dict(zip(f['H1/event_id'][:].tolist(),
                                   f['H1/template_hash'][:].tolist()))
            thash = np.array([eid_to_hash[int(e)] for e in h1_eid_net],
                             dtype=np.int64)
            out = {
                'snr': g['coherent_snr'][:],
                'time': g['end_time_gc'][:],
                'hash': thash,
                'slide': g['slide_id'][:].astype(np.int64),
                'ra': g['ra'][:],
                'dec': g['dec'][:],
            }
    except (OSError, KeyError) as e:
        raise ValueError(f"{path}: unable to read metadata ({e})")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--brute', required=True)
    ap.add_argument('--fir', required=True)
    ap.add_argument('--match-window', type=float, default=0.02)
    ap.add_argument('--report-snr', type=float, default=15.0)
    ap.add_argument('--sky-tolerance', type=float, default=1e-6)
    ap.add_argument('--max-strong-metadata-mismatch', type=int, default=0)
    args = ap.parse_args()

    try:
        b = load(args.brute)
        f = load(args.fir)
    except ValueError as e:
        print(f"METADATA: FAIL (could not load triggers): {e}")
        sys.exit(2)

    used = np.zeros(len(f['snr']), dtype=bool)
    mismatches = []
    matched = 0
    strong_matched = 0
    strong_missing = []

    for i in np.argsort(b['snr'])[::-1]:
        cand = np.flatnonzero(
            (f['hash'] == b['hash'][i])
            & (f['slide'] == b['slide'][i])
            & (~used)
            & (np.abs(f['time'] - b['time'][i]) <= args.match_window)
        )
        if len(cand) == 0:
            if b['snr'][i] >= args.report_snr:
                strong_missing.append(i)
            continue
        j = cand[np.argmax(f['snr'][cand])]
        used[j] = True
        matched += 1
        if b['snr'][i] >= args.report_snr:
            strong_matched += 1
            dra = abs(float(f['ra'][j]) - float(b['ra'][i]))
            ddec = abs(float(f['dec'][j]) - float(b['dec'][i]))
            if dra > args.sky_tolerance or ddec > args.sky_tolerance:
                mismatches.append((i, j, dra, ddec))

    print(f"brute: {len(b['snr'])} triggers, fir: {len(f['snr'])} triggers")
    print(f"matched by template_hash + time + slide: {matched}")
    print(f"strong matched (SNR >= {args.report_snr}): {strong_matched}")
    print(f"strong missing with same slide: {len(strong_missing)}")
    print(f"strong sky mismatches: {len(mismatches)}")

    for i, j, dra, ddec in mismatches[:10]:
        print(
            "  mismatch: "
            f"snr brute={b['snr'][i]:.4f} fir={f['snr'][j]:.4f} "
            f"time brute={b['time'][i]:.4f} fir={f['time'][j]:.4f} "
            f"slide={int(b['slide'][i])} hash={int(b['hash'][i])} "
            f"brute_sky=({b['ra'][i]:.8f},{b['dec'][i]:.8f}) "
            f"fir_sky=({f['ra'][j]:.8f},{f['dec'][j]:.8f}) "
            f"dsky=({dra:.3g},{ddec:.3g})"
        )

    ok = True
    if len(strong_missing) > args.max_strong_metadata_mismatch:
        print("FAIL: strong triggers missing with the same slide")
        ok = False
    if len(mismatches) > args.max_strong_metadata_mismatch:
        print("FAIL: strong matched triggers recovered a different sky bin")
        ok = False
    print("METADATA:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 2)


if __name__ == '__main__':
    main()
