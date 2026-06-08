# Copyright (C) 2012  Alex Nitz, Josh Willis
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

"""
Unit tests for PyCBC's thresholding code.
"""

import unittest
import numpy
from pycbc.types import Array, complex64
from pycbc.events import threshold, threshold_only
from utils import parse_args_all_schemes, simple_exit

_scheme, _context = parse_args_all_schemes("Threshold")

from pycbc.events.threshold_cpu import threshold_numpy as trusted_threshold
# CPU reference for the (no-clustering) threshold_only operation. This runs on
# the host regardless of the active scheme, so it is a valid reference for the
# CUDA implementation.
from pycbc.events.threshold_cpu import threshold_only as cpu_threshold_only


def _as_sorted_set(locs, vals):
    """Return (locs, vals) sorted by location, for order-independent compares."""
    locs = numpy.asarray(locs)
    vals = numpy.asarray(vals)
    order = numpy.argsort(locs, kind='stable')
    return locs[order], vals[order]


class TestThreshold(unittest.TestCase):
    def setUp(self, *args):
        self.context = _context
        self.scheme = _scheme
        numpy.random.seed(1804)
        r = numpy.random.uniform(low=-1, high=1.0, size=2**20)
        i = numpy.random.uniform(low=-1, high=1.0, size=2**20)
        v = r + i * 1.0j
        self.series = Array(v, dtype=complex64)
        self.threshold = 1.3
        self.locs, self.vals = trusted_threshold(self.series, self.threshold)
        print(f'Reference: {len(self.locs)} locs, {len(self.vals)} vals')

    def test_threshold(self):
        with self.context:
            locs, vals = threshold(self.series, self.threshold)
            print(f'Test: {len(locs)} locs, {len(vals)} vals')
            self.assertTrue((locs == self.locs).all())
            self.assertTrue((vals == self.vals).all())

    def test_threshold_only_equivalence(self):
        # threshold_only must match the CPU threshold_only over a range of
        # thresholds, both in the set of locations and their values.
        for value in (0.0, 0.5, 1.0, 1.3, 1.41):
            ref_locs, ref_vals = cpu_threshold_only(self.series, value)
            ref_locs, ref_vals = _as_sorted_set(ref_locs, ref_vals)
            with self.context:
                locs, vals = threshold_only(self.series, value)
            locs, vals = _as_sorted_set(locs, vals)
            self.assertEqual(len(locs), len(ref_locs),
                             msg=f'count mismatch at threshold {value}')
            self.assertTrue((locs == ref_locs).all(),
                            msg=f'location mismatch at threshold {value}')
            self.assertTrue((vals == ref_vals).all(),
                            msg=f'value mismatch at threshold {value}')

    def test_threshold_only_dtypes_and_order(self):
        # Regression guard: threshold_only must return (locations, values), the
        # OPPOSITE order of threshold_and_cluster which returns (values, locs).
        with self.context:
            locs, vals = threshold_only(self.series, self.threshold)
        self.assertEqual(numpy.asarray(locs).dtype, numpy.uint32)
        self.assertEqual(numpy.asarray(vals).dtype, numpy.complex64)
        # locations are real, non-negative indices into the series
        self.assertTrue((numpy.asarray(locs) < len(self.series)).all())
        # the value at each returned location must match the series there, which
        # is only true if the first return value really is the *locations*.
        host = numpy.array(self.series.data, dtype=numpy.complex64)
        self.assertTrue(numpy.allclose(host[numpy.asarray(locs)],
                                       numpy.asarray(vals)))

    def test_threshold_only_no_triggers(self):
        # A threshold above the maximum magnitude yields empty arrays.
        big = 1e9
        with self.context:
            locs, vals = threshold_only(self.series, big)
        self.assertEqual(len(locs), 0)
        self.assertEqual(len(vals), 0)
        self.assertEqual(numpy.asarray(locs).dtype, numpy.uint32)
        # The CUDA implementation is specified to return an empty complex64
        # values array. (The CPU threshold_only returns an empty float32 array
        # in the no-trigger case, a pre-existing quirk, so only assert this
        # under the CUDA scheme.)
        if self.scheme == 'cuda':
            self.assertEqual(numpy.asarray(vals).dtype, numpy.complex64)

    def test_threshold_only_known_indices(self):
        # A small series with a known, exact set of above-threshold samples.
        data = numpy.zeros(1024, dtype=numpy.complex64)
        known = numpy.array([3, 17, 500, 1023], dtype=numpy.uint32)
        data[known] = 5.0 + 0.0j  # magnitude 5 > threshold 2
        series = Array(data, dtype=complex64)
        with self.context:
            locs, vals = threshold_only(series, 2.0)
        locs, vals = _as_sorted_set(locs, vals)
        self.assertTrue((locs == known).all())
        self.assertTrue(numpy.allclose(vals, 5.0 + 0.0j))

    def test_threshold_only_overflow(self):
        # More than the historical buffer capacity (4096*256 = 1048576) of
        # above-threshold samples. The implementation must resize (or raise),
        # never silently truncate.
        capacity = 4096 * 256
        n = capacity + 1000
        data = numpy.ones(n, dtype=numpy.complex64)  # every sample magnitude 1
        series = Array(data, dtype=complex64)
        ref_locs, ref_vals = cpu_threshold_only(series, 0.5)
        try:
            with self.context:
                locs, vals = threshold_only(series, 0.5)
        except RuntimeError as e:
            # Explicitly raising on saturation is an acceptable outcome.
            self.assertIn('overflow', str(e).lower())
            return
        # If it did not raise, it must have resized and returned everything.
        self.assertEqual(len(locs), n)
        self.assertEqual(len(locs), len(ref_locs))
        locs, _ = _as_sorted_set(locs, vals)
        self.assertTrue((locs == numpy.arange(n, dtype=numpy.uint32)).all())


suite = unittest.TestSuite()
suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestThreshold))

if __name__ == '__main__':
    results = unittest.TextTestRunner(verbosity=2).run(suite)
    simple_exit(results)
