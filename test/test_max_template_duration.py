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

"""
Unit tests for the per-template duration cap used by pycbc_multi_inspiral
(the ``--max-template-duration`` option, which feeds FilterBank's
``max_template_length`` and FilterBank.find_variable_start_frequency).

These tests are CPU-only and need no GPU.
"""

import os
import types
import tempfile
import unittest

import numpy as np
import h5py

from pycbc.types import zeros, complex64
from pycbc.waveform import get_waveform_filter_length_in_time
import pycbc.waveform.bank as bankmod
from pycbc.waveform.bank import find_variable_start_frequency, FilterBank
from utils import simple_exit

APPROX = "TaylorF2"
FLOW = 15.0


def make_params(m1, m2, f_lower=FLOW):
    """A minimal template-parameter object (like a row of bank.table)."""
    p = types.SimpleNamespace()
    p.mass1 = m1
    p.mass2 = m2
    p.spin1z = 0.0
    p.spin2z = 0.0
    p.f_lower = f_lower
    return p


class TestMaxTemplateDuration(unittest.TestCase):
    def setUp(self):
        # The lightest sub-solar template: very long at the floor f_low.
        self.long_p = make_params(0.1, 0.1)
        # A heavy, short template.
        self.short_p = make_params(10.0, 10.0)
        self.cap = 3600.0

    def test_long_template_is_capped(self):
        # Sanity: uncapped duration at the floor is far longer than the cap.
        d0 = get_waveform_filter_length_in_time(APPROX, self.long_p,
                                                f_lower=FLOW)
        self.assertGreater(d0, self.cap)
        # The cap raises the start frequency above the floor ...
        f = find_variable_start_frequency(APPROX, self.long_p, FLOW, self.cap)
        self.assertGreater(f, FLOW)
        # ... and the resulting duration fits within the cap.
        d = get_waveform_filter_length_in_time(APPROX, self.long_p, f_lower=f)
        self.assertLessEqual(d, self.cap)

    def test_short_template_left_at_floor(self):
        # A template already shorter than the cap keeps the floor f_low.
        d0 = get_waveform_filter_length_in_time(APPROX, self.short_p,
                                                f_lower=FLOW)
        self.assertLess(d0, self.cap)
        f = find_variable_start_frequency(APPROX, self.short_p, FLOW, self.cap)
        self.assertEqual(f, FLOW)

    def test_none_cap_is_noop(self):
        # max_length=None must reproduce current behavior: no f_low change,
        # for both long and short templates.
        self.assertEqual(
            find_variable_start_frequency(APPROX, self.long_p, FLOW, None),
            FLOW)
        self.assertEqual(
            find_variable_start_frequency(APPROX, self.short_p, FLOW, None),
            FLOW)

    def test_taylorf2_default_pn_order_path(self):
        # find_variable_start_frequency calls get_waveform_filter_length_in_time
        # WITHOUT an explicit phase_order; the params object below has no
        # phase_order attribute. Confirm the default-order path raises no
        # None/order error and still caps the duration.
        self.assertFalse(hasattr(self.long_p, 'phase_order'))
        f = find_variable_start_frequency(APPROX, self.long_p, FLOW, 1800.0)
        self.assertGreater(f, FLOW)
        d = get_waveform_filter_length_in_time(APPROX, self.long_p, f_lower=f)
        self.assertLessEqual(d, 1800.0)

    def test_filterbank_forwards_max_template_length(self):
        # Build a tiny on-disk bank and confirm FilterBank (a) stores
        # max_template_length and (b) __getitem__ forwards it to
        # find_variable_start_frequency. We only generate the SHORT template
        # (cheap); the long one would need a multi-thousand-second segment.
        fd, path = tempfile.mkstemp(suffix='.hdf')
        os.close(fd)
        try:
            with h5py.File(path, 'w') as f:
                f.attrs['parameters'] = ['mass1', 'mass2', 'spin1z', 'spin2z']
                f['mass1'] = np.array([10.0, 0.1], dtype=np.float32)
                f['mass2'] = np.array([10.0, 0.1], dtype=np.float32)
                f['spin1z'] = np.array([0.0, 0.0], dtype=np.float32)
                f['spin2z'] = np.array([0.0, 0.0], dtype=np.float32)

            sample_rate = 2048
            seg_len = 64  # seconds; comfortably longer than the 10+10 template
            delta_f = 1.0 / seg_len
            flen = int(seg_len * sample_rate / 2) + 1
            out = zeros(flen, dtype=complex64)

            bank = FilterBank(
                path, flen, delta_f, complex64,
                low_frequency_cutoff=FLOW,
                approximant=APPROX,
                max_template_length=self.cap,
                out=out,
            )
            # (a) constructor stores the cap
            self.assertEqual(bank.max_template_length, self.cap)

            # (b) __getitem__ forwards self.max_template_length
            captured = {}
            orig = bankmod.find_variable_start_frequency

            def spy(approximant, parameters, f_start, max_length, delta_f=1):
                captured['max_length'] = max_length
                captured['f_start'] = f_start
                return orig(approximant, parameters, f_start, max_length,
                            delta_f)

            bankmod.find_variable_start_frequency = spy
            try:
                _ = bank[0]  # short (10+10) template
            finally:
                bankmod.find_variable_start_frequency = orig

            self.assertEqual(captured['max_length'], self.cap)
            self.assertEqual(captured['f_start'], FLOW)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_filterbank_none_default_is_noop(self):
        # With no cap (default None), the FilterBank passes max_length=None and
        # the short template stays at the floor frequency.
        fd, path = tempfile.mkstemp(suffix='.hdf')
        os.close(fd)
        try:
            with h5py.File(path, 'w') as f:
                f.attrs['parameters'] = ['mass1', 'mass2', 'spin1z', 'spin2z']
                f['mass1'] = np.array([10.0], dtype=np.float32)
                f['mass2'] = np.array([10.0], dtype=np.float32)
                f['spin1z'] = np.array([0.0], dtype=np.float32)
                f['spin2z'] = np.array([0.0], dtype=np.float32)

            sample_rate = 2048
            seg_len = 64
            delta_f = 1.0 / seg_len
            flen = int(seg_len * sample_rate / 2) + 1
            out = zeros(flen, dtype=complex64)

            bank = FilterBank(
                path, flen, delta_f, complex64,
                low_frequency_cutoff=FLOW,
                approximant=APPROX,
                out=out,
            )
            self.assertIsNone(bank.max_template_length)

            captured = {}
            orig = bankmod.find_variable_start_frequency

            def spy(approximant, parameters, f_start, max_length, delta_f=1):
                captured['max_length'] = max_length
                return orig(approximant, parameters, f_start, max_length,
                            delta_f)

            bankmod.find_variable_start_frequency = spy
            try:
                _ = bank[0]
            finally:
                bankmod.find_variable_start_frequency = orig

            self.assertIsNone(captured['max_length'])
        finally:
            if os.path.exists(path):
                os.remove(path)


suite = unittest.TestSuite()
suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestMaxTemplateDuration))

if __name__ == '__main__':
    results = unittest.TextTestRunner(verbosity=2).run(suite)
    simple_exit(results)
