import logging
import numpy as np
import scipy.fft as _scipy_fft
import pycbc.fft
import pycbc.scheme
from pycbc.types import zeros, complex64, Array
from pycbc.filter.matchedfilter import matched_filter_core, sigmasq
from pycbc.filter.matchedfilter_cpu import fast_multiply_analytic_cython, find_peaks_in_block_cython


def _on_cuda():
    """True if the active processing scheme is a CUDA scheme."""
    return isinstance(pycbc.scheme.mgr.state, pycbc.scheme.CUDAScheme)


_analytic_mult_kernel_cache = {}


def _get_analytic_mult_kernel():
    """Return (memoised, per-context) a PyCUDA ElementwiseKernel that performs
    the batched half-spectrum ("analytic signal") multiply used by the GPU FIR
    engine. ``bf`` holds ``n_blocks`` stacked length-``nfft`` block spectra
    (size ``n_blocks*nfft``); ``filt`` is a single length-``nfft`` filter,
    broadcast across blocks via ``j = i % nfft``:

        out[i] = bf[i] * filt[i % nfft] * inv_n   for (i % nfft) <  nhalf
        out[i] = 0                                for (i % nfft) >= nhalf

    ``bf`` is passed first so PyCUDA sizes the launch to ``n_blocks*nfft``
    (it uses the first vector arg's size and does not require equal-length
    args); ``filt`` is read only at indices ``< nfft``, so it is safe.

    The ``inv_n = 1/nfft`` factor folds in the inverse-FFT normalization: the
    GPU ifft (scikit-cuda cufft) is unnormalized, whereas the CPU path uses
    ``scipy.fft.ifft`` which carries the 1/N. Applying 1/N here makes the
    batched GPU result match the CPU result to single precision -- which the
    CPU<->CUDA parity test checks (and asserts is non-zero, to catch a silent
    CPU fallback).
    """
    from pycuda.elementwise import ElementwiseKernel
    key = id(pycbc.scheme.mgr.state)
    krnl = _analytic_mult_kernel_cache.get(key)
    if krnl is None:
        krnl = ElementwiseKernel(
            "pycuda::complex<float> *bf, pycuda::complex<float> *filt, "
            "pycuda::complex<float> *out, int nfft, int nhalf, float inv_n",
            "int j = i % nfft; "
            "out[i] = (j < nhalf) ? bf[i] * filt[j] * inv_n "
            ": pycuda::complex<float>(0.0f, 0.0f)",
            "fir_analytic_mult_batched")
        _analytic_mult_kernel_cache[key] = krnl
    return krnl


_assemble_kernel_cache = {}


def _get_assemble_kernel():
    """Return (memoised, per-context) a PyCUDA ElementwiseKernel that gathers
    the overlap-save *valid* samples of every block into a contiguous device
    SNR series -- the on-device equivalent of the CPU path's
    ``ascontiguousarray(corr[:, bad_start:bad_start+N_VALID]).ravel()`` followed
    by ``out[lo:hi] = valid_flat[...]``.

    Launched over ``i in [0, hi-lo)`` (via ``range=slice(0, hi-lo, 1)``); each
    output position ``p = i + lo`` maps back to a block ``b`` and a valid-region
    offset ``r`` in the flat ``corr`` buffer of ``n_blocks`` stacked
    length-``n_fft`` blocks:

        j = p - assemble_start;  b = j / n_valid;  r = j % n_valid
        out[p] = corr[b*n_fft + bad_start + r]

    Writing only ``[lo, hi)`` leaves the rest of ``out`` (a preallocated zero
    array) untouched, matching the CPU path which leaves samples outside the
    valid region zero. Keeping this on the device removes the per-fine-template
    device->host copy of the whole (overlap-padded) correlation buffer.
    """
    from pycuda.elementwise import ElementwiseKernel
    key = id(pycbc.scheme.mgr.state)
    krnl = _assemble_kernel_cache.get(key)
    if krnl is None:
        krnl = ElementwiseKernel(
            "pycuda::complex<float> *corr, pycuda::complex<float> *out, "
            "int n_fft, int n_valid, int bad_start, int assemble_start, int lo",
            "int p = i + lo; int j = p - assemble_start; "
            "int b = j / n_valid; int r = j - b * n_valid; "
            "out[p] = corr[b * n_fft + bad_start + r]",
            "fir_assemble_valid")
        _assemble_kernel_cache[key] = krnl
    return krnl


_block_gather_kernel_cache = {}


def _get_block_gather_kernel():
    """Return (memoised, per-context) a PyCUDA ElementwiseKernel that builds the
    overlap-save block-input buffer of the reference SNR ON THE DEVICE.

    Replaces the host-side Python loop + device upload that the original GPU
    path used to assemble ``block_in`` from a *host* ``ref_snr``. The reference
    SNR now lives on the device (see :meth:`compute_reference_snr`), so the
    ``n_blocks`` length-``nfft`` blocks (stride ``nvalid`` in time, starting at
    sample ``block0``) are gathered directly device->device:

        b = i / nfft;  col = i % nfft;  g = block0 + b*nvalid + col
        out[i] = (g < n_samples) ? ref[g] : 0

    which is exactly ``ref_snr[t_starts[b] : t_starts[b]+nfft]`` zero-padded at
    the tail, with ``t_starts[b] = block0 + b*nvalid``.
    """
    from pycuda.elementwise import ElementwiseKernel
    key = id(pycbc.scheme.mgr.state)
    krnl = _block_gather_kernel_cache.get(key)
    if krnl is None:
        krnl = ElementwiseKernel(
            "pycuda::complex<float> *ref, pycuda::complex<float> *out, "
            "int nfft, int nvalid, int block0, int n_samples",
            "int b = i / nfft; int col = i - b * nfft; "
            "int g = block0 + b * nvalid + col; "
            "out[i] = (g < n_samples) ? ref[g] "
            ": pycuda::complex<float>(0.0f, 0.0f)",
            "fir_block_gather")
        _block_gather_kernel_cache[key] = krnl
    return krnl


_batched_mult_kernel_cache = {}


def _get_batched_mult_kernel():
    """Return (memoised, per-context) the *tiled* analytic-multiply kernel: the
    K-filter generalisation of :func:`_get_analytic_mult_kernel`.

    Given ``bf`` (one shared set of ``n_blocks`` length-``nfft`` block spectra,
    size ``bspan = n_blocks*nfft``) and ``filt`` (a tile of ``K`` length-``nfft``
    conjugated filters, size ``K*nfft``), it writes the half-spectrum product for
    every (filter, block) pair into ``out`` (size ``K*bspan``):

        k = i / bspan;  m = i % bspan;  j = m % nfft
        out[i] = (j < nhalf) ? bf[m] * filt[k*nfft + j] * inv_n : 0

    ``bf`` is read modulo ``bspan`` so the SINGLE cached forward FFT of the
    reference SNR is broadcast across all ``K`` filters -- no per-filter copy.
    ``inv_n = 1/nfft`` folds the (unnormalised cufft) inverse-FFT normalisation,
    identical to the single-filter kernel, so the batched result is bit-for-bit
    the per-filter result.
    """
    from pycuda.elementwise import ElementwiseKernel
    key = id(pycbc.scheme.mgr.state)
    krnl = _batched_mult_kernel_cache.get(key)
    if krnl is None:
        krnl = ElementwiseKernel(
            "pycuda::complex<float> *bf, pycuda::complex<float> *filt, "
            "pycuda::complex<float> *out, int nfft, int nhalf, float inv_n, "
            "int bspan",
            "int k = i / bspan; int m = i - k * bspan; int j = m % nfft; "
            "out[i] = (j < nhalf) ? bf[m] * filt[k * nfft + j] * inv_n "
            ": pycuda::complex<float>(0.0f, 0.0f)",
            "fir_analytic_mult_tiled")
        _batched_mult_kernel_cache[key] = krnl
    return krnl


_batched_assemble_kernel_cache = {}


def _get_batched_assemble_kernel():
    """Return (memoised, per-context) the *tiled* overlap-save assemble+rescale
    kernel: the K-filter generalisation of :func:`_get_assemble_kernel`, with the
    per-template fine-normalisation (``snr_rescale``) division folded in.

    For each of ``K`` filters it writes the FULL length-``L`` output row inside
    ``out`` (size ``K*L``): gathering the overlap-save valid samples inside the
    valid region ``[lo, hi)`` and writing zero outside it. ``corr`` is the
    batched inverse-FFT output (size ``K*bspan``); launched over ``K*L``:

        k = i / L;  local = i % L;  p = v_start + local
        if lo <= p < hi:
            j = p - assemble_start;  b = j / nvalid;  r = j % nvalid
            out[k*L + local] = corr[k*bspan + b*nfft + bad_start + r] / rescale[k]
        else:
            out[k*L + local] = 0

    Writing the whole row (rather than only ``[lo, hi)``) means ``out`` is fully
    overwritten every tile, so the caller can REUSE one output buffer across
    tiles and coarse groups even though the valid region ``[lo, hi)`` shifts
    when the group's overlap-save tap count changes -- no stale samples, no
    per-tile pre-zeroing.

    Folding ``/ rescale[k]`` here removes the per-template device divide (and its
    allocation) the driver used to do (``series[analyze] / snr_rescale``); the
    float32 division is the same operation the CPU parity path applies on the
    host (difference is single-precision round-off, ~1e-7, far below the trigger
    gates).
    """
    from pycuda.elementwise import ElementwiseKernel
    key = id(pycbc.scheme.mgr.state)
    krnl = _batched_assemble_kernel_cache.get(key)
    if krnl is None:
        krnl = ElementwiseKernel(
            "pycuda::complex<float> *corr, pycuda::complex<float> *out, "
            "float *rescale, int nfft, int nvalid, int bad_start, "
            "int assemble_start, int v_start, int lo, int hi, int bspan, "
            "int L",
            "int k = i / L; int local = i - k * L; int p = v_start + local; "
            "if (p >= lo && p < hi) { "
            "int j = p - assemble_start; int b = j / nvalid; int r = j - b * nvalid; "
            "out[k * L + local] = "
            "corr[k * bspan + b * nfft + bad_start + r] / rescale[k]; "
            "} else { out[k * L + local] = pycuda::complex<float>(0.0f, 0.0f); }",
            "fir_assemble_tiled")
        _batched_assemble_kernel_cache[key] = krnl
    return krnl


_cufft_plan_cache = {}


def _get_cufft_plan(n_blocks, n_fft):
    """Return (memoised, per-context) a scikit-cuda batched C2C cufft plan for
    ``n_blocks`` transforms of length ``n_fft``. The same plan serves both the
    forward and inverse transforms (direction is set by the fft/ifft call)."""
    from skcuda import fft as cu_fft
    key = (id(pycbc.scheme.mgr.state), int(n_blocks), int(n_fft))
    plan = _cufft_plan_cache.get(key)
    if plan is None:
        plan = cu_fft.Plan((int(n_fft),), np.complex64, np.complex64,
                           batch=int(n_blocks))
        _cufft_plan_cache[key] = plan
    return plan


class _ScipyFFTBackend(object):
    """Drop-in replacement for the ``mkl_fft`` module interface used by the
    ratio engine.

    Historically the engine called ``mkl_fft.fft``/``mkl_fft.ifft`` directly.
    ``mkl_fft`` is not available in every environment (e.g. the ssm_pipeline
    micromamba env), so we route the CPU batched FFTs through ``scipy.fft``
    instead. ``scipy.fft`` preserves ``complex64`` precision, uses the same
    (numpy/backward) normalization convention as ``mkl_fft`` -- i.e. the 1/N
    factor lives on the inverse transform -- and supports multi-threaded
    batched transforms via the ``workers`` argument.

    This object intentionally mirrors only the small slice of the ``mkl_fft``
    API the engine relies on: ``fft(a, axis=-1)`` and
    ``ifft(a, axis=-1, out=...)``. A future CUDA backend can expose the same
    two methods (see Phase 2).
    """

    def __init__(self, workers=None):
        # ``workers=None`` lets scipy fall back to a single worker; a positive
        # integer enables threaded batched transforms (used for throughput).
        self.workers = workers

    def fft(self, a, axis=-1, out=None):
        r = _scipy_fft.fft(a, axis=axis, workers=self.workers)
        if out is not None:
            out[:] = r
            return out
        return r

    def ifft(self, a, axis=-1, out=None):
        r = _scipy_fft.ifft(a, axis=axis, workers=self.workers)
        if out is not None:
            out[:] = r
            return out
        return r

class RatioMatchedFilterControl(object):
    """
    High-performance engine for hierarchical "Ratio/FIR" matched filtering.
    Uses mkl_fft for ALL FFT operations to maximize throughput and consistency.
    """

    def __init__(self, snr_threshold, delta_f,
                 high_frequency_cutoff=None, fir_fft_length=4096, batch_size=64, tap_sample_rate=2048, engine_sample_rate=2048,
                 fft_workers=None):
        self.delta_f = delta_f
        self.snr_threshold = snr_threshold
        self.f_high = high_frequency_cutoff
    
        self.tap_sr = int(tap_sample_rate)
        self.engine_sr = int(engine_sample_rate)

        self.threshold_sq = float(snr_threshold**2)
        
        self.fir_fft_len = fir_fft_length
        self.batch_size = batch_size
        
        # 2. Intermediate Buffers (Batch x Block)
        total_batch_size = batch_size * fir_fft_length
        self.temp_freq_mult = zeros(total_batch_size, dtype=complex64).data.reshape(self.batch_size, fir_fft_length)
        self.corr_output_buffer = zeros(total_batch_size, dtype=complex64).data.reshape(self.batch_size, fir_fft_length)

        # 3. Filter Preparation Buffers
        self.filters_padded = zeros(total_batch_size, dtype=complex64)
        self.filters_f_buffer = zeros(total_batch_size, dtype=complex64)
        
        # FFT backend: scipy.fft on CPU (mkl_fft-compatible interface).
        self.fft_lib = _ScipyFFTBackend(workers=fft_workers)

        # Reusable on-device scratch buffer (K*n_blocks*N_FFT complex64) for the
        # batched GPU path, grown on demand and reused across tiles / coarse
        # groups so the per-tile multiply+ifft does not cudaMalloc/cudaFree
        # (each free synchronises the device and starves it). See
        # :meth:`fine_snr_timeseries_batch`.
        self._gpu_scratch = None

    def prepare_filters(self, fir_taps, tap_counts):
        """
        Prepare frequency-domain filters for a batch of taps.
        """
        n_filters, n_taps = fir_taps.shape
        if n_taps >= self.fir_fft_len:
             raise ValueError("FIR Taps (%d) exceed FFT block length (%d)" % 
                              (n_taps, self.fir_fft_len))
        
        # Calculate max tap count for validity logic
        n_taps_max = int(np.max(tap_counts))
        
        filters_f = self._fft_all_filters(fir_taps, tap_counts)
        return filters_f, n_taps_max

    def process_segment(self, stilde, psd, ref_template, filters_f, n_taps, indices, 
                        valid_slice=None):
        """
        Process a single data segment.
        """
        if valid_slice is None:
            valid_slice = getattr(stilde, 'analyze', None)

        # 1-2. Reference normalization + reference (coarse-template) SNR
        h_norm = self.compute_reference_snr(stilde, psd, ref_template)

        # 3. Execute Blocked Kernel
        local_idxs, t_idxs, snr_vals, tstarts = self._execute_blocked_kernel(
            self.ref_snr, filters_f, n_taps, valid_slice
        )

        # 4. Map indices
        if len(local_idxs) > 0:
            global_ids = indices[local_idxs]
            return global_ids, t_idxs, snr_vals, tstarts, h_norm
        else:
            return [], [], [], tstarts, h_norm

    def compute_reference_snr(self, stilde, psd, ref_template):
        """Compute and cache the reference (coarse-template) complex SNR time
        series for one detector's data segment.

        This is the per-detector, per-coarse-template work that the FIR ratio
        filters are applied on top of. It is factored out of
        :meth:`process_segment` so the coherent driver can compute the
        reference SNR once per detector and then reconstruct many fine-template
        SNR time series cheaply (see :meth:`fine_snr_timeseries`).

        Stores the reference SNR as ``self.ref_snr`` (a numpy complex array at
        the engine sample rate) and resets the per-segment block-FFT cache.

        Parameters
        ----------
        stilde : FrequencySeries
            Overwhitened (or raw, with ``psd`` supplied) frequency-domain data.
        psd : FrequencySeries
            The PSD for this detector/segment.
        ref_template : FrequencySeries
            The coarse reference template.

        Returns
        -------
        h_norm : float
            The reference template's sigmasq (``<h_ref|h_ref>``) over the
            filter band ``[f_lower, f_high]``.
        """
        if self.f_high is None:
            h_norm = ref_template.sigmasq(psd)
        else:
            # The filter integral below stops at f_high, so the normalization
            # must integrate the same band: the cached template sigmasq is
            # full-band and would overestimate h_norm, deflating every SNR by
            # sqrt(sigmasq_full / sigmasq_band).
            h_norm = float(sigmasq(
                ref_template, psd=psd,
                low_frequency_cutoff=ref_template.f_lower,
                high_frequency_cutoff=self.f_high))
        snr, _, norm = matched_filter_core(
            ref_template, stilde, psd=psd,
            low_frequency_cutoff=ref_template.f_lower,
            high_frequency_cutoff=self.f_high,
            h_norm=h_norm
        )
        decimate = int(np.round(self.tap_sr / self.engine_sr))
        scale = (norm * stilde.delta_t) / decimate
        if _on_cuda():
            # Keep the reference SNR ON THE DEVICE: ``matched_filter_core``
            # returns ``snr`` as a device TimeSeries under CUDA, and the batched
            # fine-template path builds its block FFTs from this device array via
            # a gather kernel (see :meth:`_build_block_f_all`). This removes the
            # device->host copy here AND the former host->device re-upload of the
            # block-input buffer -- a full host round trip per (segment,
            # detector, coarse reference). The CPU path is unchanged (host
            # numpy, the parity reference).
            self.ref_snr = snr * float(scale)
        else:
            self.ref_snr = snr.numpy() * scale
        # Block FFTs of ref_snr depend only on the reference SNR (not on the
        # fine template), so they are cached and reused across fine templates.
        self._block_fft_cache = {}
        return h_norm

    def _get_block_fft(self, t_start, ref_snr, cache):
        """Return (and cache) the FFT of the ``fir_fft_len`` data block of the
        reference SNR starting at sample ``t_start`` (zero-padded at the end).
        """
        block_f = cache.get(t_start)
        if block_f is None:
            N_FFT = self.fir_fft_len
            n_samples = len(ref_snr)
            t_end = min(t_start + N_FFT, n_samples)
            block_in = zeros(N_FFT, dtype=complex64).data
            block_in[0:t_end - t_start] = ref_snr[t_start:t_end]
            block_f = self.fft_lib.fft(block_in)
            cache[t_start] = block_f
        return block_f

    def fine_snr_timeseries(self, filter_f, n_taps_eff, valid_slice,
                            ref_snr=None, block_cache=None,
                            return_device=False):
        """Reconstruct the *full* complex SNR time series for a single fine
        template by ratio-filtering the cached reference SNR.

        This is the per-IFO SNR(t) output that the coherent search consumes
        (as opposed to :meth:`process_segment`, which returns only thresholded
        peaks). The result is the analytic (one-sided) matched-filter SNR for
        the fine template, in the *reference normalization* -- the caller must
        divide by ``bank.snr_rescale(global_id)`` to obtain the fine-template
        normalized complex SNR (matching brute-force ``snr_ts * norm``).

        Uses overlap-save blocking identical to the peak-finding kernel so the
        reconstructed values coincide with :meth:`process_segment` outputs at
        the same time indices.

        Parameters
        ----------
        filter_f : numpy.ndarray
            The conjugated frequency-domain FIR filter for this fine template,
            shape ``(fir_fft_len,)`` (one row of ``prepare_filters`` output).
        n_taps_eff : int
            The number of taps for this filter (controls the overlap-save
            valid-block length). Use the per-template ``actual_tap_count``.
        valid_slice : slice or None
            The region of interest (typically ``stilde.analyze``). Samples
            outside it are left zero.
        ref_snr : numpy.ndarray, optional
            The reference SNR to filter. Defaults to ``self.ref_snr`` (set by
            :meth:`compute_reference_snr`). Pass explicitly when reconstructing
            many fine templates against a cached per-(segment, detector)
            reference SNR.
        block_cache : dict, optional
            Cache of block FFTs of ``ref_snr`` keyed by block start sample.
            Defaults to ``self._block_fft_cache``. Pass a dedicated dict
            alongside ``ref_snr`` so the (fine-template-independent) block FFTs
            are reused across all fine templates sharing this reference SNR.
        return_device : bool, optional
            If True *and* the active scheme is CUDA, return the assembled SNR
            series as an on-device ``pycbc.types.Array`` instead of host numpy
            -- so the per-detector threshold and the coherent combine can run
            on-device and only the (sparse) triggers cross to the host. Ignored
            on the CPU path (which always returns host numpy, the parity
            reference). Defaults to False (host numpy), preserving the original
            behaviour for all existing callers (e.g. the CPU<->CUDA parity test).

        Returns
        -------
        numpy.ndarray or pycbc.types.Array
            Complex64 array of length ``len(ref_snr)`` holding the
            reference-normalized fine-template SNR over ``valid_slice``. Host
            numpy unless ``return_device`` selected the on-device CUDA return.
        """
        if ref_snr is None:
            ref_snr = self.ref_snr
        if block_cache is None:
            block_cache = self._block_fft_cache

        if _on_cuda():
            return self._fine_snr_timeseries_cuda(
                filter_f, n_taps_eff, valid_slice, ref_snr, block_cache,
                return_device=return_device)

        data = ref_snr
        n_samples = len(data)
        N_FFT = self.fir_fft_len
        n_taps_max = int(n_taps_eff)

        N_VALID = N_FFT - n_taps_max + 1
        bad_start = n_taps_max // 2  # filter support is -(n-1)//2 .. n//2 around the t=0 tap (n-1)//2, for odd AND even n

        if valid_slice is not None:
            v_start = valid_slice.start
            v_stop = valid_slice.stop
        else:
            v_start = 0
            v_stop = n_samples

        out = np.zeros(n_samples, dtype=complex64)
        # Half-spectrum (analytic) multiply writes only bins [0, N/2]; the upper
        # half of mult stays zero across the loop, which is exactly the analytic
        # signal construction. Allocate zeroed buffers once.
        filter_2d = np.ascontiguousarray(filter_f).reshape(1, N_FFT)
        mult = np.zeros((1, N_FFT), dtype=complex64)
        corr = np.zeros((1, N_FFT), dtype=complex64)

        first_block_idx = (v_start - bad_start) // N_VALID
        # Clamp to 0 so a small analyze.start (v_start < bad_start) can't make
        # the first block start negative (which would index ref_snr with a
        # negative t_start). Mirrors the CUDA path.
        loop_start = max(0, first_block_idx * N_VALID)
        for t_start in range(loop_start, n_samples, N_VALID):
            block_valid_t0 = t_start + bad_start
            if block_valid_t0 >= v_stop:
                break
            if block_valid_t0 + N_VALID <= v_start:
                continue
            roi_start = max(v_start, block_valid_t0)
            roi_stop = min(v_stop, block_valid_t0 + N_VALID)
            roi_len = roi_stop - roi_start
            if roi_len <= 0:
                continue
            buf_slice_start = roi_start - t_start

            block_f = self._get_block_fft(t_start, data, block_cache)
            fast_multiply_analytic_cython(block_f, filter_2d, mult)
            self.fft_lib.ifft(mult, axis=-1, out=corr)
            out[roi_start:roi_stop] = corr[0, buf_slice_start:buf_slice_start + roi_len]

        return out

    def _fine_snr_timeseries_cuda(self, filter_f, n_taps_eff, valid_slice,
                                  ref_snr, block_cache, return_device=False):
        """CUDA implementation of :meth:`fine_snr_timeseries`, batched on-device.

        Throughput-oriented GPU path. All overlap-save blocks are transformed
        in ONE batched forward cufft, the analytic multiply runs as a single
        broadcast ElementwiseKernel over all blocks, the inverse is ONE batched
        cufft, and the valid samples are assembled into the output SNR series
        ON THE DEVICE (a strided gather ElementwiseKernel). When
        ``return_device`` is True the device array is returned directly -- no
        per-fine-template device->host copy of the (overlap-padded) correlation
        buffer -- so the caller can threshold and coherently combine on-device
        and transfer only the sparse triggers. Otherwise the assembled series is
        copied back to host (``.numpy()``), preserving the original behaviour.
        The batched forward FFT of the reference SNR (which is fine-template
        independent) is cached on the device in ``block_cache`` and reused
        across every fine template sharing this reference SNR -- so per fine
        template the GPU cost is just one multiply + one batched ifft.

        Reproduces the CPU result to single precision: the same overlap-save
        grid and the same 1/N inverse normalization (folded into the multiply
        kernel, since cufft's ifft is unnormalized). Requires the caller to use
        a consistent ``n_taps_eff`` for all fine templates sharing a given
        ``block_cache`` (the coherent driver passes the per-coarse-group max,
        which is a valid -- and result-invariant -- overlap-save tap count).
        """
        from skcuda import fft as cu_fft
        N_FFT = self.fir_fft_len
        n_samples = len(ref_snr)
        if valid_slice is not None:
            v_start = valid_slice.start
            v_stop = valid_slice.stop
        else:
            v_start = 0
            v_stop = n_samples

        # Build (and cache) the batched forward FFT of all data blocks (shared
        # with the tiled path; reference SNR may be a device array or host numpy).
        block_f_all, t_starts, n_blocks, N_VALID, bad_start, plan = \
            self._build_block_f_all(ref_snr, n_taps_eff, valid_slice, block_cache)
        if n_blocks == 0:
            # Degenerate/empty valid region: return zeros (matches the CPU path,
            # whose block loop simply doesn't execute).
            return (zeros(n_samples, dtype=complex64) if return_device
                    else np.zeros(n_samples, dtype=complex64))

        # Per fine template: broadcast analytic multiply, batched ifft.
        filt_gpu = Array(np.ascontiguousarray(filter_f, dtype=np.complex64))
        mult = zeros(n_blocks * N_FFT, dtype=complex64)
        corr = zeros(n_blocks * N_FFT, dtype=complex64)
        krnl = _get_analytic_mult_kernel()
        krnl(block_f_all.data, filt_gpu.data, mult.data,
             np.int32(N_FFT), np.int32(N_FFT // 2 + 1), np.float32(1.0 / N_FFT))
        cu_fft.ifft(mult.data, corr.data, plan)

        # Assemble ON THE DEVICE: each block contributes valid samples
        # [bad_start, +N_VALID), which tile time contiguously from
        # t_starts[0] + bad_start. A strided gather ElementwiseKernel writes
        # those into a length-n_samples device array (zeros elsewhere) -- the
        # on-device equivalent of the former host
        # ``ascontiguousarray(corr[:, bad_start:bad_start+N_VALID]).ravel()``
        # + ``out[lo:hi] = valid_flat[...]``, with no per-template
        # device->host copy of the (overlap-padded) correlation buffer.
        assemble_start = int(t_starts[0]) + bad_start
        lo = max(v_start, assemble_start)
        hi = min(v_stop, assemble_start + n_blocks * N_VALID, n_samples)
        out_gpu = zeros(n_samples, dtype=complex64)
        if hi > lo:
            asm = _get_assemble_kernel()
            asm(corr.data, out_gpu.data,
                np.int32(N_FFT), np.int32(N_VALID), np.int32(bad_start),
                np.int32(assemble_start), np.int32(lo),
                range=slice(0, hi - lo, 1))
        if return_device:
            return out_gpu
        return out_gpu.numpy()

    def _build_block_f_all(self, ref_snr, n_taps_eff, valid_slice, block_cache):
        """Build (and cache, per ``block_cache``) the batched forward cufft of
        the overlap-save data blocks of the reference SNR, ON THE DEVICE.

        The block grid is fine-template independent (it depends only on the
        reference SNR and ``n_taps_eff``), so it is computed once per
        (segment, detector, coarse reference) and reused by every fine template
        in the coarse group -- both the single-template and the tiled GPU paths.

        ``ref_snr`` may be a device ``pycbc.types.Array``/``TimeSeries`` (the
        driver path -- the reference SNR is kept on the device) or host numpy
        (the unit tests). A host array is uploaded once; a device array is used
        in place. The block-input buffer is gathered device->device by
        :func:`_get_block_gather_kernel`, so there is no host round trip.

        Returns the tuple cached under ``block_cache['_gpu_batched']``:
        ``(block_f_all, t_starts, n_blocks, N_VALID, bad_start, plan)`` -- with
        ``block_f_all is None`` and ``n_blocks == 0`` for a degenerate/empty
        valid region.
        """
        from skcuda import fft as cu_fft
        cached = block_cache.get('_gpu_batched')
        if cached is not None:
            return cached

        N_FFT = self.fir_fft_len
        n_samples = len(ref_snr)
        n_taps_max = int(n_taps_eff)
        N_VALID = N_FFT - n_taps_max + 1
        # filter support is -(n-1)//2 .. n//2 around the t=0 tap (n-1)//2, for
        # odd AND even n.
        bad_start = n_taps_max // 2
        if valid_slice is not None:
            v_start = valid_slice.start
            v_stop = valid_slice.stop
        else:
            v_start = 0
            v_stop = n_samples

        first_block_idx = (v_start - bad_start) // N_VALID
        loop_start = max(0, first_block_idx * N_VALID)
        t_starts = []
        ts = loop_start
        while ts < n_samples:
            bvt0 = ts + bad_start
            if bvt0 >= v_stop:
                break
            if bvt0 + N_VALID > v_start:
                t_starts.append(ts)
            ts += N_VALID
        n_blocks = len(t_starts)
        if n_blocks == 0:
            cached = (None, np.asarray([], dtype=np.int64), 0,
                      N_VALID, bad_start, None)
            block_cache['_gpu_batched'] = cached
            return cached

        # Reference SNR on the device (upload host numpy once; use a device
        # array in place). The blocks tile time contiguously with stride
        # N_VALID from t_starts[0], so the gather base is t_starts[0].
        if isinstance(ref_snr, np.ndarray):
            ref_dev = Array(np.ascontiguousarray(ref_snr, dtype=np.complex64))
        else:
            ref_dev = ref_snr
        block0 = int(t_starts[0])
        block_in = zeros(n_blocks * N_FFT, dtype=complex64)
        gk = _get_block_gather_kernel()
        gk(ref_dev.data, block_in.data, np.int32(N_FFT), np.int32(N_VALID),
           np.int32(block0), np.int32(n_samples),
           range=slice(0, n_blocks * N_FFT, 1))
        block_f_all = zeros(n_blocks * N_FFT, dtype=complex64)
        plan = _get_cufft_plan(n_blocks, N_FFT)
        cu_fft.fft(block_in.data, block_f_all.data, plan)
        cached = (block_f_all, np.asarray(t_starts, dtype=np.int64),
                  n_blocks, N_VALID, bad_start, plan)
        block_cache['_gpu_batched'] = cached
        return cached

    def _get_gpu_scratch(self, n):
        """Return a reusable device complex64 scratch buffer of at least ``n``
        samples (grown on demand, never shrunk), used as the
        multiply->in-place-ifft workspace for the tiled GPU path. Reusing it
        across tiles/coarse groups avoids a per-tile cudaMalloc/cudaFree (each
        free synchronises the device)."""
        buf = self._gpu_scratch
        if buf is None or len(buf) < n:
            self._gpu_scratch = zeros(n, dtype=complex64)
            buf = self._gpu_scratch
        return buf

    def fine_snr_timeseries_batch(self, filt_dev, k_tile, n_taps_eff,
                                  valid_slice, ref_snr, block_cache,
                                  rescales_dev, out_buffer):
        """Reconstruct ``k_tile`` fine-template SNR series in ONE batched pass
        (CUDA only) -- the throughput core of the FIR engine.

        This is the tile generalisation of :meth:`_fine_snr_timeseries_cuda`:
        instead of one (multiply -> ifft -> assemble) per fine template, it
        stacks ``k_tile`` conjugated filters, broadcast-multiplies them against
        the single cached forward FFT of the reference SNR in one kernel, runs
        ONE batched inverse cufft over all ``k_tile * n_blocks`` transforms, and
        assembles + fine-normalises all ``k_tile`` SNR series in one kernel --
        turning thousands of tiny launch-bound kernel pairs into a handful of
        large batched ops, so the GPU stays fed. The per-element math is
        identical to the single-template kernel, so the result is bit-for-bit
        the per-template result (same overlap-save grid, same 1/N inverse
        normalisation, same gather).

        Parameters
        ----------
        filt_dev : pycbc.types.Array
            Device buffer of ``k_tile`` row-major length-``fir_fft_len``
            conjugated FD filters (size ``k_tile * fir_fft_len``); a contiguous
            slice of the coarse group's filters uploaded once.
        k_tile : int
            Number of filters in this tile.
        n_taps_eff : int
            Overlap-save tap count shared by the whole coarse group (the group
            max), so every fine template shares the cached block grid.
        valid_slice : slice
            The analyze region; output rows are this slice's length ``L``.
        ref_snr : pycbc.types.Array
            Device reference SNR (built by :meth:`compute_reference_snr`).
        block_cache : dict
            Per-(segment, detector) cache holding the shared block FFT.
        rescales_dev : pycbc.types.Array
            Device float32 buffer of ``k_tile`` ``snr_rescale`` values; divided
            out in the assemble kernel so each returned row is the fine-template
            normalised complex SNR (matching the CPU ``series/snr_rescale``).
        out_buffer : pycbc.types.Array
            Device buffer of length ``>= k_tile * L`` that receives the
            ``k_tile`` output rows; supplied (and reused across tiles/coarse
            groups) by the caller so it is not reallocated per tile. The
            assemble kernel writes every sample of every row (zero outside the
            valid region), so no pre-zeroing is needed.

        Returns
        -------
        list of pycbc.types.Array
            ``k_tile`` length-``L`` device views into ``out_buffer`` (row ``ki``
            is ``out_buffer[ki*L:(ki+1)*L]``), each the analyze-sliced,
            fine-normalised complex SNR for that template. The caller thresholds
            each on-device and pulls a row to the host only if a coincidence
            needs it.
        """
        from skcuda import fft as cu_fft
        N_FFT = self.fir_fft_len
        n_samples = len(ref_snr)
        if valid_slice is not None:
            v_start = valid_slice.start
            v_stop = valid_slice.stop
        else:
            v_start = 0
            v_stop = n_samples
        L = v_stop - v_start

        block_f_all, t_starts, n_blocks, N_VALID, bad_start, plan = \
            self._build_block_f_all(ref_snr, n_taps_eff, valid_slice,
                                    block_cache)
        if n_blocks == 0:
            # Empty valid region: write zero rows (full-row assemble is skipped).
            self._zero_rows(out_buffer, k_tile, L)
            return [out_buffer[ki * L:(ki + 1) * L] for ki in range(k_tile)]

        bspan = n_blocks * N_FFT
        ntot = k_tile * bspan
        scratch = self._get_gpu_scratch(ntot)

        # 1) tiled half-spectrum multiply: scratch[k,b,:] = bf[b,:]*filt[k,:]/N
        mk = _get_batched_mult_kernel()
        mk(block_f_all.data, filt_dev.data, scratch.data,
           np.int32(N_FFT), np.int32(N_FFT // 2 + 1), np.float32(1.0 / N_FFT),
           np.int32(bspan), range=slice(0, ntot, 1))
        # 2) ONE batched inverse cufft over all k_tile*n_blocks transforms,
        #    in place in the scratch buffer.
        iplan = _get_cufft_plan(k_tile * n_blocks, N_FFT)
        cu_fft.ifft(scratch.data, scratch.data, iplan)
        # 3) tiled assemble + fine-normalise into the caller's output buffer.
        #    Writes every sample of every output row (zero outside [lo, hi)).
        assemble_start = int(t_starts[0]) + bad_start
        lo = max(v_start, assemble_start)
        hi = min(v_stop, assemble_start + n_blocks * N_VALID, n_samples)
        ak = _get_batched_assemble_kernel()
        ak(scratch.data, out_buffer.data, rescales_dev.data,
           np.int32(N_FFT), np.int32(N_VALID), np.int32(bad_start),
           np.int32(assemble_start), np.int32(v_start), np.int32(lo),
           np.int32(hi), np.int32(bspan), np.int32(L),
           range=slice(0, k_tile * L, 1))
        return [out_buffer[ki * L:(ki + 1) * L] for ki in range(k_tile)]

    def _zero_rows(self, out_buffer, k_tile, L):
        """Zero the first ``k_tile`` length-``L`` rows of ``out_buffer`` (used
        only for the degenerate empty-valid-region case, where the assemble
        kernel that would otherwise overwrite every row is skipped)."""
        out_buffer[0:k_tile * L].fill(0)

    def _fft_all_filters(self, taps, counts):
        """Helper to FFT all filters using mkl_fft."""
        n_filters, n_taps_alloc = taps.shape
        filters_f = np.zeros((n_filters, self.fir_fft_len), dtype=np.complex64)
        
        # 1. Read metadata from the bank to determine the source generation rate
        bank_sample_rate = self.tap_sr
        engine_sample_rate = self.engine_sr
        # Alternatively, determine the downsampling factor directly:
        exact_ratio = (bank_sample_rate / engine_sample_rate)
        decimation_factor = int(np.round(exact_ratio))

        if abs(exact_ratio - decimation_factor) > 1e-5 or decimation_factor < 1:
            raise ValueError(
                f"Multi-rate Error: The bank sample rate ({self.tap_sr} Hz) must be "
                f"an exact integer multiple of the engine sample "
                f"rate ({self.engine_sr} Hz).\n"
                f"Calculated ratio was {exact_ratio:.4f}. Please use standard power-of-2 "
                f"downsampling scales (e.g., 2048/512)."
            )

        # 2. Establish the high-resolution FFT padding length to preserve delta_f
        # 512/4096 = 0.125   2048/(4*4096) = 0.125 preserving delta_f
        # 4096/4096 = 1      2048/(1/2*4096) = 1 for 4096 engine 2048 bank
        high_res_fft_len = self.fir_fft_len * decimation_factor
        
        # Temp allocations for high-resolution processing
        high_res_padded = np.zeros((self.batch_size, high_res_fft_len), dtype=np.complex64)

        for start in range(0, n_filters, self.batch_size):
            end = min(start + self.batch_size, n_filters)
            batch_len = end - start
            
            # Zero out processing buffer for next call
            high_res_padded[:batch_len, :] = 0.0
            
            # Copy raw 2048 Hz taps into the start of the buffer
            tmp_taps = taps[start:end]
            high_res_padded[:batch_len, :n_taps_alloc] = tmp_taps
            
            # 3. Handle Variable Time-Domain Roll Logic at the native 2048 Hz rate
            current_counts = counts[start:end]
            # pycbc_fir_bank designs tap j at time j - (K-1)//2 for BOTH odd
            # and even K (even K spans -K/2+1 .. K/2), so (K-1)//2 is the tap
            # that belongs at t=0. Rolling by K//2 (one tap too many for even
            # K) shifts the filter by a full tap-rate sample: invisible at
            # decimation 1 (integer engine-sample translation) but a
            # *fractional* engine-sample shift under decimation, which
            # samples the SNR envelope off-peak (~1.5% loss at decimation 2).
            roll_offsets = -((current_counts - 1) // 2)
            
            cols_high = np.arange(high_res_fft_len)
            rows = np.arange(batch_len)[:, None]
            shifted_cols_high = (cols_high[None, :] - roll_offsets[:, None]) % high_res_fft_len
            
            current_data = high_res_padded[:batch_len].copy()
            high_res_padded[:batch_len] = current_data[rows, shifted_cols_high]

            # 4. Transform to Frequency Domain at native resolution
            fft_high_res = self.fft_lib.fft(high_res_padded[:batch_len], axis=-1)
            
            # 5. Brick-Wall Frequency Slicing (Anti-Aliasing & Decimation Match)
            # Because the data engine goes up to 256Hz (the 512Hz Nyquist limit), only need the first 4096 bins of that spectrum
            fft_sliced = fft_high_res[:batch_len, :self.fir_fft_len]
            
            # 6. Conjugate & Store back into the 512 Hz buffer block
            filters_f[start:end] = np.conj(fft_sliced)
            
        return filters_f

    def _execute_blocked_kernel(self, data, filters_f, n_taps, valid_slice):
        """
        Inner loop: Time-Blocking + Filter-Batching using mkl_fft.
        """
        tap_groups = 3
        nsizes = np.quantile(n_taps, np.linspace(0, 1, tap_groups+1)[1:]).astype(int)
        n_samples = len(data)
        n_filters = len(filters_f)
        
        N_FFT = self.fir_fft_len
        
        all_f_idxs = []
        all_t_idxs = []
        all_snrs = []
        all_tstarts = []

        freq_mult_view = self.temp_freq_mult
        corr_out_view = self.corr_output_buffer

        if valid_slice:
            v_start = valid_slice.start
            v_stop = valid_slice.stop
        else:
            v_start = 0
            v_stop = n_samples
 
        block_f_cache = {}
        # --- OUTER LOOP: Time Blocks ---
        for f_start in range(0, n_filters, self.batch_size):  
        
            f_end = min(f_start + self.batch_size, n_filters)
            actual_batch_size = f_end - f_start
 
            current_mult_view = freq_mult_view[:actual_batch_size]
            current_corr_view = corr_out_view[:actual_batch_size]
            
            # Valid output samples per block (Overlap-Save)
            n_taps_max = n_taps[f_start:f_end].max()
            i = np.searchsorted(nsizes, n_taps_max)
            n_taps_max = nsizes[i]
            
            N_VALID = N_FFT - n_taps_max + 1
            STEP = N_VALID
            bad_start = n_taps_max // 2  # filter support is -(n-1)//2 .. n//2 around the t=0 tap (n-1)//2, for odd AND even n

            # Determine Loop Bounds
            first_block_idx = (v_start - bad_start) // STEP
            loop_start = max(0, first_block_idx * STEP)

            for t_start in range(loop_start, n_samples, STEP):

                block_valid_t0 = t_start + bad_start
                
                if block_valid_t0 >= v_stop:
                    break
                
                if block_valid_t0 + N_VALID <= v_start:
                    continue

                roi_start = max(v_start, block_valid_t0)
                roi_stop = min(v_stop, block_valid_t0 + N_VALID)
                
                roi_len = roi_stop - roi_start
                
                if roi_len <= 0: 
                    continue

                buf_slice_start = roi_start - t_start

                t_end = min(t_start + N_FFT, n_samples)
                if t_start not in block_f_cache:
                    block_in_view = np.zeros(self.fir_fft_len, dtype=complex64)
                    block_in_view[0:t_end-t_start] = data[t_start:t_end]
                    
                    block_f_view = self.fft_lib.fft(block_in_view)
                    block_f_cache[t_start] = block_f_view
                
                block_f_view = block_f_cache[t_start]
                filter_batch_f = filters_f[f_start:f_end]
                
                fast_multiply_analytic_cython(
                    block_f_view, filter_batch_f, current_mult_view
                )

                self.fft_lib.ifft(
                    current_mult_view, 
                    axis=-1, 
                    out=current_corr_view
                )
                f_list, t_list, s_list = find_peaks_in_block_cython(
                    current_corr_view, 
                    roi_start,          
                    roi_len,            
                    self.threshold_sq, 
                    f_start,
                    input_offset=buf_slice_start
                )

                if f_list:
                    all_f_idxs.extend(f_list)
                    all_t_idxs.extend(t_list)
                    all_snrs.extend(s_list)
                    all_tstarts.extend([t_start] * len(s_list)) 
                    
        return (np.array(all_f_idxs, dtype=np.int32), 
                np.array(all_t_idxs, dtype=np.int64), 
                np.array(all_snrs, dtype=np.complex64),
                np.array(all_tstarts, dtype=np.int32))
