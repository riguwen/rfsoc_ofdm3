"""Signal processing for OFDM sensing: power spectra and range-Doppler maps.

This module only depends on numpy, so it can be used (and tested) away from
the board, for example on captures saved with ``numpy.save``.

The range-Doppler processing follows the classic OFDM radar approach
(Sturm & Wiesbeck, "Waveform Design and Signal Processing Aspects for Fusion
of Wireless Communications and Radar Sensing", Proc. IEEE, 2011):

1. The transmitted (reference) and received baseband streams are aligned in
   time by cross-correlation, so the strongest propagation path (the loopback
   cable, or the direct antenna-to-antenna path) defines zero range.
2. The OFDM symbol boundaries are found from the cyclic prefix of the
   transmitted stream, the cyclic prefix is removed and a 64-point FFT is
   applied to every transmitted and received OFDM symbol.
3. The transmitted modulation symbols are removed from the received ones by
   element-wise division, ``D[m, k] = Y[m, k] / X[m, k]``, leaving only the
   channel response for each OFDM symbol ``m`` and sub-carrier ``k``.
4. An IFFT across sub-carriers gives the range (delay) profile and an FFT
   across OFDM symbols gives the Doppler spectrum.
"""

__author__ = "RFSoC OFDM contributors"
__organisation__ = "The Univeristy of Strathclyde"
__support__ = "https://github.com/strath-sdr/rfsoc_ofdm"

import numpy as np

SPEED_OF_LIGHT = 299792458.0

WINDOWS = ('hann', 'hamming', 'blackman', 'blackmanharris', 'rect')

ESTIMATORS = ('reciprocal', 'matched')


def get_window(name, n):
    """Return a length ``n`` window without zero-valued end points.

    The end points are dropped so that no sub-carrier or OFDM symbol is
    discarded when the window is applied.
    """
    n = int(n)
    name = 'rect' if name is None else str(name).lower()
    if n <= 0:
        return np.ones(0)
    if name in ('rect', 'rectangular', 'boxcar', 'none'):
        return np.ones(n)
    m = n + 2
    k = 2 * np.pi * np.arange(m) / (m - 1)
    if name in ('hann', 'hanning'):
        w = 0.5 - 0.5 * np.cos(k)
    elif name == 'hamming':
        w = 0.54 - 0.46 * np.cos(k)
    elif name == 'blackman':
        w = 0.42 - 0.5 * np.cos(k) + 0.08 * np.cos(2 * k)
    elif name in ('blackmanharris', 'blackman-harris'):
        w = (0.35875 - 0.48829 * np.cos(k) + 0.14128 * np.cos(2 * k)
             - 0.01168 * np.cos(3 * k))
    else:
        raise ValueError('Unknown window "{}". Choose one of {}.'.format(
            name, ', '.join(WINDOWS)))
    return w[1:-1]


def power_spectrum(data, fs, nfft=1024, window='blackmanharris',
                   overlap=0.5, full_scale=1.0):
    """Welch-averaged power spectrum of a complex baseband signal.

    Parameters
    ----------
    data : array_like
        Complex baseband samples.
    fs : float
        Sample rate in Hz.
    nfft : int
        FFT (segment) length. Reduced to ``len(data)`` if necessary.
    window : str
        Window applied to every segment (see ``WINDOWS``).
    overlap : float
        Fractional overlap between segments, 0 <= overlap < 1.
    full_scale : float
        Amplitude of a complex exponential that should read 0 dBFS.

    Returns
    -------
    freq : ndarray
        Frequency of each bin in Hz (centred on 0 Hz).
    power_db : ndarray
        Power per bin in dBFS. The resolution bandwidth is ``fs / nfft``.
    """
    x = np.asarray(data)
    if x.ndim != 1 or len(x) < 2:
        raise ValueError('data must be a 1-D array with at least 2 samples')
    nfft = int(min(int(nfft), len(x)))
    w = get_window(window, nfft)
    step = max(1, int(round(nfft * (1.0 - float(overlap)))))
    nseg = 1 + (len(x) - nfft) // step
    idx = np.arange(nfft)[None, :] + step * np.arange(nseg)[:, None]
    spec = np.fft.fft(x[idx] * w, axis=1)
    power = np.mean(np.abs(spec) ** 2, axis=0)
    power /= (np.sum(w) * full_scale) ** 2
    power = np.fft.fftshift(power)
    freq = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))
    return freq, 10 * np.log10(np.maximum(power, 1e-30))


def _next_pow2(n):
    return 1 << int(np.ceil(np.log2(max(1, int(n)))))


class RangeDopplerResult(object):
    """Container returned by :meth:`OFDMRadar.process`.

    Attributes
    ----------
    rd_map_db : ndarray, shape (n_doppler, n_range)
        Range-Doppler map in dB, normalised so that its peak is 0 dB.
    range_m : ndarray
        Range of every column in metres (two-way, i.e. ``c * delay / 2``),
        relative to the strongest path.
    delay_s : ndarray
        Delay of every column in seconds, relative to the strongest path.
    doppler_hz : ndarray
        Doppler frequency of every row in Hz.
    velocity_mps : ndarray
        Radial velocity of every row in m/s (``doppler * c / (2 * fc)``).
    range_profile_db : ndarray
        Row of ``rd_map_db`` that contains the peak.
    channel : ndarray, shape (n_symbols, n_fft)
        Estimated channel ``Y / X`` (or ``Y * conj(X)``) per OFDM symbol and
        sub-carrier, in FFT order. Inactive sub-carriers are zero.
    peak_range_m, peak_doppler_hz, peak_velocity_mps : float
        Location of the strongest cell.
    peak_to_noise_db : float
        Peak power relative to the median power of the map.
    n_symbols : int
        Number of OFDM symbols that were processed.
    lag_samples : int
        Delay of the received stream relative to the transmitted stream
        (includes the unknown capture offset between both inspectors).
    alignment_db : float
        Cross-correlation peak relative to the median correlation level.
        Values below about 6 dB mean that no echo of the transmitted signal
        was found in the received samples.
    cp_quality : float
        Cyclic-prefix correlation of the transmitted stream at the detected
        symbol timing (1.0 means a perfect CP-OFDM signal).
    active_subcarriers : ndarray
        Indices (FFT order) of the sub-carriers carrying energy.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return ('RangeDopplerResult(peak_range_m={:.2f}, peak_doppler_hz={:.1f}, '
                'peak_velocity_mps={:.2f}, peak_to_noise_db={:.1f}, n_symbols={}, '
                'lag_samples={}, alignment_db={:.1f})').format(
                    self.peak_range_m, self.peak_doppler_hz,
                    self.peak_velocity_mps, self.peak_to_noise_db,
                    self.n_symbols, self.lag_samples, self.alignment_db)


class OFDMRadar(object):
    """Range-Doppler processing for the RFSoC OFDM transceiver.

    The default parameters match the transceiver: 20 MSPS complex baseband,
    64-point FFT and a 16 sample cyclic prefix (IEEE 802.11a/g numerology).

    Parameters
    ----------
    fs : float
        Baseband sample rate in Hz.
    n_fft : int
        OFDM FFT size.
    n_cp : int
        Cyclic prefix length in samples.
    carrier_freq : float
        RF carrier frequency in Hz, only used to convert Doppler to velocity.
    guard : int
        The received FFT window is started ``guard`` samples early so that
        paths that arrive up to ``guard`` samples before the strongest path
        (e.g. filter pre-cursors) are still inside the cyclic prefix. Paths
        later than ``n_cp - guard`` samples are only partially captured.
    range_oversampling : int
        Zero-padding factor applied to the range IFFT.
    doppler_oversampling : int
        Zero-padding factor applied to the Doppler FFT.
    range_window, doppler_window : str
        Windows applied across sub-carriers / OFDM symbols (see ``WINDOWS``).
    estimator : str
        ``'reciprocal'`` divides by the transmitted symbols (flat, low range
        sidelobes). ``'matched'`` multiplies by their conjugate (matched
        filter, no noise enhancement for high order QAM).
    remove_static : bool
        Subtract the mean over OFDM symbols before the Doppler FFT. This
        removes everything that does not move, including the direct path.
    discard : int
        Number of samples dropped from the start of each capture (the data
        inspectors deliver their 1024 sample pre-trigger FIFO first).
    max_symbols : int or None
        Upper limit on the number of OFDM symbols that are processed.
    max_lag : int
        Largest offset (in samples) between the transmit and receive
        captures that the alignment searches for.
    align_block, align_blocks : int
        Length and maximum number of the blocks used for alignment.
    """

    def __init__(self, fs=20e6, n_fft=64, n_cp=16, carrier_freq=600e6,
                 guard=4, range_oversampling=4, doppler_oversampling=1,
                 range_window='hann', doppler_window='hann',
                 estimator='reciprocal', remove_static=False, discard=1024,
                 max_symbols=None, max_lag=2048, align_block=128,
                 align_blocks=64):
        self.fs = float(fs)
        self.n_fft = int(n_fft)
        self.n_cp = int(n_cp)
        self.carrier_freq = float(carrier_freq)
        self.guard = int(guard)
        self.range_oversampling = int(range_oversampling)
        self.doppler_oversampling = int(doppler_oversampling)
        self.range_window = range_window
        self.doppler_window = doppler_window
        self.estimator = estimator
        self.remove_static = bool(remove_static)
        self.discard = int(discard)
        self.max_symbols = max_symbols
        self.max_lag = int(max_lag)
        self.align_block = int(align_block)
        self.align_blocks = int(align_blocks)
        if not 0 <= self.guard <= self.n_cp:
            raise ValueError('guard must be between 0 and n_cp')

    # ------------------------------------------------------------------
    # Waveform properties
    # ------------------------------------------------------------------
    @property
    def symbol_length(self):
        """OFDM symbol length including the cyclic prefix, in samples."""
        return self.n_fft + self.n_cp

    @property
    def symbol_period(self):
        """OFDM symbol period in seconds."""
        return self.symbol_length / self.fs

    @property
    def subcarrier_spacing(self):
        """Sub-carrier spacing in Hz."""
        return self.fs / self.n_fft

    @property
    def range_bin(self):
        """Range covered by one baseband sample, c / (2 fs), in metres."""
        return SPEED_OF_LIGHT / (2.0 * self.fs)

    @property
    def max_range(self):
        """Range at which echoes leave the cyclic prefix, in metres."""
        return (self.n_cp - self.guard) * self.range_bin

    @property
    def max_doppler(self):
        """Largest unambiguous Doppler shift, +/- 1 / (2 T_sym), in Hz."""
        return 0.5 / self.symbol_period

    def doppler_resolution(self, n_symbols):
        """Doppler resolution for ``n_symbols`` OFDM symbols, in Hz."""
        return 1.0 / (int(n_symbols) * self.symbol_period)

    def doppler_to_velocity(self, doppler_hz):
        """Convert a Doppler shift (Hz) to a radial velocity (m/s)."""
        return np.asarray(doppler_hz) * SPEED_OF_LIGHT / (2.0 * self.carrier_freq)

    def samples_required(self, n_symbols):
        """Capture length needed to process ``n_symbols`` OFDM symbols.

        Includes a margin of ``max_lag`` samples for the unknown offset
        between the transmit and receive captures plus the loopback latency.
        """
        return int(self.discard + (int(n_symbols) + 1) * self.symbol_length
                   + self.max_lag)

    # ------------------------------------------------------------------
    # Processing steps
    # ------------------------------------------------------------------
    def align(self, tx, rx):
        """Find the lag of the strongest copy of ``tx`` inside ``rx``.

        Short blocks of ``tx`` are cross-correlated with ``rx`` over lags of
        +/- ``max_lag`` samples and the correlation power is summed over the
        blocks. Integrating non-coherently keeps the peak when the received
        signal has a Doppler (frequency) shift.

        Returns ``(lag, quality_db)`` where ``rx[n + lag] ~ tx[n]`` and
        ``quality_db`` is the correlation peak relative to the median over
        all lags.
        """
        block = int(self.align_block)
        n_min = min(len(tx), len(rx))
        max_lag = int(min(self.max_lag, (n_min - 4 * block) // 2))
        if max_lag < 1:
            raise ValueError('Captures are too short to align')
        n_blocks = (n_min - 2 * max_lag) // block
        starts = max_lag + block * np.unique(
            np.linspace(0, n_blocks - 1, min(n_blocks, int(self.align_blocks))).astype(int))
        # An FFT of seg_len samples is long enough for the linear correlation
        # at lags 0 ... 2 * max_lag (no circular wrap-around).
        seg_len = block + 2 * max_lag
        n = seg_len
        tx_blocks = tx[starts[:, None] + np.arange(block)[None, :]]
        rx_blocks = rx[starts[:, None] - max_lag + np.arange(seg_len)[None, :]]
        xc = np.fft.ifft(np.fft.fft(rx_blocks, n, axis=1)
                         * np.conj(np.fft.fft(tx_blocks, n, axis=1)), axis=1)
        # Element j of the linear correlation is the lag j - max_lag.
        power = np.sum(np.abs(xc[:, :2 * max_lag + 1]) ** 2, axis=0)
        peak = int(np.argmax(power))
        quality = 10 * np.log10(power[peak] / max(np.median(power), 1e-30))
        return peak - max_lag, float(quality)

    def symbol_timing(self, tx):
        """Locate the OFDM symbol boundaries of a transmitted stream.

        Returns ``(offset, quality, per_symbol_quality)``: ``offset`` is the
        index of the first cyclic prefix sample, ``quality`` the normalised
        CP correlation at that offset and ``per_symbol_quality`` the same
        measure for every complete symbol starting at ``offset``.
        """
        n_fft, n_cp, length = self.n_fft, self.n_cp, self.symbol_length
        prod = tx[:-n_fft] * np.conj(tx[n_fft:])
        energy = 0.5 * (np.abs(tx[:-n_fft]) ** 2 + np.abs(tx[n_fft:]) ** 2)
        kernel = np.ones(n_cp)
        corr = np.convolve(prod, kernel, 'valid')
        power = np.convolve(energy, kernel, 'valid')
        n_sym = len(corr) // length
        if n_sym < 2:
            raise ValueError('Not enough samples to find the OFDM symbol timing')
        folded_corr = corr[:n_sym * length].reshape(n_sym, length).sum(axis=0)
        folded_power = power[:n_sym * length].reshape(n_sym, length).sum(axis=0)
        metric = np.abs(folded_corr) / np.maximum(folded_power, 1e-30)
        offset = int(np.argmax(metric))
        starts = offset + length * np.arange((len(corr) - offset - 1) // length + 1)
        per_symbol = np.abs(corr[starts]) / np.maximum(power[starts], 1e-30)
        return offset, float(metric[offset]), per_symbol

    def channel_estimate(self, tx, rx):
        """Estimate the channel per OFDM symbol and sub-carrier.

        Returns ``(D, info)`` where ``D`` has shape ``(n_symbols, n_fft)``.
        """
        tx = np.asarray(tx, dtype=complex)[self.discard:]
        rx = np.asarray(rx, dtype=complex)[self.discard:]
        min_len = 4 * self.symbol_length
        if len(tx) < min_len or len(rx) < min_len:
            raise ValueError('Captures are too short for range-Doppler processing')
        if not np.any(tx):
            raise ValueError('The transmitted samples are all zero. '
                             'Is the transmitter enabled?')

        lag, alignment_db = self.align(tx, rx)
        start = max(0, -lag)
        stop = min(len(tx), len(rx) - lag)
        tx_a = tx[start:stop]
        rx_a = rx[start + lag:stop + lag]

        offset, cp_quality, sym_quality = self.symbol_timing(tx_a)
        n_sym = (len(tx_a) - offset) // self.symbol_length
        if self.max_symbols is not None:
            n_sym = min(n_sym, int(self.max_symbols))
        if n_sym < 2:
            raise ValueError('Not enough overlap between the transmit and '
                             'receive captures')

        first = offset + self.n_cp + self.symbol_length * np.arange(n_sym)
        idx = first[:, None] + np.arange(self.n_fft)[None, :]
        X = np.fft.fft(tx_a[idx], axis=1)
        Y = np.fft.fft(rx_a[idx - self.guard], axis=1)

        # Active sub-carriers are those that carry energy on average.
        px = np.abs(X) ** 2
        mean_px = px.mean(axis=0)
        active = mean_px > 10 ** (-25.0 / 10) * mean_px.max()
        ref_level = mean_px[active].mean()
        valid_sym = np.ones(n_sym, dtype=bool)
        k = min(n_sym, len(sym_quality))
        valid_sym[:k] = sym_quality[:k] > 0.5
        use = (px > 1e-3 * ref_level) & active[None, :] & valid_sym[:, None]

        estimator = str(self.estimator).lower()
        if estimator == 'reciprocal':
            D = np.where(use, Y * np.conj(X) / np.where(use, px, 1.0), 0)
        elif estimator == 'matched':
            D = np.where(use, Y * np.conj(X) / ref_level, 0)
        else:
            raise ValueError('Unknown estimator "{}". Choose one of {}.'.format(
                self.estimator, ', '.join(ESTIMATORS)))

        info = {'lag_samples': lag,
                'alignment_db': alignment_db,
                'cp_quality': cp_quality,
                'active_subcarriers': np.nonzero(active)[0],
                'valid_symbols': valid_sym}
        return D, info

    def process(self, tx, rx):
        """Compute the range-Doppler map from transmit and receive captures.

        ``tx`` and ``rx`` are complex baseband captures of the transmitter
        and receiver observation points, taken at (roughly) the same time.
        """
        D, info = self.channel_estimate(tx, rx)
        n_sym, n_fft = D.shape
        active = np.zeros(n_fft, dtype=bool)
        active[info['active_subcarriers']] = True
        valid = info['valid_symbols']

        # Work in natural frequency order (-n_fft/2 ... n_fft/2 - 1).
        shifted = np.fft.fftshift(active)
        occupied = np.nonzero(shifted)[0]
        lo, hi = occupied[0], occupied[-1]
        Ds = np.fft.fftshift(D, axes=1)

        # Fill null sub-carriers inside the occupied band (e.g. DC) by linear
        # interpolation. A hole in the spectrum would otherwise appear as a
        # pedestal at every range (about -29 dB with a Hann window).
        for hole in np.nonzero(~shifted[lo:hi + 1])[0] + lo:
            left = occupied[occupied < hole][-1]
            right = occupied[occupied > hole][0]
            frac = (hole - left) / float(right - left)
            Ds[:, hole] = (1 - frac) * Ds[:, left] + frac * Ds[:, right]

        # Window across the occupied band.
        w_range = np.zeros(n_fft)
        w_range[lo:hi + 1] = get_window(self.range_window, hi - lo + 1)
        D = np.fft.ifftshift(Ds, axes=1)
        w_range = np.fft.ifftshift(w_range)

        # Zero-padded range IFFT. Bin j corresponds to a delay of
        # j / P - guard samples relative to the strongest path.
        p = max(1, self.range_oversampling)
        n_range = n_fft * p
        half = n_fft // 2
        windowed = D * w_range
        padded = np.zeros((n_sym, n_range), dtype=complex)
        padded[:, :half] = windowed[:, :half]
        padded[:, n_range - (n_fft - half):] = windowed[:, half:]
        profiles = np.fft.ifft(padded, axis=1)

        if self.remove_static and np.count_nonzero(valid) > 1:
            profiles[valid] -= profiles[valid].mean(axis=0)

        w_doppler = get_window(self.doppler_window, n_sym) * valid
        n_doppler = _next_pow2(n_sym) * max(1, self.doppler_oversampling)
        rd = np.fft.fftshift(
            np.fft.fft(profiles * w_doppler[:, None], n=n_doppler, axis=0), axes=0)
        power = np.abs(rd) ** 2
        peak_power = power.max()
        if peak_power <= 0:
            peak_power = 1.0
        rd_db = 10 * np.log10(np.maximum(power / peak_power, 1e-30))

        delay_samples = np.arange(n_range) / float(p) - self.guard
        delay_s = delay_samples / self.fs
        range_m = delay_s * SPEED_OF_LIGHT / 2.0
        doppler_hz = np.fft.fftshift(np.fft.fftfreq(n_doppler, d=self.symbol_period))
        velocity = self.doppler_to_velocity(doppler_hz)

        i_d, i_r = np.unravel_index(int(np.argmax(power)), power.shape)
        noise = np.median(power)
        pnr = 10 * np.log10(peak_power / noise) if noise > 0 else np.inf

        return RangeDopplerResult(
            rd_map_db=rd_db,
            range_m=range_m,
            delay_s=delay_s,
            doppler_hz=doppler_hz,
            velocity_mps=velocity,
            range_profile_db=rd_db[i_d],
            channel=D,
            peak_range_m=float(range_m[i_r]),
            peak_doppler_hz=float(doppler_hz[i_d]),
            peak_velocity_mps=float(velocity[i_d]),
            peak_to_noise_db=float(pnr),
            n_symbols=int(n_sym),
            doppler_resolution_hz=self.doppler_resolution(n_sym),
            lag_samples=int(info['lag_samples']),
            alignment_db=float(info['alignment_db']),
            cp_quality=float(info['cp_quality']),
            active_subcarriers=info['active_subcarriers'])


# ----------------------------------------------------------------------
# Synthetic captures for testing and teaching
# ----------------------------------------------------------------------
_PILOTS = (-21, -7, 7, 21)


def _qam_symbols(order, n, rng):
    if order == 2:
        return 2.0 * rng.randint(0, 2, n) - 1.0
    m = int(round(np.sqrt(order)))
    if m * m != order:
        raise ValueError('modulation_order must be 2 or a square QAM order')
    levels = 2.0 * np.arange(m) - (m - 1)
    sym = levels[rng.randint(0, m, n)] + 1j * levels[rng.randint(0, m, n)]
    return sym / np.sqrt(np.mean(np.abs(levels) ** 2) * 2)


def simulate_ofdm(n_symbols, modulation_order=4, n_fft=64, n_cp=16, seed=None):
    """Generate an IEEE 802.11a/g-like CP-OFDM stream.

    48 data sub-carriers, 4 pilots and 12 null sub-carriers (including DC).
    Returns ``(samples, symbols)`` where ``symbols`` has shape
    ``(n_symbols, n_fft)`` in FFT order.
    """
    rng = np.random.RandomState(seed)
    used = [k for k in range(-26, 27) if k != 0]
    data = [k for k in used if k not in _PILOTS]
    X = np.zeros((n_symbols, n_fft), dtype=complex)
    X[:, np.array(data) % n_fft] = _qam_symbols(
        modulation_order, n_symbols * len(data), rng).reshape(n_symbols, -1)
    X[:, np.array(_PILOTS) % n_fft] = 1.0
    x = np.fft.ifft(X, axis=1) * np.sqrt(n_fft) / np.sqrt(len(used) / float(n_fft))
    x = np.concatenate([x[:, -n_cp:], x], axis=1).ravel()
    return x, X


def simulate_capture(n_samples=24576, targets=((0.0, 0.0, 0.0),), fs=20e6,
                     n_fft=64, n_cp=16, carrier_freq=600e6, snr_db=30.0,
                     modulation_order=16, latency=57, capture_offset=-29,
                     amplitude=0.25, seed=None):
    """Synthesise transmit/receive inspector captures with radar targets.

    Parameters
    ----------
    targets : iterable of (range_m, velocity_mps, power_db)
        Propagation paths. Range is two-way (``c * delay / 2``); power is
        relative to 0 dB. Include a 0 m, 0 dB path to model the direct
        (loopback) path that the processing uses as its range reference.
    latency : float
        Fixed transmit-to-receive latency in samples (filters, converters).
    capture_offset : int
        Start of the receive capture relative to the transmit capture, in
        samples. Models the unknown delay between arming both inspectors.
    snr_db : float
        Signal-to-noise ratio of the 0 dB path at the receiver.

    Returns
    -------
    tx, rx : ndarray
        Complex captures, scaled like the hardware inspectors.
    """
    rng = np.random.RandomState(seed)
    length = n_fft + n_cp
    pad = 4096
    total = n_samples + 2 * pad
    x, _ = simulate_ofdm(total // length + 1, modulation_order, n_fft, n_cp,
                         None if seed is None else seed + 1)
    x = x[:total] * amplitude
    t = np.arange(total) / fs
    freq = np.fft.fftfreq(total, d=1.0 / fs)
    spectrum = np.fft.fft(x)
    y = np.zeros(total, dtype=complex)
    for rng_m, vel, p_db in targets:
        delay = latency / fs + 2.0 * rng_m / SPEED_OF_LIGHT
        doppler = 2.0 * vel * carrier_freq / SPEED_OF_LIGHT
        path = np.fft.ifft(spectrum * np.exp(-2j * np.pi * freq * delay))
        y += 10 ** (p_db / 20.0) * path * np.exp(2j * np.pi * doppler * t)
    sigma = amplitude * 10 ** (-snr_db / 20.0) / np.sqrt(2)
    y += sigma * (rng.randn(total) + 1j * rng.randn(total))
    start = pad
    tx = x[start:start + n_samples]
    rx = y[start + capture_offset:start + capture_offset + n_samples]
    return tx, rx
