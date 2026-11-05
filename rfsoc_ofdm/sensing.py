"""Spectrum analyser and range-Doppler (OFDM radar) applications.

Both applications work with the existing OFDM overlay. They only use the data
inspectors that are already in the design, so no new bitstream is needed:

* The transmitter inspector taps the OFDM transmitter output at 20 MSPS
  ("Symbols") and the interpolator at 240 MSPS ("Interpolated").
* The receiver inspector taps the decimator output at 20 MSPS ("Symbols",
  the samples that enter the OFDM receiver) and the first decimation stage
  at 240 MSPS ("Decimated").

The range-Doppler map captures the 20 MSPS transmitter and receiver streams at
the same time and uses the transmitted samples as the radar reference (see
rfsoc_ofdm.radar for the signal processing).
"""

__author__ = "RFSoC OFDM contributors"
__organisation__ = "The Univeristy of Strathclyde"
__support__ = "https://github.com/strath-sdr/rfsoc_ofdm"

import contextlib
import threading
import time

import numpy as np
import ipywidgets as ipw
from pynq import allocate

from .radar import OFDMRadar, power_spectrum, WINDOWS
from .sdr_plots import PowerSpectrumPlot, RangeDopplerPlot, RangeProfilePlot
from .dma_timer import DmaTimer

SAMPLE_SCALE = 2 ** -14   # Scaling used by Inspector.get_frame
FULL_SCALE = 2.0          # Amplitude of a full-scale int16 sample after scaling
MAX_SAMPLES = 2 ** 22     # 16 MB per inspector buffer
ALIGNMENT_THRESHOLD_DB = 6.0   # Below this, no echo of the transmit signal
BASEBAND_RATE = 20e6      # 'Symbols' observation points
WIDEBAND_RATE = 240e6     # 'Interpolated' / 'Decimated' observation points

OBSERVATION_POINTS = [('Baseband, 20 MSPS (Symbols)', 0),
                      ('Wideband, 240 MSPS (Interpolated / Decimated)', 1)]

_DESCRIPTION_STYLE = {'description_width': '150px'}
_CONTROL_LAYOUT = {'width': '340px'}


def _inspector_lock(inspector):
    lock = getattr(inspector, '_lock', None)
    if lock is None:
        lock = threading.RLock()
        inspector._lock = lock
    return lock


def _route(inspector, channel):
    """Select an observation point without touching the inspector's widgets."""
    if channel is None or not hasattr(inspector, 'axis_switch'):
        return
    inspector.axis_switch.write(0x40, int(channel))
    inspector.axis_switch.write(0x00, 2)


def _reset_dma_channel(channel):
    """Best-effort recovery of a DMA channel whose transfer never finished."""
    try:
        channel._mmio.write(channel._offset, 0x4)
        for _ in range(10000):
            if not channel._mmio.read(channel._offset) & 0x4:
                break
        channel.start()
        channel._first_transfer = True
    except Exception:
        pass


def _to_complex(buffer):
    raw = np.array(buffer, dtype=np.float64)
    return (raw[0::2] + 1j * raw[1::2]) * SAMPLE_SCALE


def _mixer_freq(block):
    """Fine mixer frequency of a data converter block in MHz, or None."""
    try:
        return float(block.MixerSettings['Freq'])
    except Exception:
        return None


def _error_html(err):
    return '<span style="color:tomato"><b>{}:</b> {}</span>'.format(
        type(err).__name__, err)


class SimultaneousCapture():
    """Capture blocks of contiguous samples from several data inspectors.

    Every inspector is armed before any of them is triggered, so the captures
    overlap in time. Their start times still differ by a few microseconds
    (the triggers are separate AXI4-Lite writes); the range-Doppler
    processing removes this offset by cross-correlation.
    """

    def __init__(self, inspectors, timeout=2.0):
        self.inspectors = list(inspectors)
        self.timeout = timeout
        self._buffers = [None] * len(self.inspectors)

    def _get_buffer(self, index, n_samples):
        buffer = self._buffers[index]
        if buffer is None or buffer.shape[0] != 2 * n_samples:
            if buffer is not None:
                buffer.freebuffer()
            buffer = allocate(shape=(2 * n_samples,), dtype=np.int16)
            self._buffers[index] = buffer
        return buffer

    def _wait(self):
        channels = [insp.axi_dma.recvchannel for insp in self.inspectors]
        deadline = time.time() + self.timeout
        pending = list(channels)
        while pending:
            pending = [ch for ch in pending if not ch.idle]
            if not pending:
                break
            if time.time() > deadline:
                for ch in pending:
                    _reset_dma_channel(ch)
                raise TimeoutError(
                    'No samples arrived from the data inspector within {} s. '
                    'Check that the transmitter is enabled, e.g. '
                    'ofdm_hw.initialise_transmitter().'.format(self.timeout))
            time.sleep(0.0002)
        # wait() returns immediately now; it checks for errors and
        # invalidates the cache of the receive buffers.
        for ch in channels:
            ch.wait()

    def capture(self, n_samples, channels=None):
        """Capture ``n_samples`` complex samples from every inspector.

        channels: observation point (axis switch input) per inspector, or
        None to keep the current one. The previous observation point and
        packet size are restored afterwards.

        Returns a list of complex numpy arrays, one per inspector.
        """
        n = int(n_samples)
        if not 64 <= n <= MAX_SAMPLES:
            raise ValueError('n_samples must be between 64 and {}'.format(MAX_SAMPLES))
        if channels is None:
            channels = [None] * len(self.inspectors)

        with contextlib.ExitStack() as stack:
            for insp in self.inspectors:
                stack.enter_context(_inspector_lock(insp))
            buffers = [self._get_buffer(i, n) for i in range(len(self.inspectors))]
            saved = []
            for insp, channel in zip(self.inspectors, channels):
                dim = insp.data_inspector_module
                saved.append((dim.packetsize, getattr(insp, '_current_channel', None)))
                dim.enable = 0
                dim.reset = 1
                _route(insp, channel)
                dim.packetsize = n
            try:
                for insp in self.inspectors:
                    insp.data_inspector_module.reset = 0
                for insp, buffer in zip(self.inspectors, buffers):
                    insp.axi_dma.recvchannel.transfer(buffer)
                for insp in self.inspectors:
                    insp.data_inspector_module.enable = 1
                self._wait()
            finally:
                for insp, (size, channel) in zip(self.inspectors, saved):
                    dim = insp.data_inspector_module
                    dim.enable = 0
                    dim.reset = 1
                    _route(insp, channel)
                    dim.packetsize = size
            return [_to_complex(buffer) for buffer in buffers]

    def close(self):
        """Free the DMA buffers."""
        for i, buffer in enumerate(self._buffers):
            if buffer is not None:
                buffer.freebuffer()
                self._buffers[i] = None


class SpectrumApplication():
    """Averaged power spectra of the transmitted and received signals.

    Example
    -------
    >>> app = SpectrumApplication(ofdm_hw)
    >>> app.get_widget()          # interactive plot, press play
    >>> result = app.measure()    # or one measurement as numpy arrays
    """

    def __init__(self, overlay, n_samples=16384, nfft=1024,
                 window='blackmanharris', averaging=0.5, observation=0,
                 plotting_rate=0.5):
        self._ol = overlay
        tx = overlay.inspectors['transmitter']
        rx = overlay.inspectors['receiver']
        self._capture = SimultaneousCapture([tx, rx])
        self._sample_freq = list(getattr(rx, '_sample_freq',
                                         [BASEBAND_RATE, WIDEBAND_RATE]))
        self.n_samples = int(n_samples)
        self.nfft = int(nfft)
        self.window = window
        self.averaging = float(averaging)
        self.observation = int(observation)
        self.result = None
        self._avg = None
        self._avg_key = None
        self._display_lock = threading.Lock()

        self._plot = PowerSpectrumPlot(names=('Transmit', 'Receive'))
        self._build_widgets()
        self._timer = DmaTimer(self._show, self._acquire, plotting_rate)

    # ------------------------------------------------------------------
    def _build_widgets(self):
        def dropdown(options, value, description):
            return ipw.Dropdown(options=options, value=value,
                                description=description,
                                style=_DESCRIPTION_STYLE, layout=_CONTROL_LAYOUT)

        self._w_obs = dropdown(OBSERVATION_POINTS, self.observation,
                               'Observation Point:')
        self._w_nfft = dropdown([256, 512, 1024, 2048, 4096], self.nfft, 'FFT Size:')
        self._w_window = dropdown(list(WINDOWS), self.window, 'Window:')
        self._w_avg = ipw.FloatSlider(value=self.averaging, min=0.0, max=0.95,
                                      step=0.05, description='Trace Averaging:',
                                      style=_DESCRIPTION_STYLE, layout=_CONTROL_LAYOUT)
        self._w_tx = ipw.Checkbox(value=True, description='Show Transmit')
        self._w_rx = ipw.Checkbox(value=True, description='Show Receive')
        self._w_single = ipw.Button(description='Single Capture')
        self._w_status = ipw.HTML(value='Press play to start the spectrum analyser.')

        def on_settings(change):
            self.observation = self._w_obs.value
            self.nfft = self._w_nfft.value
            self.window = self._w_window.value
            self.averaging = self._w_avg.value
            self._avg = None

        def on_visibility(change):
            if self.result is not None:
                self._show(self.result)

        for w in (self._w_obs, self._w_nfft, self._w_window, self._w_avg):
            w.observe(on_settings, names='value')
        for w in (self._w_tx, self._w_rx):
            w.observe(on_visibility, names='value')
        self._w_single.on_click(lambda _: self.update())

    # ------------------------------------------------------------------
    def capture(self):
        """Capture the transmit and receive observation points.

        Returns ``(tx, rx, fs)``.
        """
        channel = self.observation
        tx, rx = self._capture.capture(self.n_samples, [channel, channel])
        return tx, rx, self._sample_freq[channel]

    def measure(self):
        """Capture and compute (averaged) spectra.

        Returns a dict with ``freq`` (Hz), ``transmit`` and ``receive``
        (dBFS per bin), ``transmit_power`` and ``receive_power`` (dBFS) and
        the sample rate ``fs``.
        """
        tx, rx, fs = self.capture()
        freq, p_tx = power_spectrum(tx, fs, self.nfft, self.window,
                                    full_scale=FULL_SCALE)
        _, p_rx = power_spectrum(rx, fs, self.nfft, self.window,
                                 full_scale=FULL_SCALE)
        linear = 10 ** (np.vstack([p_tx, p_rx]) / 10)
        key = (self.observation, self.nfft, self.window)
        if self._avg is None or self._avg_key != key or self._avg.shape != linear.shape:
            self._avg = linear
        else:
            self._avg = self.averaging * self._avg + (1 - self.averaging) * linear
        self._avg_key = key
        avg_db = 10 * np.log10(np.maximum(self._avg, 1e-30))

        def total_power(x):
            return 10 * np.log10(max(np.mean(np.abs(x) ** 2), 1e-30) / FULL_SCALE ** 2)

        self.result = {'freq': freq,
                       'transmit': avg_db[0],
                       'receive': avg_db[1],
                       'transmit_power': total_power(tx),
                       'receive_power': total_power(rx),
                       'fs': fs,
                       'nfft': self.nfft}
        return self.result

    def _acquire(self):
        try:
            return self.measure()
        except Exception as err:
            return err

    def _show(self, result):
        with self._display_lock:
            if isinstance(result, Exception):
                self._w_status.value = _error_html(result)
                return
            traces = {'Transmit': (result['freq'], result['transmit']) if self._w_tx.value else None,
                      'Receive': (result['freq'], result['receive']) if self._w_rx.value else None}
            dac = _mixer_freq(self._ol.dac_block)
            adc = _mixer_freq(self._ol.adc_block)
            centre = ''
            if dac is not None and adc is not None:
                centre = ' | Centre: TX {:.3f} MHz, RX {:.3f} MHz'.format(abs(dac), abs(adc))
            title = 'Power Spectrum (fs = {:g} MSPS, RBW = {:.1f} kHz{})'.format(
                result['fs'] * 1e-6, result['fs'] / result['nfft'] * 1e-3, centre)
            self._plot.update_data(traces, title)
            self._w_status.value = ('Total power: transmit {:.1f} dBFS, receive {:.1f} dBFS'
                                    .format(result['transmit_power'], result['receive_power']))

    # ------------------------------------------------------------------
    def update(self):
        """Capture once and refresh the plot."""
        self._show(self._acquire())

    def start(self):
        self._timer.start()

    def stop(self):
        self._timer.stop()

    def get_widget(self):
        controls = ipw.HBox([ipw.VBox([self._w_obs, self._w_nfft, self._w_window]),
                             ipw.VBox([self._w_avg, self._w_tx, self._w_rx])])
        return ipw.VBox([self._plot.get_widget(), controls,
                         ipw.HBox([self._timer.get_widget(), self._w_single]),
                         self._w_status])


class RangeDopplerApplication():
    """Live range-Doppler map using the OFDM signal as a radar waveform.

    The transmitter output is used as the reference, so any modulation
    scheme works. Zero range is the strongest received path (the loopback
    cable or the direct path between the antennas) because the capture
    offset between the transmitter and receiver inspectors is unknown.

    Example
    -------
    >>> app = RangeDopplerApplication(ofdm_hw)
    >>> app.get_widget()          # interactive plot, press play
    >>> result = app.measure()    # or one measurement (RangeDopplerResult)
    """

    def __init__(self, overlay, n_symbols=256, plotting_rate=0.5,
                 dynamic_range=60, max_range=150.0, **radar_kwargs):
        self._ol = overlay
        tx = overlay.inspectors['transmitter']
        rx = overlay.inspectors['receiver']
        self._capture = SimultaneousCapture([tx, rx])
        self._channels = [0, 0]   # 20 MSPS 'Symbols' observation points
        self._dac_freq = _mixer_freq(overlay.dac_block)
        carrier = abs(self._dac_freq) * 1e6 if self._dac_freq else 600e6
        radar_kwargs.setdefault('carrier_freq', carrier)
        self.radar = OFDMRadar(fs=BASEBAND_RATE, **radar_kwargs)
        self.n_symbols = int(n_symbols)
        self.dynamic_range = dynamic_range
        self.max_range = max_range
        self.max_doppler = self.radar.max_doppler
        self.units = 'doppler'
        self.nco_offset = 0.0
        self.result = None
        self._display_lock = threading.Lock()

        self._rd_plot = RangeDopplerPlot(dynamic_range=dynamic_range)
        self._profile_plot = RangeProfilePlot(dynamic_range=dynamic_range)
        self._build_widgets()
        self._timer = DmaTimer(self._show, self._acquire, plotting_rate)

    # ------------------------------------------------------------------
    def _build_widgets(self):
        radar = self.radar

        def dropdown(options, value, description):
            return ipw.Dropdown(options=options, value=value,
                                description=description,
                                style=_DESCRIPTION_STYLE, layout=_CONTROL_LAYOUT)

        def slider(cls, value, vmin, vmax, step, description, **kwargs):
            return cls(value=value, min=vmin, max=vmax, step=step,
                       description=description, continuous_update=False,
                       style=_DESCRIPTION_STYLE, layout=_CONTROL_LAYOUT, **kwargs)

        self._w_symbols = dropdown([64, 128, 256, 512, 1024], self.n_symbols,
                                   'OFDM Symbols:')
        self._w_pad = dropdown([1, 2, 4, 8], radar.range_oversampling,
                               'Range Zero-Padding:')
        self._w_window = dropdown(list(WINDOWS), radar.range_window, 'Window:')
        self._w_estimator = dropdown([('Reciprocal (Y / X)', 'reciprocal'),
                                      ('Matched (Y X*)', 'matched')],
                                     radar.estimator, 'Estimator:')
        self._w_static = ipw.Checkbox(value=radar.remove_static,
                                      description='Remove static clutter (zero Doppler)',
                                      style={'description_width': 'initial'})
        self._w_units = dropdown([('Doppler (kHz)', 'doppler'),
                                  ('Velocity (m/s)', 'velocity')],
                                 self.units, 'Doppler Axis:')
        self._w_dr = slider(ipw.IntSlider, self.dynamic_range, 20, 100, 5,
                            'Dynamic Range (dB):')
        max_range = (radar.n_fft - radar.guard) * radar.range_bin
        self._w_range = slider(ipw.FloatSlider, min(self.max_range, max_range),
                               radar.range_bin, max_range, radar.range_bin,
                               'Max Range (m):', readout_format='.1f')
        max_doppler_khz = radar.max_doppler * 1e-3
        self._w_doppler = slider(ipw.FloatSlider, max_doppler_khz, 0.5,
                                 max_doppler_khz, 0.5, 'Max |Doppler| (kHz):',
                                 readout_format='.1f')
        self._w_nco = slider(ipw.FloatSlider, 0.0, -50.0, 50.0, 0.5,
                             'DAC NCO Offset (kHz):', readout_format='.1f')
        self._w_single = ipw.Button(description='Single Capture')
        self._w_status = ipw.HTML(value='Press play to start the range-Doppler map.')

        def on_processing(change):
            self.n_symbols = self._w_symbols.value
            radar.range_oversampling = self._w_pad.value
            radar.range_window = self._w_window.value
            radar.doppler_window = self._w_window.value
            radar.estimator = self._w_estimator.value
            radar.remove_static = self._w_static.value

        def on_display(change):
            self.units = self._w_units.value
            self.dynamic_range = self._w_dr.value
            self.max_range = self._w_range.value
            self.max_doppler = self._w_doppler.value * 1e3
            self._rd_plot.set_dynamic_range(self.dynamic_range)
            self._profile_plot.set_dynamic_range(self.dynamic_range)
            if self.result is not None:
                self._show(self.result)

        def on_nco(change):
            try:
                self.set_nco_offset(change['new'] * 1e3)
            except Exception as err:
                self._w_status.value = _error_html(err)

        for w in (self._w_symbols, self._w_pad, self._w_window,
                  self._w_estimator, self._w_static):
            w.observe(on_processing, names='value')
        for w in (self._w_units, self._w_dr, self._w_range, self._w_doppler):
            w.observe(on_display, names='value')
        self._w_nco.observe(on_nco, names='value')
        self._w_single.on_click(lambda _: self.update())

    # ------------------------------------------------------------------
    def set_nco_offset(self, offset_hz):
        """Offset the DAC fine mixer to emulate a Doppler shift.

        The whole received signal is shifted by ``offset_hz``, which appears
        as a Doppler shift in the range-Doppler map. Use 0 to restore the
        original carrier frequency. The hardware OFDM receiver will see the
        same frequency offset, so the constellation may rotate.
        """
        if self._dac_freq is None:
            raise RuntimeError('The DAC mixer frequency could not be read.')
        import xrfdc
        block = self._ol.dac_block
        settings = dict(block.MixerSettings)
        settings['Freq'] = self._dac_freq + offset_hz * 1e-6
        block.MixerSettings = settings
        block.UpdateEvent(xrfdc.EVENT_MIXER)
        self.nco_offset = float(offset_hz)
        if abs(self._w_nco.value - offset_hz * 1e-3) > 1e-6:
            self._w_nco.value = offset_hz * 1e-3

    def capture(self):
        """Capture the 20 MSPS transmit and receive streams together.

        Returns ``(tx, rx)`` complex numpy arrays.
        """
        n = self.radar.samples_required(self.n_symbols)
        tx, rx = self._capture.capture(n, self._channels)
        return tx, rx

    def process(self, tx, rx):
        """Compute a RangeDopplerResult from captured samples."""
        self.radar.max_symbols = self.n_symbols
        return self.radar.process(tx, rx)

    def measure(self):
        """Capture and process one range-Doppler map."""
        tx, rx = self.capture()
        self.result = self.process(tx, rx)
        return self.result

    def _acquire(self):
        try:
            return self.measure()
        except Exception as err:
            return err

    def _status_html(self, r):
        radar = self.radar
        lines = [
            '<b>Peak:</b> range {:.1f} m, Doppler {:.2f} kHz ({:.1f} m/s), '
            '{:.1f} dB above the median level'.format(
                r.peak_range_m, r.peak_doppler_hz * 1e-3, r.peak_velocity_mps,
                r.peak_to_noise_db),
            '<b>Resolution:</b> {} OFDM symbols ({:.2f} ms), Doppler {:.0f} Hz '
            '({:.1f} m/s), range bin {:.2f} m, CP-limited range {:.1f} m, '
            'carrier {:.3f} MHz'.format(
                r.n_symbols, r.n_symbols * radar.symbol_period * 1e3,
                r.doppler_resolution_hz,
                float(radar.doppler_to_velocity(r.doppler_resolution_hz)),
                radar.range_bin, radar.max_range, radar.carrier_freq * 1e-6),
            '<b>Alignment:</b> receive lag {} samples, correlation peak {:.1f} dB, '
            'CP quality {:.2f}'.format(r.lag_samples, r.alignment_db, r.cp_quality)]
        if r.alignment_db < ALIGNMENT_THRESHOLD_DB:
            lines.append('<span style="color:tomato"><b>Warning:</b> no echo of the '
                         'transmitted signal was found in the received samples. '
                         'Check the loopback cable or antennas.</span>')
        return '<br>'.join(lines)

    def _show(self, result):
        with self._display_lock:
            if isinstance(result, Exception):
                self._w_status.value = _error_html(result)
                return
            min_range = -self.radar.guard * self.radar.range_bin
            title = 'Range-Doppler Map ({} symbols)'.format(result.n_symbols)
            if self.nco_offset:
                title += ' | NCO offset {:+.1f} kHz'.format(self.nco_offset * 1e-3)
            self._rd_plot.update_data(result, max_range=self.max_range,
                                      min_range=min_range,
                                      max_doppler=self.max_doppler,
                                      units=self.units, title=title)
            self._profile_plot.update_data(result, max_range=self.max_range,
                                           min_range=min_range)
            self._w_status.value = self._status_html(result)

    # ------------------------------------------------------------------
    def update(self):
        """Capture once and refresh the plots."""
        self._show(self._acquire())

    def start(self):
        self._timer.start()

    def stop(self):
        self._timer.stop()

    def get_widget(self, compact=False):
        """Return the application widget.

        compact: only show the range-Doppler map (no range profile).
        """
        plots = self._rd_plot.get_widget() if compact else ipw.HBox(
            [self._rd_plot.get_widget(), self._profile_plot.get_widget()])
        controls = ipw.HBox([
            ipw.VBox([self._w_symbols, self._w_pad, self._w_window,
                      self._w_estimator, self._w_static]),
            ipw.VBox([self._w_units, self._w_dr, self._w_range,
                      self._w_doppler, self._w_nco])])
        return ipw.VBox([plots, controls,
                         ipw.HBox([self._timer.get_widget(), self._w_single]),
                         self._w_status])
