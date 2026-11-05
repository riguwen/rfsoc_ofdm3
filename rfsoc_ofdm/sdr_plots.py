__author__ = "David Northcote"
__organisation__ = "The Univeristy of Strathclyde"
__support__ = "https://github.com/strath-sdr/rfsoc_radio"

import numpy as np
import plotly.graph_objs as go
import ipywidgets as ipw
from contextlib import contextmanager

import plotly.io as pio

pio.templates.default = "seaborn"
                
class SpectrumAnalyser():
    def __init__(self,
                 data,
                 fs,
                 animation_period=50,
                 width=800,
                 height=600,
                 autosize=True):
        """Create a new Spectrum Analyser object for plotting frequency against power."""
        
        self._fs = fs
        self._window = np.array(np.blackman(len(data))[:])
        self._data = self._data = [{'y' : self._compute_spectrum(data),
                                    'name' : 'IQ Spectrum',
                                    'showlegend' : True}]
        self._animation_period = animation_period
        self._width = width
        self._height = height
        self._yaxisrange = [-100, 10]
        lim = int(self._fs/2)
        self._xaxisrange = [-lim, lim - (self._fs/len(data))]
        self._autosize = autosize
        self._complex = isinstance(data[0], complex)
        self._yrange_counter = 0
        
        self._layout = {
            'height' : self._height,
            'width' : self._width,
            'autosize' : True,
            'xaxis' : {
                'range' : self._xaxisrange,
                'title' : 'Frequency (Hz)',
            },
            'yaxis' : {
                'range' : self._yaxisrange,
                'title' : 'Normalised Amplitude (dB)',
            },
            'title' : 'Frequency Spectrum'
        }
        
        self._plot = go.FigureWidget(
            layout = self._layout,
            data = self._data
        )
        
        self._plot.layout.xaxis.range = self._xaxisrange
        self._plot.data[0].x = [x for x in np.arange(-lim, lim, (self._fs/len(data)))]   
        
    def _update_yrange(self):
        if self._yrange_counter > 4:
            min_data = min(self._data)
            max_data = max(self._data)
            self._yaxisrange = [min_data-10, max_data+10]
            self._plot.layout.yaxis.range = self._yaxisrange
            self._yrange_counter = 0
        else:
            self._yrange_counter += 1
        
    def _compute_spectrum(self, data):
        data = data * self._window
        data = np.fft.fftshift(np.fft.fft(data))
        with np.errstate(divide='ignore', invalid='ignore'):
            data = abs(data)/max(abs(data))
        return 20*np.where(data > 0, np.log10(data), 0)
        
    def set_frequency(self, fs):
        self._fs = fs
    
    def update_data(self, data):
        self._window = np.array(np.blackman(len(data))[:])
        lim = int(self._fs/2)
        self._xaxisrange = [-lim, lim - (self._fs/len(data))]
        self._plot.layout.xaxis.range = self._xaxisrange
        self._data = self._compute_spectrum(data)
        self._update_yrange()
        self._plot.data[0].y = self._data
        self._plot.data[0].x = [x for x in np.arange(-lim, lim, (self._fs/len(data)))]
            
    def get_widget(self):
        return self._plot    
        

class TimePlot():
    def __init__(self,
                 data,
                 animation_period=50,
                 width=800,
                 height=600,
                 autosize=True):
        """Create a new plot object for plotting data against time."""
        
        self._data = np.empty(len(data))
        self._animation_period = animation_period
        self._width = width
        self._height = height
        self._yaxisrange = [-0.3, 0.3]
        self._xaxisrange = [-0.5, len(data)-0.5]
        self._autosize = autosize
        self._complex = isinstance(data[0], complex)
        self._yrange_counter = 0
        
        if self._complex:
            self._data = [{'y' : np.real(data),
                           'name' : 'Real Signal'},
                          {'y' : np.imag(data),
                           'name' : 'Imag Signal'}]
        else:
            self._data = [{'y' : data, 'name' : 'Time Signal'}]
        
        self._layout = {
            'height' : self._height,
            'width' : self._width,
            'autosize' : True,
            'xaxis' : {
                'range' : self._xaxisrange,
                'title' : 'Samples',
            },
            'yaxis' : {
                'range' : self._yaxisrange,
                'title' : 'Amplitude',
            },
            'title' : 'Time Domain Signal'
        }
        
        self._plot = go.FigureWidget(
            layout = self._layout,
            data = self._data
        )
        
    def _update_yrange(self):
        if self._yrange_counter > 4:
            if self._complex:
                min_data = min(self._plot.data[0].y)
                max_data = max(self._plot.data[0].y)
            else:
                min_data = min(self._plot.data[0].y)
                max_data = max(self._plot.data[0].y)
            self._yaxisrange = [min_data-0.05, max_data+0.05]
            self._plot.layout.yaxis.range = self._yaxisrange
            self._yrange_counter = 0
        else:
            self._yrange_counter += 1
        
    def set_axisrange(self, axisrange):
        self._yaxisrange = axisrange
        self._plot.layout.yaxis.range = axisrange
        
    def update_data(self, data):
        """Update the frame of data currently on the canvas
        """
        self._xaxisrange = len(data)
        self._plot.layout.xaxis.range = [-0.5, self._xaxisrange-0.5]
        self._update_yrange()
        
        if self._complex:
            self._data = [{'y' : np.real(data)},
                          {'y' : np.imag(data)}]
        else:
            self._data = [{'y' : data}]

        self._plot.data[0].y = self._data[0].get('y')
        if self._complex:
            self._plot.data[1].y = self._data[1].get('y')
                
    def get_widget(self):
        return self._plot

    
class ConstellationPlot():
    def __init__(self,
                 data,
                 animation_period=50,
                 height=600,
                 width=600,
                 autosize=True):
        """Creates a new plot object for plotting IQ constellations."""
        
        self._data = data
        self._animation_period = 50
        self._width = width
        self._height = height
        self._axisrange = [-1.5, 1.5]
        self._autosize = autosize
        self._complex = isinstance(self._data[0], complex)
        
        if not self._complex:
            raise Exception('Input data is not of type complex.')
        
        self._layout = {
            'hovermode' : 'closest',
            'height' : self._height,
            'width' : self._width,
            'autosize' : self._autosize,
            'xaxis' : {
                'range' : self._axisrange,
                'title' : 'In-phase Amplitude'
            },
            'yaxis' : {
                'range' : self._axisrange,
                'title' : 'Quadrature Amplitude'
            },
            'title' : 'Constellation Plot'
        }
        
        self._plot = go.FigureWidget(
            layout = self._layout,
            data = [{
                'mode' : 'markers',
                'x' : np.real(self._data),
                'y' : np.imag(self._data)
            }])
        
    def set_axisrange(self, axisrange):
        self._axisrange = axisrange
        self._plot.layout.yaxis.range = axisrange
        self._plot.layout.xaxis.range = axisrange
        
    def update_data(self, data):
        """Update the frame of data currently on the canvas
        """
        
        if not isinstance(data[0], complex):
            raise Exception('Input data is not of type complex.')
        self._data = data
        
        self._plot.data[0].update({'x' : np.real(self._data),
                                   'y' : np.imag(self._data)})        
        
    def get_widget(self):
        return self._plot
    

class PowerSpectrumPlot():
    """Averaged power spectrum of several signals (e.g. transmit and receive)."""

    def __init__(self,
                 names=('Transmit', 'Receive'),
                 width=800,
                 height=500,
                 yrange=(-120, 0)):
        self._names = list(names)
        self._plot = go.FigureWidget(
            layout={
                'height' : height,
                'width' : width,
                'uirevision' : 'spectrum',
                'xaxis' : {'title' : 'Frequency Offset (MHz)'},
                'yaxis' : {'title' : 'Power (dBFS per bin)',
                           'range' : list(yrange)},
                'title' : 'Power Spectrum',
                'legend' : {'x' : 0.99, 'xanchor' : 'right', 'y' : 0.99},
            },
            data=[{'type' : 'scatter',
                   'mode' : 'lines',
                   'name' : name,
                   'x' : [0],
                   'y' : [yrange[0]]} for name in self._names])

    def update_data(self, traces, title=None):
        """Update the plot.

        traces: dict mapping a trace name to ``(freq_hz, power_db)`` or None
        to hide the trace.
        """
        with self._plot.batch_update():
            for trace in self._plot.data:
                data = traces.get(trace.name)
                if data is None:
                    trace.visible = False
                    continue
                freq, power = data
                trace.x = np.asarray(freq) * 1e-6
                trace.y = power
                trace.visible = True
            if title is not None:
                self._plot.layout.title = title

    def set_yrange(self, yrange):
        self._plot.layout.yaxis.range = list(yrange)

    def get_widget(self):
        return self._plot


class RangeDopplerPlot():
    """Heatmap of a range-Doppler map (see rfsoc_ofdm.radar.OFDMRadar)."""

    def __init__(self,
                 width=600,
                 height=500,
                 dynamic_range=60,
                 colorscale='Viridis'):
        self._dynamic_range = dynamic_range
        self._plot = go.FigureWidget(
            layout={
                'height' : height,
                'width' : width,
                'uirevision' : 'range-doppler',
                'xaxis' : {'title' : 'Doppler (kHz)'},
                'yaxis' : {'title' : 'Range (m)'},
                'title' : 'Range-Doppler Map',
            },
            data=[{'type' : 'heatmap',
                   'x' : [0],
                   'y' : [0],
                   'z' : [[-dynamic_range]],
                   'zmin' : -dynamic_range,
                   'zmax' : 0,
                   'colorscale' : colorscale,
                   'colorbar' : {'title' : 'dB'}}])

    def set_dynamic_range(self, dynamic_range):
        self._dynamic_range = dynamic_range
        self._plot.data[0].zmin = -dynamic_range

    def update_data(self, result, max_range=None, min_range=None,
                    max_doppler=None, units='doppler', title=None):
        """Draw a RangeDopplerResult.

        max_range, min_range: range limits in metres.
        max_doppler: Doppler limit (absolute value) in Hz.
        units: 'doppler' (kHz) or 'velocity' (m/s) for the Doppler axis.
        """
        rng = result.range_m
        r_sel = np.ones(len(rng), dtype=bool)
        if max_range is not None:
            r_sel &= rng <= max_range
        if min_range is not None:
            r_sel &= rng >= min_range
        d_sel = np.ones(len(result.doppler_hz), dtype=bool)
        if max_doppler is not None:
            d_sel &= np.abs(result.doppler_hz) <= max_doppler
        if not np.any(d_sel):
            d_sel[np.argmin(np.abs(result.doppler_hz))] = True
        if units == 'velocity':
            x = result.velocity_mps[d_sel]
            xtitle = 'Radial Velocity (m/s)'
        else:
            x = result.doppler_hz[d_sel] * 1e-3
            xtitle = 'Doppler (kHz)'
        z = np.maximum(result.rd_map_db[np.ix_(d_sel, r_sel)].T,
                       -self._dynamic_range).astype(np.float32)
        with self._plot.batch_update():
            heatmap = self._plot.data[0]
            heatmap.x = x
            heatmap.y = rng[r_sel]
            heatmap.z = z
            heatmap.zmin = -self._dynamic_range
            self._plot.layout.xaxis.title = xtitle
            if title is not None:
                self._plot.layout.title = title

    def get_widget(self):
        return self._plot


class RangeProfilePlot():
    """Range profile (power against range) at the Doppler bin of the peak."""

    def __init__(self,
                 width=400,
                 height=500,
                 dynamic_range=60):
        self._plot = go.FigureWidget(
            layout={
                'height' : height,
                'width' : width,
                'uirevision' : 'range-profile',
                'xaxis' : {'title' : 'Power (dB)',
                           'range' : [-dynamic_range, 3]},
                'yaxis' : {'title' : 'Range (m)'},
                'title' : 'Range Profile',
                'showlegend' : False,
            },
            data=[{'type' : 'scatter',
                   'mode' : 'lines',
                   'name' : 'Range Profile',
                   'x' : [-dynamic_range],
                   'y' : [0]}])

    def set_dynamic_range(self, dynamic_range):
        self._plot.layout.xaxis.range = [-dynamic_range, 3]

    def update_data(self, result, max_range=None, min_range=None):
        rng = result.range_m
        sel = np.ones(len(rng), dtype=bool)
        if max_range is not None:
            sel &= rng <= max_range
        if min_range is not None:
            sel &= rng >= min_range
        with self._plot.batch_update():
            self._plot.data[0].x = result.range_profile_db[sel]
            self._plot.data[0].y = rng[sel]

    def get_widget(self):
        return self._plot
