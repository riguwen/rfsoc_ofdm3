<img src="strathsdr_banner.png" width="100%">

<table border="0" align="center">
    <tr border="0">
        <td align="center" width="50%" border="0">
            <img src="https://www.rfsocbook.com/wp-content/uploads/2022/12/RFSoC3D_v4-1433x1536.png" alt="oscthumb" style="width: 60%" border="0"/>
        </td>
        <td align="center" width="50%" border="0">
            <font size=7><b>Available Now!</b></font size> <br> <font size=5>Software Defined Radio with Zynq® UltraScale+ RFSoC</font size> <br> <font size=4><a href="https://rfsocbook.com/">Free Download</a> <br><a href="https://www.amazon.com/Software-Defined-Radio-Ultrascale-RFSoC/dp/1739588606?keywords=zynq+rfsoc&qid=1673452844&sprefix=%2Caps%2C137&sr=8-1&linkCode=ll1&tag=thzybo-20&linkId=0bf245a543fd4af4625086df4c190928&language=en_US&ref_=as_li_ss_tl">Printed Edition</a></font size>
        </td>
    </tr>
</table>

# RFSoC OFDM Transceiver
This repository contains an RFSoC demonstration of an Orthogonal Frequency Division Multiplexing (OFDM) transceiver. The OFDM system is only compatible with [PYNQ images v3.1.1](https://github.com/Xilinx/PYNQ/releases) and greater for the following RFSoC development boards:
* [ZCU208](https://www.xilinx.com/products/boards-and-kits/zcu208.html),
* [ZCU111](https://www.xilinx.com/products/boards-and-kits/zcu111.html),
* [RFSoC4x2](http://rfsoc-pynq.io/),
* [RFSoC2x2](http://rfsoc-pynq.io/),

The following boards are limited to PYNQ version 2.7 and Vivado 2020.2 since this board is not officially supported by PYNQ and no official SD card image exists:
* [ZCU216](https://www.xilinx.com/products/boards-and-kits/zcu216.html).

The SD card image for this board has been made available from the following repository: [sarafs1926/ZCU216-PYNQ](https://github.com/sarafs1926/ZCU216-PYNQ).

<p align="center">
  <img src="demonstration.gif"/>
<p/>

## Quick Start
Follow the instructions below to install the OFDM demonstrator on your development board. **You will need to give your board access to the internet**.
* Power on your RFSoC development board with an SD Card containing a fresh PYNQ v2.7 image or greater.
* Navigate to Jupyter Labs by opening a browser (preferably Chrome) and connecting to `http://<board_ip_address>:9090/lab`.
* We need to open a terminal in Jupyter Lab. Firstly, open a launcher window as shown in the figure below:

<p align="center">
  <img src="./open_jupyter_launcher.jpg" width="50%" height="50%" />
<p/>

* Now open a terminal in Jupyter as illustrated below:

<p align="center">
  <img src="./open_terminal_window.jpg" width="50%" height="50%" />
<p/>

* Now execute the following command in the terminal:

```sh
pip3 install https://github.com/strath-sdr/rfsoc_ofdm/releases/download/v0.4.0/rfsoc_ofdm.tar.gz
python -m rfsoc_ofdm install
```

The notebook should now be available in the `rfsoc_ofdm` folder in your Jupyter Workspace.
It is important that you use the Chrome web browser if possible as rendering performance is important.

## Spectrum Analyser and Range-Doppler Map
This adds two visualisations to the OFDM demonstrator. Both are software only and run on the existing bitstream of the v0.3.5 release (and v0.4.0), so no Vivado rebuild is needed:
* **Spectrum analyser**: averaged power spectra (in dBFS) of the transmitted and received signals at the 20 MSPS baseband or the 240 MSPS wideband observation points.
* **Range-Doppler map**: the OFDM signal is used as a radar waveform. The transmitter and receiver 20 MSPS streams are captured at the same time. The transmitted symbols are divided out of the received symbols on every sub-carrier, then an IFFT across sub-carriers (range) and an FFT across OFDM symbols (Doppler) are applied. A Doppler shift can be emulated by offsetting the DAC NCO.

### Installing on a board that already runs v0.3.5 (or v0.4.0)
Open a terminal in Jupyter Lab and run:

```sh
git clone -b claude/adoring-cerf-l8376v https://github.com/riguwen/rfsoc_ofdm3.git /tmp/rfsoc_ofdm_sensing
PKG=$(python3 -c "import os, rfsoc_ofdm; print(os.path.dirname(rfsoc_ofdm.__file__))")
SRC=/tmp/rfsoc_ofdm_sensing/rfsoc_ofdm
cp $SRC/radar.py $SRC/sensing.py $SRC/sdr_plots.py $SRC/inspector.py $SRC/overlay.py $PKG/
cp $SRC/notebooks/02_rfsoc_ofdm_demonstrator.ipynb $PKG/notebooks/
cp $SRC/notebooks/02_rfsoc_ofdm_demonstrator.ipynb $PYNQ_JUPYTER_NOTEBOOKS/rfsoc_ofdm/
```

Only copy the files listed above. In particular, keep the installed `ofdm.py`: it names the OFDM IP version of the installed bitstream (v0.4 IP for release v0.3.5, v0.5 IP for release v0.4.0). The last command replaces notebook `02_rfsoc_ofdm_demonstrator.ipynb` in your Jupyter workspace, so back it up first if you have changed it. Restart the kernel of any open OFDM notebook afterwards.

### Usage
Notebook `02_rfsoc_ofdm_demonstrator.ipynb` has new *Spectrum Analyser* and *Range-Doppler Map* sections, and the Voila dashboard (notebook 03) has a *Range-Doppler* tab. From Python:

```python
from rfsoc_ofdm.overlay import Overlay
ofdm_hw = Overlay()

ofdm_hw.spectrum_application()        # live transmit/receive spectrum widget
ofdm_hw.range_doppler_application()   # live range-Doppler map widget

result = ofdm_hw.range_doppler.measure()   # one map as numpy arrays
ofdm_hw.range_doppler.set_nco_offset(10e3) # emulate a +10 kHz Doppler shift
```

Zero range is the strongest received path (the loopback cable or the direct antenna path), because the start times of the transmit and receive captures differ by an unknown few microseconds. At 20 MSPS one range bin is 7.5 m, the cyclic prefix covers about 90 m, and 256 OFDM symbols give a Doppler resolution of 977 Hz (244 m/s at 600 MHz). The signal processing is in `rfsoc_ofdm/radar.py` and only needs numpy, so it can also process saved captures, or synthetic ones from `radar.simulate_capture`, on a PC.

## Using the Project Files
The following software is required to use the project files in this repository.
- Vivado Design Suite 2020.2
- MATLAB R2020a

### Vivado
This project can be built with Vivado from the command line. Open Vivado 2020.1 and execute the following into the tcl console:
```sh
cd /<repository-location>/boards/<board-name>/rfsoc_ofdm/
make block_design
make bitstream
```

Alternatively, you can run the entire project build by executing the following into the tcl console:
```sh
make all
```

## License
[BSD 3-Clause](/LICENSE)
