# Refrigerator Compressor ON/OFF Detector

Python app that learns the spectral fingerprint of your fridge compressor
and later detects whether it is running, using the microphone via **PyAudio**.

## What it does

1. **Train** – record two short samples:
   - compressor **ON**
   - compressor **OFF**
   → saves `templates/compressor_on.npz` and `templates/compressor_off.npz`
   → **automatically calibrates** the decision threshold → `templates/threshold.json`

2. **Calibrate** (optional) – re-compute threshold from existing templates
   without recording again.

3. **Detect** – continuous live monitoring.  
   Each ~1.5 s window is compared (cosine distance of Welch PSD) to both templates.
   Uses the calibrated threshold automatically. Smoothed decision is printed.

## Install

```bash
# System library (Debian / Ubuntu / Raspberry Pi OS)
sudo apt update
sudo apt install -y portaudio19-dev python3-pyaudio

# or pure pip (after portaudio19-dev is installed)
pip install pyaudio numpy scipy
```

## Usage

```bash
# See available microphones
python compressor_detector.py devices

# Interactive training + automatic threshold calibration
python compressor_detector.py train

# Use a specific mic and longer recordings
python compressor_detector.py train -d 1 -t 12

# Re-calibrate only (no new recordings)
python compressor_detector.py calibrate

# Live detection (loads calibrated threshold automatically)
python compressor_detector.py detect

# Override threshold manually if needed
python compressor_detector.py detect --threshold 0.50
```

## Files produced

```
templates/
  compressor_on.npz    # freqs + normalized PSD of running compressor
  compressor_off.npz   # freqs + normalized PSD of idle / fan only
  threshold.json       # auto-calibrated decision threshold + diagnostics
```

You can copy the whole `templates/` folder + the script to another machine
and run only the `detect` mode there.

## How automatic calibration works

After the two templates are saved the script:

1. Computes the spectral distance between ON and OFF templates.
2. Places the decision boundary near the midpoint in distance space.
3. Applies a small **safety margin** (default 0.18) that biases slightly
   toward OFF → fewer false positives.
4. Saves the resulting threshold into `threshold.json`.

Decision rule used at runtime:

```text
is_on = (distance_to_ON < distance_to_OFF * threshold)
```

- Lower threshold → more sensitive (easier to declare ON)
- Higher threshold → stricter (harder to declare ON)

You can still override it with `--threshold` on the command line.

## Tips

- Place the microphone 20–80 cm from the compressor (or against the side of the fridge).
- During **OFF** training capture typical background (fan noise, room noise).
- After training/calibration look at the printed “Spectral distance ON ↔ OFF”:
  - **> 0.20** → good separation
  - **0.12–0.20** → usable but watch for false triggers
  - **< 0.12** → re-record closer or longer
- If detection flickers, increase `--window` (e.g. 2.0) or the internal `SMOOTH_WINDOWS`.
"""
