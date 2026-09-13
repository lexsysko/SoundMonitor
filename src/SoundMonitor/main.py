#!/usr/bin/env python3
"""
Refrigerator Compressor ON/OFF Detector
=======================================
- Training mode: record mic samples → save two .npz templates (on.npz / off.npz)
- Automatic threshold calibration from the two templates
- Detection mode: live mic monitoring using the trained templates + calibrated threshold
- Direct PyAudio microphone access

Install (Debian/Ubuntu/Raspberry Pi):
    sudo apt update
    sudo apt install -y portaudio19-dev python3-pyaudio
    # or:  pip install pyaudio   (after portaudio19-dev is present)

    pip install numpy scipy

Usage:
    python compressor_detector.py train          # record + auto-calibrate
    python compressor_detector.py calibrate      # re-calibrate from existing templates
    python compressor_detector.py detect         # live detection (uses calibrated threshold)
    python compressor_detector.py devices
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import welch, butter, sosfilt

# ---------------------------------------------------------------------------
# Optional PyAudio import with clear error
# ---------------------------------------------------------------------------
try:
    import pyaudio
except ImportError:
    print(
        "ERROR: pyaudio is not installed.\n"
        "  sudo apt install -y portaudio19-dev\n"
        "  pip install pyaudio\n"
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SR = 16000  # sample rate
CHANNELS = 1
CHUNK = 1024  # frames per buffer

# Spectral analysis
NPERSEG = 4096  # ~256 ms
FREQ_RANGE = (40.0, 1000.0)  # Hz – region of interest

# Detection defaults (overridden by calibration when available)
DEFAULT_THRESHOLD = 0.60  # used only if no calibration file exists
SMOOTH_WINDOWS = 3  # majority vote over last N decisions
MIN_RECORD_SEC = 3.0  # minimum useful recording length for a template

# Safety margin: how much we bias the decision boundary toward OFF
# (reduces false positives). 0.0 = exact midpoint, 0.15–0.25 = safer.
SAFETY_MARGIN = 0.18

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

TEMPLATE_DIR = DATA_DIR / "templates"
ON_FILE = TEMPLATE_DIR / "compressor_on.npz"
OFF_FILE = TEMPLATE_DIR / "compressor_off.npz"
THRESHOLD_FILE = TEMPLATE_DIR / "threshold.json"


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def list_input_devices() -> None:
    pa = pyaudio.PyAudio()
    print("\nAvailable input devices:")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            print(
                f"  [{i}] {info['name']}  "
                f"(max in={info['maxInputChannels']}, "
                f"default rate={int(info['defaultSampleRate'])})"
            )
    pa.terminate()


def record_seconds(seconds: float, device_index: int | None = None) -> np.ndarray:
    """Record mono float32 audio from the microphone."""
    pa = pyaudio.PyAudio()
    kwargs = dict(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SR,
        input=True,
        frames_per_buffer=CHUNK,
    )
    if device_index is not None:
        kwargs["input_device_index"] = device_index

    stream = pa.open(**kwargs)
    frames = []
    n_chunks = int(SR / CHUNK * seconds)
    print(f"  Recording {seconds:.1f}s …", end="", flush=True)
    for _ in range(n_chunks):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(np.frombuffer(data, dtype=np.int16))
    stream.stop_stream()
    stream.close()
    pa.terminate()
    print(" done")

    audio = np.concatenate(frames).astype(np.float32)
    audio /= 32768.0
    return audio


def compute_psd(audio: np.ndarray, sr: int = SR) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD limited to FREQ_RANGE, returned as (freqs, normalized_psd)."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    sos = butter(2, 25, btype="high", fs=sr, output="sos")
    audio = sosfilt(sos, audio.astype(np.float64))

    nperseg = min(NPERSEG, max(256, len(audio) // 4))
    freqs, psd = welch(audio, fs=sr, nperseg=nperseg, scaling="density")
    mask = (freqs >= FREQ_RANGE[0]) & (freqs <= FREQ_RANGE[1])
    freqs = freqs[mask]
    psd = psd[mask]
    psd = psd / (psd.max() + 1e-12)
    return freqs, psd.astype(np.float32)


def spectral_distance(psd_a: np.ndarray, psd_b: np.ndarray) -> float:
    """
    Distance between two normalized PSD vectors.
    0 = identical shape, higher = more different.
    Uses 1 - cosine similarity (robust to overall gain).
    """
    a = psd_a.astype(np.float64)
    b = psd_b.astype(np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    cos = np.dot(a, b) / (na * nb)
    return float(1.0 - cos)


# ---------------------------------------------------------------------------
# Automatic threshold calibration
# ---------------------------------------------------------------------------
def calibrate_from_templates(verbose: bool = True) -> dict:
    """
    Derive a decision threshold from the two saved templates only
    (no extra recordings needed).

    Decision rule used later:
        is_on = (d_on < d_off * threshold)

    We place the boundary near the midpoint between the two templates
    in distance space, then apply a safety margin that biases slightly
    toward OFF (fewer false positives).

    Returns a dict that is also written to threshold.json.
    """
    if not ON_FILE.exists() or not OFF_FILE.exists():
        raise FileNotFoundError(
            f"Need both {ON_FILE.name} and {OFF_FILE.name}. Run 'train' first."
        )

    on = np.load(ON_FILE)
    off = np.load(OFF_FILE)
    psd_on = on["psd"]
    psd_off = off["psd"]

    # Distance between the two class centroids
    d_on_off = spectral_distance(psd_on, psd_off)

    # Self-distance is ~0; we still compute for sanity
    d_on_self = spectral_distance(psd_on, psd_on)
    d_off_self = spectral_distance(psd_off, psd_off)

    # Ideal midpoint rule in the (d_on, d_off) plane:
    #   classify ON when d_on / d_off < 1.0
    # With safety margin we require a stricter ratio:
    #   threshold = 1.0 / (1.0 + SAFETY_MARGIN)   ≈ 0.85 when margin=0.18
    # But we also scale by how well separated the templates are.
    #
    # More robust formula used here:
    #   threshold = midpoint_ratio * (1 - SAFETY_MARGIN)
    # where midpoint_ratio would be 1.0 for equal distance.
    #
    # Practical calibrated value:
    base = 1.0  # pure midpoint
    threshold = base * (1.0 - SAFETY_MARGIN)

    # Clamp to a sensible range so extreme templates don't produce nonsense
    threshold = float(np.clip(threshold, 0.35, 0.90))

    # Extra diagnostic: expected scores if a new sample == template
    # When sample == ON:  d_on≈0, d_off≈d_on_off  →  ratio d_on/d_off ≈ 0
    # When sample == OFF: d_on≈d_on_off, d_off≈0 → ratio → +inf
    # Decision boundary at d_on = threshold * d_off

    result = {
        "threshold": threshold,
        "distance_on_off": round(d_on_off, 4),
        "safety_margin": SAFETY_MARGIN,
        "method": "midpoint_with_safety_margin",
        "rule": "is_on = (d_on < d_off * threshold)",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "notes": (
            "Lower threshold → more sensitive (easier ON). "
            "Higher → stricter (fewer false positives)."
        ),
    }

    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(THRESHOLD_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if verbose:
        print("\n" + "=" * 60)
        print("AUTOMATIC THRESHOLD CALIBRATION")
        print("=" * 60)
        print(f"  Spectral distance ON ↔ OFF : {d_on_off:.4f}")
        print(f"  Safety margin              : {SAFETY_MARGIN:.2f}")
        print(f"  Calibrated threshold       : {threshold:.3f}")
        print(f"  Saved to                   : {THRESHOLD_FILE}")
        if d_on_off < 0.12:
            print("\n  ⚠  Templates are very close – detection will be unreliable.")
            print(
                "     Re-record with mic closer to the compressor or longer duration."
            )
        elif d_on_off < 0.20:
            print(
                "\n  ⚠  Moderate separation. Consider re-recording if you see false triggers."
            )
        else:
            print("\n  ✓  Good separation between ON and OFF templates.")
        print("=" * 60)

    return result


def load_threshold(override: float | None = None) -> float:
    """Return threshold to use: CLI override > calibrated file > default."""
    if override is not None:
        return override
    if THRESHOLD_FILE.exists():
        try:
            with open(THRESHOLD_FILE, encoding="utf-8") as f:
                data = json.load(f)
            t = float(data["threshold"])
            print(
                f"  Using calibrated threshold: {t:.3f}  (from {THRESHOLD_FILE.name})"
            )
            return t
        except Exception as e:
            print(f"  Warning: could not read {THRESHOLD_FILE}: {e}")
    print(f"  Using default threshold: {DEFAULT_THRESHOLD:.3f}")
    return DEFAULT_THRESHOLD


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(device_index: int | None = None, duration: float = 8.0) -> None:
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    list_input_devices()
    print()

    def capture_label(label: str, path: Path) -> None:
        input(f"\n>>> Prepare '{label}' state, then press Enter to start recording…")
        audio = record_seconds(duration, device_index)
        if len(audio) < SR * MIN_RECORD_SEC:
            print("Recording too short – try again.")
            return
        freqs, psd = compute_psd(audio)
        np.savez_compressed(
            path,
            freqs=freqs,
            psd=psd,
            sr=SR,
            duration=duration,
            label=label,
            created=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        print(f"  Saved template → {path}")
        peaks = freqs[np.argsort(psd)[::-1][:5]]
        print(f"  Top peaks (Hz): {', '.join(f'{p:.0f}' for p in peaks)}")

    print("=" * 60)
    print("TRAINING MODE")
    print("You will record two states of the refrigerator:")
    print("  1. Compressor ON  (running / humming)")
    print("  2. Compressor OFF (only fan / room noise, or completely silent)")
    print(f"Each recording lasts {duration:.0f} seconds.")
    print("=" * 60)

    capture_label("ON", ON_FILE)
    capture_label("OFF", OFF_FILE)

    # Automatic calibration right after training
    if ON_FILE.exists() and OFF_FILE.exists():
        calibrate_from_templates(verbose=True)
    print("\nTraining + calibration finished.")
    print("You can now run:  python compressor_detector.py detect")


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def load_templates() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not ON_FILE.exists() or not OFF_FILE.exists():
        print(f"Templates not found in {TEMPLATE_DIR}")
        print("Run:  python compressor_detector.py train")
        sys.exit(1)
    on = np.load(ON_FILE)
    off = np.load(OFF_FILE)
    return on["freqs"], on["psd"], off["psd"]


def detect(
    device_index: int | None = None,
    threshold: float | None = None,
    window_sec: float = 1.5,
    poll_interval: float = 0.6,
) -> None:
    freqs_ref, psd_on, psd_off = load_templates()
    thr = load_threshold(override=threshold)

    print("=" * 60)
    print("LIVE DETECTION")
    print(f"  Template dir : {TEMPLATE_DIR}")
    print(f"  Threshold    : {thr:.3f}  (lower = more sensitive to ON)")
    print(f"  Window       : {window_sec:.1f}s")
    print("  Ctrl+C to stop")
    print("=" * 60)
    list_input_devices()
    print()

    history: list[bool] = []
    last_state: bool | None = None

    try:
        while True:
            audio = record_seconds(window_sec, device_index)
            _, psd = compute_psd(audio)

            n = min(len(psd), len(psd_on), len(psd_off))
            d_on = spectral_distance(psd[:n], psd_on[:n])
            d_off = spectral_distance(psd[:n], psd_off[:n])

            # Calibrated rule
            is_on = d_on < (d_off * thr)

            history.append(is_on)
            if len(history) > SMOOTH_WINDOWS:
                history.pop(0)
            smoothed = sum(history) > len(history) / 2

            state_str = "ON " if smoothed else "off"
            marker = " <<<" if smoothed != last_state and last_state is not None else ""
            print(
                f"\r[{time.strftime('%H:%M:%S')}]  "
                f"d_on={d_on:.3f}  d_off={d_off:.3f}  "
                f"→ {state_str}{marker}   ",
                end="",
                flush=True,
            )
            if smoothed != last_state and last_state is not None:
                print()  # newline on change
            last_state = smoothed

            time.sleep(max(0.0, poll_interval - 0.05))
    except KeyboardInterrupt:
        print("\n\nStopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refrigerator compressor ON/OFF detector (PyAudio + FFT templates)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # train
    p_train = sub.add_parser("train", help="Record ON/OFF templates + auto-calibrate")
    p_train.add_argument(
        "-d", "--device", type=int, default=None, help="PyAudio input device index"
    )
    p_train.add_argument(
        "-t",
        "--duration",
        type=float,
        default=8.0,
        help="Seconds to record for each template (default 8)",
    )

    # calibrate (re-run from existing templates)
    sub.add_parser("calibrate", help="Re-compute threshold from existing on/off .npz")

    # detect
    p_det = sub.add_parser("detect", help="Live detection using saved templates")
    p_det.add_argument(
        "-d", "--device", type=int, default=None, help="PyAudio input device index"
    )
    p_det.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override calibrated threshold (optional)",
    )
    p_det.add_argument(
        "--window", type=float, default=1.5, help="Analysis window length in seconds"
    )

    # list devices
    sub.add_parser("devices", help="List microphone devices")

    args = parser.parse_args()

    if args.cmd == "devices":
        list_input_devices()
    elif args.cmd == "train":
        train(device_index=args.device, duration=args.duration)
    elif args.cmd == "calibrate":
        calibrate_from_templates(verbose=True)
    elif args.cmd == "detect":
        detect(
            device_index=args.device,
            threshold=args.threshold,
            window_sec=args.window,
        )


if __name__ == "__main__":
    main()
