#!/usr/bin/env python3
"""
Compare TTS samples against a reference with visual diagnostics.

Outputs:
- One PNG per candidate with waveform/spectrogram/diff/F0 diagnostics
- One JSON file with scalar metrics for quick ranking
"""

from __future__ import annotations

import argparse
import json
import math
import os
import wave
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.io import wavfile as scipy_wavfile
except Exception:  # pragma: no cover
    scipy_wavfile = None


def _safe_name(path: str) -> str:
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)


def load_wav_mono(path: str) -> Tuple[np.ndarray, int]:
    """Load wav as mono float32 in [-1, 1]."""
    if scipy_wavfile is not None:
        sr, data = scipy_wavfile.read(path)
        if data.ndim == 2:
            data = data.mean(axis=1)
        if np.issubdtype(data.dtype, np.integer):
            maxv = np.iinfo(data.dtype).max
            data = data.astype(np.float32) / float(maxv)
        else:
            data = data.astype(np.float32)
        return np.clip(data, -1.0, 1.0), int(sr)

    # Fallback path without scipy
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    if sampwidth != 2:
        raise ValueError(f"Unsupported sample width ({sampwidth}) in {path}")
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return np.clip(data, -1.0, 1.0), int(sr)


def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    if x.size == 0:
        return x
    dur = len(x) / float(sr_in)
    n_out = max(1, int(round(dur * sr_out)))
    t_in = np.linspace(0.0, dur, num=len(x), endpoint=False)
    t_out = np.linspace(0.0, dur, num=n_out, endpoint=False)
    return np.interp(t_out, t_in, x).astype(np.float32)


def rms_db(x: np.ndarray, eps: float = 1e-9) -> float:
    return float(20.0 * np.log10(np.sqrt(np.mean(np.square(x)) + eps) + eps))


def peak_db(x: np.ndarray, eps: float = 1e-9) -> float:
    return float(20.0 * np.log10(np.max(np.abs(x)) + eps))


def frame_signal(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    if len(x) < frame_len:
        pad = np.zeros(frame_len - len(x), dtype=x.dtype)
        x = np.concatenate([x, pad], axis=0)
    n_frames = 1 + (len(x) - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n_frames)[:, None]
    return x[idx]


def stft_mag_db(
    x: np.ndarray, sr: int, n_fft: int = 1024, hop: int = 256
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = frame_signal(x, n_fft, hop)
    win = np.hanning(n_fft).astype(np.float32)
    windowed = frames * win[None, :]
    spec = np.fft.rfft(windowed, axis=1)
    mag = np.abs(spec).T  # [freq, time]
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-8))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    times = (np.arange(mag.shape[1]) * hop) / float(sr)
    return mag_db, freqs, times


def estimate_f0_autocorr(
    x: np.ndarray,
    sr: int,
    frame_len: int = 1024,
    hop: int = 256,
    fmin: float = 70.0,
    fmax: float = 400.0,
) -> Tuple[np.ndarray, np.ndarray]:
    frames = frame_signal(x, frame_len, hop)
    win = np.hanning(frame_len).astype(np.float32)
    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)
    f0 = np.zeros(frames.shape[0], dtype=np.float32)
    for i, fr in enumerate(frames):
        y = fr * win
        e = float(np.mean(y * y))
        if e < 1e-6:
            continue
        ac = np.correlate(y, y, mode="full")[frame_len - 1 :]
        if max_lag >= len(ac):
            continue
        seg = ac[min_lag:max_lag]
        if seg.size == 0:
            continue
        lag = int(np.argmax(seg)) + min_lag
        peak = float(ac[lag])
        if peak <= 0.15 * float(ac[0]):
            continue
        f0[i] = float(sr / lag)
    times = (np.arange(len(f0)) * hop) / float(sr)
    return f0, times


def trim_silence_edges(x: np.ndarray, threshold_db: float = -45.0) -> np.ndarray:
    frame_len, hop = 1024, 256
    frames = frame_signal(x, frame_len, hop)
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-9)
    frame_db = 20.0 * np.log10(frame_rms + 1e-9)
    keep = np.where(frame_db >= threshold_db)[0]
    if keep.size == 0:
        return x
    start = int(max(0, keep[0] * hop))
    end = int(min(len(x), keep[-1] * hop + frame_len))
    return x[start:end]


def normalize_for_compare(x: np.ndarray, target_rms_db: float = -20.0) -> np.ndarray:
    cur = rms_db(x)
    gain = 10.0 ** ((target_rms_db - cur) / 20.0)
    y = x * gain
    mx = np.max(np.abs(y)) + 1e-8
    if mx > 1.0:
        y = y / mx
    return y.astype(np.float32)


@dataclass
class SingleMetrics:
    duration_s: float
    peak_dbfs: float
    rms_dbfs: float
    crest_db: float
    clipping_pct: float
    silence_pct: float
    hf_energy_ratio: float
    voiced_pct: float
    f0_med_hz: float
    f0_std_hz: float


def compute_single_metrics(x: np.ndarray, sr: int) -> SingleMetrics:
    frame_len, hop = 1024, 256
    frames = frame_signal(x, frame_len, hop)
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-9)
    frame_db = 20.0 * np.log10(frame_rms + 1e-9)
    silence_pct = float(np.mean(frame_db < (np.max(frame_db) - 35.0)) * 100.0)
    clipping_pct = float(np.mean(np.abs(x) >= 0.999) * 100.0)

    mag_db, freqs, _ = stft_mag_db(x, sr, n_fft=1024, hop=256)
    mag = 10.0 ** (mag_db / 20.0)
    mean_spec = np.mean(mag, axis=1)
    lo = float(np.sum(mean_spec[freqs < 4000.0]) + 1e-9)
    hi = float(np.sum(mean_spec[freqs >= 4000.0]))
    hf_ratio = hi / lo

    f0, _ = estimate_f0_autocorr(x, sr)
    voiced = f0[f0 > 0]
    voiced_pct = float(100.0 * np.mean(f0 > 0))
    f0_med = float(np.median(voiced)) if voiced.size else 0.0
    f0_std = float(np.std(voiced)) if voiced.size else 0.0

    peak = peak_db(x)
    rms = rms_db(x)
    return SingleMetrics(
        duration_s=float(len(x) / sr),
        peak_dbfs=peak,
        rms_dbfs=rms,
        crest_db=float(peak - rms),
        clipping_pct=clipping_pct,
        silence_pct=silence_pct,
        hf_energy_ratio=hf_ratio,
        voiced_pct=voiced_pct,
        f0_med_hz=f0_med,
        f0_std_hz=f0_std,
    )


def resize_time(mat: np.ndarray, target_t: int) -> np.ndarray:
    # mat: [freq, time]
    if mat.shape[1] == target_t:
        return mat
    x_in = np.linspace(0.0, 1.0, mat.shape[1], endpoint=False)
    x_out = np.linspace(0.0, 1.0, target_t, endpoint=False)
    out = np.stack([np.interp(x_out, x_in, row) for row in mat], axis=0)
    return out


def pair_metrics(
    ref: np.ndarray, cand: np.ndarray, sr: int
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = max(1, min(len(ref), len(cand)))
    ref_n = ref[:n]
    cand_n = cand[:n]

    err = ref_n - cand_n
    num = float(np.mean(ref_n * ref_n) + 1e-9)
    den = float(np.mean(err * err) + 1e-9)
    snr_like = 10.0 * math.log10(num / den)
    corr = float(np.corrcoef(ref_n, cand_n)[0, 1]) if n > 8 else 0.0

    ref_db, freqs, times_ref = stft_mag_db(ref_n, sr)
    cand_db, _, times_cand = stft_mag_db(cand_n, sr)
    t = min(ref_db.shape[1], cand_db.shape[1])
    ref_db = resize_time(ref_db, t)
    cand_db = resize_time(cand_db, t)
    diff_db = cand_db - ref_db

    l1 = float(np.mean(np.abs(diff_db)))
    rmse = float(np.sqrt(np.mean(np.square(diff_db))))
    ref_lin = 10.0 ** (ref_db / 20.0)
    cand_lin = 10.0 ** (cand_db / 20.0)
    cos = float(
        np.sum(ref_lin * cand_lin)
        / (np.sqrt(np.sum(ref_lin**2)) * np.sqrt(np.sum(cand_lin**2)) + 1e-9)
    )

    out = {
        "wave_corr": corr,
        "snr_like_db": snr_like,
        "stft_logmag_l1_db": l1,
        "stft_logmag_rmse_db": rmse,
        "stft_mag_cosine": cos,
        "duration_ratio_cand_over_ref": float(len(cand) / max(1, len(ref))),
    }
    return out, ref_db, cand_db, diff_db, freqs


def plot_pair(
    ref: np.ndarray,
    cand: np.ndarray,
    sr: int,
    ref_name: str,
    cand_name: str,
    out_png: str,
) -> None:
    show_s = 6.0
    n_show = int(show_s * sr)
    ref_s = ref[:n_show]
    cand_s = cand[:n_show]
    t_ref = np.arange(len(ref_s)) / float(sr)
    t_cand = np.arange(len(cand_s)) / float(sr)

    _, ref_db, cand_db, diff_db, freqs = pair_metrics(ref, cand, sr)
    f0_ref, tf0_ref = estimate_f0_autocorr(ref, sr)
    f0_cand, tf0_cand = estimate_f0_autocorr(cand, sr)

    fig, ax = plt.subplots(3, 2, figsize=(16, 11), constrained_layout=True)

    ax[0, 0].plot(t_ref, ref_s, lw=0.8)
    ax[0, 0].set_title(f"Reference waveform (first {show_s:.0f}s): {ref_name}")
    ax[0, 0].set_xlabel("time (s)")
    ax[0, 0].set_ylabel("amp")

    ax[0, 1].plot(t_cand, cand_s, lw=0.8, color="tab:orange")
    ax[0, 1].set_title(f"Candidate waveform (first {show_s:.0f}s): {cand_name}")
    ax[0, 1].set_xlabel("time (s)")
    ax[0, 1].set_ylabel("amp")

    im1 = ax[1, 0].imshow(
        ref_db,
        origin="lower",
        aspect="auto",
        extent=[0, ref_db.shape[1], float(freqs[0]), float(freqs[-1])],
        vmin=-90,
        vmax=-10,
        cmap="magma",
    )
    ax[1, 0].set_title("Reference log-magnitude spectrogram")
    ax[1, 0].set_xlabel("frame")
    ax[1, 0].set_ylabel("Hz")
    fig.colorbar(im1, ax=ax[1, 0], fraction=0.046, pad=0.04)

    im2 = ax[1, 1].imshow(
        cand_db,
        origin="lower",
        aspect="auto",
        extent=[0, cand_db.shape[1], float(freqs[0]), float(freqs[-1])],
        vmin=-90,
        vmax=-10,
        cmap="magma",
    )
    ax[1, 1].set_title("Candidate log-magnitude spectrogram")
    ax[1, 1].set_xlabel("frame")
    ax[1, 1].set_ylabel("Hz")
    fig.colorbar(im2, ax=ax[1, 1], fraction=0.046, pad=0.04)

    im3 = ax[2, 0].imshow(
        diff_db,
        origin="lower",
        aspect="auto",
        extent=[0, diff_db.shape[1], float(freqs[0]), float(freqs[-1])],
        vmin=-20,
        vmax=20,
        cmap="coolwarm",
    )
    ax[2, 0].set_title("Spectrogram delta (candidate - reference, dB)")
    ax[2, 0].set_xlabel("frame")
    ax[2, 0].set_ylabel("Hz")
    fig.colorbar(im3, ax=ax[2, 0], fraction=0.046, pad=0.04)

    ax[2, 1].plot(tf0_ref, f0_ref, label="ref", lw=1.0)
    ax[2, 1].plot(tf0_cand, f0_cand, label="cand", lw=1.0, alpha=0.9)
    ax[2, 1].set_ylim(60, 420)
    ax[2, 1].set_title("F0 contour proxy (autocorrelation)")
    ax[2, 1].set_xlabel("time (s)")
    ax[2, 1].set_ylabel("Hz")
    ax[2, 1].legend()

    fig.suptitle(f"TTS comparison: {cand_name} vs {ref_name}", fontsize=14)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)

    missing: List[str] = []
    if not os.path.isfile(args.reference):
        missing.append(args.reference)
    for cand_path in args.candidates:
        if not os.path.isfile(cand_path):
            missing.append(cand_path)
    if missing:
        missing_block = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            "Input wav file(s) not found.\n"
            "Replace placeholder paths like /path/to/cand1.wav with real files.\n"
            f"Missing:\n{missing_block}"
        )

    ref_raw, ref_sr = load_wav_mono(args.reference)
    ref = resample_linear(ref_raw, ref_sr, args.sample_rate)
    ref = trim_silence_edges(ref)
    ref = normalize_for_compare(ref)
    ref_single = compute_single_metrics(ref, args.sample_rate)

    report: Dict[str, object] = {
        "reference": args.reference,
        "sample_rate": args.sample_rate,
        "reference_metrics": ref_single.__dict__,
        "candidates": [],
    }

    ref_name = _safe_name(args.reference)
    for cand_path in args.candidates:
        cand_raw, cand_sr = load_wav_mono(cand_path)
        cand = resample_linear(cand_raw, cand_sr, args.sample_rate)
        cand = trim_silence_edges(cand)
        cand = normalize_for_compare(cand)

        single = compute_single_metrics(cand, args.sample_rate)
        pair, _, _, _, _ = pair_metrics(ref, cand, args.sample_rate)

        cand_name = _safe_name(cand_path)
        out_png = os.path.join(args.out_dir, f"diag_{cand_name}_vs_{ref_name}.png")
        plot_pair(ref, cand, args.sample_rate, ref_name, cand_name, out_png)

        report["candidates"].append(
            {
                "path": cand_path,
                "metrics": single.__dict__,
                "vs_reference": pair,
                "plot_png": out_png,
            }
        )
        print(f"[ok] wrote {out_png}")

    out_json = os.path.join(args.out_dir, "diagnostics_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[ok] wrote {out_json}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visual diagnostics for TTS sample comparisons")
    p.add_argument("--reference", required=True, help="Reference wav path")
    p.add_argument(
        "--candidates",
        nargs="+",
        required=True,
        help="Candidate wav paths to compare against reference",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for PNG diagnostics + JSON report",
    )
    p.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="Internal sample rate for consistent comparisons",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
