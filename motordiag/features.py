"""Caractéristiques "à la main" pour la baseline (forêt aléatoire / gradient boosting).

Calcul vectorisé sur un lot de fenêtres (n, L). Les caractéristiques sont choisies pour
être physiquement interprétables :
- temporelles : RMS, crête, kurtosis, facteurs de forme...
- spectrales : énergie par bande (bandes log), centroïde spectral ;
- vibration : amplitudes du spectre d'enveloppe aux harmoniques de BPFO, BPFI, BSF, FTF, fr ;
- courant : raies latérales f_s ± BPFO, f_s ± BPFI, f_s ± fr relatives au fondamental,
  harmoniques 3, 5, 7.
"""
from __future__ import annotations

import numpy as np

from .signal_analysis import bearing_fault_frequencies


def _time_features(x):
    xc = x - x.mean(axis=1, keepdims=True)
    std = xc.std(axis=1) + 1e-12
    rms = np.sqrt((x ** 2).mean(axis=1)) + 1e-12
    peak = np.abs(x).max(axis=1)
    absmean = np.abs(x).mean(axis=1) + 1e-12
    kurt = (xc ** 4).mean(axis=1) / std ** 4
    skew = (xc ** 3).mean(axis=1) / std ** 3
    clear = peak / (np.sqrt(np.abs(x)).mean(axis=1) ** 2 + 1e-12)
    feats = [std, rms, peak, x.max(axis=1) - x.min(axis=1), peak / rms, kurt, skew, rms / absmean,
             peak / absmean, clear]
    names = ["std", "rms", "peak", "p2p", "crest", "kurtosis", "skewness", "shape", "impulse", "clearance"]
    return np.stack(feats, axis=1), names


def _band_features(amp, freqs, n_bands=12, fmin=10.0):
    fmax = freqs[-1]
    edges = np.geomspace(fmin, fmax, n_bands + 1)
    total = (amp ** 2).sum(axis=1)
    feats, names = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (freqs >= a) & (freqs < b)
        e = (amp[:, m] ** 2).sum(axis=1)
        feats.append(np.log10(e / (total + 1e-20) + 1e-12))
        names.append(f"band_{a:.0f}_{b:.0f}Hz")
    centroid = (amp * freqs).sum(axis=1) / (amp.sum(axis=1) + 1e-20)
    feats.append(centroid); names.append("spectral_centroid")
    return np.stack(feats, axis=1), names


def _peak_near(amp, freqs, f_target, tol_bins=2):
    """Amplitude max autour de f_target (une fréquence par fenêtre)."""
    df = freqs[1] - freqs[0]
    k = np.round(f_target / df).astype(int)
    out = np.zeros(amp.shape[0])
    for d in range(-tol_bins, tol_bins + 1):
        kk = np.clip(k + d, 0, amp.shape[1] - 1)
        out = np.maximum(out, amp[np.arange(amp.shape[0]), kk])
    return out


def window_features(x: np.ndarray, fs: float, modality: str, rpm: np.ndarray, geometry: dict | None = None):
    """Matrice (n, F) de caractéristiques et liste des noms. `rpm` : vitesse par fenêtre."""
    x = np.asarray(x, dtype=np.float64)
    n, L = x.shape
    rpm = np.broadcast_to(np.asarray(rpm, dtype=float), (n,))
    tf, tn = _time_features(x)
    w = np.hanning(L)
    xc = x - x.mean(axis=1, keepdims=True)
    amp = np.abs(np.fft.rfft(xc * w, axis=1)) * 2 / w.sum()
    freqs = np.fft.rfftfreq(L, 1 / fs)
    bf, bn = _band_features(amp, freqs)
    feats, names = [tf, bf], [f"{modality}_{s}" for s in tn + bn]
    ff = {k: np.array([bearing_fault_frequencies(r, geometry)[k] for r in rpm]) for k in ("fr", "BPFO", "BPFI", "BSF", "FTF")}

    if modality == "vib":
        # Spectre d'enveloppe vectorisé (passe-bande 1 kHz – 0,45·fs puis Hilbert par FFT).
        full_f = np.fft.fftfreq(L, 1 / fs)
        h = np.where((full_f >= 1000) & (full_f <= 0.45 * fs), 2.0, 0.0)
        env = np.abs(np.fft.ifft(np.fft.fft(x, axis=1) * h, axis=1))
        env -= env.mean(axis=1, keepdims=True)
        eamp = np.abs(np.fft.rfft(env * w, axis=1)) * 2 / w.sum()
        ref = np.median(eamp[:, (freqs > 5) & (freqs < 1000)], axis=1) + 1e-12
        cols = []
        for name in ("fr", "FTF", "BSF", "BPFO", "BPFI"):
            for k in (1, 2, 3):
                cols.append(np.log10(_peak_near(eamp, freqs, k * ff[name]) / ref + 1e-12))
                names.append(f"vib_env_{k}x{name}")
        feats.append(np.stack(cols, axis=1))
    else:
        # Fondamental du courant : pic max entre 20 et 150 Hz (moteur à onduleur).
        band = (freqs >= 20) & (freqs <= 150)
        kf = np.argmax(amp[:, band], axis=1)
        f_s = freqs[band][kf]
        a_s = amp[np.arange(n), np.flatnonzero(band)[kf]] + 1e-12
        cols = [f_s, np.log10(a_s)]
        names += ["cur_f_supply", "cur_log_amp_supply"]
        for name in ("fr", "BPFO", "BPFI"):
            for sign, s in ((-1, "m"), (1, "p")):
                ft = np.abs(f_s + sign * ff[name])
                cols.append(20 * np.log10(_peak_near(amp, freqs, ft) / a_s + 1e-12))
                names.append(f"cur_sb_{s}{name}_dB")
        for h_ in (3, 5, 7):
            cols.append(20 * np.log10(_peak_near(amp, freqs, h_ * f_s) / a_s + 1e-12))
            names.append(f"cur_h{h_}_dB")
        feats.append(np.stack(cols, axis=1))
    return np.concatenate(feats, axis=1), names
