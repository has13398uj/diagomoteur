"""Analyse de signal pour le diagnostic de roulements (numpy / scipy uniquement).

Toutes les fonctions prennent un signal 1D et sa fréquence d'échantillonnage `fs` (Hz)
et renvoient des grandeurs physiques avec unités explicites. Elles sont utilisées par
le dashboard (graphiques, indicateurs) et par la baseline "caractéristiques + forêt
aléatoire" du notebook : le même code sert donc aux deux.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps
from scipy import stats

from .constants import BEARING_6203


# --------------------------------------------------------------------------------
# Indicateurs temporels
# --------------------------------------------------------------------------------
def indicators(x: np.ndarray) -> dict:
    """RMS, crête, crête-à-crête, facteur de crête, kurtosis, skewness...

    La kurtosis renvoyée est la kurtosis de Pearson (= 3 pour un bruit gaussien) : c'est
    la convention habituelle en surveillance vibratoire (un roulement sain est proche de 3,
    des chocs périodiques la font monter nettement au-dessus).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        raise ValueError("Signal vide.")
    mean = float(np.mean(x))
    xc = x - mean
    rms = float(np.sqrt(np.mean(x ** 2)))
    std = float(np.std(x))
    peak = float(np.max(np.abs(x)))
    p2p = float(np.max(x) - np.min(x))
    abs_mean = float(np.mean(np.abs(x)))
    return {
        "mean": mean,
        "std": std,
        "rms": rms,
        "peak": peak,
        "peak_to_peak": p2p,
        "crest_factor": peak / rms if rms > 0 else float("nan"),
        "kurtosis": float(stats.kurtosis(xc, fisher=False)) if std > 0 else float("nan"),
        "skewness": float(stats.skew(xc)) if std > 0 else float("nan"),
        "shape_factor": rms / abs_mean if abs_mean > 0 else float("nan"),
        "impulse_factor": peak / abs_mean if abs_mean > 0 else float("nan"),
    }


# --------------------------------------------------------------------------------
# Spectres
# --------------------------------------------------------------------------------
def amplitude_spectrum(x: np.ndarray, fs: float, detrend: bool = True):
    """Spectre d'amplitude monolatéral (fenêtre de Hann, amplitude corrigée).

    Une sinusoïde d'amplitude A donne un pic de hauteur ~A (même unité que le signal).
    Renvoie (fréquences en Hz, amplitudes).
    """
    x = np.asarray(x, dtype=np.float64)
    if detrend:
        x = x - np.mean(x)
    n = x.size
    w = np.hanning(n)
    spec = np.fft.rfft(x * w)
    amp = np.abs(spec) * 2.0 / np.sum(w)
    amp[0] /= 2.0
    if n % 2 == 0:
        amp[-1] /= 2.0
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, amp


def bandpass(x: np.ndarray, fs: float, f_low: float, f_high: float, order: int = 4) -> np.ndarray:
    """Filtre passe-bande Butterworth à phase nulle (sos + filtfilt)."""
    nyq = fs / 2.0
    f_low = max(float(f_low), 1e-3)
    f_high = min(float(f_high), nyq * 0.999)
    if not f_low < f_high:
        raise ValueError(f"Bande invalide : {f_low:.1f}–{f_high:.1f} Hz (Nyquist = {nyq:.1f} Hz).")
    sos = sps.butter(order, [f_low, f_high], btype="bandpass", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, np.asarray(x, dtype=np.float64))


def envelope_spectrum(x: np.ndarray, fs: float, band: tuple[float, float] | None = None,
                      fmax: float | None = 1000.0):
    """Spectre d'enveloppe : passe-bande autour d'une résonance, enveloppe de Hilbert, FFT.

    Les chocs d'un défaut de roulement excitent des résonances haute fréquence ; leur
    cadence (BPFO, BPFI...) apparaît dans le spectre de l'enveloppe. Renvoie
    (fréquences en Hz, amplitudes de l'enveloppe).
    """
    y = np.asarray(x, dtype=np.float64)
    if band is not None:
        y = bandpass(y, fs, band[0], band[1])
    env = np.abs(sps.hilbert(y))
    freqs, amp = amplitude_spectrum(env, fs, detrend=True)
    if fmax is not None:
        keep = freqs <= fmax
        freqs, amp = freqs[keep], amp[keep]
    return freqs, amp


def spectrogram(x: np.ndarray, fs: float, nperseg: int = 1024, noverlap: int | None = None,
                fmax: float | None = None, max_frames: int = 400):
    """STFT : renvoie (temps en s, fréquences en Hz, densité en dB).

    Le nombre de trames est limité à `max_frames` (hop agrandi si besoin) pour garder
    une taille raisonnable à afficher.
    """
    x = np.asarray(x, dtype=np.float64)
    nperseg = int(min(nperseg, x.size))
    if noverlap is None:
        noverlap = nperseg // 2
    hop = nperseg - noverlap
    n_frames = 1 + max(0, (x.size - nperseg) // hop)
    if n_frames > max_frames:
        hop = int(np.ceil((x.size - nperseg) / (max_frames - 1)))
        noverlap = max(0, nperseg - hop)
    f, t, sxx = sps.spectrogram(x - np.mean(x), fs=fs, window="hann", nperseg=nperseg,
                                noverlap=noverlap, scaling="spectrum", mode="psd")
    if t.size > max_frames:
        # Même sans recouvrement il y a trop de trames : moyenne par groupes de k trames.
        k = int(np.ceil(t.size / max_frames))
        n = (t.size // k) * k
        sxx = sxx[:, :n].reshape(sxx.shape[0], -1, k).mean(axis=2)
        t = t[:n].reshape(-1, k).mean(axis=1)
    if fmax is not None:
        keep = f <= fmax
        f, sxx = f[keep], sxx[keep]
    sxx_db = 10.0 * np.log10(sxx + 1e-20)
    return t, f, sxx_db


# --------------------------------------------------------------------------------
# Cinématique du roulement
# --------------------------------------------------------------------------------
def bearing_fault_frequencies(rpm: float, geometry: dict | None = None) -> dict:
    """Fréquences caractéristiques de défaut (Hz) à partir de la géométrie et de la vitesse.

    fr   = rpm / 60                                   (fréquence de rotation de l'arbre)
    FTF  = fr/2 · (1 − d/D·cosφ)                       (cage)
    BPFO = n·fr/2 · (1 − d/D·cosφ)                     (bague extérieure)
    BPFI = n·fr/2 · (1 + d/D·cosφ)                     (bague intérieure)
    BSF  = D·fr/(2d) · (1 − (d/D·cosφ)²)               (bille, rotation propre)
    """
    g = dict(BEARING_6203)
    if geometry:
        g.update({k: v for k, v in geometry.items() if v is not None})
    n = float(g["n_balls"])
    d = float(g["ball_diameter_mm"])
    D = float(g["pitch_diameter_mm"])
    phi = np.deg2rad(float(g.get("contact_angle_deg", 0.0)))
    if rpm is None or rpm <= 0:
        raise ValueError("La vitesse de rotation doit être > 0 tr/min.")
    if not (0 < d < D):
        raise ValueError("Géométrie invalide : il faut 0 < diamètre de bille < diamètre primitif.")
    fr = rpm / 60.0
    r = d / D * np.cos(phi)
    return {
        "fr": fr,
        "FTF": fr / 2.0 * (1 - r),
        "BPFO": n * fr / 2.0 * (1 - r),
        "BPFI": n * fr / 2.0 * (1 + r),
        "BSF": D * fr / (2.0 * d) * (1 - r ** 2),
    }


def estimate_supply_frequency(x: np.ndarray, fs: float, fmin: float = 5.0, fmax: float = 500.0) -> float:
    """Fréquence fondamentale du courant = pic le plus haut du spectre entre fmin et fmax.

    Le moteur Paderborn est alimenté par onduleur : la fréquence d'alimentation dépend de
    la vitesse et ne vaut pas forcément 50 Hz. On l'estime donc sur le signal ; une
    interpolation parabolique affine le pic sous la résolution FFT.
    """
    freqs, amp = amplitude_spectrum(x, fs)
    band = (freqs >= fmin) & (freqs <= min(fmax, fs / 2))
    if not np.any(band):
        raise ValueError("Bande de recherche de la fréquence d'alimentation vide.")
    idx = np.flatnonzero(band)
    k = idx[np.argmax(amp[idx])]
    if 0 < k < len(amp) - 1:
        a, b, c = np.log(amp[k - 1] + 1e-20), np.log(amp[k] + 1e-20), np.log(amp[k + 1] + 1e-20)
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
        return float(freqs[k] + delta * (freqs[1] - freqs[0]))
    return float(freqs[k])


def current_sideband_markers(f_supply: float, fault_freqs: dict, orders=(1, 2)) -> list[dict]:
    """Fréquences attendues dans le courant : f_s ± k·f_défaut (modèle de Schoen, 1995)."""
    markers = [{"label": "f_s", "freq": f_supply, "kind": "supply"}]
    for name in ("fr", "BPFO", "BPFI"):
        f = fault_freqs.get(name)
        if f is None:
            continue
        for k in orders:
            for sign, s in ((-1, "−"), (1, "+")):
                fk = f_supply + sign * k * f
                if fk > 0:
                    lab = f"f_s{s}{k if k > 1 else ''}{name}"
                    markers.append({"label": lab, "freq": float(fk), "kind": name})
    return markers


# --------------------------------------------------------------------------------
# Réduction pour l'affichage
# --------------------------------------------------------------------------------
def minmax_downsample(x: np.ndarray, fs: float, t0: float = 0.0, max_points: int = 4000):
    """Réduit un signal long pour l'affichage en gardant min et max de chaque paquet.

    Contrairement à une simple décimation, les pics (chocs) restent visibles.
    Renvoie (temps en s, valeurs).
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    t = t0 + np.arange(n) / fs
    if n <= max_points:
        return t, x
    n_bins = max_points // 2
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    out_t = np.empty(2 * n_bins)
    out_x = np.empty(2 * n_bins)
    for i in range(n_bins):
        a, b = edges[i], edges[i + 1]
        seg = x[a:b]
        i_min, i_max = int(np.argmin(seg)), int(np.argmax(seg))
        first, second = (i_min, i_max) if i_min < i_max else (i_max, i_min)
        out_t[2 * i], out_x[2 * i] = t[a + first], seg[first]
        out_t[2 * i + 1], out_x[2 * i + 1] = t[a + second], seg[second]
    return out_t, out_x


def reduce_spectrum(freqs: np.ndarray, amp: np.ndarray, max_points: int = 3000):
    """Réduit un spectre pour l'affichage en gardant le maximum de chaque paquet de raies."""
    n = freqs.size
    if n <= max_points:
        return freqs, amp
    edges = np.linspace(0, n, max_points + 1).astype(int)
    idx = np.array([a + int(np.argmax(amp[a:b])) for a, b in zip(edges[:-1], edges[1:]) if b > a])
    return freqs[idx], amp[idx]
