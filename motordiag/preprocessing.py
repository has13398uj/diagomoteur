"""Prétraitement commun à l'entraînement (notebook) et à l'inférence (dashboard).

Principe : il n'y a qu'UNE implémentation de chaque étape.
  1. `resample`        : ré-échantillonnage polyphase avec filtre anti-repliement ;
  2. `window_starts`   : position des fenêtres (centre en secondes -> indice de début) ;
  3. `extract_windows` : découpage en fenêtres.
La normalisation (moyenne / écart-type calculés sur les roulements d'entraînement) et la
représentation (signal brut, FFT, enveloppe, STFT) font partie du modèle PyTorch
(voir `models.py`) : elles sont donc sauvegardées avec les poids et appliquées à
l'identique dans le dashboard.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction

import numpy as np
from scipy import signal as sps


@dataclass
class ModalitySpec:
    """Paramètres d'entrée d'une modalité (vibration ou courant)."""
    fs: int            # fréquence d'échantillonnage après ré-échantillonnage (Hz)
    window: int        # longueur de fenêtre (échantillons à `fs`)

    @property
    def duration(self) -> float:
        return self.window / self.fs

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ModalitySpec":
        return ModalitySpec(fs=int(d["fs"]), window=int(d["window"]))


def resample(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Ré-échantillonne le long du dernier axe (polyphase + anti-repliement, float32)."""
    x = np.asarray(x)
    if fs_in <= 0 or fs_out <= 0:
        raise ValueError("Les fréquences d'échantillonnage doivent être > 0.")
    if abs(fs_in - fs_out) < 1e-9:
        return x.astype(np.float32, copy=True)
    ratio = Fraction(float(fs_out) / float(fs_in)).limit_denominator(1000)
    y = sps.resample_poly(x.astype(np.float64), ratio.numerator, ratio.denominator, axis=-1)
    return y.astype(np.float32)


def window_starts(centers_s: np.ndarray, fs: float, window: int) -> np.ndarray:
    """Indice de début de chaque fenêtre centrée sur `centers_s` (secondes)."""
    centers_s = np.asarray(centers_s, dtype=np.float64)
    return np.round(centers_s * fs).astype(np.int64) - window // 2


def eval_centers(duration_s: float, max_window_s: float, n_windows: int | None = None,
                 hop_s: float | None = None) -> np.ndarray:
    """Centres des fenêtres d'évaluation, régulièrement espacés sur le signal.

    Soit un nombre fixe de fenêtres (`n_windows`, utilisé pour comparer équitablement des
    configurations de longueurs différentes), soit un pas fixe (`hop_s`).
    """
    lo = max_window_s / 2.0
    hi = duration_s - max_window_s / 2.0
    if hi < lo - 1e-9:
        raise ValueError(
            f"Signal trop court ({duration_s:.3f} s) pour une fenêtre de {max_window_s:.3f} s.")
    if hi <= lo:
        return np.array([lo])
    if n_windows is not None:
        return np.linspace(lo, hi, int(n_windows))
    if hop_s is None or hop_s <= 0:
        raise ValueError("Donner n_windows ou hop_s.")
    return np.arange(lo, hi + 1e-9, hop_s)


def extract_windows(x: np.ndarray, fs: float, centers_s: np.ndarray, window: int) -> np.ndarray:
    """Découpe des fenêtres (n, window) d'un signal 1D déjà à la fréquence `fs`."""
    x = np.asarray(x, dtype=np.float32)
    starts = window_starts(centers_s, fs, window)
    if starts.min() < 0 or starts.max() + window > x.size:
        raise ValueError("Une fenêtre dépasse les bornes du signal.")
    idx = starts[:, None] + np.arange(window)[None, :]
    return x[idx]


def prepare_modality(x: np.ndarray, fs_in: float, spec: ModalitySpec) -> np.ndarray:
    """Signal brut -> signal ré-échantillonné à `spec.fs` (float32)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if not np.all(np.isfinite(x)):
        raise ValueError("Le signal contient des valeurs non finies (NaN ou infini).")
    return resample(x, fs_in, spec.fs)
