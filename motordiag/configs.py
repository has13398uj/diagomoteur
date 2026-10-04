"""Configurations de référence (point de départ de l'optimisation dans le notebook)."""
from __future__ import annotations

import copy

from .constants import CLASS_NAMES_FR

REFERENCE_INPUTS = {
    # Vibration : 64 kHz conservé (les résonances excitées par les chocs sont hautes en
    # fréquence) ; 4096 échantillons = 64 ms ≈ 1,6 tour d'arbre à 1500 tr/min.
    "vib": {"fs": 64000, "window": 4096, "repr": "raw", "norm": "global"},
    # Courant : l'information utile est autour du fondamental (60–100 Hz) et de ses raies
    # latérales ; 8 kHz suffisent et une fenêtre longue (0,5 s) donne une résolution de 2 Hz.
    "cur": {"fs": 8000, "window": 4096, "repr": "fft", "norm": "global", "fmax": 2000.0},
}


def make_model_cfg(arch: str = "resnet1d", modalities=("vib", "cur"), inputs: dict | None = None,
                   width: int = 32, emb_dim: int = 128, dropout: float = 0.2, fusion: str = "attention",
                   modality_dropout: float = 0.0) -> dict:
    ins = copy.deepcopy(REFERENCE_INPUTS)
    for m, d in (inputs or {}).items():
        ins.setdefault(m, {}).update(d)
    return {
        "arch": arch,
        "modalities": list(modalities),
        "inputs": {m: ins[m] for m in modalities},
        "width": width,
        "emb_dim": emb_dim,
        "dropout": dropout,
        "fusion": fusion,
        "modality_dropout": modality_dropout,
        "n_classes": len(CLASS_NAMES_FR),
        "norm_stats": {},
    }


def original_model_cfg() -> dict:
    """Modèle du notebook d'origine : fenêtres de 1024 échantillons à 64 kHz, 2 capteurs."""
    return {
        "arch": "original",
        "modalities": ["vib", "cur"],
        "inputs": {"vib": {"fs": 64000, "window": 1024, "repr": "raw", "norm": "global"},
                   "cur": {"fs": 64000, "window": 1024, "repr": "raw", "norm": "global"}},
        "width": 32,
        "dropout": 0.4,
        "n_classes": len(CLASS_NAMES_FR),
        "norm_stats": {},
    }
