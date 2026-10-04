"""Petits utilitaires : graines aléatoires, JSON robuste."""
from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Fixe les graines Python / NumPy / PyTorch pour des résultats reproductibles."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj, path) -> None:
    """Écrit un JSON (NaN -> null) de façon atomique."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
