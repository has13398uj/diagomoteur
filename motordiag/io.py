"""Lecture des fichiers de signaux : Paderborn .mat (format natif), CSV, NPY.

Toutes les erreurs de format lèvent `SignalFileError` avec un message en français,
affiché tel quel dans le dashboard.
"""
from __future__ import annotations

import io as _io
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.io

from .constants import (OPERATING_CONDITIONS, PADERBORN_BEARINGS, PADERBORN_CUR_CHANNEL,
                        PADERBORN_FS, PADERBORN_SPEED_CHANNEL, PADERBORN_VIB_CHANNEL)

MIN_SAMPLES = 256

_PADERBORN_RE = re.compile(r"(N\d{2}_M\d{2}_F\d{2})_(K\d{3}|K[AIB]\d{2})_(\d+)")


class SignalFileError(ValueError):
    """Fichier illisible ou non conforme (message destiné à l'utilisateur)."""


@dataclass
class SignalData:
    values: np.ndarray                 # signal 1D (float64)
    fs: float | None                   # Hz (None = inconnue, à saisir)
    unit: str | None = None            # unité physique si connue
    source: str = ""                   # nom du fichier
    channel: str = ""                  # voie lue
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float | None:
        return None if not self.fs else self.values.size / self.fs


# --------------------------------------------------------------------------------
# Paderborn
# --------------------------------------------------------------------------------
def parse_paderborn_name(name: str) -> dict | None:
    """'N15_M07_F10_KA04_1.mat' -> condition, roulement, n° d'enregistrement, classe..."""
    m = _PADERBORN_RE.search(Path(name).name)
    if not m:
        return None
    cond, bearing, rec = m.group(1), m.group(2), int(m.group(3))
    info = {"condition": cond, "bearing": bearing, "recording": rec}
    if cond in OPERATING_CONDITIONS:
        info["rpm_nominal"] = OPERATING_CONDITIONS[cond]["rpm"]
    if bearing in PADERBORN_BEARINGS:
        label, origin = PADERBORN_BEARINGS[bearing]
        info["label"], info["origin"] = label, origin
    return info


def _raster_to_fs(raster: str | None, n: int, n_ref: int | None) -> float | None:
    """Fréquence d'échantillonnage d'une voie Paderborn d'après son 'Raster'."""
    if raster:
        r = raster.lower()
        if "hostservice" in r:
            return float(PADERBORN_FS)
        m = re.search(r"(\d+(?:\.\d+)?)\s*khz", r)
        if m:
            return float(m.group(1)) * 1000.0
        m = re.search(r"(\d+(?:\.\d+)?)\s*hz", r)
        if m:
            return float(m.group(1))
    if n_ref:
        # Déduction par la longueur : toutes les voies couvrent la même durée (~4 s).
        return PADERBORN_FS * n / n_ref
    return None


def _field_str(entry, name: str) -> str | None:
    try:
        v = entry[name]
    except (ValueError, KeyError, IndexError):
        return None
    v = np.asarray(v).ravel()
    if v.size == 0:
        return None
    s = v[0]
    return str(s.item() if hasattr(s, "item") else s).strip() or None


def read_paderborn_struct(mat: dict) -> dict | None:
    """Renvoie {nom_voie: {"data", "fs", "unit"}} si `mat` a la structure Paderborn."""
    keys = [k for k in mat if not k.startswith("_")]
    for k in keys:
        s = mat[k]
        if not (isinstance(s, np.ndarray) and s.dtype.names and "Y" in s.dtype.names):
            continue
        rec = s[0, 0]
        ys = rec["Y"]
        if not (ys.dtype.names and "Data" in ys.dtype.names and "Name" in ys.dtype.names):
            continue
        channels = {}
        for i in range(ys.shape[1]):
            e = ys[0, i]
            name = _field_str(e, "Name")
            data = np.asarray(e["Data"]).ravel().astype(np.float64)
            channels[name] = {"data": data, "raster": _field_str(e, "Raster"),
                              "unit": _field_str(e, "Unit")}
        n_ref = None
        if PADERBORN_VIB_CHANNEL in channels:
            n_ref = channels[PADERBORN_VIB_CHANNEL]["data"].size
        for ch in channels.values():
            ch["fs"] = _raster_to_fs(ch.pop("raster"), ch["data"].size, n_ref)
        return channels
    return None


def load_paderborn_file(path) -> dict:
    """Charge un .mat Paderborn depuis le disque : voies + métadonnées du nom de fichier."""
    mat = scipy.io.loadmat(str(path))
    channels = read_paderborn_struct(mat)
    if channels is None:
        raise SignalFileError(f"{Path(path).name} : structure Paderborn introuvable.")
    return {"channels": channels, "meta": parse_paderborn_name(Path(path).name) or {}}


def paderborn_rpm(channels: dict) -> float | None:
    """Vitesse moyenne (tr/min) d'après la voie 'speed' si elle existe."""
    sp = channels.get(PADERBORN_SPEED_CHANNEL)
    if sp is None or sp["data"].size == 0:
        return None
    v = float(np.median(sp["data"]))
    return v if v > 0 else None


# --------------------------------------------------------------------------------
# Lecture générique (dashboard)
# --------------------------------------------------------------------------------
_VIB_HINTS = ("vib", "acc", "de_time", "fe_time", "accel")
_CUR_HINTS = ("cur", "courant", "current", "phase", "i_", "ia", "amp")


def _check_signal(x: np.ndarray, source: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < MIN_SAMPLES:
        raise SignalFileError(f"{source} : signal trop court ({x.size} échantillons, minimum {MIN_SAMPLES}).")
    if not np.all(np.isfinite(x)):
        n_bad = int(np.sum(~np.isfinite(x)))
        raise SignalFileError(f"{source} : {n_bad} valeurs non numériques (NaN / infini) dans le signal.")
    if np.ptp(x) == 0:
        raise SignalFileError(f"{source} : le signal est constant (aucune variation).")
    return x


def _fs_from_time(t: np.ndarray) -> float | None:
    d = np.diff(np.asarray(t, dtype=np.float64))
    if d.size < 2 or np.any(d <= 0):
        return None
    dt = float(np.median(d))
    if np.max(np.abs(d - dt)) > 0.01 * dt:
        return None
    return 1.0 / dt


def _pick_column(names: list[str], kind: str) -> int | None:
    hints = _VIB_HINTS if kind == "vib" else _CUR_HINTS
    for i, n in enumerate(names):
        low = n.lower()
        if any(h in low for h in hints):
            return i
    return None


def _read_mat(content: bytes, source: str, kind: str) -> SignalData:
    try:
        mat = scipy.io.loadmat(_io.BytesIO(content))
    except NotImplementedError:
        raise SignalFileError(f"{source} : fichier MATLAB v7.3 (HDF5) non supporté ; "
                              "réenregistrez-le avec save(..., '-v7').")
    except Exception as e:  # noqa: BLE001
        raise SignalFileError(f"{source} : fichier .mat illisible ({e}).")

    channels = read_paderborn_struct(mat)
    if channels is not None:
        want = PADERBORN_VIB_CHANNEL if kind == "vib" else PADERBORN_CUR_CHANNEL
        if want not in channels:
            raise SignalFileError(f"{source} : voie '{want}' absente du fichier Paderborn "
                                  f"(voies : {', '.join(channels)}).")
        ch = channels[want]
        meta = parse_paderborn_name(source) or {}
        meta["format"] = "paderborn"
        rpm = paderborn_rpm(channels)
        if rpm:
            meta["rpm_measured"] = rpm
        return SignalData(_check_signal(ch["data"], source), ch["fs"], ch["unit"], source, want, meta)

    # .mat générique : une seule variable numérique longue (ex. CWRU : X097_DE_time).
    arrays = {k: np.asarray(v) for k, v in mat.items()
              if not k.startswith("_") and isinstance(v, np.ndarray) and v.dtype.kind in "fiu"}
    longs = {k: v for k, v in arrays.items() if v.size >= MIN_SAMPLES and min(v.shape) == 1}
    if not longs:
        raise SignalFileError(f"{source} : aucune variable 1D numérique trouvée dans le .mat.")
    name = None
    if len(longs) > 1:
        idx = _pick_column(list(longs), kind)
        if idx is None:
            raise SignalFileError(f"{source} : plusieurs signaux possibles ({', '.join(longs)}) ; "
                                  "gardez une seule variable dans le fichier.")
        name = list(longs)[idx]
    else:
        name = next(iter(longs))
    fs = None
    for key in ("fs", "Fs", "FS", "sampling_rate", "sample_rate"):
        if key in arrays and arrays[key].size == 1:
            fs = float(arrays[key].ravel()[0])
    meta = {"format": "mat"}
    for key in ("RPM", "rpm"):
        if key in arrays and arrays[key].size == 1:
            meta["rpm_measured"] = float(arrays[key].ravel()[0])
    return SignalData(_check_signal(longs[name], source), fs, None, source, name, meta)


def _read_csv(content: bytes, source: str, kind: str) -> SignalData:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = content.decode("latin-1")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < MIN_SAMPLES:
        raise SignalFileError(f"{source} : trop peu de lignes ({len(lines)}).")
    head, sample = lines[0], lines[min(1, len(lines) - 1)]
    head_is_text = re.search(r"[A-Za-zÀ-ÿ_]", head) is not None and not re.fullmatch(r"[\s\d.,;eE+\-\t]+", head)
    delim = None  # espaces / une seule colonne
    for cand in ("\t", ";", ","):
        # Le séparateur doit apparaître dans l'en-tête (s'il y en a un) et dans les données.
        if cand in sample and (not head_is_text or cand in head):
            delim = cand
            break
    decimal_comma = delim in (";", "\t", None) and re.search(r"\d,\d", sample) is not None

    def split(ln):
        parts = ln.split(delim) if delim else ln.split()
        return [p.strip().strip('"') for p in parts]

    first = split(lines[0])

    def is_num(s):
        try:
            float(s.replace(",", ".") if decimal_comma else s)
            return True
        except ValueError:
            return False

    header = None
    body = lines
    if not all(is_num(c) for c in first if c != ""):
        header, body = first, lines[1:]
    rows = []
    for k, ln in enumerate(body):
        parts = split(ln)
        if all(p == "" for p in parts):  # ligne vide « ,, » laissée par Excel
            continue
        if decimal_comma:
            parts = [p.replace(",", ".") for p in parts]
        try:
            rows.append([float(p) for p in parts if p != ""])
        except ValueError:
            raise SignalFileError(f"{source} : valeur non numérique à la ligne {k + 2 if header else k + 1}.")
    n_cols = min(len(r) for r in rows)
    if n_cols == 0:
        raise SignalFileError(f"{source} : aucune colonne numérique.")
    data = np.array([r[:n_cols] for r in rows], dtype=np.float64)

    fs = None
    col = 0
    names = list(header[:n_cols]) if header else []
    names += [f"col{i}" for i in range(len(names), n_cols)]
    if n_cols == 1 and re.search(r"\b(time|temps|t)\b", names[0].lower()) and np.all(np.diff(data[:, 0]) > 0):
        raise SignalFileError(f"{source} : le fichier ne contient qu'une colonne de temps (« {names[0]} »), "
                              "sans valeurs de signal. Il faut une colonne de mesures (vibration ou courant).")
    if n_cols >= 2:
        fs = _fs_from_time(data[:, 0])
        candidates = list(range(1, n_cols)) if fs else list(range(n_cols))
        pick = _pick_column([names[i] for i in candidates], kind)
        col = candidates[pick] if pick is not None else candidates[0]
        if fs:
            tname = names[0].lower()
            if "ms" in tname:
                fs *= 1000.0
    return SignalData(_check_signal(data[:, col], source), fs, None, source, names[col],
                      {"format": "csv", "time_column": bool(fs)})


def _read_npy(content: bytes, source: str, kind: str) -> SignalData:
    try:
        arr = np.load(_io.BytesIO(content), allow_pickle=False)
    except Exception as e:  # noqa: BLE001
        raise SignalFileError(f"{source} : fichier .npy illisible ({e}).")
    if isinstance(arr, np.lib.npyio.NpzFile):
        raise SignalFileError(f"{source} : archive .npz non supportée, utilisez un .npy.")
    if arr.dtype.kind not in "fiu":
        raise SignalFileError(f"{source} : le tableau doit être numérique (type {arr.dtype}).")
    arr = np.squeeze(arr)
    fs = None
    if arr.ndim == 2:
        if arr.shape[0] == 2 and arr.shape[1] > 2:
            arr = arr.T
        if arr.shape[1] == 2:
            fs = _fs_from_time(arr[:, 0])
            if fs is None:
                raise SignalFileError(f"{source} : tableau (N, 2) dont la 1re colonne n'est pas un temps "
                                      "régulier ; fournissez un tableau 1D.")
            arr = arr[:, 1]
        else:
            raise SignalFileError(f"{source} : tableau de forme {arr.shape} ; attendu un signal 1D "
                                  "ou un tableau (N, 2) [temps, valeur].")
    elif arr.ndim != 1:
        raise SignalFileError(f"{source} : tableau à {arr.ndim} dimensions ; attendu un signal 1D.")
    return SignalData(_check_signal(arr, source), fs, None, source, "npy", {"format": "npy"})


def load_signal(filename: str, content: bytes, kind: str) -> SignalData:
    """Lit un fichier téléversé. `kind` = 'vib' ou 'cur' (choix de la voie Paderborn)."""
    if kind not in ("vib", "cur"):
        raise ValueError("kind doit valoir 'vib' ou 'cur'.")
    ext = Path(filename).suffix.lower()
    if not content:
        raise SignalFileError(f"{filename} : fichier vide.")
    if ext == ".mat":
        return _read_mat(content, filename, kind)
    if ext in (".csv", ".txt"):
        return _read_csv(content, filename, kind)
    if ext == ".npy":
        return _read_npy(content, filename, kind)
    raise SignalFileError(f"{filename} : extension '{ext}' non supportée (formats acceptés : .mat, .csv, .txt, .npy).")


def load_signal_path(path, kind: str) -> SignalData:
    path = Path(path)
    return load_signal(path.name, path.read_bytes(), kind)
