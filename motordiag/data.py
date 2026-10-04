"""Données Paderborn côté entraînement : cache, découpage par roulement, fenêtres sur GPU.

Le cache stocke chaque enregistrement ENTIER (4 s) une seule fois :
    vib.npy : (n_fichiers, N) à 64 kHz      cur.npy : (n_fichiers, N) à 64 kHz
    meta.csv : une ligne par fichier (roulement, classe, condition, n° d'enregistrement...)
Les fenêtres ne sont pas stockées : elles sont découpées à la volée (indices), ce qui
évite de dupliquer les données et permet de changer longueur / recouvrement sans
reconstruire le cache.
"""
from __future__ import annotations

import csv
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .constants import (CLASS_NAMES_FR, PADERBORN_BEARINGS, PADERBORN_CUR_CHANNEL, PADERBORN_FS,
                        PADERBORN_VIB_CHANNEL)
from .io import load_paderborn_file, paderborn_rpm, parse_paderborn_name
from .preprocessing import resample, window_starts

CACHE_SAMPLES = 256000  # 4,0 s à 64 kHz (les fichiers Paderborn font ~256 823 échantillons)
META_FIELDS = ["file_idx", "file", "bearing", "label", "class_name", "origin", "condition",
               "recording", "rpm_nominal", "rpm_measured", "n_samples_raw"]


# --------------------------------------------------------------------------------
# Inventaire et cache
# --------------------------------------------------------------------------------
def scan_paderborn(data_dir) -> list[dict]:
    """Liste les fichiers .mat Paderborn (dossiers par roulement ou à plat), triés."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dossier de données introuvable : {data_dir}")
    files = []
    for p in sorted(data_dir.rglob("*.mat")):
        info = parse_paderborn_name(p.name)
        if info is None or info.get("bearing") not in PADERBORN_BEARINGS:
            continue
        files.append({"path": str(p), "file": p.name, **info})
    files.sort(key=lambda r: (r["bearing"], r["condition"], r["recording"]))
    return files


def _load_one(rec: dict, n_samples: int):
    try:
        d = load_paderborn_file(rec["path"])
    except Exception as e:  # noqa: BLE001
        return rec, None, None, None, f"lecture impossible : {e}"
    ch = d["channels"]
    if PADERBORN_VIB_CHANNEL not in ch or PADERBORN_CUR_CHANNEL not in ch:
        return rec, None, None, None, "voie vibration ou courant absente"
    v, c = ch[PADERBORN_VIB_CHANNEL], ch[PADERBORN_CUR_CHANNEL]
    if v["fs"] != PADERBORN_FS or c["fs"] != PADERBORN_FS:
        return rec, None, None, None, f"fréquence inattendue ({v['fs']}, {c['fs']})"
    n_raw = min(v["data"].size, c["data"].size)
    if n_raw < n_samples:
        return rec, None, None, None, f"trop court ({n_raw} < {n_samples})"
    extra = {"rpm_measured": paderborn_rpm(ch), "n_samples_raw": n_raw}
    return rec, v["data"][:n_samples].astype(np.float32), c["data"][:n_samples].astype(np.float32), extra, None


def build_cache(data_dir, cache_dir, n_samples: int = CACHE_SAMPLES, dtype=np.float16,
                n_workers: int = 8, tmp_dir: str | None = None, files: list[dict] | None = None,
                log=print) -> dict:
    """Lit tous les .mat et écrit vib.npy, cur.npy, meta.csv dans `cache_dir`.

    Aucun filtrage ni bruit n'est ajouté : le cache contient les signaux bruts. Les
    tableaux sont écrits d'abord sur le disque local (`tmp_dir`), puis copiés (Drive).
    float16 divise la taille par 2 ; la quantification (~-66 dB par échantillon) reste
    bien sous le niveau des raies utiles, car le bruit se répartit sur toutes les raies FFT.
    """
    files = files if files is not None else scan_paderborn(data_dir)
    if not files:
        raise RuntimeError(f"Aucun fichier Paderborn trouvé dans {data_dir}")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tmp_dir) if tmp_dir else cache_dir
    tmp.mkdir(parents=True, exist_ok=True)
    n = len(files)
    vib = np.lib.format.open_memmap(tmp / "vib.npy", mode="w+", dtype=dtype, shape=(n, n_samples))
    cur = np.lib.format.open_memmap(tmp / "cur.npy", mode="w+", dtype=dtype, shape=(n, n_samples))
    meta, errors, k = [], [], 0
    try:
        from tqdm.auto import tqdm
        progress = tqdm(total=n, desc="Lecture des .mat")
    except ImportError:  # pragma: no cover
        progress = None
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for rec, v, c, extra, err in ex.map(lambda r: _load_one(r, n_samples), files):
            if progress:
                progress.update(1)
            if err:
                errors.append((rec["file"], err))
                continue
            vib[k], cur[k] = v, c
            label, origin = PADERBORN_BEARINGS[rec["bearing"]]
            meta.append({
                "file_idx": k, "file": rec["file"], "bearing": rec["bearing"], "label": label,
                "class_name": CLASS_NAMES_FR[label], "origin": origin, "condition": rec["condition"],
                "recording": rec["recording"], "rpm_nominal": rec.get("rpm_nominal"),
                "rpm_measured": extra["rpm_measured"], "n_samples_raw": extra["n_samples_raw"]})
            k += 1
    if progress:
        progress.close()
    vib.flush(); cur.flush()
    del vib, cur
    if k < n:  # fichiers rejetés : on retaille les tableaux
        for name in ("vib.npy", "cur.npy"):
            a = np.load(tmp / name, mmap_mode="r")
            b = np.lib.format.open_memmap(tmp / f"_{name}", mode="w+", dtype=dtype, shape=(k, n_samples))
            b[:] = a[:k]
            b.flush(); del a, b
            os.replace(tmp / f"_{name}", tmp / name)
    if tmp != cache_dir:
        for name in ("vib.npy", "cur.npy"):
            shutil.copyfile(tmp / name, cache_dir / name)
    write_meta(meta, cache_dir / "meta.csv")
    for f, e in errors:
        log(f"  fichier ignoré : {f} ({e})")
    log(f"Cache : {k} fichiers valides / {n} ; {len(errors)} ignorés.")
    return {"n_files": k, "errors": errors}


def write_meta(meta: list[dict], path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=META_FIELDS)
        w.writeheader()
        for row in meta:
            w.writerow({k: row.get(k) for k in META_FIELDS})


def read_meta(path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["file_idx"] = int(r["file_idx"]); r["label"] = int(r["label"])
            r["recording"] = int(r["recording"])
            for key in ("rpm_nominal", "rpm_measured"):
                r[key] = float(r[key]) if r.get(key) not in (None, "", "None") else None
            r["n_samples_raw"] = int(float(r["n_samples_raw"]))
            rows.append(r)
    return rows


def load_cache(cache_dir, mmap: bool = True) -> dict:
    cache_dir = Path(cache_dir)
    mode = "r" if mmap else None
    return {"vib": np.load(cache_dir / "vib.npy", mmap_mode=mode),
            "cur": np.load(cache_dir / "cur.npy", mmap_mode=mode),
            "meta": read_meta(cache_dir / "meta.csv"), "fs": PADERBORN_FS}


def resample_array(arr: np.ndarray, fs_in: int, fs_out: int, chunk: int = 32, dtype=np.float16) -> np.ndarray:
    """Ré-échantillonne un tableau (n_fichiers, N) par paquets (mémoire limitée)."""
    if fs_in == fs_out:
        return np.asarray(arr)
    first = resample(np.asarray(arr[:1], dtype=np.float32), fs_in, fs_out)
    out = np.empty((arr.shape[0], first.shape[-1]), dtype=dtype)
    for i in range(0, arr.shape[0], chunk):
        out[i:i + chunk] = resample(np.asarray(arr[i:i + chunk], dtype=np.float32), fs_in, fs_out)
    return out


# --------------------------------------------------------------------------------
# Découpage par roulement (aucun roulement à la fois en apprentissage et en test)
# --------------------------------------------------------------------------------
def bearing_table(meta: list[dict]) -> dict:
    """{roulement: (classe, origine)} pour les roulements présents dans le cache."""
    out = {}
    for r in meta:
        out[r["bearing"]] = (r["label"], r["origin"])
    return out


def make_bearing_folds(meta: list[dict], n_folds: int = 3, seed: int = 0, val_fraction: float = 0.25) -> list[dict]:
    """Validation croisée stratifiée et groupée par roulement.

    Pour chaque classe, les roulements sont mélangés (graine fixe), triés par origine
    (artificiel / réel) pour alterner les deux, puis distribués à tour de rôle entre les
    plis : chaque pli de test contient des roulements de chaque classe.
    Dans les roulements restants, une partie (`val_fraction`, au moins 3 roulements dans
    la classe) sert de validation (arrêt anticipé, choix des réglages). Avec seulement
    3 roulements "combinés", aucune validation n'est prise dans cette classe : les 2
    restants servent à l'apprentissage.
    """
    rng = np.random.default_rng(seed)
    table = bearing_table(meta)
    by_class: dict[int, list[str]] = {}
    for b, (label, origin) in sorted(table.items()):
        by_class.setdefault(label, []).append(b)
    test_sets = [[] for _ in range(n_folds)]
    offset = 0
    for label in sorted(by_class):
        bs = list(by_class[label])
        rng.shuffle(bs)
        bs.sort(key=lambda b: table[b][1])  # tri stable par origine -> alternance
        for i, b in enumerate(bs):
            test_sets[(i + offset) % n_folds].append(b)
        offset += len(bs)
    folds = []
    for k in range(n_folds):
        test = sorted(test_sets[k])
        pool = [b for b in table if b not in test]
        val = []
        for label in sorted(by_class):
            cands = [b for b in pool if table[b][0] == label]
            n_val = int(round(val_fraction * len(cands))) if len(cands) >= 3 else 0
            if n_val:
                cands = list(rng.permutation(sorted(cands)))
                val += [str(b) for b in cands[:n_val]]
        train = sorted(b for b in pool if b not in val)
        folds.append({"fold": k, "train": train, "val": sorted(val), "test": test})
    return folds


def files_of(meta: list[dict], bearings) -> np.ndarray:
    bs = set(bearings)
    return np.array([r["file_idx"] for r in meta if r["bearing"] in bs], dtype=np.int64)


def check_no_overlap(fold: dict) -> None:
    tr, va, te = set(fold["train"]), set(fold["val"]), set(fold["test"])
    assert not (tr & te) and not (va & te) and not (tr & va), "Un roulement apparaît dans deux ensembles !"


# --------------------------------------------------------------------------------
# Banque de signaux sur GPU et découpage de fenêtres
# --------------------------------------------------------------------------------
class SignalBank:
    """Signaux entiers en mémoire (GPU si possible) ; découpe des fenêtres par indices."""

    def __init__(self, arrays: dict, fs: dict, device="cpu"):
        import torch
        self.device = torch.device(device)
        self.fs = dict(fs)
        self.data = {}
        for m, a in arrays.items():
            a = np.ascontiguousarray(a)
            if not a.flags.writeable:  # tableau en lecture seule (memmap) : copie pour PyTorch
                a = a.copy()
            self.data[m] = torch.from_numpy(a).to(self.device)
        self.n_samples = {m: t.shape[1] for m, t in self.data.items()}
        self.duration = min(self.n_samples[m] / self.fs[m] for m in self.data)

    def windows(self, modality: str, file_idx: np.ndarray, centers_s: np.ndarray, window: int):
        import torch
        fs = self.fs[modality]
        starts = window_starts(centers_s, fs, window)
        if starts.min() < 0 or starts.max() + window > self.n_samples[modality]:
            raise ValueError("Fenêtre hors du signal (centre trop près du bord).")
        fi = torch.as_tensor(file_idx, device=self.device)
        st = torch.as_tensor(starts, device=self.device)
        idx = st[:, None] + torch.arange(window, device=self.device)[None, :]
        return self.data[modality][fi[:, None], idx].float()


def random_centers(file_idx: np.ndarray, per_file: int, duration: float, half_window: float,
                   rng: np.random.Generator):
    """Centres aléatoires (décalage temporel aléatoire = augmentation naturelle)."""
    f = np.repeat(file_idx, per_file)
    c = rng.uniform(half_window, duration - half_window, size=f.size)
    return f, c


def grid_centers(file_idx: np.ndarray, per_file: int, duration: float, half_window: float):
    """Centres régulièrement espacés : mêmes positions pour toutes les configurations."""
    c1 = np.linspace(half_window, duration - half_window, per_file)
    return np.repeat(file_idx, per_file), np.tile(c1, file_idx.size)
