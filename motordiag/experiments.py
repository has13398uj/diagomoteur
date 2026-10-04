"""Outils d'expérimentation pour le notebook : banques de signaux, exécutions reprenables.

Chaque expérience est identifiée par une empreinte (hash) de sa configuration : une
configuration déjà entraînée n'est jamais relancée (résultat relu sur Drive), et une
exécution interrompue reprend à son dernier epoch.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import SignalBank, resample_array
from .signal_analysis import bandpass
from .training import train_run
from .utils import load_json, save_json


def cfg_hash(*objs) -> str:
    s = json.dumps(objs, sort_keys=True, default=str)
    return hashlib.sha1(s.encode()).hexdigest()[:10]


def _lowpass_rows(arr, fs, fc):
    from scipy import signal as sps
    sos = sps.butter(4, fc, btype="low", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, np.asarray(arr, dtype=np.float64), axis=-1).astype(np.float16)


class BankManager:
    """Fournit des `SignalBank` (sur GPU) aux fréquences demandées, avec mise en cache RAM."""

    def __init__(self, cache: dict, device):
        self.raw = {"vib": cache["vib"], "cur": cache["cur"]}
        self.fs_raw = cache["fs"]
        self.device = device
        self._arrays: dict = {}
        self._bank = None
        self._bank_key = None

    def array(self, modality: str, fs: int, filters: str | None = None):
        key = (modality, fs, filters)
        if key not in self._arrays:
            a = np.asarray(self.raw[modality])
            if filters == "original":
                # Filtres prévus dans H_M_100 : vibration 100–5000 Hz, courant passe-bas 3 kHz.
                if modality == "vib":
                    a = np.stack([bandpass(r, self.fs_raw, 100, 5000).astype(np.float16) for r in a])
                else:
                    a = np.concatenate([_lowpass_rows(a[i:i + 64], self.fs_raw, 3000) for i in range(0, a.shape[0], 64)])
            out = resample_array(a, self.fs_raw, fs)
            if isinstance(out, np.memmap):  # lecture complète en RAM (accès aléatoire lent sur Drive)
                out = np.array(out)
            self._arrays[key] = out
        return self._arrays[key]

    def bank(self, model_cfg: dict, filters: str | None = None) -> SignalBank:
        need = tuple(sorted((m, int(model_cfg["inputs"][m]["fs"])) for m in model_cfg["modalities"]))
        key = (need, filters)
        if key != self._bank_key:
            self._bank = None  # libère la mémoire GPU de la banque précédente
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            arrays = {m: self.array(m, fs, filters) for m, fs in need}
            self._bank = SignalBank(arrays, {m: fs for m, fs in need}, self.device)
            self._bank_key = key
        return self._bank

    def drop_cached(self, keep_raw_rates=True):
        """Libère la RAM (tableaux ré-échantillonnés / filtrés) et la banque GPU courante."""
        self._arrays = {k: v for k, v in self._arrays.items() if keep_raw_rates and k[1] == self.fs_raw and k[2] is None}
        self._bank, self._bank_key = None, None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def run_or_load(tag: str, model_cfg: dict, train_cfg: dict, banks: BankManager, meta, split: dict, work_dir,
                device, log=print, filters: str | None = None, fixed_train=None, fixed_val=None) -> dict:
    """Entraîne (ou relit) une configuration. Renvoie le résumé + le dossier du run."""
    h = cfg_hash(model_cfg, train_cfg, {k: list(map(int, v)) for k, v in split.items()}, filters,
                 None if fixed_train is None else cfg_hash(fixed_train[0].tolist()[:50], len(fixed_train[0])))
    run_dir = Path(work_dir) / f"{tag}_{h}"
    done = run_dir / "train_summary.json"
    if done.exists():
        summ = load_json(done)
        summ["run_dir"] = str(run_dir)
        summ["cached"] = True
        return summ
    log(f"[{tag}] entraînement -> {run_dir.name}")
    bank = banks.bank(model_cfg, filters)
    res = train_run(model_cfg, train_cfg, bank, meta, split, run_dir=run_dir, device=device, log=log,
                    fixed_train=fixed_train, fixed_val=fixed_val)
    summ = load_json(done)
    summ["run_dir"] = str(run_dir)
    summ["cached"] = False
    summ["_model"] = res["model"]
    return summ


def set_path(d: dict, path: str, value):
    """Modifie une clé imbriquée : set_path(cfg, 'inputs.vib.fs', 16000)."""
    d = copy.deepcopy(d)
    cur = d
    keys = path.split(".")
    for k in keys[:-1]:
        cur = cur[k]
    cur[keys[-1]] = value
    return d


def selection_score(summary: dict, k: int = 3) -> float:
    """Score de sélection d'une configuration : moyenne des k meilleurs F1 de validation.

    Avec peu de roulements en validation, le F1 varie fortement d'un epoch à l'autre ;
    le seul meilleur epoch récompense un pic chanceux. La moyenne des 3 meilleurs epochs
    départage les configurations sur un niveau atteint de façon répétée.
    """
    vals = [v for v in summary.get("history", {}).get("val_macro_f1", []) if v is not None]
    if not vals:
        return float(summary.get("best_val_f1") or 0.0)
    return float(np.mean(sorted(vals, reverse=True)[:k]))


class StepwiseSearch:
    """Optimisation « un changement à la fois », jugée sur le F1 macro de VALIDATION.

    Pour chaque étape, on essaie plusieurs variantes d'un seul facteur ; la meilleure
    n'est gardée que si elle dépasse la configuration courante d'au moins `threshold`.
    Le score comparé est `selection_score` (moyenne des 3 meilleurs epochs de validation).
    """

    def __init__(self, name, model_cfg, train_cfg, runner, threshold=0.005, log=print):
        self.name = name
        self.model_cfg, self.train_cfg = copy.deepcopy(model_cfg), copy.deepcopy(train_cfg)
        self.runner, self.threshold, self.log = runner, threshold, log
        self.history = []
        self.current = None

    def _score(self, mcfg, tcfg, label):
        s = self.runner(mcfg, tcfg)
        f1 = selection_score(s)
        self.log(f"   {label:<45} F1 val (moy. 3 meilleurs epochs) = {f1:.4f} ; meilleur epoch = "
                 f"{s['best_val_f1']:.4f}{' (déjà calculé)' if s.get('cached') else ''}")
        return f1, s

    def baseline(self):
        f1, s = self._score(self.model_cfg, self.train_cfg, "référence")
        self.current = f1
        self.history.append({"search": self.name, "step": "référence", "variant": "référence",
                             "val_macro_f1": f1, "val_macro_f1_best_epoch": s["best_val_f1"],
                             "kept": True, "run_dir": s["run_dir"]})
        return f1

    def step(self, step_name: str, variants: dict):
        """variants : {libellé: (fonction(model_cfg, train_cfg) -> (model_cfg, train_cfg))}."""
        if self.current is None:
            self.baseline()
        self.log(f"-- {self.name} / étape : {step_name} (courant = {self.current:.4f})")
        results = []
        for label, fn in variants.items():
            mcfg, tcfg = fn(copy.deepcopy(self.model_cfg), copy.deepcopy(self.train_cfg))
            f1, s = self._score(mcfg, tcfg, label)
            results.append((f1, label, mcfg, tcfg, s))
        best = max(results, key=lambda r: r[0])
        keep = best[0] >= self.current + self.threshold
        for f1, label, mcfg, tcfg, s in results:
            self.history.append({"search": self.name, "step": step_name, "variant": label, "val_macro_f1": f1,
                                 "val_macro_f1_best_epoch": s["best_val_f1"],
                                 "kept": bool(keep and label == best[1]), "run_dir": s["run_dir"]})
        if keep:
            self.model_cfg, self.train_cfg, self.current = best[2], best[3], best[0]
            self.log(f"   => gardé : {best[1]} ({best[0]:.4f})")
        else:
            self.log(f"   => aucun gain ≥ {self.threshold:.3f} : configuration inchangée")
        return keep
