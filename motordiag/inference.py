"""Inférence : charge artifacts/ et diagnostique des signaux avec le prétraitement d'entraînement.

Utilisé par le dashboard ET par le notebook (vérification finale) : c'est le même code.

    from motordiag.inference import Diagnoser
    diag = Diagnoser("artifacts")
    result = diag.diagnose(vib=signal_vib, cur=signal_cur, rpm=1500)

Deux familles de modèles peuvent être exportées :
- « torch »    : réseaux PyTorch (CNN, fusion par attention), fichiers models/<nom>.pt ;
- « features » : caractéristiques physiques + gradient boosting / forêt aléatoire (scikit-learn),
                 fichiers models/<nom>.joblib. Ils ont besoin de la vitesse de rotation
                 (fréquences de défaut BPFO, BPFI…).
Le modèle utilisé par défaut est celui qui a le meilleur F1 macro en test (`config["primary"]`).
"""
from __future__ import annotations

import csv
import datetime as _dt
from pathlib import Path

import contextlib

import numpy as np

try:  # PyTorch est facultatif : la version en ligne légère n'utilise que les modèles scikit-learn
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .constants import BEARING_6203, CLASS_KEYS, CLASS_NAMES_FR, PADERBORN_FS
from .features import window_features
from .io import SignalData
from .preprocessing import ModalitySpec, eval_centers, extract_windows, prepare_modality
from .utils import load_json, save_json

SCHEMA_VERSION = 2
MODEL_LABELS = {"fusion": "CNN fusion vibration + courant", "vib": "CNN vibration seule", "cur": "CNN courant seul"}
DEFAULT_PRIMARY = {"both": "fusion", "vib": "vib", "cur": "cur"}


class DiagnosisUnavailable(RuntimeError):
    """Diagnostic impossible avec les signaux / modèles disponibles (message utilisateur)."""


def library_versions() -> dict:
    import sklearn
    return {"torch": torch.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__}


# ================================================================================
# Export des artefacts (appelé par le notebook)
# ================================================================================
def export_artifacts(out_dir, models: dict, metrics: dict, history: dict | None = None,
                     confusion: dict | None = None, predictions: list[dict] | None = None,
                     extra_config: dict | None = None, demo: bool = False,
                     feature_models: dict | None = None, primary: dict | None = None) -> Path:
    """Écrit artifacts/ : config.json, models/*, metrics.json, history.json,
    confusion_matrix.json, test_predictions.csv.

    models         : {"fusion" | "vib" | "cur": torch.nn.Module (avec .cfg)}
    feature_models : {nom: {"estimator": modèle scikit-learn, "feature_cfg": {...}, "label": "..."}}
    primary        : modèle par défaut selon les signaux fournis {"both", "vib", "cur"}
    """
    import joblib
    from .models import count_parameters
    out = Path(out_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    model_entries = {}
    for name, model in models.items():
        path = out / "models" / f"{name}.pt"
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)
        model_entries[name] = {"kind": "torch", "file": f"models/{name}.pt", "label": MODEL_LABELS.get(name, name),
                               "model_cfg": model.cfg, "n_parameters": count_parameters(model)}
    for name, fm in (feature_models or {}).items():
        path = out / "models" / f"{name}.joblib"
        joblib.dump(fm["estimator"], path)
        model_entries[name] = {"kind": "features", "file": f"models/{name}.joblib", "label": fm["label"],
                               "feature_cfg": fm["feature_cfg"], "n_parameters": None}
    config = {
        "schema_version": SCHEMA_VERSION,
        "demo": bool(demo),
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "dataset": "Paderborn University Bearing DataCenter (KAt)",
        "class_keys": CLASS_KEYS,
        "class_names": CLASS_NAMES_FR,
        "raw_fs": PADERBORN_FS,
        "bearing_geometry": BEARING_6203,
        "models": model_entries,
        "primary": {**DEFAULT_PRIMARY, **(primary or {})},
        "versions": library_versions(),
        "inference": {"hop_fraction": 0.25, "max_windows": 400},
    }
    if extra_config:
        config.update(extra_config)
    save_json(config, out / "config.json")
    save_json({"demo": bool(demo), **metrics}, out / "metrics.json")
    if history is not None:
        save_json(history, out / "history.json")
    if confusion is not None:
        save_json(confusion, out / "confusion_matrix.json")
    if predictions:
        keys = list(predictions[0].keys())
        for r in predictions:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(out / "test_predictions.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(predictions)
    return out


def create_demo_artifacts(out_dir, seed: int = 0) -> Path:
    """Artefacts de DÉMONSTRATION : modèles NON entraînés (poids aléatoires), aucune métrique.

    Sert uniquement à tester la chaîne (lecture, prétraitement, affichage) avant d'avoir
    les vrais artefacts exportés par Colab. Le dashboard affiche un bandeau "DÉMO".
    """
    from .configs import make_model_cfg
    from .models import build_model
    torch.manual_seed(seed)
    models = {}
    for name, mods in (("fusion", ("vib", "cur")), ("vib", ("vib",)), ("cur", ("cur",))):
        cfg = make_model_cfg("resnet1d", mods, width=16, emb_dim=64)
        cfg["norm_stats"] = {m: {"mean": 0.0, "std": 1.0} for m in mods}
        m = build_model(cfg)
        m.eval()
        models[name] = m
    return export_artifacts(out_dir, models, metrics={"available": False,
                            "note": "Modèles non entraînés : aucune métrique."},
                            extra_config={"note": "DÉMO : poids aléatoires, prédictions sans valeur."}, demo=True)


# ================================================================================
# Modèle « caractéristiques physiques » (scikit-learn)
# ================================================================================
class FeatureModel:
    """Enveloppe d'un classifieur scikit-learn entraîné sur `features.window_features`.

    Reproduit exactement le calcul du notebook (section 5) : ré-échantillonnage, fenêtres
    centrées, caractéristiques vibration puis courant concaténées, NaN -> 0.
    """

    def __init__(self, estimator, feature_cfg: dict, n_classes: int):
        self.estimator = estimator
        self.cfg = {"modalities": list(feature_cfg["modalities"]), "inputs": feature_cfg["inputs"]}
        self.n_classes = n_classes

    def predict_proba(self, prepared: dict, centers: np.ndarray, rpm: float) -> np.ndarray:
        blocks = []
        for m in self.cfg["modalities"]:
            x, spec = prepared[m]
            w = extract_windows(x, spec.fs, centers, spec.window).astype(np.float64)
            f, _ = window_features(w, spec.fs, m, np.full(centers.size, rpm))
            blocks.append(f)
        X = np.nan_to_num(np.concatenate(blocks, axis=1))
        p = self.estimator.predict_proba(X)
        out = np.zeros((X.shape[0], self.n_classes))
        out[:, np.asarray(self.estimator.classes_, dtype=int)] = p
        return out


# ================================================================================
# Diagnostic
# ================================================================================
class Diagnoser:
    def __init__(self, artifacts_dir, device: str = "cpu"):
        self.dir = Path(artifacts_dir)
        cfg_path = self.dir / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"config.json introuvable dans {self.dir}")
        self.config = load_json(cfg_path)
        self.device = torch.device(device) if torch is not None else None
        self.skipped = []  # CNN ignorés si PyTorch n'est pas installé (version en ligne légère)
        self.class_names = self.config.get("class_names", CLASS_NAMES_FR)
        self.demo = bool(self.config.get("demo", False))
        self.primary = {**DEFAULT_PRIMARY, **self.config.get("primary", {})}
        self.models, self.labels, self.kinds, self.load_errors = {}, {}, {}, {}
        for name, entry in self.config.get("models", {}).items():
            path = self.dir / entry["file"]
            kind = entry.get("kind", "torch")
            if kind == "torch" and (torch is None or not path.exists()):
                self.skipped.append(name)  # version en ligne légère : CNN non inclus
                continue
            if not path.exists():
                continue
            try:
                if kind == "features":
                    import joblib
                    model = FeatureModel(joblib.load(path), entry["feature_cfg"], len(self.class_names))
                else:
                    from .models import build_model
                    model = build_model(entry["model_cfg"])
                    model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
                    model.to(self.device).eval()
            except Exception as e:  # noqa: BLE001
                wanted = self.config.get("versions", {}).get("sklearn")
                self.load_errors[name] = (f"Modèle « {entry.get('label', name)} » illisible ({type(e).__name__}). "
                                          + (f"Il a été exporté avec scikit-learn {wanted} : relancez start.py, qui "
                                             "installe cette version." if kind == "features" and wanted else ""))
                continue
            self.models[name] = model
            self.labels[name] = entry.get("label", MODEL_LABELS.get(name, name))
            self.kinds[name] = kind

    @property
    def available(self) -> list[str]:
        return list(self.models)

    def _modalities(self, name: str) -> list[str]:
        return list(self.models[name].cfg["modalities"])

    def choose_model(self, has_vib: bool, has_cur: bool, rpm_known: bool = True,
                     override: str | None = None) -> tuple[str, list[str]]:
        """Choisit le modèle selon les signaux fournis. Jamais de signal inventé."""
        notes: list[str] = []
        given = {m for m, ok in (("vib", has_vib), ("cur", has_cur)) if ok}
        if not given:
            raise DiagnosisUnavailable("Aucun signal fourni.")
        if override:
            if override not in self.models:
                raise DiagnosisUnavailable(self.load_errors.get(override, f"Modèle « {override} » indisponible."))
            missing = set(self._modalities(override)) - given
            if missing:
                raise DiagnosisUnavailable(
                    f"Le modèle « {self.labels[override]} » a besoin du signal "
                    f"{' et '.join('de vibration' if m == 'vib' else 'de courant' for m in sorted(missing))}.")
            if self.kinds[override] == "features" and not rpm_known:
                raise DiagnosisUnavailable(f"Le modèle « {self.labels[override]} » a besoin de la vitesse de rotation "
                                           "(fréquences de défaut) : saisissez-la.")
            return override, notes

        key = "both" if given == {"vib", "cur"} else next(iter(given))
        order = [self.primary.get(key), DEFAULT_PRIMARY[key]]
        if key == "both":
            order += [self.primary.get("vib"), "vib", self.primary.get("cur"), "cur"]
        for name in dict.fromkeys(n for n in order if n):
            if name not in self.models or not set(self._modalities(name)) <= given:
                continue
            if self.kinds[name] == "features" and not rpm_known:
                notes.append(f"Vitesse de rotation inconnue : le modèle « {self.labels[name]} » (meilleur en test) "
                             "a besoin des fréquences de défaut ; diagnostic avec un autre modèle.")
                continue
            if key == "both" and len(self._modalities(name)) == 1:
                notes.append(f"Pas de modèle à deux signaux disponible : diagnostic avec « {self.labels[name]} » ; "
                             "l'autre signal n'est pas utilisé par le modèle.")
            elif key != "both" and any(len(self._modalities(n)) == 2 for n in self.models):
                notes.append("Un seul signal fourni : le modèle de fusion a besoin des deux signaux ; "
                             f"diagnostic avec le modèle « {self.labels[name]} ».")
            return name, notes
        single = {"vib": "vibration", "cur": "courant"}
        if key != "both":
            raise DiagnosisUnavailable(
                f"Un seul signal fourni ({single[key]}), mais aucun modèle « {single[key]} seul » n'est disponible dans "
                "artifacts/. Le modèle de fusion a besoin des deux signaux (vibration ET courant).")
        raise DiagnosisUnavailable("Aucun modèle compatible avec les signaux fournis.")

    def diagnose(self, *args, **kwargs) -> dict:
        ctx = torch.no_grad() if torch is not None else contextlib.nullcontext()
        with ctx:
            return self._diagnose(*args, **kwargs)

    def _diagnose(self, vib: SignalData | None = None, cur: SignalData | None = None, rpm: float | None = None,
                 model: str | None = None, hop_s: float | None = None, max_windows: int | None = None) -> dict:
        if rpm is None:
            for s in (vib, cur):
                if s is not None and (s.meta.get("rpm_measured") or s.meta.get("rpm_nominal")):
                    rpm = float(s.meta.get("rpm_measured") or s.meta.get("rpm_nominal"))
                    break
        name, notes = self.choose_model(vib is not None, cur is not None, rpm_known=bool(rpm), override=model)
        mdl = self.models[name]
        cfg = mdl.cfg
        signals = {"vib": vib, "cur": cur}
        prepared, durations = {}, []
        for m in cfg["modalities"]:
            sig = signals[m]
            if sig.fs is None:
                raise DiagnosisUnavailable(f"Fréquence d'échantillonnage inconnue pour le signal « {m} ».")
            spec = ModalitySpec.from_dict(cfg["inputs"][m])
            x = prepare_modality(sig.values, sig.fs, spec)
            if abs(sig.fs - spec.fs) > 1e-6:
                label = "vibration" if m == "vib" else "courant"
                notes.append(f"Signal {label} ré-échantillonné de {sig.fs:g} Hz à {spec.fs:g} Hz (comme à l'entraînement).")
            prepared[m] = (x, spec)
            durations.append(x.size / spec.fs)
        duration = min(durations)
        max_win = max(spec.duration for _, spec in prepared.values())
        if duration < max_win:
            raise DiagnosisUnavailable(
                f"Signal trop court : {duration:.3f} s alors que le modèle a besoin d'au moins {max_win:.3f} s.")
        hop = hop_s or max_win * self.config.get("inference", {}).get("hop_fraction", 0.25)
        limit = max_windows or self.config.get("inference", {}).get("max_windows", 400)
        centers = eval_centers(duration, max_win, hop_s=hop)
        if centers.size > limit:
            centers = eval_centers(duration, max_win, n_windows=limit)
            notes.append(f"Signal long : {limit} fenêtres réparties sur toute la durée.")

        att = None
        if self.kinds[name] == "features":
            probs = mdl.predict_proba(prepared, centers, rpm)
        else:
            out, atts = [], []
            for i in range(0, centers.size, 128):
                c = centers[i:i + 128]
                inputs = {m: torch.from_numpy(extract_windows(x, spec.fs, c, spec.window)).to(self.device)
                          for m, (x, spec) in prepared.items()}
                logits, det = mdl(inputs, return_details=True)
                out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
                if det.get("attention") is not None and len(cfg["modalities"]) > 1:
                    atts.append(det["attention"].float().cpu().numpy())
            probs = np.concatenate(out)
            att = np.concatenate(atts) if atts else None
        preds = probs.argmax(1)
        mean_p = probs.mean(axis=0)
        k = int(mean_p.argmax())
        windows = []
        for j in range(centers.size):
            w = {"t": float(centers[j]), "pred": int(preds[j]), "probs": [float(v) for v in probs[j]]}
            if att is not None:
                w["attention"] = {m: float(att[j, q]) for q, m in enumerate(cfg["modalities"])}
            windows.append(w)
        if any(s is not None and s.meta.get("format") != "paderborn" for s in (vib, cur)):
            notes.append("Modèle entraîné sur le banc Paderborn (roulement 6203, moteur 425 W) : pour un signal "
                         "d'une autre machine, le diagnostic est hors domaine et doit être interprété avec prudence.")
        return {
            "model": name,
            "model_label": self.labels[name],
            "model_kind": self.kinds[name],
            "is_primary": name in self.primary.values(),
            "demo": self.demo,
            "class_names": self.class_names,
            "window_s": {m: spec.duration for m, (_, spec) in prepared.items()},
            "rpm_used": rpm,
            "windows": windows,
            "summary": {
                "pred": k,
                "class_name": self.class_names[k],
                "probs": [float(v) for v in mean_p],
                "agreement": float(np.mean(preds == k)),
                "n_windows": int(centers.size),
                "attention_mean": ({m: float(att[:, q].mean()) for q, m in enumerate(cfg["modalities"])}
                                   if att is not None else None),
            },
            "notes": notes,
        }
