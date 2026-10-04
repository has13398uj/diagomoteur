"""API du dashboard (FastAPI). Toute l'analyse passe par le module `motordiag`,
le même que celui du notebook d'entraînement.

Lancement : `python start.py` à la racine du projet (sert aussi l'interface web).
"""
from __future__ import annotations

import csv
import json
import os
import sys
import threading
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motordiag import signal_analysis as sa  # noqa: E402
from motordiag.constants import BEARING_6203, CLASS_NAMES_FR  # noqa: E402
from motordiag.inference import DiagnosisUnavailable, Diagnoser, create_demo_artifacts  # noqa: E402
from motordiag.io import SignalData, SignalFileError, load_signal, parse_paderborn_name  # noqa: E402
from motordiag.utils import load_json  # noqa: E402

ARTIFACTS = Path(os.environ.get("PFE_ARTIFACTS", ROOT / "artifacts"))
DEMO_ARTIFACTS = Path(os.environ.get("PFE_DEMO_ARTIFACTS", ROOT / "artifacts_demo"))
SAMPLES = Path(os.environ.get("PFE_SAMPLES", ROOT / "samples"))
DIST = ROOT / "dashboard" / "dist"
MAX_UPLOAD_MB = 300
SIGNAL_EXT = {".mat", ".csv", ".txt", ".npy"}

app = FastAPI(title="Diagnostic moteur — API", version="1.0")
_lock = threading.Lock()
_demo_lock = threading.Lock()
_diag_cache: dict = {}
_analyses: "OrderedDict[str, dict]" = OrderedDict()


# ================================================================================
# Artefacts
# ================================================================================
def active_artifacts() -> tuple[Path, bool]:
    """Artefacts réels s'ils existent, sinon artefacts DÉMO (modèles non entraînés)."""
    if (ARTIFACTS / "config.json").exists():
        return ARTIFACTS, False
    with _demo_lock:  # plusieurs requêtes simultanées : une seule création
        if not (DEMO_ARTIFACTS / "config.json").exists():
            create_demo_artifacts(DEMO_ARTIFACTS)
    return DEMO_ARTIFACTS, True


def get_diagnoser() -> Diagnoser:
    path, _ = active_artifacts()
    key = (str(path), (path / "config.json").stat().st_mtime)
    with _lock:
        if _diag_cache.get("key") != key:
            _diag_cache["key"] = key
            _diag_cache["diag"] = Diagnoser(path)
        return _diag_cache["diag"]


def _read_json(path: Path):
    return load_json(path) if path.exists() else None


def _per_bearing(pred_csv: Path) -> list[dict]:
    if not pred_csv.exists():
        return []
    stats: dict = {}
    with open(pred_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("model", "fusion") != "fusion":
                continue
            s = stats.setdefault(r["bearing"], {"bearing": r["bearing"], "true": int(r["true"]), "n": 0, "ok": 0,
                                                "fold": r.get("fold")})
            s["n"] += 1
            s["ok"] += int(r["true"] == r["pred"])
    return [{**s, "accuracy": s["ok"] / s["n"]} for s in sorted(stats.values(), key=lambda s: s["bearing"])]


@app.get("/api/status")
def status():
    path, using_demo = active_artifacts()
    cfg = load_json(path / "config.json")
    metrics = _read_json(path / "metrics.json") or {}
    diag = get_diagnoser()
    models = []
    for n, e in cfg.get("models", {}).items():
        if n not in diag.available:
            continue
        mc = e.get("model_cfg") or e.get("feature_cfg") or {}
        models.append({"name": n, "label": diag.labels[n], "kind": diag.kinds[n], "n_parameters": e.get("n_parameters"),
                       "arch": mc.get("arch", "features"), "modalities": mc.get("modalities", []),
                       "inputs": mc.get("inputs", {}), "primary": n in diag.primary.values()})
    return {"artifacts_dir": str(path), "using_demo_artifacts": using_demo, "demo": bool(cfg.get("demo")),
            "smoke_test": bool(metrics.get("smoke_test")), "created": cfg.get("created"),
            "metrics_available": bool(metrics.get("available")), "models": models,
            "primary": diag.primary, "load_errors": diag.load_errors, "skipped_models": diag.skipped,
            "rul_available": (path / "rul" / "rul_config.json").exists(),
            "class_names": cfg.get("class_names", CLASS_NAMES_FR), "n_samples": len(list_samples())}


@app.get("/api/results")
def results():
    path, using_demo = active_artifacts()
    cfg = load_json(path / "config.json")
    metrics = _read_json(path / "metrics.json") or {}
    out = {"available": bool(metrics.get("available")), "demo": bool(cfg.get("demo")),
           "using_demo_artifacts": using_demo, "smoke_test": bool(metrics.get("smoke_test")),
           "artifacts_dir": str(path),
           "config": {k: cfg.get(k) for k in ("created", "dataset", "class_names", "folds", "exported_fold",
                                              "exported_test_bearings", "source_notebook", "bearing_geometry")},
           "models": {n: {"label": e.get("label"), "kind": e.get("kind", "torch"), "n_parameters": e.get("n_parameters"),
                          "model_cfg": e.get("model_cfg"), "feature_cfg": e.get("feature_cfg")}
                      for n, e in cfg.get("models", {}).items()},
           "primary": cfg.get("primary")}
    if not out["available"]:
        out["reason"] = ("Aucun résultat d'entraînement : exécutez le notebook PFE_Diagnostic_v2 dans Colab puis "
                         "copiez son dossier artifacts/ à la racine du projet.")
        return out
    out["metrics"] = metrics
    out["history"] = _read_json(path / "history.json")
    out["confusion"] = _read_json(path / "confusion_matrix.json")
    out["per_bearing"] = _per_bearing(path / "test_predictions.csv")
    rul_dir = path / "rul"
    if (rul_dir / "rul_metrics.json").exists():
        rows = []
        if (rul_dir / "rul_predictions.csv").exists():
            with open(rul_dir / "rul_predictions.csv", newline="", encoding="utf-8") as f:
                rows = [{"bearing": r["bearing"], "t": int(r["t_min"]), "true": float(r["rul_true_min"]),
                         "pred": float(r["rul_pred_min"])} for r in csv.DictReader(f)]
        step = max(1, len(rows) // 4000)
        out["rul"] = {"metrics": load_json(rul_dir / "rul_metrics.json"),
                      "config": {k: v for k, v in load_json(rul_dir / "rul_config.json").items()
                                 if k not in ("feature_mean", "feature_std", "feature_names")},
                      "predictions": rows[::step]}
    return out


# ================================================================================
# Exemples (mode démo)
# ================================================================================
def list_samples() -> list[dict]:
    if not SAMPLES.exists():
        return []
    listed = {d["file"]: d for d in (_read_json(SAMPLES / "samples.json") or [])}
    path, using_demo = active_artifacts()
    cfg = _read_json(path / "config.json") or {}
    test_bearings = set(cfg.get("exported_test_bearings") or [])
    out = []
    for p in sorted(SAMPLES.iterdir()):
        if p.suffix.lower() not in SIGNAL_EXT:
            continue
        info = {"file": p.name, "size_mb": round(p.stat().st_size / 1e6, 2)}
        pad = parse_paderborn_name(p.name)
        if pad:
            info.update({"bearing": pad["bearing"], "label": pad.get("label"), "condition": pad["condition"],
                         "origin": pad.get("origin"), "rpm_nominal": pad.get("rpm_nominal")})
            if pad.get("label") is not None:
                info["class_name"] = CLASS_NAMES_FR[pad["label"]]
        info.update(listed.get(p.name, {}))
        if test_bearings and info.get("bearing"):
            info["status"] = "test" if info["bearing"] in test_bearings else "train"
        else:
            info["status"] = "unknown"
        out.append(info)
    return out


@app.get("/api/samples")
def samples():
    return {"samples": list_samples()}


# ================================================================================
# Analyse
# ================================================================================
def _arr(x, dec=6):
    return np.round(np.asarray(x, dtype=np.float64), dec).tolist()


def _spectrogram_payload(x, fs, fmax):
    nper = 1024 if fs >= 16000 else 512
    t, f, db = sa.spectrogram(x, fs, nperseg=min(nper, x.size), fmax=fmax, max_frames=300)
    if f.size > 256:  # max par paquets de fréquences pour l'affichage
        k = int(np.ceil(f.size / 256))
        n = (f.size // k) * k
        db = db[:n].reshape(-1, k, db.shape[1]).max(axis=1)
        f = f[:n].reshape(-1, k).mean(axis=1)
    lo, hi = np.percentile(db, 5), np.max(db)
    return {"t": _arr(t, 4), "f": _arr(f, 2), "db": np.round(db, 1).tolist(), "db_min": float(lo), "db_max": float(hi)}


def _signal_block(sig: SignalData, kind: str, params: dict, fault: dict | None, warnings: list) -> dict:
    x, fs = sig.values, float(sig.fs)
    unit = params.get(f"{kind}_unit") or sig.unit or "u.a."
    t, y = sa.minmax_downsample(x, fs, max_points=4000)
    f, a = sa.amplitude_spectrum(x, fs)
    fr_, ar_ = sa.reduce_spectrum(f, a, 3000)
    low = f <= min(1000.0, fs / 2)
    fl, al = sa.reduce_spectrum(f[low], a[low], 4000)
    block = {"source": sig.source, "channel": sig.channel, "fs": fs, "n": int(x.size), "duration": x.size / fs,
             "unit": unit, "indicators": sa.indicators(x),
             "waveform": {"t": _arr(t, 6), "y": _arr(y, 6)},
             "spectrum": {"f": _arr(fr_, 3), "a": _arr(ar_, 8)},
             "spectrum_low": {"f": _arr(fl, 3), "a": _arr(al, 8)}}
    if kind == "vib":
        band = params.get("env_band") or [1000.0, min(0.45 * fs, 20000.0)]
        band = [float(band[0]), min(float(band[1]), 0.49 * fs)]
        fmax_env = max(500.0, 5.5 * fault["BPFI"]) if fault else 1000.0
        try:
            ef, ea = sa.envelope_spectrum(x, fs, band=band, fmax=fmax_env)
            block["envelope"] = {"f": _arr(ef, 3), "a": _arr(ea, 8), "band": band}
        except ValueError as e:
            warnings.append(f"Spectre d'enveloppe : {e}")
        block["spectrogram"] = _spectrogram_payload(x, fs, fmax=min(fs / 2, 20000.0))
    else:
        f_s = params.get("supply_freq")
        source = "saisie"
        if not f_s:
            f_s = sa.estimate_supply_frequency(x, fs)
            source = "estimée sur le spectre"
        span = max(150.0, 3.2 * fault["BPFI"]) if fault else 200.0
        m = (f >= max(0.0, f_s - span)) & (f <= f_s + span)
        a_s = a[np.argmin(np.abs(f - f_s))] or 1e-12
        db = 20 * np.log10(np.maximum(a[m], 1e-12) / a_s)
        block["supply"] = {"f_s": float(f_s), "source": source}
        block["zoom"] = {"f": _arr(f[m], 3), "db": _arr(db, 2), "span": span}
        block["markers"] = sa.current_sideband_markers(f_s, fault or {}, orders=(1, 2)) if fault else \
            [{"label": "f_s", "freq": float(f_s), "kind": "supply"}]
        block["spectrogram"] = _spectrogram_payload(x, fs, fmax=min(fs / 2, 2000.0))
    return block


def _resolve_fs(sig: SignalData, user_fs, kind, warnings):
    label = "vibration" if kind == "vib" else "courant"
    if sig.fs:
        if user_fs and abs(user_fs - sig.fs) / sig.fs > 0.01:
            warnings.append(f"Fréquence d'échantillonnage du fichier {label} ({sig.fs:g} Hz) utilisée à la place de "
                            f"la valeur saisie ({user_fs:g} Hz).")
        return sig
    if not user_fs:
        raise HTTPException(422, f"Fréquence d'échantillonnage du signal {label} inconnue : le fichier "
                                 f"« {sig.source} » ne la contient pas, saisissez-la.")
    sig.fs = float(user_fs)
    return sig


def analyze(vib: SignalData | None, cur: SignalData | None, params: dict) -> dict:
    warnings: list[str] = []
    if vib is None and cur is None:
        raise HTTPException(422, "Aucun signal fourni : chargez un fichier de vibration et/ou de courant.")
    if vib is not None:
        vib = _resolve_fs(vib, params.get("fs_vib"), "vib", warnings)
    if cur is not None:
        cur = _resolve_fs(cur, params.get("fs_cur"), "cur", warnings)

    # Vitesse : fichier (voie speed, puis nom de fichier) sinon valeur saisie
    rpm, rpm_source = None, None
    for s in (vib, cur):
        if s is not None and s.meta.get("rpm_measured"):
            rpm, rpm_source = float(s.meta["rpm_measured"]), "voie « speed » du fichier"
            break
    if rpm is None:
        for s in (vib, cur):
            if s is not None and s.meta.get("rpm_nominal"):
                rpm, rpm_source = float(s.meta["rpm_nominal"]), "nom du fichier (condition nominale)"
                break
    user_rpm = params.get("rpm")
    if rpm is None and user_rpm:
        rpm, rpm_source = float(user_rpm), "saisie"
    elif rpm is not None and user_rpm and abs(user_rpm - rpm) / rpm > 0.05:
        warnings.append(f"Vitesse du fichier ({rpm:.0f} tr/min) utilisée à la place de la valeur saisie ({user_rpm:g}).")
    geometry = {**BEARING_6203, **(params.get("geometry") or {})}
    fault = None
    if rpm:
        try:
            fault = sa.bearing_fault_frequencies(rpm, geometry)
        except ValueError as e:
            warnings.append(str(e))
    else:
        warnings.append("Vitesse de rotation inconnue : saisissez-la pour afficher les fréquences de défaut "
                        "(BPFO, BPFI, BSF, FTF).")
    for s in (vib, cur):
        if s is not None and not s.unit and not params.get(f"{'vib' if s is vib else 'cur'}_unit"):
            warnings.append(f"Unité non précisée dans « {s.source} » : les amplitudes sont affichées en unités du fichier.")
    if vib is not None and cur is not None and abs(vib.duration - cur.duration) > 0.05:
        warnings.append(f"Durées différentes (vibration {vib.duration:.2f} s, courant {cur.duration:.2f} s) : "
                        "le diagnostic utilise la partie commune.")

    out = {"signals": {}, "rpm": {"value": rpm, "source": rpm_source}, "geometry": geometry,
           "fault_freqs": fault, "warnings": warnings}
    if vib is not None:
        out["signals"]["vib"] = _signal_block(vib, "vib", params, fault, warnings)
    if cur is not None:
        out["signals"]["cur"] = _signal_block(cur, "cur", params, fault, warnings)
    try:
        out["diagnosis"] = get_diagnoser().diagnose(vib=vib, cur=cur, rpm=rpm, model=params.get("model"))
    except DiagnosisUnavailable as e:
        out["diagnosis"] = {"error": str(e)}
    except Exception as e:  # noqa: BLE001
        out["diagnosis"] = {"error": f"Erreur pendant le diagnostic : {e}"}
    out["rul"] = {"available": False,
                  "reason": "La durée de vie résiduelle ne peut pas être estimée à partir d'un enregistrement isolé : "
                            "il faut l'historique de dégradation d'un même roulement (série de mesures dans le temps). "
                            "Voir la page « Pronostic RUL »."}
    out["warnings"] = list(dict.fromkeys(warnings))  # sans doublons (même fichier pour les deux voies)
    aid = uuid.uuid4().hex[:12]
    with _lock:
        _analyses[aid] = {"vib": vib, "cur": cur}
        while len(_analyses) > 8:
            _analyses.popitem(last=False)
    out["id"] = aid
    return out


def _parse_params(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        p = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(422, "Paramètres invalides (JSON attendu).")
    clean = {}
    for k in ("fs_vib", "fs_cur", "rpm", "supply_freq"):
        v = p.get(k)
        if v not in (None, ""):
            try:
                v = float(v)
            except (TypeError, ValueError):
                raise HTTPException(422, f"Valeur numérique attendue pour « {k} ».")
            if v <= 0:
                raise HTTPException(422, f"« {k} » doit être strictement positif.")
            clean[k] = v
    for k in ("vib_unit", "cur_unit"):
        if p.get(k):
            clean[k] = str(p[k])[:20]
    if p.get("model") and p.get("model") != "auto":
        clean["model"] = str(p["model"])[:40]
    if p.get("env_band"):
        try:
            lo, hi = float(p["env_band"][0]), float(p["env_band"][1])
        except (TypeError, ValueError, IndexError):
            raise HTTPException(422, "Bande d'enveloppe invalide.")
        if not 0 < lo < hi:
            raise HTTPException(422, "Bande d'enveloppe invalide : il faut 0 < f basse < f haute.")
        clean["env_band"] = [lo, hi]
    if p.get("geometry"):
        g = {}
        for k in ("n_balls", "ball_diameter_mm", "pitch_diameter_mm", "contact_angle_deg"):
            if p["geometry"].get(k) not in (None, ""):
                try:
                    g[k] = float(p["geometry"][k])
                except (TypeError, ValueError):
                    raise HTTPException(422, f"Géométrie : valeur numérique attendue pour « {k} ».")
        clean["geometry"] = g
    return clean


async def _read_upload(f: UploadFile | None, kind: str) -> SignalData | None:
    if f is None or not f.filename:
        return None
    content = await f.read()
    if len(content) > MAX_UPLOAD_MB * 1e6:
        raise HTTPException(413, f"{f.filename} : fichier trop volumineux (> {MAX_UPLOAD_MB} Mo).")
    try:
        return load_signal(f.filename, content, kind)
    except SignalFileError as e:
        raise HTTPException(422, str(e))


@app.post("/api/analyze")
async def analyze_upload(vib_file: UploadFile | None = File(None), cur_file: UploadFile | None = File(None),
                         params: str | None = Form(None)):
    p = _parse_params(params)
    vib = await _read_upload(vib_file, "vib")
    cur = await _read_upload(cur_file, "cur")
    return JSONResponse(analyze(vib, cur, p))


@app.post("/api/analyze_sample")
async def analyze_sample(payload: dict):
    name = Path(str(payload.get("name", ""))).name
    path = SAMPLES / name
    if not name or not path.exists() or path.suffix.lower() not in SIGNAL_EXT:
        raise HTTPException(404, f"Exemple introuvable : {name}")
    p = _parse_params(json.dumps(payload.get("params") or {}))
    content = path.read_bytes()
    use = payload.get("signals") or ["vib", "cur"]
    try:
        vib = load_signal(name, content, "vib") if "vib" in use else None
        cur = load_signal(name, content, "cur") if "cur" in use else None
    except SignalFileError as e:
        raise HTTPException(422, str(e))
    res = analyze(vib, cur, p)
    res["sample"] = next((s for s in list_samples() if s["file"] == name), {"file": name})
    return JSONResponse(res)


@app.get("/api/analysis/{aid}/segment")
def segment(aid: str, signal: str, t0: float, t1: float):
    with _lock:
        a = _analyses.get(aid)
    if a is None:
        raise HTTPException(404, "Analyse expirée : relancez l'analyse.")
    sig = a.get(signal)
    if sig is None:
        raise HTTPException(404, f"Signal « {signal} » absent de cette analyse.")
    fs = sig.fs
    i0 = max(0, int(t0 * fs)); i1 = min(sig.values.size, int(np.ceil(t1 * fs)) + 1)
    if i1 - i0 < 2:
        raise HTTPException(422, "Intervalle trop court.")
    t, y = sa.minmax_downsample(sig.values[i0:i1], fs, t0=i0 / fs, max_points=4000)
    return {"t": _arr(t, 7), "y": _arr(y, 6), "full_resolution": (i1 - i0) <= 4000}


# ================================================================================
# Pronostic RUL (série d'enregistrements XJTU-SY)
# ================================================================================
@app.post("/api/rul")
async def rul_series(files: list[UploadFile] = File(...), rpm: float = Form(...), fs: float | None = Form(None)):
    path, _ = active_artifacts()
    rul_dir = path / "rul"
    if not (rul_dir / "rul_config.json").exists():
        raise HTTPException(409, "Aucun modèle RUL dans artifacts/rul/ (section 10.2 du notebook, données XJTU-SY).")
    from motordiag.rul import RULPredictor
    items = []
    for f in files:
        content = await f.read()
        stem = Path(f.filename).stem
        try:
            lines = content.decode("utf-8-sig").splitlines()
            skip = 1 if lines and any(ch.isalpha() for ch in lines[0]) else 0  # en-tête éventuel
            data = np.loadtxt(lines[skip:], delimiter=",", dtype=np.float64)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(422, f"{f.filename} : CSV illisible ({e}).")
        if data.ndim != 2 or data.shape[1] != 2:
            raise HTTPException(422, f"{f.filename} : 2 colonnes attendues (vibrations horizontale et verticale).")
        items.append((int(stem) if stem.isdigit() else len(items), f.filename, data))
    items.sort(key=lambda t: t[0])
    try:
        res = RULPredictor(rul_dir).predict_series([d for _, _, d in items], rpm=rpm, fs=fs)
    except ValueError as e:
        raise HTTPException(422, str(e))
    res["files"] = [n for _, n, _ in items]
    return res


# ================================================================================
# Interface web (build Vite)
# ================================================================================
if (DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


@app.get("/{path:path}")
def spa(path: str):
    if path.startswith("api/"):
        raise HTTPException(404, "Route API inconnue.")
    target = DIST / path
    if path and target.is_file():
        return FileResponse(target)
    index = DIST / "index.html"
    if not index.exists():
        return JSONResponse({"message": "Interface non construite : lancez `python start.py` (build automatique)."},
                            status_code=503)
    return FileResponse(index)
