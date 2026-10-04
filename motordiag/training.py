"""Entraînement, prédiction et métriques (utilisé par le notebook).

Points clés :
- les statistiques de normalisation sont calculées UNIQUEMENT sur les fichiers
  d'apprentissage, puis stockées dans le modèle ;
- le choix du meilleur epoch (arrêt anticipé) se fait sur la VALIDATION, jamais sur le test ;
- un point de reprise est écrit à chaque epoch : si Colab se déconnecte, relancer la
  cellule reprend l'entraînement là où il s'était arrêté.
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from torch import nn

from .data import grid_centers, random_centers
from .models import build_model, count_parameters
from .utils import save_json, set_seed

DEFAULT_TRAIN = {
    "epochs": 30,
    "batch_size": 128,
    "optimizer": "adamw",        # adamw | adam
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "schedule": "onecycle",      # onecycle | cosine | constant
    "label_smoothing": 0.05,
    "class_balance": "weights",  # weights | balanced_sampling | none
    "augment": True,
    "aug_scale": [0.8, 1.25],
    "aug_noise_prob": 0.5,
    "aug_snr_db": [20.0, 40.0],
    "train_windows_per_file": 12,
    "train_sampling": "random",  # random (décalage aléatoire) | grid (positions fixes)
    "eval_windows_per_file": 15,
    "patience": 6,
    "grad_clip": 1.0,
    "seed": 0,
    "amp": True,
}


def train_config(**overrides) -> dict:
    cfg = copy.deepcopy(DEFAULT_TRAIN)
    cfg.update(overrides)
    return cfg


# ================================================================================
# Métriques
# ================================================================================
def classification_metrics(y_true, y_pred, n_classes: int, class_names=None) -> dict:
    """Exactitude, F1 macro, précision / rappel / F1 par classe, matrice de confusion.

    Le F1 macro est calculé sur les classes présentes dans y_true (utile en validation,
    où la classe "combiné" peut être absente).
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    labels = list(range(n_classes))
    present = sorted(set(y_true.tolist()))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    names = class_names or [str(i) for i in labels]
    return {
        "n": int(y_true.size),
        "accuracy": float(np.mean(y_true == y_pred)) if y_true.size else float("nan"),
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
        "per_class": {names[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                 "f1": float(f[i]), "support": int(s[i])} for i in labels},
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def grouped_metrics(y_true, y_pred, groups) -> dict:
    """Exactitude et F1 macro par groupe (ex. condition de fonctionnement)."""
    y_true, y_pred, groups = map(np.asarray, (y_true, y_pred, groups))
    out = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        present = sorted(set(y_true[m].tolist()))
        out[str(g)] = {"n": int(m.sum()), "accuracy": float(np.mean(y_true[m] == y_pred[m])),
                       "macro_f1": float(f1_score(y_true[m], y_pred[m], labels=present,
                                                  average="macro", zero_division=0))}
    return out


def recording_level(probs, y_true, file_idx):
    """Décision par enregistrement : moyenne des probabilités de ses fenêtres."""
    file_idx = np.asarray(file_idx)
    files = np.unique(file_idx)
    p = np.stack([probs[file_idx == f].mean(axis=0) for f in files])
    t = np.array([np.asarray(y_true)[file_idx == f][0] for f in files])
    return files, t, p.argmax(axis=1), p


def mean_std(values) -> dict:
    v = np.asarray([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=float)
    if v.size == 0:
        return {"mean": None, "std": None, "n": 0, "values": []}
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "n": int(v.size), "values": v.tolist()}


# ================================================================================
# Utilitaires d'entraînement
# ================================================================================
def max_half_window(model_cfg: dict) -> float:
    return max(model_cfg["inputs"][m]["window"] / model_cfg["inputs"][m]["fs"]
               for m in model_cfg["modalities"]) / 2.0


def compute_norm_stats(bank, modality: str, file_idx: np.ndarray, max_files: int = 300, seed: int = 0) -> dict:
    """Moyenne et écart-type d'une voie, calculés sur des fichiers d'APPRENTISSAGE."""
    rng = np.random.default_rng(seed)
    idx = np.asarray(file_idx)
    if idx.size > max_files:
        idx = rng.choice(idx, max_files, replace=False)
    t = bank.data[modality]
    s, s2, n = 0.0, 0.0, 0
    for i in range(0, idx.size, 32):
        x = t[torch.as_tensor(idx[i:i + 32], device=t.device)].double()
        s += float(x.sum()); s2 += float((x ** 2).sum()); n += x.numel()
    mean = s / n
    std = math.sqrt(max(s2 / n - mean ** 2, 1e-12))
    return {"mean": mean, "std": std}


def class_weights(labels_per_file: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(labels_per_file, minlength=n_classes).astype(float)
    w = np.where(counts > 0, counts.sum() / (n_classes * np.maximum(counts, 1)), 0.0)
    return w


def augment_batch(x: torch.Tensor, cfg: dict, gen: torch.Generator) -> torch.Tensor:
    """Augmentation (apprentissage seulement) : gain aléatoire + bruit gaussien à SNR aléatoire."""
    b = x.shape[0]
    lo, hi = cfg["aug_scale"]
    scale = lo + (hi - lo) * torch.rand(b, 1, device=x.device, generator=gen)
    x = x * scale
    if cfg["aug_noise_prob"] > 0:
        apply = (torch.rand(b, 1, device=x.device, generator=gen) < cfg["aug_noise_prob"]).float()
        s_lo, s_hi = cfg["aug_snr_db"]
        snr = s_lo + (s_hi - s_lo) * torch.rand(b, 1, device=x.device, generator=gen)
        x_c = x - x.mean(dim=1, keepdim=True)
        rms = x_c.pow(2).mean(dim=1, keepdim=True).sqrt()
        noise = torch.randn(x.shape, device=x.device, generator=gen) * rms * 10 ** (-snr / 20)
        x = x + apply * noise
    return x


def batch_inputs(model_cfg, bank, f_idx, centers):
    return {m: bank.windows(m, f_idx, centers, model_cfg["inputs"][m]["window"]) for m in model_cfg["modalities"]}


@torch.no_grad()
def predict(model, bank, file_idx, centers, batch_size: int = 512):
    """Probabilités (n, C) et poids d'attention (n, M) éventuels pour des fenêtres données."""
    model.eval()
    cfg = model.cfg
    probs, atts = [], []
    for i in range(0, len(file_idx), batch_size):
        inp = batch_inputs(cfg, bank, file_idx[i:i + batch_size], centers[i:i + batch_size])
        logits, det = model(inp, return_details=True)
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        if det.get("attention") is not None:
            atts.append(det["attention"].float().cpu().numpy())
    return np.concatenate(probs), (np.concatenate(atts) if atts else None)


def _torch_save_atomic(obj, path: Path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ================================================================================
# Entraînement
# ================================================================================
def train_run(model_cfg: dict, train_cfg: dict, bank, meta: list[dict], split: dict,
              run_dir=None, device=None, log=print, fixed_train=None, fixed_val=None) -> dict:
    """Entraîne un modèle.

    split       : {"train": idx_fichiers, "val": idx_fichiers} (indices de `meta`)
    fixed_train : (file_idx, centres) fixes au lieu d'un tirage aléatoire à chaque epoch
    fixed_val   : (file_idx, centres) de validation imposés (sinon grille sur split["val"])
    Renvoie {"model", "model_cfg", "history", "best_epoch", "best_val_f1"}.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tcfg = train_config(**train_cfg)
    set_seed(tcfg["seed"])
    labels = np.array([r["label"] for r in meta])
    n_cls = model_cfg.get("n_classes", 4)
    train_files = np.asarray(split["train"])
    run_dir = Path(run_dir) if run_dir else None
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)

    # --- normalisation : statistiques des fichiers d'apprentissage uniquement ---
    model_cfg = copy.deepcopy(model_cfg)
    model_cfg["norm_stats"] = {m: compute_norm_stats(bank, m, train_files, seed=tcfg["seed"])
                               for m in model_cfg["modalities"]}
    model = build_model(model_cfg).to(device)
    half = max_half_window(model_cfg)

    # --- fenêtres de validation ---
    if fixed_val is not None:
        val_f, val_c = fixed_val
    elif len(split.get("val", [])):
        val_f, val_c = grid_centers(np.asarray(split["val"]), tcfg["eval_windows_per_file"], bank.duration, half)
    else:
        val_f, val_c = np.array([], dtype=np.int64), np.array([])
    has_val = len(val_f) > 0

    # --- pondération des classes ---
    w = None
    if tcfg["class_balance"] == "weights":
        w = torch.tensor(class_weights(labels[train_files], n_cls), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=tcfg["label_smoothing"])

    n_train = len(fixed_train[0]) if fixed_train is not None else train_files.size * tcfg["train_windows_per_file"]
    steps = max(1, math.ceil(n_train / tcfg["batch_size"]))
    params = model.parameters()
    if tcfg["optimizer"] == "adamw":
        opt = torch.optim.AdamW(params, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    else:
        opt = torch.optim.Adam(params, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    total = tcfg["epochs"] * steps
    if tcfg["schedule"] == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tcfg["lr"], total_steps=total, pct_start=0.15)
    elif tcfg["schedule"] == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=tcfg["lr"] * 1e-3)
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    use_amp = bool(tcfg["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {k: [] for k in ("epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1", "lr", "time_s")}
    best = {"f1": -1.0, "loss": float("inf"), "epoch": -1, "state": None}
    start_epoch, bad = 0, 0

    # --- reprise après déconnexion ---
    ckpt_path = run_dir / "last.pt" if run_dir else None
    if ckpt_path and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        history, best, bad, start_epoch = ck["history"], ck["best"], ck["bad"], ck["epoch"] + 1
        # Le planning du taux d'apprentissage est reconstruit puis avancé du nombre de pas
        # déjà faits : cela reste correct même si le nombre d'epochs a été modifié.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(min(start_epoch * steps, total)):
                sched.step()
        log(f"  reprise à l'epoch {start_epoch}")

    log(f"  modèle {model_cfg['arch']} {model_cfg['modalities']} : {count_parameters(model):,} paramètres ; "
        f"{n_train} fenêtres/epoch ; val = {len(val_f)} fenêtres")
    gen = torch.Generator(device=device)
    for epoch in range(start_epoch, tcfg["epochs"]):
        t0 = time.time()
        rng = np.random.default_rng(tcfg["seed"] * 100003 + epoch)
        gen.manual_seed(tcfg["seed"] * 100003 + epoch)
        if fixed_train is not None:
            perm = rng.permutation(len(fixed_train[0]))
            tr_f, tr_c = fixed_train[0][perm], fixed_train[1][perm]
        else:
            files = train_files
            if tcfg["class_balance"] == "balanced_sampling":
                lab = labels[train_files]
                pw = 1.0 / np.bincount(lab, minlength=n_cls)[lab]
                files = rng.choice(train_files, size=train_files.size, p=pw / pw.sum())
            if tcfg["train_sampling"] == "grid":
                tr_f, tr_c = grid_centers(files, tcfg["train_windows_per_file"], bank.duration, half)
            else:
                tr_f, tr_c = random_centers(files, tcfg["train_windows_per_file"], bank.duration, half, rng)
            perm = rng.permutation(tr_f.size)
            tr_f, tr_c = tr_f[perm], tr_c[perm]

        model.train()
        tot_loss, tot_ok, tot_n = 0.0, 0, 0
        for i in range(0, tr_f.size, tcfg["batch_size"]):
            bf, bc = tr_f[i:i + tcfg["batch_size"]], tr_c[i:i + tcfg["batch_size"]]
            inp = batch_inputs(model_cfg, bank, bf, bc)
            if tcfg["augment"]:
                inp = {m: augment_batch(x, tcfg, gen) for m, x in inp.items()}
            y = torch.as_tensor(labels[bf], device=device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(inp)
                loss = loss_fn(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            scaler.step(opt); scaler.update()
            sched.step()
            tot_loss += loss.item() * y.numel()
            tot_ok += int((logits.argmax(1) == y).sum()); tot_n += y.numel()

        row = {"epoch": epoch, "train_loss": tot_loss / tot_n, "train_acc": tot_ok / tot_n,
               "lr": opt.param_groups[0]["lr"], "val_loss": None, "val_acc": None, "val_macro_f1": None}
        if has_val:
            probs, _ = predict(model, bank, val_f, val_c)
            yv = labels[val_f]
            row["val_loss"] = float(nn.functional.nll_loss(torch.log(torch.as_tensor(probs) + 1e-9),
                                                           torch.as_tensor(yv)))
            m = classification_metrics(yv, probs.argmax(1), n_cls)
            row["val_acc"], row["val_macro_f1"] = m["accuracy"], m["macro_f1"]
        row["time_s"] = time.time() - t0
        for k in history:
            history[k].append(row[k])

        # --- meilleur modèle selon la validation (F1 macro, puis perte) ---
        if has_val:
            better = (row["val_macro_f1"] > best["f1"] + 1e-4) or (
                abs(row["val_macro_f1"] - best["f1"]) <= 1e-4 and row["val_loss"] < best["loss"])
        else:
            better = True  # pas de validation : on garde le dernier epoch (nombre d'epochs fixé)
        if better:
            best = {"f1": row["val_macro_f1"] if has_val else None, "loss": row["val_loss"] if has_val else None,
                    "epoch": epoch, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
        vtxt = f"val F1={row['val_macro_f1']:.4f} acc={row['val_acc']:.4f}" if has_val else "pas de validation"
        log(f"  epoch {epoch:3d} | perte={row['train_loss']:.4f} acc={row['train_acc']:.4f} | {vtxt} | "
            f"{row['time_s']:.0f} s{' *' if better else ''}")
        if ckpt_path:
            _torch_save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                                "scaler": scaler.state_dict(), "history": history, "best": best, "bad": bad,
                                "epoch": epoch, "model_cfg": model_cfg}, ckpt_path)
        if has_val and bad >= tcfg["patience"]:
            log(f"  arrêt anticipé (pas d'amélioration depuis {tcfg['patience']} epochs)")
            break

    model.load_state_dict(best["state"])
    result = {"model": model, "model_cfg": model_cfg, "train_cfg": tcfg, "history": history,
              "best_epoch": best["epoch"], "best_val_f1": best["f1"]}
    if run_dir:
        torch.save(model.state_dict(), run_dir / "best_model.pt")
        save_json({"model_cfg": model_cfg, "train_cfg": tcfg, "history": history,
                   "best_epoch": best["epoch"], "best_val_f1": best["f1"]}, run_dir / "train_summary.json")
    return result


def load_trained(run_dir, device=None):
    """Recharge un modèle entraîné (best_model.pt + train_summary.json)."""
    run_dir = Path(run_dir)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    with open(run_dir / "train_summary.json", encoding="utf-8") as f:
        summ = json.load(f)
    model = build_model(summ["model_cfg"]).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True))
    model.eval()
    return model, summ


# ================================================================================
# Évaluation sur des fichiers de test
# ================================================================================
def evaluate_files(model, bank, meta, file_idx, windows_per_file: int, class_names, extra_cols=None):
    """Prédit sur une grille de fenêtres et renvoie métriques + tableau de prédictions."""
    cfg = model.cfg
    half = max_half_window(cfg)
    f, c = grid_centers(np.asarray(file_idx), windows_per_file, bank.duration, half)
    probs, att = predict(model, bank, f, c)
    labels = np.array([meta[i]["label"] for i in f])
    pred = probs.argmax(1)
    n_cls = len(class_names)
    metrics = classification_metrics(labels, pred, n_cls, class_names)
    metrics["per_condition"] = grouped_metrics(labels, pred, [meta[i]["condition"] for i in f])
    metrics["per_origin"] = grouped_metrics(labels, pred, [meta[i]["origin"] for i in f])
    rf, rt, rp, _ = recording_level(probs, labels, f)
    metrics["recording_level"] = classification_metrics(rt, rp, n_cls, class_names)
    rows = []
    for k in range(len(f)):
        r = meta[f[k]]
        row = {"file": r["file"], "bearing": r["bearing"], "condition": r["condition"], "origin": r["origin"],
               "center_s": round(float(c[k]), 5), "true": int(labels[k]), "pred": int(pred[k])}
        for j, name in enumerate(class_names):
            row[f"p_{j}"] = round(float(probs[k, j]), 5)
        if att is not None:
            for j, m in enumerate(cfg["modalities"]):
                row[f"att_{m}"] = round(float(att[k, j]), 4)
        if extra_cols:
            row.update(extra_cols)
        rows.append(row)
    return metrics, rows
