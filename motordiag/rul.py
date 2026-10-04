"""Pronostic (RUL, durée de vie résiduelle).

1) `original_rul_audit` : montre pourquoi la cible RUL du notebook d'origine est synthétique.
   Paderborn ne contient pas d'historiques jusqu'à la défaillance : chaque roulement est
   mesuré dans un état fixe. La cible d'origine est une formule
       RUL = 180 · (1 − sévérité[classe]) · (1 − rang/total)
   où le rang suit l'ordre alphabétique des fichiers (donc la condition de fonctionnement).

2) RUL réelle sur XJTU-SY (optionnel) : 15 roulements menés jusqu'à la défaillance
   (Wang et al., IEEE Trans. Reliability, 2020), un enregistrement de 1,28 s par minute,
   accéléromètres horizontal + vertical à 25,6 kHz. Ici la cible est le vrai temps
   restant avant la fin de l'essai.
"""
from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import numpy as np

from .features import window_features

# ================================================================================
# 1) Audit de la cible RUL d'origine (H_M_100.ipynb, cellule 45)
# ================================================================================
ORIGINAL_SEVERITY = {0: 0.05, 1: 0.60, 2: 0.55, 3: 0.85}
ORIGINAL_MAX_RUL = 180.0


def original_rul_audit(meta: list[dict], windows_per_file: int = 200) -> dict:
    """Recalcule la cible d'origine et mesure la part expliquée par (classe, condition).

    R² ≈ 1 signifie que la "RUL" se déduit presque entièrement de la classe et de la
    condition de fonctionnement, sans aucune notion de dégradation dans le temps.
    """
    by_bearing: dict[str, list[dict]] = {}
    for r in meta:
        by_bearing.setdefault(r["bearing"], []).append(r)
    target, label, cond = [], [], []
    for b, rows in by_bearing.items():
        rows = sorted(rows, key=lambda r: r["file"])  # os.listdir trié, comme à l'origine
        total = len(rows) * windows_per_file
        lab = rows[0]["label"]
        rul_start = ORIGINAL_MAX_RUL * (1.0 - ORIGINAL_SEVERITY[lab])
        pos = 0
        for r in rows:
            for _ in range(windows_per_file):
                target.append(rul_start * (1.0 - pos / max(total - 1, 1)))
                label.append(lab); cond.append(r["condition"])
                pos += 1
    target, label, cond = np.array(target), np.array(label), np.array(cond)
    keys = np.char.add(label.astype(str), cond)
    pred = np.empty_like(target)
    for k in np.unique(keys):
        m = keys == k
        pred[m] = target[m].mean()
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    healthy_end = float(target[label == 0].min()) if np.any(label == 0) else None
    return {"r2_class_condition": 1 - ss_res / ss_tot,
            "rmse_lookup_min": math.sqrt(ss_res / target.size),
            "healthy_min_rul": healthy_end, "n_windows": int(target.size)}


# ================================================================================
# 2) XJTU-SY
# ================================================================================
XJTU_FS = 25600
XJTU_BEARING = {"name": "LDK UER204 (XJTU-SY)", "n_balls": 8, "ball_diameter_mm": 7.92,
                "pitch_diameter_mm": 34.55, "contact_angle_deg": 0.0}
_BEARING_RE = re.compile(r"Bearing(\d)_(\d)")
_COND_RE = re.compile(r"(\d+(?:\.\d+)?)Hz")


def scan_xjtu(root) -> list[dict]:
    """Repère les dossiers BearingX_Y et leurs fichiers 1.csv … N.csv (ordre numérique)."""
    root = Path(root)
    out = []
    for d in sorted(root.rglob("Bearing*_*")):
        if not d.is_dir() or not _BEARING_RE.fullmatch(d.name):
            continue
        files = sorted((p for p in d.glob("*.csv") if p.stem.isdigit()), key=lambda p: int(p.stem))
        if not files:
            continue
        m = _COND_RE.search(d.parent.name)
        fr = float(m.group(1)) if m else None
        out.append({"bearing": d.name, "condition": d.parent.name, "shaft_hz": fr,
                    "rpm": fr * 60 if fr else None, "files": [str(p) for p in files]})
    return out


def read_xjtu_csv(path) -> np.ndarray:
    """(32768, 2) : vibrations horizontale et verticale.

    pandas (présent dans Colab) lit ~20 fois plus vite que np.loadtxt : important pour
    les ~9 000 fichiers du jeu complet.
    """
    try:
        import pandas as pd
        return pd.read_csv(path, dtype=np.float32).to_numpy()[:, :2]
    except ImportError:
        return np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)


def snapshot_features(x: np.ndarray, rpm: float, fs: float = XJTU_FS):
    """Caractéristiques d'un enregistrement (2 voies) -> vecteur 1D + noms."""
    f, names = window_features(np.asarray(x, dtype=np.float64).T, fs, "vib", np.full(2, rpm), XJTU_BEARING)
    return f.reshape(-1), [f"{axis}_{n}" for axis in ("h", "v") for n in names]


def build_xjtu_features(root, cache_file=None, log=print) -> dict:
    """Caractéristiques de chaque enregistrement de chaque roulement (mises en cache .npz)."""
    if cache_file and Path(cache_file).exists():
        z = np.load(cache_file, allow_pickle=True)
        return z["data"].item()
    data, names = {}, None
    for b in scan_xjtu(root):
        feats = []
        for p in b["files"]:
            v, names = snapshot_features(read_xjtu_csv(p), b["rpm"])
            feats.append(v)
        data[b["bearing"]] = {"features": np.stack(feats).astype(np.float32), "condition": b["condition"],
                              "rpm": b["rpm"], "n": len(b["files"])}
        log(f"  {b['bearing']} ({b['condition']}) : {len(b['files'])} enregistrements")
    out = {"bearings": data, "feature_names": names}
    if cache_file:
        np.savez_compressed(cache_file, data=np.array(out, dtype=object))
    return out


def rul_targets(n: int, interval_min: float = 1.0, cap: float | None = None) -> np.ndarray:
    """RUL vraie (minutes) de chaque enregistrement : temps restant jusqu'au dernier."""
    rul = (n - 1 - np.arange(n)) * interval_min
    return np.minimum(rul, cap) if cap else rul


def make_sequences(features: np.ndarray, targets: np.ndarray, seq_len: int):
    """Séquences glissantes des `seq_len` derniers enregistrements -> RUL du dernier."""
    idx = np.arange(seq_len - 1, features.shape[0])
    X = np.stack([features[i - seq_len + 1:i + 1] for i in idx])
    return X, targets[idx], idx


def build_gru(n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
    import torch
    from torch import nn

    class GRURUL(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(n_features, hidden, num_layers=layers, batch_first=True,
                              dropout=dropout if layers > 1 else 0.0)
            self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        def forward(self, x):
            out, _ = self.gru(x)
            return torch.sigmoid(self.head(out[:, -1])).squeeze(-1)  # RUL / cap, dans [0, 1]

    return GRURUL()


def train_gru(X, y_scaled, epochs=60, lr=1e-3, batch=64, seed=0, device="cpu", hidden=64, layers=2):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    model = build_gru(X.shape[2], hidden, layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_scaled, dtype=torch.float32, device=device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(Xt.shape[0], generator=g).to(device)
        for i in range(0, Xt.shape[0], batch):
            b = perm[i:i + batch]
            loss = nn.functional.smooth_l1_loss(model(Xt[b]), yt[b], beta=0.05)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def leave_one_bearing_out(feat_data: dict, seq_len=30, cap=120.0, epochs=60, seed=0, device="cpu", log=print) -> dict:
    """Validation croisée « un roulement exclu » : on prédit la RUL d'un roulement jamais vu."""
    import torch
    bearings = sorted(feat_data["bearings"])
    rows, per_bearing = [], {}
    for test_b in bearings:
        tr = [b for b in bearings if b != test_b]
        Xtr, ytr = [], []
        all_tr = np.concatenate([feat_data["bearings"][b]["features"] for b in tr])
        mu, sd = all_tr.mean(0), all_tr.std(0) + 1e-6  # normalisation : roulements d'apprentissage
        for b in tr:
            f = (feat_data["bearings"][b]["features"] - mu) / sd
            X, y, _ = make_sequences(f, rul_targets(f.shape[0], cap=cap), seq_len)
            Xtr.append(X); ytr.append(y)
        model = train_gru(np.concatenate(Xtr), np.concatenate(ytr) / cap, epochs=epochs, seed=seed, device=device)
        f = (feat_data["bearings"][test_b]["features"] - mu) / sd
        X, y, idx = make_sequences(f, rul_targets(f.shape[0], cap=cap), seq_len)
        model.eval()
        with torch.no_grad():
            pred = model(torch.as_tensor(X, dtype=torch.float32, device=device)).cpu().numpy() * cap
        rmse = float(np.sqrt(np.mean((pred - y) ** 2))); mae = float(np.mean(np.abs(pred - y)))
        per_bearing[test_b] = {"rmse_min": rmse, "mae_min": mae, "n": int(y.size),
                               "condition": feat_data["bearings"][test_b]["condition"]}
        log(f"  {test_b} : RMSE = {rmse:.2f} min, MAE = {mae:.2f} min ({y.size} séquences)")
        for i, t, p in zip(idx, y, pred):
            rows.append({"bearing": test_b, "t_min": int(i), "rul_true_min": float(t), "rul_pred_min": float(p)})
    all_t = np.array([r["rul_true_min"] for r in rows]); all_p = np.array([r["rul_pred_min"] for r in rows])
    return {"per_bearing": per_bearing, "rows": rows,
            "rmse_min": float(np.sqrt(np.mean((all_p - all_t) ** 2))), "mae_min": float(np.mean(np.abs(all_p - all_t))),
            "rmse_per_bearing": [v["rmse_min"] for v in per_bearing.values()],
            "mae_per_bearing": [v["mae_min"] for v in per_bearing.values()]}


def export_rul(out_dir, feat_data: dict, lobo: dict, seq_len: int, cap: float, epochs: int, seed: int = 0,
               device="cpu", hidden: int = 64, layers: int = 2) -> Path:
    """Entraîne le GRU final sur tous les roulements et écrit artifacts/rul/."""
    import torch
    from .utils import save_json
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bearings = sorted(feat_data["bearings"])
    allf = np.concatenate([feat_data["bearings"][b]["features"] for b in bearings])
    mu, sd = allf.mean(0), allf.std(0) + 1e-6
    Xs, ys = [], []
    for b in bearings:
        f = (feat_data["bearings"][b]["features"] - mu) / sd
        X, y, _ = make_sequences(f, rul_targets(f.shape[0], cap=cap), seq_len)
        Xs.append(X); ys.append(y)
    model = train_gru(np.concatenate(Xs), np.concatenate(ys) / cap, epochs=epochs, seed=seed, device=device,
                      hidden=hidden, layers=layers)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / "rul_model.pt")
    save_json({"dataset": "XJTU-SY", "fs": XJTU_FS, "interval_min": 1.0, "seq_len": seq_len, "cap_min": cap,
               "hidden": hidden, "layers": layers, "feature_names": feat_data["feature_names"],
               "feature_mean": mu, "feature_std": sd, "bearing_geometry": XJTU_BEARING,
               "bearings": {b: {"condition": v["condition"], "rpm": v["rpm"], "n": v["n"]}
                            for b, v in feat_data["bearings"].items()}}, out / "rul_config.json")
    save_json({k: v for k, v in lobo.items() if k != "rows"}, out / "rul_metrics.json")
    write_rows_csv(lobo["rows"], out / "rul_predictions.csv")
    return out


class RULPredictor:
    """Prédit une trajectoire de RUL à partir d'une SÉRIE d'enregistrements d'un même roulement."""

    def __init__(self, rul_dir, device="cpu"):
        import torch
        from .utils import load_json
        self.dir = Path(rul_dir)
        self.cfg = load_json(self.dir / "rul_config.json")
        self.model = build_gru(len(self.cfg["feature_names"]), self.cfg["hidden"], self.cfg["layers"])
        self.model.load_state_dict(torch.load(self.dir / "rul_model.pt", map_location=device, weights_only=True))
        self.model.eval()
        self.mu = np.asarray(self.cfg["feature_mean"]); self.sd = np.asarray(self.cfg["feature_std"])

    def predict_series(self, snapshots: list[np.ndarray], rpm: float, fs: float | None = None) -> dict:
        """snapshots : liste ordonnée de tableaux (N, 2) [horizontal, vertical], un par minute."""
        import torch
        fs = fs or self.cfg["fs"]
        L = self.cfg["seq_len"]
        if len(snapshots) < L:
            raise ValueError(f"Il faut au moins {L} enregistrements successifs (reçu {len(snapshots)}).")
        feats = []
        for x in snapshots:
            x = np.asarray(x, dtype=np.float64)
            if x.ndim != 2 or x.shape[1] != 2:
                raise ValueError("Chaque enregistrement doit avoir 2 colonnes (vibrations horizontale et verticale).")
            feats.append(snapshot_features(x, rpm, fs)[0])
        f = (np.stack(feats) - self.mu) / self.sd
        X, _, idx = make_sequences(f, np.zeros(len(f)), L)
        with torch.no_grad():
            pred = self.model(torch.as_tensor(X, dtype=torch.float32)).numpy() * self.cfg["cap_min"]
        return {"t_min": idx.tolist(), "rul_min": pred.tolist(), "cap_min": self.cfg["cap_min"]}


def write_rows_csv(rows: list[dict], path) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
