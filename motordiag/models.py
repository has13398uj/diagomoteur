"""Modèles PyTorch : représentations d'entrée, branches CNN, fusion vibration + courant.

Tout ce qui transforme le signal (normalisation, FFT, enveloppe, STFT) est un module
PyTorch avec ses paramètres en "buffers" : c'est sauvegardé dans le fichier de poids et
appliqué exactement de la même façon à l'entraînement et dans le dashboard.

Un modèle est entièrement décrit par un dictionnaire de configuration (`model_cfg`),
sauvegardé dans artifacts/config.json :

    {
      "arch": "resnet1d" | "cnn_gap" | "multiscale" | "original",
      "modalities": ["vib"] | ["cur"] | ["vib", "cur"],
      "inputs": {"vib": {"fs": 64000, "window": 4096, "repr": "raw", "norm": "global", ...}},
      "width": 32, "emb_dim": 128, "dropout": 0.2,
      "fusion": "attention" | "concat", "modality_dropout": 0.0,
      "n_classes": 4,
      "norm_stats": {"vib": {"mean": 0.0, "std": 1.0}}
    }
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

DEFAULT_INPUT = {
    "repr": "raw",          # raw | fft | envelope | stft
    "norm": "global",       # global (stats d'entraînement) | window (z-score par fenêtre) | none
    "fmax": None,           # coupure haute (Hz) pour fft / envelope / stft
    "env_band": None,       # bande (Hz) du passe-bande avant enveloppe ; défaut [1000, 0.45·fs]
    "stft_nfft": 256,
    "stft_hop": 64,
}


def input_cfg(model_cfg: dict, modality: str) -> dict:
    cfg = dict(DEFAULT_INPUT)
    cfg.update(model_cfg["inputs"][modality])
    return cfg


# ================================================================================
# Normalisation et représentations
# ================================================================================
class Normalizer(nn.Module):
    def __init__(self, mode: str = "global", mean: float = 0.0, std: float = 1.0):
        super().__init__()
        if mode not in ("global", "window", "none"):
            raise ValueError(f"Normalisation inconnue : {mode}")
        self.mode = mode
        self.register_buffer("mean", torch.tensor(float(mean)))
        self.register_buffer("std", torch.tensor(float(std) if std > 0 else 1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "global":
            return (x - self.mean) / self.std
        if self.mode == "window":
            mu = x.mean(dim=-1, keepdim=True)
            sd = x.std(dim=-1, keepdim=True)
            return (x - mu) / (sd + 1e-6)
        return x


class RawRepr(nn.Module):
    """Signal temporel brut : (B, L) -> (B, 1, L)."""

    def forward(self, x):
        return x.unsqueeze(1)


class FFTRepr(nn.Module):
    """log(1 + |FFT|) de la fenêtre (Hann) : (B, L) -> (B, 1, F)."""

    def __init__(self, fs: float, window: int, fmax: float | None):
        super().__init__()
        self.register_buffer("win", torch.hann_window(window, periodic=False))
        n_bins = window // 2 + 1
        if fmax is not None:
            n_bins = min(n_bins, int(fmax * window / fs) + 1)
        self.n_bins = n_bins

    def forward(self, x):
        spec = torch.fft.rfft(x * self.win, dim=-1).abs()[..., : self.n_bins]
        return torch.log1p(spec).unsqueeze(1)


class EnvelopeRepr(nn.Module):
    """Spectre d'enveloppe : passe-bande fréquentiel, signal analytique, |FFT(enveloppe)|."""

    def __init__(self, fs: float, window: int, band, fmax: float | None):
        super().__init__()
        if band is None:
            band = (1000.0, 0.45 * fs)
        freqs = torch.fft.fftfreq(window, d=1.0 / fs)
        # Filtre analytique (h) combiné au passe-bande, appliqué au spectre complet.
        h = torch.zeros(window)
        pos = (freqs > 0) & (freqs >= band[0]) & (freqs <= band[1])
        h[pos] = 2.0
        self.register_buffer("h", h)
        self.register_buffer("win", torch.hann_window(window, periodic=False))
        n_bins = window // 2 + 1
        fmax = 1000.0 if fmax is None else fmax
        self.n_bins = min(n_bins, int(fmax * window / fs) + 1)

    def forward(self, x):
        analytic = torch.fft.ifft(torch.fft.fft(x, dim=-1) * self.h, dim=-1)
        env = analytic.abs()
        env = env - env.mean(dim=-1, keepdim=True)
        spec = torch.fft.rfft(env * self.win, dim=-1).abs()[..., : self.n_bins]
        return torch.log1p(spec).unsqueeze(1)


class STFTRepr(nn.Module):
    """Spectrogramme log : les bandes de fréquence deviennent des canaux (B, F, T)."""

    def __init__(self, fs: float, n_fft: int, hop: int, fmax: float | None):
        super().__init__()
        self.n_fft, self.hop = int(n_fft), int(hop)
        self.register_buffer("win", torch.hann_window(self.n_fft))
        n_bins = self.n_fft // 2 + 1
        if fmax is not None:
            n_bins = min(n_bins, int(fmax * self.n_fft / fs) + 1)
        self.n_bins = n_bins

    def forward(self, x):
        s = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.win,
                       center=False, return_complex=True).abs()
        return torch.log1p(s[:, : self.n_bins, :])


def build_repr(cfg: dict) -> nn.Module:
    fs, window, kind = cfg["fs"], cfg["window"], cfg["repr"]
    if kind == "raw":
        return RawRepr()
    if kind == "fft":
        return FFTRepr(fs, window, cfg.get("fmax"))
    if kind == "envelope":
        return EnvelopeRepr(fs, window, cfg.get("env_band"), cfg.get("fmax"))
    if kind == "stft":
        return STFTRepr(fs, cfg.get("stft_nfft", 256), cfg.get("stft_hop", 64), cfg.get("fmax"))
    raise ValueError(f"Représentation inconnue : {kind}")


# ================================================================================
# Blocs de convolution
# ================================================================================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation (repris du notebook d'origine)."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(channels, reduced, bias=False), nn.ReLU(),
            nn.Linear(reduced, channels, bias=False), nn.Sigmoid())

    def forward(self, x):
        return x * self.se(x).unsqueeze(-1)


class ConvBlockSE(nn.Module):
    """2 × (Conv-BN-ReLU) + SE + MaxPool + Dropout (bloc du notebook d'origine)."""

    def __init__(self, in_ch, out_ch, kernel_size=3, dropout=0.4, se_reduction=8):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad), nn.BatchNorm1d(out_ch), nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad), nn.BatchNorm1d(out_ch), nn.ReLU())
        self.se = SEBlock(out_ch, se_reduction)
        self.pool = nn.MaxPool1d(2, ceil_mode=True)  # ceil_mode : pas d'erreur sur une entrée très courte
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.pool(self.se(self.block(x))))


def _stem_params(length: int) -> tuple[int, int]:
    """Noyau / pas de la première convolution selon la longueur d'entrée.

    Pour une entrée longue, un premier filtre large à pas > 1 (type WDCNN) réduit la
    longueur tôt et coûte peu de calcul.
    """
    if length <= 1024:
        stride = 1
    elif length <= 2048:
        stride = 2
    elif length <= 8192:
        stride = 4
    else:
        stride = 8
    kernel = max(7, 4 * stride + 1)
    return kernel, stride


class Stem(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, length: int):
        super().__init__()
        k, s = _stem_params(length)
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_ch),
            nn.Conv1d(in_ch, out_ch, k, stride=s, padding=k // 2, bias=False),
            nn.BatchNorm1d(out_ch), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class CNNGAPBranch(nn.Module):
    """CNN du notebook d'origine, mais terminé par un pooling global moyen (GAP).

    Le GAP remplace le Flatten -> Linear(8192, 512) (4,2 M paramètres) : moins de
    paramètres, moins de sur-apprentissage, indépendance à la longueur d'entrée.
    """

    def __init__(self, in_ch, length, width=32, dropout=0.2):
        super().__init__()
        w = width
        self.stem = Stem(in_ch, w, length)
        self.blocks = nn.Sequential(
            ConvBlockSE(w, w, 5, dropout), ConvBlockSE(w, 2 * w, 3, dropout),
            ConvBlockSE(2 * w, 2 * w, 3, dropout), ConvBlockSE(2 * w, 4 * w, 3, dropout))
        self.out_dim = 4 * w

    def forward(self, x):
        return self.blocks(self.stem(x)).mean(dim=-1)


class BasicBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel=7, dropout=0.0):
        super().__init__()
        p = kernel // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=p, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=p, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SEBlock(out_ch)
        self.drop = nn.Dropout(dropout)
        self.short = None
        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                                       nn.BatchNorm1d(out_ch))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.se(self.bn2(self.conv2(out)))
        sc = x if self.short is None else self.short(x)
        return F.relu(out + sc)


class ResNet1DBranch(nn.Module):
    """ResNet 1D (4 étages de 2 blocs résiduels avec SE), pooling global moyen."""

    def __init__(self, in_ch, length, width=32, dropout=0.1):
        super().__init__()
        w = width
        chans = [w, 2 * w, 4 * w, 4 * w]
        self.stem = nn.Sequential(Stem(in_ch, w, length), nn.MaxPool1d(3, 2, 1))
        layers, c_in = [], w
        for i, c in enumerate(chans):
            stride = 1 if i == 0 else 2
            layers += [BasicBlock1D(c_in, c, stride, dropout=dropout), BasicBlock1D(c, c, 1, dropout=dropout)]
            c_in = c
        self.layers = nn.Sequential(*layers)
        self.out_dim = chans[-1]

    def forward(self, x):
        return self.layers(self.stem(x)).mean(dim=-1)


class MultiScaleBlock(nn.Module):
    """Bloc multi-échelle type Inception : noyaux 3, 9, 27 en parallèle + max-pool."""

    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        b = out_ch // 4
        self.bottleneck = nn.Conv1d(in_ch, b, 1, bias=False)
        self.convs = nn.ModuleList([nn.Conv1d(b, b, k, padding=k // 2, bias=False) for k in (3, 9, 27)])
        self.pool_branch = nn.Sequential(nn.MaxPool1d(3, 1, 1), nn.Conv1d(in_ch, out_ch - 3 * b, 1, bias=False))
        self.bn = nn.BatchNorm1d(out_ch)
        self.se = SEBlock(out_ch)
        self.short = nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, bias=False), nn.BatchNorm1d(out_ch))
        self.pool = nn.MaxPool1d(2, ceil_mode=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        z = self.bottleneck(x)
        out = torch.cat([c(z) for c in self.convs] + [self.pool_branch(x)], dim=1)
        out = F.relu(self.se(self.bn(out)) + self.short(x))
        return self.drop(self.pool(out))


class MultiScaleBranch(nn.Module):
    def __init__(self, in_ch, length, width=32, dropout=0.1):
        super().__init__()
        w = width
        self.stem = Stem(in_ch, w, length)
        self.blocks = nn.Sequential(MultiScaleBlock(w, 2 * w, dropout), MultiScaleBlock(2 * w, 4 * w, dropout),
                                    MultiScaleBlock(4 * w, 4 * w, dropout))
        self.out_dim = 4 * w

    def forward(self, x):
        return self.blocks(self.stem(x)).mean(dim=-1)


BRANCHES = {"cnn_gap": CNNGAPBranch, "resnet1d": ResNet1DBranch, "multiscale": MultiScaleBranch}


# ================================================================================
# Modèle de diagnostic (1 ou 2 modalités)
# ================================================================================
class ModalityEncoder(nn.Module):
    """Normalisation -> représentation -> branche CNN -> projection (emb_dim)."""

    def __init__(self, model_cfg: dict, modality: str):
        super().__init__()
        cfg = input_cfg(model_cfg, modality)
        stats = model_cfg.get("norm_stats", {}).get(modality, {"mean": 0.0, "std": 1.0})
        self.norm = Normalizer(cfg["norm"], stats["mean"], stats["std"])
        self.repr = build_repr(cfg)
        with torch.no_grad():
            dummy = self.repr(torch.zeros(2, cfg["window"]))
        in_ch, length = dummy.shape[1], dummy.shape[2]
        branch_cls = BRANCHES[model_cfg["arch"]]
        self.branch = branch_cls(in_ch, length, model_cfg.get("width", 32), model_cfg.get("dropout", 0.2))
        d = model_cfg.get("emb_dim", 128)
        self.proj = nn.Sequential(nn.Linear(self.branch.out_dim, d), nn.LayerNorm(d), nn.GELU())

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = self.repr(self.norm(x.float()))
        return self.proj(self.branch(z))


class DiagnosisNet(nn.Module):
    """Classifieur à 1 modalité ou fusion de 2 modalités (concaténation ou attention).

    Fusion par attention : pour chaque fenêtre, un petit réseau note l'embedding de chaque
    capteur ; un softmax donne des poids α_vib + α_cur = 1 ; l'embedding fusionné est la
    somme pondérée. Contrairement aux deux scalaires du modèle d'origine, α dépend de
    l'échantillon et est directement comparable entre capteurs (même espace d'embedding).
    """

    def __init__(self, model_cfg: dict):
        super().__init__()
        self.cfg = copy.deepcopy(model_cfg)
        self.modalities = list(model_cfg["modalities"])
        self.encoders = nn.ModuleDict({m: ModalityEncoder(model_cfg, m) for m in self.modalities})
        d = model_cfg.get("emb_dim", 128)
        n_cls = model_cfg.get("n_classes", 4)
        p = model_cfg.get("dropout", 0.2)
        self.fusion = model_cfg.get("fusion", "attention") if len(self.modalities) > 1 else "none"
        self.modality_dropout = float(model_cfg.get("modality_dropout", 0.0))
        if self.fusion == "attention":
            self.score = nn.Sequential(nn.Linear(d, d // 2), nn.Tanh(), nn.Linear(d // 2, 1))
            head_in = d
        elif self.fusion == "concat":
            head_in = d * len(self.modalities)
        elif self.fusion == "none":
            head_in = d
        else:
            raise ValueError(f"Fusion inconnue : {self.fusion}")
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(head_in, d), nn.GELU(), nn.Dropout(p), nn.Linear(d, n_cls))

    def _modality_mask(self, batch: int, device) -> torch.Tensor:
        """Masque (B, M) : pendant l'entraînement, retire parfois un capteur entier."""
        m = len(self.modalities)
        mask = torch.ones(batch, m, device=device)
        if self.training and self.modality_dropout > 0 and m > 1:
            drop = torch.rand(batch, device=device) < self.modality_dropout
            which = torch.randint(0, m, (batch,), device=device)
            mask[drop, which[drop]] = 0.0
        return mask

    def forward(self, inputs: dict, return_details: bool = False):
        embs = [self.encoders[m](inputs[m]) for m in self.modalities]
        alpha = None
        if self.fusion == "none":
            fused = embs[0]
        else:
            e = torch.stack(embs, dim=1)                                    # (B, M, d)
            mask = self._modality_mask(e.shape[0], e.device)
            if self.fusion == "attention":
                scores = self.score(e).squeeze(-1)                          # (B, M)
                scores = scores.masked_fill(mask == 0, -1e4)
                alpha = torch.softmax(scores, dim=1)
                fused = (alpha.unsqueeze(-1) * e).sum(dim=1)
            else:
                fused = (e * mask.unsqueeze(-1)).flatten(1)
        logits = self.head(fused)
        if return_details:
            return logits, {"attention": alpha}
        return logits


# ================================================================================
# Modèle d'origine (H_M_100.ipynb), reproduit à l'identique pour la comparaison avant/après
# ================================================================================
class OriginalBranch(nn.Module):
    def __init__(self, hidden_units, seq_len, dropout_rate=0.4, branch_type="vib"):
        super().__init__()
        k1 = 7 if branch_type == "cur" else 3
        h = hidden_units
        self.block1 = ConvBlockSE(1, h, kernel_size=k1, dropout=dropout_rate)
        self.block2 = ConvBlockSE(h, h * 2, kernel_size=5, dropout=dropout_rate)
        self.block3 = ConvBlockSE(h * 2, h * 2, kernel_size=3, dropout=dropout_rate)
        self.block4 = ConvBlockSE(h * 2, h * 2, kernel_size=3, dropout=dropout_rate)
        with torch.no_grad():
            self.flat_size = self._features(torch.ones(1, 1, seq_len)).numel()

    def _features(self, x):
        return self.block4(self.block3(self.block2(self.block1(x))))

    def forward(self, x):
        return self._features(x).flatten(1)


class OriginalFusionModel(nn.Module):
    """CNN 2 branches + "WeightedFusion" (2 scalaires) + MLP, comme dans H_M_100.ipynb.

    Seule différence : la normalisation est un scalaire par capteur (et non 1024 moyennes
    par position d'échantillon), pour pouvoir l'intégrer au modèle.
    """

    def __init__(self, model_cfg: dict):
        super().__init__()
        self.cfg = copy.deepcopy(model_cfg)
        self.modalities = ["vib", "cur"]
        seq = model_cfg["inputs"]["vib"]["window"]
        h = model_cfg.get("width", 32)
        p = model_cfg.get("dropout", 0.4)
        stats = model_cfg.get("norm_stats", {})
        self.norm = nn.ModuleDict({m: Normalizer("global", **stats.get(m, {"mean": 0.0, "std": 1.0}))
                                   for m in self.modalities})
        self.branch_vib = OriginalBranch(h, seq, p, "vib")
        self.branch_cur = OriginalBranch(h, model_cfg["inputs"]["cur"]["window"], p, "cur")
        self.w_vib = nn.Parameter(torch.tensor(0.5))
        self.w_cur = nn.Parameter(torch.tensor(0.5))
        fused = self.branch_vib.flat_size + self.branch_cur.flat_size
        self.classifier = nn.Sequential(
            nn.Dropout(p), nn.Linear(fused, 512), nn.ReLU(),
            nn.Dropout(p * 0.75), nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(p * 0.5), nn.Linear(256, model_cfg.get("n_classes", 4)))

    def forward(self, inputs: dict, return_details: bool = False):
        fv = self.branch_vib(self.norm["vib"](inputs["vib"].float()).unsqueeze(1))
        fc = self.branch_cur(self.norm["cur"](inputs["cur"].float()).unsqueeze(1))
        w = torch.softmax(torch.stack([self.w_vib, self.w_cur]), dim=0)
        logits = self.classifier(torch.cat([fv * w[0], fc * w[1]], dim=1))
        if return_details:
            return logits, {"attention": w.expand(logits.shape[0], 2)}
        return logits


def build_model(model_cfg: dict) -> nn.Module:
    if model_cfg["arch"] == "original":
        return OriginalFusionModel(model_cfg)
    return DiagnosisNet(model_cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
