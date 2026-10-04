"""motordiag : diagnostic de défauts de roulements (vibration + courant moteur).

Module partagé par le notebook d'entraînement et le dashboard :
- io.py              lecture des fichiers (Paderborn .mat, CSV, NPY)
- preprocessing.py   ré-échantillonnage et fenêtrage (identiques entraînement / inférence)
- signal_analysis.py FFT, enveloppe, STFT, indicateurs, fréquences de défaut
- models.py          réseaux PyTorch (normalisation et représentation incluses)
- inference.py       chargement de artifacts/ et diagnostic
- data.py, training.py, features.py : côté entraînement (notebook)
- rul.py             pronostic RUL sur données run-to-failure (XJTU-SY, optionnel)
"""
__version__ = "1.0.0"
