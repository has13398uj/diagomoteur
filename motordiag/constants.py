"""Constantes du projet : jeu de données Paderborn (KAt-DataCenter) et roulement 6203.

Sources : documentation officielle du KAt-DataCenter (Université de Paderborn) et
Lessmeier et al., "Condition Monitoring of Bearing Damage in Electromechanical Drive
Systems by Using Motor Current Signals of Electric Motors", PHM Europe 2016.
"""

# --- Classes -----------------------------------------------------------------------
# Paderborn ne contient AUCUN défaut de bille. Les roulements KB23/KB24/KB27 ont des
# dommages combinés bague intérieure + bague extérieure : la classe s'appelle donc
# "Combiné (BI+BE)" et non "Ball".
CLASS_KEYS = ["healthy", "inner", "outer", "combined"]
CLASS_NAMES_FR = ["Sain", "Bague intérieure (BI)", "Bague extérieure (BE)", "Combiné (BI+BE)"]
CLASS_SHORT_FR = ["Sain", "BI", "BE", "BI+BE"]

# --- Roulements Paderborn -> (classe, origine du dommage) -------------------------
# origine : "none" (sain), "artificial" (EDM / gravure / perçage),
#           "real" (dommage obtenu par essai de durée de vie accéléré)
PADERBORN_BEARINGS = {
    "K001": (0, "none"), "K002": (0, "none"), "K003": (0, "none"),
    "K004": (0, "none"), "K005": (0, "none"), "K006": (0, "none"),
    "KI01": (1, "artificial"), "KI03": (1, "artificial"), "KI05": (1, "artificial"),
    "KI07": (1, "artificial"), "KI08": (1, "artificial"),
    "KI04": (1, "real"), "KI14": (1, "real"), "KI16": (1, "real"),
    "KI17": (1, "real"), "KI18": (1, "real"), "KI21": (1, "real"),
    "KA01": (2, "artificial"), "KA03": (2, "artificial"), "KA05": (2, "artificial"),
    "KA06": (2, "artificial"), "KA07": (2, "artificial"), "KA08": (2, "artificial"),
    "KA09": (2, "artificial"),
    "KA04": (2, "real"), "KA15": (2, "real"), "KA16": (2, "real"),
    "KA22": (2, "real"), "KA30": (2, "real"),
    "KB23": (3, "real"), "KB24": (3, "real"), "KB27": (3, "real"),
}

# --- Conditions de fonctionnement (codées dans le nom de fichier) -----------------
# N = vitesse (x100 tr/min), M = couple de charge (x0.1 N.m), F = force radiale (x100 N)
OPERATING_CONDITIONS = {
    "N15_M07_F10": {"rpm": 1500, "torque_Nm": 0.7, "radial_force_N": 1000},
    "N09_M07_F10": {"rpm": 900, "torque_Nm": 0.7, "radial_force_N": 1000},
    "N15_M01_F10": {"rpm": 1500, "torque_Nm": 0.1, "radial_force_N": 1000},
    "N15_M07_F04": {"rpm": 1500, "torque_Nm": 0.7, "radial_force_N": 400},
}

PADERBORN_FS = 64000  # Hz, vibration et courants moteur
PADERBORN_VIB_CHANNEL = "vibration_1"
PADERBORN_CUR_CHANNEL = "phase_current_1"
PADERBORN_SPEED_CHANNEL = "speed"

# --- Géométrie du roulement 6203 (banc Paderborn) ---------------------------------
# 8 billes, diamètre de bille 6.75 mm, diamètre primitif 28.55 mm, angle de contact 0°.
BEARING_6203 = {
    "name": "6203 (Paderborn)",
    "n_balls": 8,
    "ball_diameter_mm": 6.75,
    "pitch_diameter_mm": 28.55,
    "contact_angle_deg": 0.0,
}
