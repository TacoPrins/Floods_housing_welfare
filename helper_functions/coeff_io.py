# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:27:22 2026

@author: tprins
"""

"""
coeff_io.py

Round-trips model coefficient sets to/from an Excel file so that solved
coefficients from one run can warm-start the next run's guesses.

Layout: one sheet, one row per named coefficient set.
Columns: name, c0, c1, c2, c3, c4
"""

import os
import numpy as np
import pandas as pd

COEFF_FILE = "coefficients.xlsx"
N_COEFF = 5  # c0..c4

# Hardcoded fallback guesses, used when a name is absent from the Excel file.
DEFAULTS = {
    "vCoeff_C_initial_HE":  [0.63017547, 0., 0., 0., 0.],
    "vCoeff_NC_initial_HE": [0.7047178,  0., 0., 0., 0.],
    "vCoeff_C_initial_RE":  [0.63017547, 0., 0., 0., 0.],
    "vCoeff_NC_initial_RE": [0.7047178,  0., 0., 0., 0.],
    "vCoeff_C_HE":  [0.56764294, -0.06088019,  0.00295728,  0.0084201,   0.0014617],
    "vCoeff_NC_HE": [0.73139782,  0.02410277, -0.00152038, -0.00262021,  0.00075086],
    "vCoeff_C_RE": [ 0.56068014, -0.06077558,  0.0046855,   0.00715913,  0.00060609],
    "vCoeff_NC_RE": [0.7336555,   0.02041293, -0.00074853, -0.00112876, -0.00116299],
    "vCoeff_C_BR":  [5.75967988e-01, -2.35610169e-02, 1.78769229e-04, 1.58161220e-02, -4.34140202e-03],
    "vCoeff_NC_BR": [7.29983289e-01, 2.56602216e-02, -7.85866181e-03, -1.40065192e-03, -1.36957527e-04],
    "vCoeff_C_MP":  [0.56776576, -0.06110282,  0.00295317,  0.00840279,  0.001457],
    "vCoeff_NC_MP": [0.73157062,  0.02379444, -0.00153431, -0.00262569,  0.00075946],
    "vCoeff_C_terminal_RE": [0.58944375, 0., 0., 0., 0.],
    "vCoeff_NC_terminal_RE": [0.85491565, 0., 0., 0., 0.],
    "vCoeff_C_terminal_HE": [0.64583997, 0., 0., 0., 0.],
    "vCoeff_NC_terminal_HE": [0.81916869, 0., 0., 0., 0.],
    "vCoeff_C_terminal_BR": [0.69186954, 0., 0., 0., 0.],
    "vCoeff_NC_terminal_BR": [0.83346934, 0., 0., 0., 0.],
    "vCoeff_C_terminal_MP": [0.64583997, 0., 0., 0., 0.],
    "vCoeff_NC_terminal_MP": [0.81916869, 0., 0., 0., 0.],
}


def load_with_defaults(path=COEFF_FILE):
    """Load coefficients, filling any missing name from DEFAULTS."""
    saved = load_coefficients(path)
    return {name: get_guess(saved, name, default)
            for name, default in DEFAULTS.items()}


def load_coefficients(path=COEFF_FILE):
    """Return {name: np.array([...])} from the Excel file, or {} if absent."""
    if not os.path.exists(path):
        return {}
    df = pd.read_excel(path)
    cols = [f"c{i}" for i in range(N_COEFF)]
    return {row["name"]: row[cols].to_numpy(dtype=float) for _, row in df.iterrows()}


def get_guess(saved, name, default):
    """Saved coefficients for `name` if present, else the hardcoded default."""
    return saved.get(name, np.asarray(default, dtype=float))


def save_coefficients(coeff_dict, path=COEFF_FILE):
    """Write {name: array} to Excel, one labelled row per set."""
    rows = []
    for name, arr in coeff_dict.items():
        arr = np.asarray(arr, dtype=float).ravel()
        row = {"name": name}
        row.update({f"c{i}": arr[i] for i in range(N_COEFF)})
        rows.append(row)
    df = pd.DataFrame(rows, columns=["name"] + [f"c{i}" for i in range(N_COEFF)])
    df.to_excel(path, index=False)
    return path