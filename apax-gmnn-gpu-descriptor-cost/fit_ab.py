"""Linear and log2-linear fits of final val force MAE against A (n_radial) and B (n_radial_tensor).

usage: python fit_ab.py   (reads logs/<run>.csv, last row = epoch 399)
"""

import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# run: (A, B)
FULL = {"p1_B2": (12, 2), "p1_B3": (12, 3), "v2B4_s1": (12, 4), "p1_B5": (12, 5), "p1_A16B5": (16, 5),
        "p1_B6": (12, 6), "p1_B8": (12, 8)}
L2 = {"v2L3_s1": (12, 4), "p1_L3B6": (12, 6), "p1_L3A24": (24, 4), "p1_L3A32": (32, 4), "p1_L3B10": (12, 10)}


def val_f(run):
    rows = list(csv.DictReader(open(f"{HERE}/logs/{run}.csv")))
    return 1000 * float(rows[-1]["val_forces_mae"])


def fit(runs, cols, names):
    A, B = (np.array([runs[r][i] for r in runs], dtype=float) for i in (0, 1))
    F = np.array([val_f(r) for r in runs])
    X = np.column_stack([np.ones_like(A)] + [g(A, B) for g in cols])
    c = np.linalg.lstsq(X, F, rcond=None)[0]
    res = F - X @ c
    print("  " + "  ".join(f"{n}={v:+.2f}" for n, v in zip(["const"] + names, c)),
          f"  rms {np.sqrt((res**2).mean()):.2f}  max {abs(res).max():.2f}")


for name, runs in (("l<=3", FULL), ("l<=2", L2)):
    print(name)
    fit(runs, [lambda A, B: A, lambda A, B: B], ["A", "B"])
    fit(runs, [lambda A, B: np.log2(A), lambda A, B: np.log2(B)], ["log2A", "log2B"])
