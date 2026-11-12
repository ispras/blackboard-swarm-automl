import numpy as np


def params_single(b, variant="A"):
    if b is None:
        return 0.0, 0.0, 0.0, 0
    cx = float(b[0])
    if variant == "A":
        cy = float(b[1]) + 0.9 * float(b[3])
        side = max(4.5 * max(float(b[2]), float(b[3])), 56.0)
    elif variant == "B":
        cy = float(b[1]) + 1.7 * float(b[3])
        side = max(3.2 * max(float(b[2]), float(b[3])), 48.0)
    elif variant == "C":
        cy = float(b[1]) + 0.6 * float(b[3])
        side = max(6.5 * max(float(b[2]), float(b[3])), 72.0)
    else:
        raise ValueError(variant)
    return cx, cy, side, 1
