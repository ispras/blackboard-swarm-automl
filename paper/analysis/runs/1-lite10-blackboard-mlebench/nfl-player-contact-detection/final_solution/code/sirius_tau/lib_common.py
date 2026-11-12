import numpy as np

def best_thr(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(-p)
    ys = y[order]; ps = p[order]
    P = np.cumsum(ys)
    totalP = P[-1]; N_total = len(ys) - totalP
    TP = P
    FP = (np.arange(len(ys)) + 1.0) - P
    FN = totalP - TP
    TN = N_total - FP
    denom = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = np.where(denom > 0, (TP * TN - FP * FN) / denom, 0.0)
    k = int(np.argmax(mcc))
    return float(ps[k]), float(mcc[k])
