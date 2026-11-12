import os, sys
import numpy as np
from common import WS, load_data, logloss

# usage: python make_seedavg.py NAME USE_EPAVG tag1 tag2 ...
name = sys.argv[1]
use_ep = sys.argv[2] == "epavg"
tags = sys.argv[3:]

tr, te, y = load_data()
oofs, tests = [], []
for t in tags:
    sfx = "_epavg" if use_ep else ""
    op = os.path.join(WS, f"oof_{t}{sfx}.npy")
    tp = os.path.join(WS, f"test_{t}{sfx}.npy")
    if not (os.path.exists(op) and os.path.exists(tp)):
        print("MISSING", op); sys.exit(1)
    o = np.load(op); tt = np.load(tp)
    print(t, "OOF", round(logloss(y, o), 5))
    oofs.append(o); tests.append(tt)

so = np.mean(oofs, 0)
st = np.mean(tests, 0)
print("seedavg", name, "OOF", round(logloss(y, so), 5))
np.save(os.path.join(WS, f"oof_{name}.npy"), so)
np.save(os.path.join(WS, f"test_{name}.npy"), st)
print("saved", f"oof_{name}.npy")
