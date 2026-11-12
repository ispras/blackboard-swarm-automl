import json
import os
import numpy as np
import common as C

M = "/download_job_code/payload/sandbox/workspaces/mirfak_iota/"
BASE_REG = "models_final_c.json"
SEED_TAGS = {
    "struct_scored": ["s42", "s100", "s7", "s13", "s17", "s23", "s29", "s31", "s37", "s41", "s43"],
    "struct_all5": ["all5_s42", "all5_s100", "all5_s7", "all5_s13", "all5_s17", "all5_s23", "all5_s29", "all5_s31", "all5_s41", "all5_s43", "all5_s47"],
    "struct_huber": ["huber_s42", "huber_s100"],
}

tr = C.load_train()
Y = C.make_targets(tr)
sn = np.array([r["SN_filter"] for r in tr], dtype=bool)

avgs = []
for name, tags in SEED_TAGS.items():
    oofs, tests, used = [], [], []
    for t in tags:
        op = M + f"pred_struct2_{t}_oof.npy"
        tp = M + f"pred_struct2_{t}_test.npy"
        if os.path.exists(op) and os.path.exists(tp):
            oofs.append(np.load(op))
            tests.append(np.load(tp))
            used.append(t)
    if not oofs:
        continue
    o = np.mean(oofs, 0).astype(np.float32)
    te = np.mean(tests, 0).astype(np.float32)
    m, r = C.mcrmse(Y[sn], o[sn])
    print(f"{name}: seeds={used} n={len(used)} sn1={m:.5f} cols={['%.4f' % x for x in r]}")
    np.save(M + f"pred_{name}_oof.npy", o)
    np.save(M + f"pred_{name}_test.npy", te)
    avgs.append({"name": "my_" + name, "oof": M + f"pred_{name}_oof.npy",
                 "test": M + f"pred_{name}_test.npy"})

reg = json.load(open(M + BASE_REG))
reg = [m for m in reg if not m["name"].startswith("my_struct")]
reg += avgs
json.dump(reg, open(M + "models_final.json", "w"), indent=1)
print("final registry models:", len(reg), [m["name"] for m in avgs])
