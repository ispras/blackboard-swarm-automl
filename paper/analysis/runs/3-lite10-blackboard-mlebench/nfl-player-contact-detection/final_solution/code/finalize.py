import json, os, subprocess, sys
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "workspaces/alkaid_psi")
FIN = os.path.join(ROOT, "final_solution")
CAN = "workspaces/canopus_pi/preds"
CAP = "workspaces/capella_pi"

REG = {
    # name: (test path, column, is_raw_logit)
    "capstmix": (f"{CAP}/test_stack_mix.parquet", "stack_pred", True),
    "capst7": (f"{CAP}/test_stack_v7.parquet", "stack_pred", False),
    "capst6": (f"{CAP}/test_stack_v6.parquet", "stack_pred", False),
    "capst5": (f"{CAP}/test_stack_v5.parquet", "stack_pred", False),
    "capst4": (f"{CAP}/test_stack_v4.parquet", "stack_pred", False),
    "capst3": (f"{CAP}/test_stack_v3.parquet", "stack_pred", False),
    "capst": (f"{CAP}/test_stack_v2.parquet", "stack_pred", False),
    "capst1": (f"{CAP}/test_stack.parquet", "stack_pred", False),
    "can_ens4pp": (f"{CAN}/ens_test_probs_v4.parquet", "p_pp", False),
    "can_ens4gr": (f"{CAN}/ens_test_probs_v4.parquet", "p_gr", False),
    "can_pp5": (f"{CAN}/test_pp_v5.parquet", "pred", False),
    "pp_xgb": (f"{CAN}/test_pp_xgb1.parquet", "pred", False),
    "pp_gru3": (f"{CAN}/test_pp_gru_v3.parquet", "pred", False),
    "pp_gru2": (f"{CAN}/test_pp_gru_v2.parquet", "pred", False),
    "pp_gru": (f"{CAN}/test_pp_gru_v1.parquet", "pred", False),
    "cnn_ppW": (f"workspaces/alkaid_psi/test_cnn_cand_vW1m.parquet", "pred", False),
    "can_pp": (f"{CAN}/test_pp_v2.parquet", "pred", False),
    "cnn_pp": (f"workspaces/alkaid_psi/test_cnn_cand_v2m.parquet", "pred", False),
    "cap": (f"{CAP}/test_lgbm.parquet", "pred", False),
    "can_gr4": (f"{CAN}/test_ground_v4.parquet", "pred", False),
    "gr_gru4": (f"{CAN}/test_ground_gru_v4.parquet", "pred", False),
    "gr_gru3": (f"{CAN}/test_ground_gru_v3.parquet", "pred", False),
    "gr_gru2": (f"{CAN}/test_ground_gru_v2.parquet", "pred", False),
    "gr_gru": (f"{CAN}/test_ground_gru_v1.parquet", "pred", False),
    "gr_xgb": (f"{CAN}/test_ground_xgb1.parquet", "pred", False),
    "can_gr": (f"{CAN}/test_ground_v2.parquet", "pred", False),
    "cnn_gr": (f"workspaces/alkaid_psi/test_ground_cnn_v1m.parquet", "pred", False),
    "cnn_grB": (f"workspaces/alkaid_psi/test_ground_cnn_vB1m.parquet", "pred", False),
    "cnn_grC": (f"workspaces/alkaid_psi/test_ground_cnn_vC1m.parquet", "pred", False),
}


def main():
    apply = json.load(open(f"{OUT}/blend3_apply.json"))
    used = set(apply["pp"]["weights"]) | set(apply["gr"]["weights"])
    missing = [u for u in used if u not in REG]
    assert not missing, missing
    sources = {}
    raw = []
    for u in sorted(used):
        path, col, israw = REG[u]
        assert os.path.exists(os.path.join(ROOT, path)), path
        sources[u] = {"path": path, "column": col}
        if israw:
            raw.append(u)
    cfg = {
        "sources": sources,
        "raw_logit": raw,
        "keys": {"pp": ["game_play", "p1", "p2s"], "gr": ["game_play", "p1"]},
        "pp": apply["pp"],
        "gr": apply["gr"],
    }
    json.dump(cfg, open(f"{FIN}/final_config.json", "w"), indent=2)
    print("config written; sources:", list(sources))
    r = subprocess.run([sys.executable, f"{FIN}/code/make_submission.py"], check=True)
    sub = pd.read_csv(f"{FIN}/submission.csv")
    ss = pd.read_csv(os.path.join(ROOT, "data/sample_submission.csv"))
    assert len(sub) == len(ss) and (sub.contact_id.values == ss.contact_id.values).all()
    assert set(sub.contact.unique()) <= {0, 1}
    print("verified submission", sub.shape, "pos", int(sub.contact.sum()), "rate", sub.contact.mean())


if __name__ == "__main__":
    main()
