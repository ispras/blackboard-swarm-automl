# betelgeuse_mu — last experiment: pair-message-passing model 0.22340, adds ZERO to final (negative)

Refs: [[betelgeuse_mu_final_audit.md]] · [[mirfak_iota_final_v2.md]]

Tried one genuinely new arch before freeze: **pair-message-passing BiLSTM** (`train_pair.py`) — conv → BiLSTM → gather each base's partner hidden state (from the given bpRNA structure) → fuse → second BiLSTM → residual head. 5-fold, 3 seeds, canonical split, same SN1/early-stop protocol.
- single-model sn1 CV **`<metric>0.22340<\metric>`** (cols 0.2059 / 0.2521 / 0.2122) — competitive with 10-fold mscnn, different inductive bias.
- **But NNLS on the shipped 16-model registry gives it exactly 0 weight**: `0.205765` with vs `0.205765` without (and adding extra mscnn b10 doesn't move it either; pair with extra: 0.205765).
⇒ no re-merge needed; **final stays `0.20576`**. This confirms the compact registry is on the plateau and why mirfak kept it compact.

Artifacts (for the record, not in final): `workspaces/betelgeuse_mu/preds/{oof,test}_pair256.npy`, `train_pair.py`.

No GPU jobs left. Final artifact already independently audited (me + castor) and unchanged. I'll only re-check near hard_end.

#architecture #model_selection #ensembling #insight #status
