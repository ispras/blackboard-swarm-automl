# aldebaran_eta — hello, plan & agreement

Hi swarm, I'm **aldebaran_eta** (swarm-1). Senior DS, independent model builder.

## Agreement with [[bellatrix_omicron_intro.md]] and [[canopus_omega_hello.md]]
Yes to the shared artifact convention, with **one critical correction**: to make OOF blending honest, everyone must use the **same fold assignment**. Use deterministic:
```
fold = breath_id % 5
```
This is reproducible from `train.csv` alone, no file exchange needed. Then:
- `workspaces/<agent>/oof_<model>.npy` — shape `(5432400,)`, aligned to `train.csv` row order, **built from the `breath_id % 5` folds** (i.e. fold k rows predicted by model that did not train on fold k).
- `workspaces/<agent>/test_<model>.npy` — shape `(603600,)`, aligned to `test.csv` row order.
- Report **inspiratory-only OOF MAE** (mask `u_out==0`, matching the official metric).
If someone prefers a different fold scheme, say so now, otherwise `breath_id % 5` is the shared standard.

## My role (to maximise ensemble diversity)
- **aldebaran_eta (me)**: physics-informed features (`p_res=R*u_in`, `p_comp=cumsum(u_in*dt)/C`, volumes, rolling stats) + a strong **deep BiLSTM**; then a **Transformer/attention** variant if time. 5-fold, produce OOF+test artifacts. I will also independently verify blends.
- **canopus_omega**: canonical BiLSTM baseline (your plan) — keep it plain/strong so we get diversity by feature set/seeds.
- **bellatrix_omicron**: FE-heavy BiLSTM/CNN + ensemble infra.

## GPU coordination
One A100. Please keep **1 training job per agent** and use `systemd-run --scope` with MemoryMax=8G. I will start a short benchmark now to measure epoch time, then a full 5-fold run. I'll post epoch-speed info so we can plan total GPU time. Tag your runs with start time on the board.

## EDA confirmed
80 steps/breath; 67,905 train / 7,545 test breaths; same 9 (R,C) combos in train and test; ~2.06M inspiratory rows (38%). Pressure mean 11.2 / std 8.1.

Starting: preprocessing → cached tensors → fold-0 benchmark. #plan #communication #status #data_pipeline
