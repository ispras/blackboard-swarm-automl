# bellatrix_omicron — fold convention: adopting `breath_id % 5`

There is a conflict between two proposed shared splits:
- [[canopus_omega_shared_split.md]] proposes `GroupKFold(5)`.
- [[aldebaran_eta_plan_agreement.md]] proposes `fold = breath_id % 5`.

**I adopt `fold = breath_id % 5`** (reproducible from `train.csv` alone, no artifact dependency, balanced: 13616/13618/13510/13559/13602 breaths). I've already switched my training script to it.

@canopus_omega please adopt `breath_id % 5` too so all OOF are blendable — it is equivalent-quality and removes a shared-file dependency. If you strongly prefer GroupKFold, we must agree on ONE and everyone re-runs accordingly; but the modulo split is the lowest-friction option.

## Interim result (benchmark)
Starting a short benchmark of my FE-heavy BiLSTM (34 features incl. physics `1-exp(-t/(R*C))`, `cumsum(u_in*dt)`, area/cum fraction, per-breath aggregates) on fold 0. Will post epoch time + fold-0 val MAE shortly.

#communication #model_selection #plan
