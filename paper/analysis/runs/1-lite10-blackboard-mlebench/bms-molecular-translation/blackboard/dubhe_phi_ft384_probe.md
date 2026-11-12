# dubhe_phi — probing a higher-res (384) continuation (bounded, low-risk)

The shipped `final_solution` ([[mirfak_mu_solution_placed.md]]) is **frozen** and untouched — this is a parallel experiment; I will only propose a re-decode on a *clear* win.

## Why
[[altair_lambda_error_analysis.md]] (directional): errors concentrate in **long molecules**; model capacity/data-limited. Highest-res images are ~500px but we trained at **320**. A 384 view could read long structures better — the only untested lever left.

## What I'm running now (`finetune384.py`, GPU ~47 min, ends ~17:10)
- Init from `run1_last.pt` (= what's shipped), **interpolate encoder pos-emb 10×10 → 12×12**, train @ **384** (batch 96, low LR 6e-5 cosine), quickval on `val_true` every 3k steps.
- **Sanity result**: the 320 model decoded at 384 **without adaptation = 51.2** (exact 0%) → resolution change needs re-adaptation, as expected. The continuation is already back to loss ~0.77 and adapting.

## Go / no-go (I'll post the val_true numbers)
- **GO** only if `val_true` mean Lev **< 2.45** (clear ≥0.13 win over 2.581 / [[mirfak_mu_ship_decision.md]]).
- **NO-GO** otherwise → ship current submission unchanged.
- If GO, re-decode cost at 384 ≈ 55 min (17:10→18:05) then merge — fits before 18:35, and the current CSV stays as backup until a verified replacement exists.

@mirfak_mu: **hold the re-decode decision** until I post the val number; don't switch the artifact yet. If you'd rather not risk the window, say NO-GO and I'll stop — zero downside to the current submission either way.
@altair_lambda: your 2k independent cross-check on the **current** shipped CSV is unaffected; it may run slightly slow while I hold the GPU (~47 min). Prefer to run it now or after 17:10.

#insight #architecture #inference #model_selection #communication #status
