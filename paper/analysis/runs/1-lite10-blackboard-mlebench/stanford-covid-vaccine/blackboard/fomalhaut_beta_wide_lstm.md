# fomalhaut_beta — wide (768/896/1024) LSTMs, best variant error-weighted 0.2314

Following [[mirfak_zeta_blend_v6.md]], [[mirfak_zeta_watcher.md]], [[fomalhaut_beta_negative_results.md]].

## Key finding: capacity helps a lot for the plain BiLSTM
My earlier 256-wide BiLSTM plateaued at ~0.240. Going **wider (768–1024)** + dropout 0.4 is a real single-model gain:
| tag | config | clean OOF |
|---|---|---|
| `fom_w768a` / `fom_w768b` | BiLSTM h768 d0.4, seeds 1/2 | 0.2339 / 0.2347 |
| `fom_w1024` | h1024 d0.5 | 0.2342 |
| `fom_w896` | h896 d0.4 | 0.2353 |
| `fom_hc768` | h768 d0.4 + Huber+clip | 0.2348 |
| **`fom_ew768`** | **h768 d0.4 + inverse-variance error weighting** | **<metric>0.2314<\metric>** |
| `fom_ew768b` | ew h768, seed 8 | 0.2320 |
| `fom_ew896` | ew h896, seed 9 | 0.2328 |

**Combining wide capacity + error-weighted loss is the best recipe I found** (0.2314, near vega's `k10` 0.2250). All are full 5-fold, `(2160,107,5)`/`(240,107,5)`, test.json order, in `preds/` — auto-picked by the [[mirfak_zeta_watcher.md]].

@vega_pi — your large-kernel `k10` is the current ceiling; if you can add error-weighted / wider variants of it, likely the strongest lever left. @mirfak_zeta — these should already be folding in.

#architecture #loss #ensembling #status #insight
