# capella_kappa — status: Swin-Base launched (3rd job)

Refs: [[arcturus_kappa_split_alignment.md]], [[achernar_delta_role_claim.md]], [[capella_kappa_split_fix.md]]

Ack: confirmed `folds5_seed42.csv` == arcturus `folds.csv` (StratifiedKFold 5, seed42). Good.

@arcturus_kappa: my Swin launch crossed your message — sorry. It's already running. Current GPU jobs (3):
- arcturus: `effb4_512 --allfolds` (5x12ep)
- achernar: `convnext_small @384 --folds 0,1,2,3,4` (f0 ep1 val 0.8021 ✓ correct split now)
- me: `swin_base_patch4_window12_384.ms_in22k_ft_in1k` @384, fold 0 (train on folds!=0)

3 jobs on one A100 is the sensible max (mem ~50GB/80GB). **Proposal: nobody launches a 4th heavy job without announcing.** Total throughput is roughly constant; diversity is worth it.

## My plan
1. Swin-Base @384 fold0, 10 ep -> val_acc + test probs to `workspaces/capella_kappa/swin_base_384/`.
2. If time: 2nd transformer (`swin_large` or `vit_base`) or fold-bagging. Will announce.
3. I'm happy to also help with the ensembler/stacker if achernar wants a second pair of eyes.

## Reminder to all — artifact contract
`val_probs.npy` (rows = fold-0 val, aligned to `folds5_seed42.csv` fold==0 order) + `test_probs.npy` (2676x5, sample_submission order) + report `<metric>val_acc<\metric>`.

#status #communication #plan #ensembling #architecture
