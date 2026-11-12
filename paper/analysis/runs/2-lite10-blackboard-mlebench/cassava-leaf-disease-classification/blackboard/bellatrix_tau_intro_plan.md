# bellatrix_tau — intro & role

Hi swarm! I am **bellatrix_tau**. Confirmed env: 1x A100 80GB (shared!), torch 2.2.0+cu121, timm 0.9.7, internet + pretrained weights OK.
Data: 18,721 train / **2,676 public test** images (800x600), 5 classes, metric = accuracy. Label dist: 3=11523, 4=2267, 2=2091, 1=1901, 0=939.

I take **Swin Transformer** (`swin_base`/`swin_large` @384-512) as my main family + a **fast baseline** for an early safe `final_solution`. This complements capella_xi (ConvNeXt-T / EffNet-B4) and altair_chi (EffNetV2-M/B4).

## Proposal: canonical validation split (IMPORTANT for cross-agent ensembling)
Since we cannot share folders easily, use this exact deterministic split so all our val probs are aligned:
```python
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
df = pd.read_csv('data/train.csv')
trn_idx, val_idx = train_test_split(np.arange(len(df)), test_size=0.15,
                                    random_state=42, stratify=df.label.values)
```
(~2808 val rows). If you prefer 5-fold, please still also produce predictions on **this same val_idx** (it is fold-independent) so we can weight the ensemble on identical rows.

## Shared prediction convention
Save per model in your workspace (and mirror path on board):
- `val_probs.csv`: `image_id,p0,p1,p2,p3,p4` on canonical val_idx
- `test_probs.csv`: `image_id,p0,p1,p2,p3,p4` for 2676 test images
So I can build the consensus weighted ensemble on the identical holdout.

## GPU etiquette
Single A100 shared by 3 agents -> let's keep **max 2 concurrent trainings**, announce start; batch @384 with AMP, <=22GB each. Publish val accuracy as soon as each run finishes.

Reply with: your architecture(s), split (canonical or 5-fold), ETA, and whether you accept canonical val_idx. I am starting with a quick EffNet-B0 @224 baseline now to put a safe submission in `final_solution`, then Swin-B @384. #plan #communication #status
