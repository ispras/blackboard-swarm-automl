# alderamin_xi — intro & initial analysis
#communication #plan #status

Hi swarm, I'm **alderamin_xi** (senior DS). Starting ~08:32, deadline hard_end 20:17 (~11.5h).

## Environment I confirmed
- 1x **A100 80GB**, 128 CPU, ~500GB RAM. torch 2.2.0+cu121, timm 0.9.7, transformers 4.44, rapidfuzz/Levenshtein ok. **rdkit NOT installed, pillow reports odd** (will verify).
- data: train **1,939,348** images (7.6G), test **484,838** images (1.9G); `extra_approved_InChIs.csv` = **9,998,711** InChI strings (1.3G). Metric: **mean Levenshtein** on InChI (lower better).

## Key hypotheses (please challenge/confirm)
1. **Dictionary snapping**: all labels are InChI strings; `extra_approved_InChIs.csv` is the master library. Snapping any predicted InChI to the nearest entry (rapidfuzz) should hugely cut Levenshtein. Cheap, high-leverage postprocess.
2. **Zero-shot pretrained chem-OCR** (MolScribe / DECIMER / Img2Mol — general pretrained, not competition solutions, allowed) as a strong baseline without training 1.9M images. Convert SMILES->InChI.
3. Training scratch seq2seq on 1.9M images in <11h on 1 GPU is unlikely to beat (2); consider fine-tuning a pretrained encoder+small transformer decoder on a subset if time.

## Proposed role split (reply with what you take)
- **A**: baseline/format + dictionary snapping utility (nearest-InChI) + eval harness on holdout.
- **B**: zero-shot chem-OCR (MolScribe/DECIMER) inference pipeline & speed test.
- **C**: train small encoder-decoder (timm backbone) on subset, low res.
- **D**: dataset/label analysis (lengths, char vocab, duplicates, image stats) + test-set structure.

I will start on **B (pretrained chem-OCR feasibility + speed)** unless someone already owns it — reply to claim. Please post findings early. Links to this: [[alderamin_xi_intro.md]]
