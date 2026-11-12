# Cassava Leaf Disease Classification — swarm solution

Metric: **accuracy**. Submission: `submission.csv` (`image_id,label`).

## Pipeline
1. **Training** (`train.py`): timm ImageNet-pretrained backbones, mixed precision (bf16), AdamW,
   cosine LR + warmup, label smoothing 0.1, augmentation (RandomResizedCrop, flip, rotate,
   brightness/contrast, HSV, coarse dropout). Stratified 5-fold, seed 42. Every per-fold model
   writes validation and test probabilities (hflip TTA).
2. **Member assembly** (`prep_members.py`): standardizes every agent's probability files to
   `image_id,p0..p4`. For members trained by other swarm agents the same `image_id` join is used.
3. **Stage A** (`stageA_stack.py`): the four full-OOF CNN members are combined by a multinomial
   logistic-regression stacker (C=0.3) on their 20 probability features, 5 seeds x 5-fold bagged.
   The stacker OOF predictions are used for weight selection downstream (E1 OOF accuracy 0.8995).
4. **Stage B** (`final_blend.py`, `stageB_analysis.py`): E1 plus the strong single-split members
   (Swin-B seeds 42/123, EffNetV2-L, ConvNeXt-Base, EffNet-B0) are combined on the shared
   canonical holdout (2,809 rows, `train_test_split(0.15, seed=42)`). Equal weights were selected
   by repeated 3x5-fold CV against log-loss / greedy / temperature+bias / geometric / power-mean.
5. `argmax` of the final probabilities is written to `submission.csv`.

## Members used in the shipped submission
- ConvNeXt-Tiny @384, 5-fold (OOF acc 0.8908)
- ConvNeXt-Tiny @384 -> 512 fine-tune, 5-fold (OOF acc 0.8907)
- ConvNeXt-Small @384, 5-fold (OOF acc 0.8911)
- EfficientNetV2-M @384, 5-fold (agent altair_chi, OOF acc 0.8933)
- Swin-Base @384 seed42, 8-view rotation TTA, test = mean(85%-model, full-data model)
  (agent bellatrix_tau, holdout acc 0.9071)
- Swin-Base @384 seed123, same recipe (holdout acc 0.9057)
- EfficientNetV2-L @384, canonical split (agent altair_chi, holdout acc 0.9071)
- ConvNeXt-Base @384, canonical split (agent altair_chi, holdout acc 0.8996)
- EfficientNet-B0 @224 (agent bellatrix_tau, small diversity weight)

Stage-A is a **logistic-regression stacker** (C=0.3, 5x5-fold bagged) over the four full-OOF
CNN members (ConvNeXt-T @384, ConvNeXt-T 512FT, ConvNeXt-Small, EffNetV2-M); it raises E1 OOF
accuracy from 0.8964 (weighted average) to **0.8995** (verified on all 18,721 OOF rows).

Stage-B uses **equal weights** over E1 + Swin42 + Swin123 + EffNetV2-L + ConvNeXt-Base +
EffNet-B0 (selected by repeated 3x5-fold CV on the 2,809-row canonical holdout:
equal 0.9167 vs log-loss 0.9133, logloss+temp 0.9129, greedy 0.9124).
Final holdout accuracy **0.9167**. Swin test probabilities are the mean of the 85%-trained
and 100%-data fine-tuned models (validation probs remain the honest 85%-model, so no leakage).

## Files
- `train.py` — training / per-fold inference
- `infer.py` — checkpoint inference with configurable TTA
- `assemble.py` — merge per-fold probs into OOF/test CSVs
- `prep_members.py` — align all swarm members onto `image_id`
- `stageA_stack.py` — Stage-A logistic-regression stacker (OOF + test)
- `final_blend.py` — Stage-B ensemble + CV strategy selection (also contains Stage-A utilities)
- `stageB_analysis.py` — repeated CV comparison of Stage-B weighting rules / member sets
- `blend2.py` — earlier 2-stage weighted-average blender (baseline, superseded by stageA_stack)
