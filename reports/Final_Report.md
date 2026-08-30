# The final model, what we built and what it scores

Author: Harutyun
Date: 30 August 2026
Branch: `final_arch_a_fixes`

## Short version

The experiment phase left a ResNet34 at 224 pixels scoring ROC-AUC 0.887, and a
list of things that would probably make it better. We built all of them into one
model: a bigger pretrained backbone at a higher resolution, GeM pooling, the
patient's sex, age and body site fused into the image features, mixup and cutmix,
a two stage fine tuning schedule, and flip test time augmentation.

It works. ROC-AUC goes from 0.887 to 0.909, and it wins on all five folds.

Out of fold, meaning every one of the 33,126 competition photos scored exactly
once by the fold model that never trained on it. This is the number to quote:

| | ROC-AUC | PR-AUC | Sens@95%Spec |
| --- | ---: | ---: | ---: |
| Random guessing | 0.500 | 0.0176 | 0.050 |
| resnet34 @224, no external | 0.879 | 0.170 | 0.462 |
| resnet34 @224, the baseline | 0.886 | 0.216 | 0.522 |
| **efficientnet-b4 @300 + metadata** | **0.9076** | **0.2507** | **0.5479** |
| change against the baseline | **+0.0216** | **+0.0347** | **+0.0259** |

In melanomas rather than decimals: at a fixed 5% false alarm rate the baseline
finds 305 of the 584 cancers and this model finds 320. Fifteen more, for the same
number of false alarms. Against where the project started, without the external
data, it is fifty more.

It cost 11.6 GPU hours against the baseline's 1.9. That is 6.2x the compute for
+0.021 ROC-AUC, which is the honest price of the last two points.

One thing stops this being the final word, and it is written up below: the epoch
selection has a test time augmentation confound that makes the per fold numbers
slightly inconsistent with each other. It does not affect the out of fold score,
which is measured on predictions rather than on the selection rule.

## What the model actually is

`src/models/final_model.py`. Four pieces.

**Backbone.** `tf_efficientnet_b4` from timm, ImageNet pretrained, 300x300 input,
classifier head removed and global pooling disabled so the raw feature map comes
out.

**GeM pooling.** Generalised mean pooling with a learnable exponent `p`, initialised
at 3.0. `p=1` is average pooling and `p→∞` is max pooling, so the layer learns
where between the two it wants to sit. Melanoma is a local texture problem inside a
mostly uninformative field of skin, which is exactly the case where average pooling
dilutes the signal. It is forced to run in fp32; see the caveats.

**Metadata branch.** Entity embeddings for sex (3 categories) and anatomical site
(7), plus a small MLP on normalised age, projected to 256 dimensions. These are the
three columns the EDA found actually carry signal.

**Gated fusion.**

```
fused = image_features + gate(meta) * project(meta)
```

The gate scales an *additive* metadata contribution rather than multiplying the
image features. That direction matters: metadata on this dataset is missing and
occasionally wrong, and with this arrangement the worst a bad metadata row can do
is contribute nothing (`gate → 0`). It can never zero out the image branch, which is
what a multiplicative gate would allow.

A LayerNorm and a single linear layer turn the fused 256 vector into one logit.

## How it was trained

Five folds, grouped by patient, the same folds and the same seed as the baseline.
ISIC 2019 external data in training only; the assertion in `run_final_kaggle.py`
refuses to run if an external row reaches a validation fold.

| | |
| --- | --- |
| backbone / input | `tf_efficientnet_b4`, 300x300, ImageNet pretrained |
| epochs | 12 total: 2 with the backbone frozen, then 10 fine tuning all of it |
| schedule | 1 epoch linear warmup into cosine decay, over stage 2 only |
| learning rates | head `1e-3` both stages, backbone `3e-5` once unfrozen |
| optimiser | AdamW, weight decay `1e-4`, gradient clipped at norm 5 |
| loss | `BCEWithLogitsLoss`, `pos_weight` computed per split (≈9.41) |
| batch size | 32, mixed precision |
| augmentation | h/v flip, affine (±30°, ±5% shift, 0.9 to 1.1 scale), brightness/contrast jitter |
| mixup / cutmix | mixup α 0.4, cutmix α 1.0, applied to 50% of batches |
| TTA | 4 views (identity, h-flip, v-flip, both), last 3 epochs only |
| checkpoint selection | best validation ROC-AUC |
| hardware | Kaggle T4, two concurrent sessions, ~139 min per fold |

Two details worth naming because they are easy to get wrong:

Freezing the backbone in stage 1 also puts it in `.eval()`. Setting
`requires_grad=False` alone leaves BatchNorm updating its running statistics from
the new data, so a "frozen" backbone quietly drifts anyway.

Mixup and cutmix mix the images and the labels but leave the metadata index
aligned with the original batch. Interpolating a categorical embedding index is
meaningless, so the metadata is not mixed.

## Results

![Per-fold comparison against the baseline](figures/final_vs_baseline.png)

Per fold, validating on the fold named and training on the other four plus all of
ISIC 2019:

| fold | ROC-AUC | PR-AUC | Sens@95%Spec | best epoch | threshold | F1 | val photos | val melanomas | min |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.8863 | 0.2253 | 0.5726 | 10 | 0.526 | 0.320 | 6,627 | 117 | 139 |
| 1 | 0.9012 | 0.1689 | 0.4655 | 8 | 0.610 | 0.289 | 6,625 | 116 | 140 |
| 2 | 0.9182 | 0.2901 | 0.6207 | 11 | 0.577 | 0.368 | 6,621 | 116 | 139 |
| 3 | 0.9119 | 0.2763 | 0.5299 | 10 | 0.679 | 0.349 | 6,628 | 117 | 139 |
| 4 | 0.9255 | 0.3044 | 0.6102 | 11 | 0.447 | 0.354 | 6,625 | 118 | 139 |
| **mean** | **0.9086** | **0.2530** | **0.5598** | | | **0.336** | 33,126 | 584 | 696 |
| std dev | 0.0153 | 0.0557 | 0.0636 | | | | | | |

Against the baseline, fold by fold:

| fold | ROC base | ROC final | Δ | PR base | PR final | Δ | Sens base | Sens final | Δ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.8846 | 0.8863 | +0.0017 | 0.2123 | 0.2253 | +0.0130 | 0.5043 | 0.5726 | +0.0684 |
| 1 | 0.8704 | 0.9012 | +0.0307 | 0.1589 | 0.1689 | +0.0100 | 0.4655 | 0.4655 | 0.0000 |
| 2 | 0.9039 | 0.9182 | +0.0143 | 0.2863 | 0.2901 | +0.0039 | 0.5862 | 0.6207 | +0.0345 |
| 3 | 0.8972 | 0.9119 | +0.0147 | 0.2539 | 0.2763 | +0.0223 | 0.5299 | 0.5299 | 0.0000 |
| 4 | 0.8805 | 0.9255 | +0.0450 | 0.2311 | 0.3044 | +0.0733 | 0.5085 | 0.6102 | +0.1017 |
| **mean** | 0.8873 | **0.9086** | **+0.0213** | 0.2285 | **0.2530** | **+0.0245** | 0.5189 | **0.5598** | **+0.0409** |

Five wins out of five on ROC-AUC and on PR-AUC. Four wins and two exact ties on
sensitivity, which never goes backwards. The two ties are not a coincidence of
rounding: sensitivity at a fixed specificity is a count of melanomas above a
cutoff, so with 116 and 117 positives it can only move in steps of about 0.009 and
landing on the same count twice is ordinary.

## Is the gain real?

Same folds, same seed, same data, so this is a paired comparison and the fold to
fold difficulty cancels out.

| metric | mean Δ | wins | paired t | Wilcoxon |
| --- | ---: | ---: | ---: | ---: |
| ROC-AUC | +0.0213 | 5/5 | p = 0.047 | p = 0.063 |
| PR-AUC | +0.0245 | 5/5 | p = 0.123 | p = 0.063 |
| Sens@95%Spec | +0.0409 | 4/5, 2 ties | p = 0.108 | p = 0.188 |

Read this carefully, because only one of those cells clears 0.05.

**ROC-AUC is significant, PR-AUC is not.** The PR-AUC improvement is +0.0245 on
average but the per fold deltas run from +0.004 to +0.073, and five samples with
that spread cannot produce a small p value. The mean gain is real in the sense
that it happened every time; it is not established at the 0.05 level.

**The Wilcoxon test cannot go below 0.0625 with five pairs.** With n=5 there are
2⁵ = 32 possible sign arrangements, so the smallest two sided p the test can ever
return is 2/32 = 0.0625. Both ROC-AUC and PR-AUC hit that floor, meaning the sign
was unanimous and the test has nothing left to say. Quoting "p = 0.063, not
significant" for these would be misreading the test.

**The strongest evidence here is the unanimity, not any p value.** Five folds,
five positive ROC-AUC differences: under a null of no effect that is a 1 in 32
outcome one sided, p = 0.031. PR-AUC gives the same 5/5 and sensitivity never goes
backwards, but those are not independent tests. They are three views of the same
five model pairs on the same five patient groups, so they cannot be multiplied
together into a smaller number. One clean sign test on the competition metric, and
two other metrics that agree with it, is the claim to defend.

### What it means in practice

At 95% specificity, meaning we accept a 5% false alarm rate on benign lesions,
measured on the pooled out of fold predictions:

- without the external data, the model catches **270** of the 584 melanomas
- the resnet34 baseline with ISIC 2019 catches **305**
- this model catches **320**

Fifteen more cancers than the baseline, and fifty more than where the project
started, for no extra false alarms. The ISIC 2019 data bought 35 of those and the
architecture bought the other 15, which is worth knowing: the cheaper change was
the bigger one.

## The test time augmentation confound

This is the one methodological problem in the run and it needs stating plainly.

`src/run_final_kaggle.py:177`:

```python
tta = args.tta if epoch > args.epochs - 3 else 1
```

TTA is only applied on the last three epochs, to save forward passes while the
model is still moving. Reasonable on its own. But the checkpoint is then selected
as the best validation ROC-AUC across *all twelve* epochs, and epochs 10 to 12 are
scored with a four view average while epochs 1 to 9 are scored with one view. TTA
is worth a few thousandths of ROC-AUC for free. So the selection is biased toward
the last three epochs, and it is not surprising that four of the five folds picked
one:

| fold | best epoch | scored with TTA? |
| ---: | ---: | --- |
| 0 | 10 | yes |
| 1 | 8 | **no** |
| 2 | 11 | yes |
| 3 | 10 | yes |
| 4 | 11 | yes |

Three consequences.

**The obvious "it was still improving at the cutoff" reading is not supported.**
Best epochs clustering at 10 and 11 out of 12 normally means the model was still
climbing and more epochs would help. Here it also has a much duller explanation:
those are the epochs that got a scoring bonus. The learning curves would settle
it, and we should look at them before spending another 12 GPU hours on longer
training.

**Fold 1's row is not measured the same way as the other four.** It reports
`tta,4` in `final_results.csv`, because that column records the `--tta` argument
rather than what the winning epoch actually used, but its metrics come from a
single view forward pass. Fold 1 also has by some distance the worst PR-AUC of the
five (0.169 against a 0.253 mean). Part of that gap is the missing TTA.

**Fold 1 was already the hardest fold for the baseline too** (PR-AUC 0.159, the
lowest of its five), so this is not purely an artefact. But the two effects are
tangled together and the run as it stands cannot separate them.

The fix is one line: score every epoch the same way. Either run TTA on every epoch
and pay for it, or select the checkpoint on plain single view predictions and
apply TTA once, afterwards, to the chosen checkpoint. The second is cheaper and
more correct.

## Out of fold

All 33,126 competition photos, each scored exactly once by the fold model that did
not train on it. `reports/final_oof.csv`, summarised by `src/final_report.py`.

| | photos | ROC-AUC | PR-AUC | Sens@95%Spec |
| --- | ---: | ---: | ---: | ---: |
| resnet34 @224, no external | 33,126 | 0.879 | 0.170 | 0.462 |
| resnet34 @224, the baseline | 33,126 | 0.886 | 0.216 | 0.522 |
| **efficientnet-b4 @300 + metadata** | 33,126 | **0.9076** | **0.2507** | **0.5479** |

The out of fold ROC-AUC is 0.9076 against a per fold mean of 0.9086. The gap is
small and in the expected direction: averaging five per fold numbers is slightly
optimistic, because each fold's epoch was chosen partly on that fold's noise.
Pooling removes some of that. Quote 0.9076.

**The deployment threshold is 0.6245.** That is the F1 optimal cutoff on the pooled
predictions. It replaces the per fold thresholds, which ran from 0.447 to 0.679 and
none of which was usable on its own. `predict.py` uses this number.

The file is also clean, which settles an open question. `predict()` replaces any
NaN or inf prediction with a neutral 0.5 and prints a warning; `final_oof.csv`
contains no NaN and not a single probability of exactly 0.5 across 33,126 rows. The
fp16 GeM overflow did not touch this run.

## Error analysis

Where the model fails, from the same out of fold predictions.

**It misses 264 of the 584 melanomas** at the 5% false alarm operating point. That
is the headline failure and no amount of ROC-AUC hides it.

**Twelve melanomas were scored below 0.01**, which is not a near miss but a
confident wrong answer. The worst is `ISIC_5733748` at 0.00306, ranked below 5.3%
of all 33,126 photos: the model placed a confirmed melanoma among the most
obviously benign moles in the dataset. Three of the ten worst misses are in fold 0,
which is also the fold with the smallest gain over the baseline.

| image | probability | ranked below |
| --- | ---: | ---: |
| `ISIC_5733748` | 0.00306 | 5.3% of all photos |
| `ISIC_5331102` | 0.00469 | 12.7% |
| `ISIC_7948290` | 0.00518 | 15.1% |
| `ISIC_3028754` | 0.00581 | 18.5% |

**118 benign lesions were scored above 0.90.** The worst, `ISIC_4973831` at 0.996,
the model is more confident about than almost any true melanoma. In a triage queue
these are the cases that waste a specialist's time, and at 1.76% prevalence they
outnumber the true positives at every high threshold.

Pull those eight image names out of `data/processed/train_512` and look at them.
That is the honest way to answer "why did it get confused", and it is the one thing
this report cannot do for you, because the images are not in the repository.

## What is missing

**No held out test fold.** The run used `--test_fold -99`, meaning nothing was
held back. Every fold's score is the best epoch measured on the same fold that
selected it, which is normal practice for choosing a checkpoint but makes the
number slightly optimistic. The baseline has exactly the same property, so the
*comparison* between them is fair even though the absolute numbers are not clean.
`src/evaluate_test.py` scores a genuinely untouched fold, and it cannot be run on
this run's checkpoints: with `--test_fold -99` there is no fold that some
checkpoint did not either train on or select on. It refuses every combination,
correctly. Getting a clean number means either retraining four folds with
`--test_fold 4`, about nine GPU hours, or submitting to the competition, where the
10,982 test photos have labels nobody on this project has ever seen.

## Honest caveats

**584 melanomas in total, about 117 per fold.** Every number here rests on a small
number of positive cases. PR-AUC in particular has a standard deviation of 0.056
across folds, which is more than twice the average improvement being claimed.
Treat any single fold difference under about 0.05 PR-AUC as noise unless it is
consistent across folds, which is the whole reason the analysis above is paired.

**The focal loss was built and not used.** `src/losses/focal_loss.py` exists,
`train_final.py` defaults to it, and `run_final_kaggle.py` defaults to `bce`, so
the run that produced these numbers used `BCEWithLogitsLoss` with a computed
`pos_weight`. Focal loss versus weighted BCE on this imbalance is an untested
one-line ablation.

**The backbone is not the one the code intends.** `final_model.py` defaults to
`tf_efficientnet_b4_ns`, the noisy student weights, which are generally worth a
little over plain ImageNet weights on this task. The run used `tf_efficientnet_b4`.
In current timm the noisy student checkpoint is named
`tf_efficientnet_b4.ns_jft_in1k`; the old `_ns` suffix no longer resolves. This is
free performance left on the table.

**Metadata fusion is not separately measured.** The model gained a bigger
backbone, more resolution, GeM, metadata, mixup/cutmix, a two stage schedule and
TTA all at once, against a baseline that had none of them. The +0.021 belongs to
that bundle. `--no_metadata` and `--no_gem` exist precisely to take it apart and
neither has been run.

**Cost.** 11.6 GPU hours for five folds against 1.9 for the baseline. On Kaggle's
30 hour weekly quota this is roughly a third of a week for one configuration, so
the ablations above have to be chosen rather than swept.

**Still not a medical device.** Same as everywhere else in this repository: a
triage aid on dermoscopic images, not a diagnosis, and the ISIC archive barely
contains dark skin. Sensitivity on skin tones the data does not cover is unmeasured
and should be assumed worse.

## What to do next, in order

1. **Look at the eight images named in the error analysis.** Free, and it is the
   only item on this list that changes what we understand rather than what we
   score. It is also the part of a presentation nobody can fake.
2. **Fix the TTA selection line.** One line, and it makes the five folds
   comparable to each other.
3. **Submit to the competition.** The 2020 leaderboard still accepts late
   submissions and scores against 10,982 photos whose labels nobody here has ever
   had. About 40 minutes of T4 time, and it is the only genuinely held out number
   this project can get without retraining.
4. **Switch to `tf_efficientnet_b4.ns_jft_in1k`.** Same cost, probably better
   weights. The `_ns` suffix the code still defaults to no longer resolves in timm.
5. **Run the `--no_metadata` ablation on two folds.** Roughly 4.5 GPU hours, and it
   answers whether the metadata branch earned its place, the part of this
   architecture that a reader will ask about first.
6. **Train longer, but only if the curves say so.** 16 to 18 epochs is the obvious
   next step and it is about 3.5 extra GPU hours per fold. Decide it from the
   learning curves after step 2, not from the current best-epoch column.
7. **Ensemble.** Different backbone, different resolution, rank average with
   `src/ensemble_oof.py`. Historically worth more than any single architecture
   change, and `final_oof.csv` now exists to measure it against.

## Reproducing this

On Kaggle, `kaggle/final_model_kaggle.ipynb`. Set the constants in cell 1 to what
was actually run. The committed defaults say `tf_efficientnet_b3`:

```python
FOLDS      = "0,1,2"    # and "3,4" in a second session
BACKBONE   = "tf_efficientnet_b4"
IMAGE_SIZE = 300
EPOCHS     = 12
BATCH_SIZE = 32
TTA        = 4
```

Accelerator T4, not P100: Kaggle's PyTorch build no longer compiles for Pascal, and
a P100 reports `cuda.is_available() == True` before failing on the first kernel
launch. Internet on, so timm can fetch the pretrained weights. Two sessions in
parallel halve the wall clock to about six hours.

Locally, or anywhere with a GPU:

```bash
python src/run_final_kaggle.py \
    --img_dir data/processed/train_512 \
    --metadata_csv data/processed/metadata_clean.csv \
    --out_dir reports \
    --backbone tf_efficientnet_b4 \
    --image_size 300 \
    --folds 0,1,2,3,4 \
    --epochs 12 \
    --tta 4 \
    --save_weights

python src/final_report.py --in_dir reports
```

`final_report.py` writes `reports/final_model_report.md`, the machine-generated
table. This document is the written-up version of it; the two are deliberately
different filenames because macOS filesystems are case-insensitive and
`final_report.md` would overwrite `Final_Report.md`.

Every finished fold is appended to `final_results.csv` and `final_oof.csv`
immediately and a restart skips what is already there, which matters when a
Kaggle session can die at any point in a twelve hour run. Seed is fixed at 42.

Raw per fold numbers: [`reports/final_results.csv`](final_results.csv). Baseline
numbers for the comparison: [`reports/results.csv`](results.csv), rows
`resnet34_224_ext`.
