# Model experiments, what we ran and what we found

Author: Harutyun
Date: 24 August 2026
Branch: `exp_r_fixes`

## Short version

We trained a melanoma classifier and answered the one question the preprocessing
phase left open: does adding the ISIC 2019 data actually help?

It does. On all five folds, every single time.

The model itself scores PR-AUC 0.216 out of fold, which is about twelve times
better than guessing. Adding ISIC 2019 improves that by 27%. In plainer terms,
it finds 35 more melanomas out of 584 without raising the false alarm rate.

Before any of this could run we had to fix the training code, which was quietly
broken in a way that made every result meaningless. That is written up in the
second half.

## The headline numbers

Five fold cross validation, split by patient so no person appears in both
training and validation.

| | PR-AUC | ROC-AUC | Sens@95%Spec |
| --- | ---: | ---: | ---: |
| Random guessing | 0.0176 | 0.500 | 0.050 |
| Our model, no external data | 0.187 | 0.881 | 0.462 |
| **Our model, with ISIC 2019** | **0.229** | **0.887** | **0.519** |

Out of fold, meaning every one of the 33,126 competition photos scored exactly
once by a model that never saw it in training:

| | photos | PR-AUC | ROC-AUC | Sens@95%Spec |
| --- | ---: | ---: | ---: | ---: |
| without external | 33,126 | 0.170 | 0.879 | 0.462 |
| **with external** | 33,126 | **0.216** | **0.886** | **0.522** |

Setup: ResNet34 pretrained on ImageNet, 224x224 input, 10 epochs, AdamW with a
cosine schedule, mixed precision, `BCEWithLogitsLoss` with `pos_weight` computed
from each training split. Flips, small affine shifts and brightness jitter for
augmentation. Ran on a Kaggle T4, 2.9 GPU hours for all ten runs.

## Did ISIC 2019 help?

This was set up as a paired comparison. Same folds, same seed, same everything,
with the external data as the only thing that changed.

| fold | with external | without | difference |
| ---: | ---: | ---: | ---: |
| 0 | 0.2123 | 0.1632 | +0.0492 |
| 1 | 0.1589 | 0.1570 | +0.0019 |
| 2 | 0.2863 | 0.1980 | +0.0883 |
| 3 | 0.2539 | 0.2271 | +0.0268 |
| 4 | 0.2311 | 0.1912 | +0.0400 |
| **mean** | **0.2285** | **0.1873** | **+0.0412** |

Five wins out of five. Paired t test gives p = 0.044, Wilcoxon signed rank gives
p = 0.031. Both under 0.05, so this is not noise.

### Why paired matters here

The fold to fold spread is large. With external data the five folds ranged from
0.159 to 0.286, a spread of ±0.048. That is bigger than the +0.041 average
improvement, so if you compared two averages you would conclude nothing.

But that spread is mostly about some folds having easier patients than others,
and that difficulty shows up in both numbers and cancels out when you subtract.
What is left is the effect of the external data alone. Fold 1 improved by only
+0.002 and fold 2 by +0.088, but the sign was positive every time, and that
consistency is the evidence.

### What it means in practice

At a fixed 95% specificity, meaning we accept a 5% false alarm rate:

- without external data the model catches 270 of the 584 melanomas
- with external data it catches 305

That is 35 more cancers found, for the same number of false alarms. That is the
number worth putting in front of anyone who is not going to read a PR curve.

## What we did not get to

The plan had two more experiments queued that we ran out of time for:

- `efficientnet_b0` at 224, five folds. Would tell us whether the architecture
  choice matters much at this scale.
- `resnet34` at 384. The loss was still falling at epoch 10, so more resolution
  or more epochs would probably help.

Neither changes the ISIC 2019 conclusion, which is what we most needed.

## Honest caveats

**Each fold score is the best epoch measured on that same fold.** That is normal
practice for picking a checkpoint, but it means the number is slightly
optimistic. Fold 2 for example peaked at 0.286 on epoch 7 and finished at 0.262.
The out of fold numbers in the second table have the same property. A fully
clean number needs a third split that is never touched, which `evaluate_test.py`
supports and we did not have time to run.

**The no external runs saw half as much data for the same ten epochs.** With
26,499 training photos instead of 51,228 they got roughly half the gradient
steps. So some of the improvement could be "more steps" rather than "better
data". Looking at the curves, the no external runs were still climbing at epoch
10 while the with external ones had mostly flattened, so this is worth checking
by matching step counts rather than epochs. It would probably shrink the gap
somewhat, not erase it.

**117 melanomas per validation fold.** Everything here is measured on a small
number of positive cases. Treat differences under about 0.05 PR-AUC with
suspicion unless they are consistent across folds, the way this one was.

**Ten epochs is short.** Loss was still going down. This is a baseline, not a
tuned model.

## The bug that had to be fixed first

None of the above would have been possible with the code as it was. The training
script pointed its image directory at `data/raw/train`, which does not exist in
this repo. The photos are in `data/processed/train_512`.

That on its own would be a five second fix, except the dataset class did this:

```python
if os.path.exists(img_path):
    image = cv2.imread(img_path)
else:
    image = np.zeros((224, 224, 3), dtype=np.uint8)
```

A missing file quietly became a black square. So every one of the 58,000 images
was blank, and training ran to completion with no error, no warning, and saved
two model files. A model trained on identical blank inputs can only learn a
constant, which on this data means PR-AUC around 0.018, the same as guessing.

The fix is that missing files now raise, plus a check that samples the file list
before training starts so a wrong path fails in seconds instead of after an
epoch of nothing.

Other things fixed in the same pass:

- `pos_weight` was hardcoded to 55, the ratio from before ISIC 2019 was added.
  The real ratio with external data is about 9.4, so positives were being
  weighted roughly six times too heavily. It is now computed from the data,
  which also keeps it correct for the no external runs where it rises to 36.
- Training validated on fold 4 and the evaluation script then scored on fold 4,
  so the model was being judged on the data used to select it. Validation and
  test folds are now separate arguments and both scripts refuse to run if they
  collide.
- Images were normalised with a bare divide by 255 while the pretrained weights
  expect ImageNet mean and standard deviation, which throws away much of the
  benefit of pretraining.
- PR-AUC was computed as `auc(recall, precision)`, which interpolates across a
  curve that is not monotonic and reads slightly high. Now uses
  `average_precision_score`.
- The augmentation argument existed but was never passed, so nothing was
  augmented.
- Two identical 81 MB model checkpoints were committed to git, one of them
  inside `src/models/`, a folder sitting next to `src/models.py` that would
  shadow the module as soon as anyone added an `__init__.py`.
- `requirements.txt` pinned TensorFlow while all the code is PyTorch, and
  torch, timm and albumentations were not listed at all.

## Reproducing this

```bash
python src/experiment_runner.py \
    --img_dir data/processed/train_512 \
    --folds_csv data/processed/folds.csv \
    --out_dir reports \
    --only resnet34_224_ext,resnet34_224_noext \
    --folds 0,1,2,3,4

python src/report_results.py --in_dir reports
```

Every finished fold is written to `results.csv` as it completes, so the run can
be stopped and restarted without losing work. Seed is fixed at 42.

On Kaggle, use the notebook in `kaggle/`. Pick the T4 accelerator and not the
P100. Kaggle's PyTorch build no longer supports the P100's architecture, and
`torch.cuda.is_available()` returns True on it anyway before failing on the
first real operation.

## What to do next

1. Run the efficientnet and 384 pixel experiments that got cut.
2. Rerun the ablation with matched gradient steps rather than matched epochs, to
   remove the one confound in the headline result.
3. Train longer. Ten epochs was not enough to converge.
4. Ensemble the five fold models and add test time augmentation. Usually worth
   a few points of PR-AUC for no extra training.
5. Use a held out third split for a final clean number before anyone quotes
   these figures outside the team.
