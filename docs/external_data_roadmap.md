# Roadmap: adding ISIC 2019 as external data

## The problem in one line

Our 2020 competition data is **33,126 photos with 584 melanomas — 1.76%**. A model
that always guesses "not cancer" scores 98.24% accuracy and is worthless. There
are not enough positive examples to learn from.

## The plan

Add the **ISIC 2019** challenge training set as extra *training* material. Same
disease, same kind of dermoscopic photo, different year. It contributes **4,522
melanoma photos** — roughly **8x** our current positive count.

| | photos | melanoma | melanoma rate |
| --- | ---: | ---: | ---: |
| ISIC 2020 (ours) | 33,126 | 584 | 1.76% |
| ISIC 2019 (adding) | 25,331 | 4,522 | 17.85% |
| **Combined training pool** | **58,457** | **5,106** | **8.73%** |

1.76% → 8.73% is a 5x improvement in positive density. It does not make the
problem balanced, and we still need class weights or focal loss on top — but it
gives the model far more melanoma to actually look at.

## What ISIC 2019 contains, exactly

25,331 photos, labelled across 8 diagnoses as one-hot columns in
`ISIC_2019_Training_GroundTruth.csv`:

| Column | Meaning | Count | Our `target` |
| --- | --- | ---: | :---: |
| `MEL` | Melanoma | 4,522 | **1** |
| `NV` | Melanocytic nevus (mole) | 12,875 | 0 |
| `BCC` | Basal cell carcinoma | 3,323 | 0 |
| `AK` | Actinic keratosis | 867 | 0 |
| `BKL` | Benign keratosis | 2,624 | 0 |
| `DF` | Dermatofibroma | 239 | 0 |
| `VASC` | Vascular lesion | 253 | 0 |
| `SCC` | Squamous cell carcinoma | 628 | 0 |

**BCC and SCC are skin cancers but they are not melanoma.** Our target column
means melanoma specifically, so they map to 0. Getting this wrong would silently
corrupt every label.

`ISIC_2019_Training_Metadata.csv` carries `image`, `age_approx`,
`anatom_site_general`, `lesion_id`, `sex`.

## Five differences that have to be handled

### 1. Body site names do not match

2020 has one `torso`. 2019 splits it three ways. Unmapped, step 2 halts with an
unexpected-category error.

| ISIC 2019 | maps to 2020 |
| --- | --- |
| `anterior torso` | `torso` |
| `posterior torso` | `torso` |
| `lateral torso` | `torso` |
| `upper extremity` | `upper extremity` |
| `lower extremity` | `lower extremity` |
| `head/neck` | `head/neck` |
| `palms/soles` | `palms/soles` |
| `oral/genital` | `oral/genital` |

### 2. There is no `patient_id` in 2019

2019 has `lesion_id`, and it is only partly filled. We generate a synthetic id
per row prefixed `EXT19_`, so it can never collide with a real 2020 patient and
accidentally merge two people into one group.

This is safe **only because external rows never enter a validation fold.** The
sole job of the id is namespace isolation.

### 3. ISIC 2019 already contains ISIC 2018 (HAM10000)

The 2019 training set is built from HAM10000 + BCN20000 + MSK. Adding ISIC 2018
separately would duplicate ~10,015 photos. **Add 2019 only.**

### 4. The same photo can appear in both years

The ISIC archive is cumulative. A photo present in both sets, where one copy is
in training and its twin is in a validation fold, is exactly the leakage we
built patient-level folds to prevent.

`--check-duplicates` fingerprints every photo in both sets (average hash: shrink
to 8x8 grey, threshold at the mean, keep the 64 bits) and drops any 2019 photo
whose fingerprint already exists in 2020.

### 5. Different cameras, different hospitals

2019 comes from different sites and equipment than 2020. This is why external
data is **train-only**, enforced by `fold = -1`.

## The decision: all of 2019, or melanoma only?

| | Positive rate | Risk |
| --- | ---: | --- |
| `--which all` (default) | 8.73% | Adds 20,809 benign photos. Slower training. |
| `--which malignant-only` | 13.57% | **Model can learn a shortcut.** |

The shortcut is the reason for the default. If *every* external photo is
malignant, the model can learn "this looks like a 2019-era photo, therefore
cancer" — 2019 photos have different colour balance and optics. That rule scores
brilliantly in training and collapses on the 2020 test set.

Including all of 2019 puts 2019-looking photos in **both** classes, so source
style stops being predictive. Start with `all`.

## Cost

| | |
| --- | --- |
| Download | ~9 GB (`ISIC_2019_Training_Input.zip`) |
| Unzipped | ~9 GB more |
| Resized to 512 | ~2 GB |
| Duplicate check | ~10 min on 20 threads |
| Resize | ~15 min on 20 threads |

Source is the official ISIC S3 bucket, not a third-party mirror, so it will not
disappear.

## Commands, in order

```bash
python src/step4_add_external.py --download
python src/step4_add_external.py --prepare --which all
python src/step4_add_external.py --check-duplicates

python src/step3_resize_images.py \
    --input-folder data/raw/isic2019/ISIC_2019_Training_Input \
    --image-list data/processed/external_2019.csv \
    --output-folder data/processed/train_512

python src/step2_make_folds.py --external-csv data/processed/external_2019.csv
python src/step5_package.py --archive
```

Order matters: `--check-duplicates` must run **before** the final
`step2_make_folds.py`, or dropped duplicates stay in the folds.

## How we will know it worked

Not from the training loss — extra positives always improve that. The check is
**PR-AUC on a competition-only validation fold**, compared against the same fold
trained without external data. Same fold, same seed, one variable changed.

If PR-AUC does not improve, the external data is adding noise or the model has
found the domain shortcut. That is the experiment to hand the modelling person
along with the data.

## Later options, not now

- **ISIC 2018 separately** — no, it is inside 2019 already.
- **ISIC 2017 / 2016** — a few hundred melanomas each, much older equipment.
  Poor value relative to the domain shift they introduce.
- **Upsampling the positives we have** — cheap, no download, but adds no new
  information. Worth combining with external data, not instead of it.
