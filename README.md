# Melanoma Classification

Binary classifier for dermoscopic skin lesion images. Feed it one JPEG, it returns the probability that the lesion is melanoma.

## Overview

Melanoma is the skin cancer that actually kills people. It is also rare compared to the benign moles it resembles, and the visual gap between an early melanoma and an ordinary nevus is small enough that trained dermatologists disagree with each other. A model that scores a dermoscopic photo is a triage aid. It sorts a queue, it does not make a diagnosis.

The SIIM-ISIC data is hard for reasons that have little to do with which network you pick. Only 1.76% of the training rows are malignant, so a model that answers "benign" every time scores 98.2% accuracy and is worthless. The images come from many different cameras and clinics, with resolutions running from 640x480 up to 6000x4000 in the same training set. Patients repeat, heavily: one patient contributes 115 images. Split the rows at random and the same person's skin lands on both sides of the split, which quietly inflates every validation number you look at.

This repository currently holds the exploratory analysis in [notebooks/eda.ipynb](notebooks/eda.ipynb), which produced every dataset number quoted below. The training and inference code is not committed yet. Sections below marked TODO are the parts that are still missing, and they are marked rather than guessed at.

## Dataset

SIIM-ISIC Melanoma Classification, from the 2020 Kaggle competition.

| Split | Rows | Metadata columns |
|---|---|---|
| train | 33,126 | 8 |
| test | 10,982 | 5 |

Class balance in the training set:

| target | label | count | share |
|---|---|---|---|
| 0 | benign | 32,542 | 98.24% |
| 1 | malignant | 584 | 1.76% |

Metadata columns are `image_name`, `patient_id`, `sex`, `age_approx`, `anatom_site_general_challenge`, `diagnosis`, `benign_malignant`, `target`. The test set drops the last three.

### Missing values

| Column | Missing (train) | Missing (test) |
|---|---|---|
| sex | 65 | 0 |
| age_approx | 68 | 0 |
| anatom_site_general_challenge | 527 | 351 |
| everything else | 0 | 0 |

Missingness in `sex` and `age_approx` is not independent. 65 rows are missing both, 3 rows are missing only `age_approx`, and no row is missing only `sex`. Whatever went wrong went wrong at the row level, so imputing the two columns separately is a bit of a lie.

### Feature distributions

Anatomical site, training set:

| Site | Count |
|---|---|
| torso | 16,845 |
| lower extremity | 8,417 |
| upper extremity | 4,983 |
| head/neck | 1,855 |
| palms/soles | 375 |
| oral/genital | 124 |

The test set uses the same six categories. `head/neck` has the highest malignant rate of any site, but with 1,855 samples that estimate is shakier than the torso one. Exact per-site malignant rates: TODO (the notebook plots them, it does not print the numbers).

Sex: 17,080 male, 15,981 female, 65 missing. Roughly balanced. Malignant rate is higher for male patients.

Age: mean 48.9, median 50, range 0 to 90, recorded in 5 year buckets. Skewness is 0.081, so it is effectively symmetric and needs no transform. Malignant rate rises with age. Two rows have `age_approx = 0` and both belong to patient `IP_1300691`, who has a third row recording age 10. Either a data entry error or the dataset spans several years of visits.

### Patients

2,056 patients have more than one image. The largest have 115 images each (`IP_7279968`, `IP_4479736`, `IP_4938382`, `IP_4382720`). Spot checking one patient's images confirms they are distinct lesions and angles, not copies of the same photo, but they still share skin tone, hair, camera, and clinic. Total unique patients in train: TODO.

There are zero exact duplicate rows in either split.

### Image resolution

From a random sample of 200 training images:

| Resolution | Count |
|---|---|
| 6000x4000 | 96 |
| 1872x1053 | 30 |
| 640x480 | 25 |
| 5184x3456 | 24 |
| 3264x2448 | 11 |
| 2592x1936 | 9 |
| 4288x2848 | 4 |
| 4032x3024 | 1 |

Everything gets resized to a fixed input size during preprocessing.

### Getting the data

Images are not in this repository. `data/` is in [.gitignore](.gitignore) and the raw download is far too large for git.

```bash
pip install kaggle
# put your kaggle.json in ~/.kaggle/ first, and accept the competition rules on the website
kaggle competitions download -c siim-isic-melanoma-classification -p data/raw
unzip data/raw/siim-isic-melanoma-classification.zip -d data/raw
```

Expected layout after unzipping:

```
data/
  raw/
    train.csv
    test.csv
    jpeg/
      train/          # ISIC_0015719.jpg, ...
      test/
```

The notebook reads these as `../data/raw/train.csv` and `../data/raw/jpeg/train`, relative to `notebooks/`. Download size on disk: TODO.

## Approach

Nothing in this section is implemented yet. It is the plan the EDA points to, and the split logic is the one part of it the EDA already settles.

**Baseline.** TODO. Intended as a small CNN trained from scratch on resized images, to establish a floor worth beating.

**Final architecture.** TODO. Intended to be an ImageNet-pretrained backbone with the classifier head replaced, fine tuned on the lesion images.

**Why transfer learning.** 584 positive examples is not enough to learn edges, texture, and color from scratch. A pretrained backbone arrives already knowing those, so training only has to learn what separates a melanoma from a nevus. It also cuts training time to something four students can iterate on.

**Validation split.** Group by `patient_id`, using `StratifiedGroupKFold` or `GroupKFold` from scikit-learn, so that no patient appears in both the training and validation folds.

This matters more here than in a typical image task. 2,056 patients contribute multiple images and one contributes 115. Under a random row split, a model can memorize a patient's skin tone, hair pattern, ruler markings, and camera artifacts from their training images, then recognize the same patient in validation and score their remaining lesions correctly for the wrong reason. Validation AUC goes up, real performance does not. Grouping by patient forces the model to generalize to skin it has never seen.

Stratification on `target` is still needed on top of the grouping. At 1.76% positive, an unstratified fold can end up with almost no malignant cases in it.

**Class imbalance handling.** TODO. Candidates the EDA raises: class weighting, oversampling the malignant rows, augmentation applied more aggressively to the minority class.

**Number of folds, image size, augmentation, optimizer, schedule, epochs.** TODO.

## Results

No trained model has been committed, so there is no `reports/results.csv` to read from. The table below is the shape the results should take, with every cell left as TODO rather than filled in with a plausible looking number.

| Model | ROC-AUC | PR-AUC | Sensitivity @ 95% specificity | Notes |
|---|---|---|---|---|
| Baseline CNN | TODO | TODO | TODO | TODO |
| Final model | TODO | TODO | TODO | TODO |

ROC-AUC is the competition metric, but at 1.76% positives it is generous. PR-AUC and sensitivity at a fixed specificity are the numbers that say whether the model is useful, so report all three. Baseline PR-AUC for a random model on this data is 0.0176.

Figures: `reports/figures/` does not exist yet. Once training runs and writes it, embed the plots here with relative paths, for example:

```markdown
![ROC curve](reports/figures/roc_curve.png)
![Precision-recall curve](reports/figures/pr_curve.png)
![Confusion matrix](reports/figures/confusion_matrix.png)
```

Expected figure filenames: TODO. The EDA plots (target distribution, malignant rate by site, malignant rate by age, resolution scatter) currently live inline in the notebook and are not exported to files.

## Quickstart

**Read this first:** `predict.py`, `requirements.txt`, and the trained weights are not in the repository yet. The commands below are the intended interface and will fail today. What does work right now is opening the EDA notebook, which is the last block in this section.

Clone:

```bash
git clone https://github.com/Harut200/melanoma-classification.git
cd melanoma-classification
```

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` is TODO. The EDA notebook imports pandas, numpy, seaborn, matplotlib, missingno, scipy, scikit-learn, tensorflow, and pillow. Pinned versions: TODO.

Download the trained weights:

```bash
# TODO: weights are not published yet. Intended:
# curl -L -o models/final_model.h5 <TODO_WEIGHTS_URL>
```

Run a prediction on a single image:

```bash
python predict.py --image path/to/lesion.jpg
```

Expected output format: TODO. Suggested: the malignant probability and the predicted label at the chosen threshold.

### Reproducing training

```bash
# 1. Put the Kaggle data in data/raw/ (see the Dataset section)
# 2. Train
python src/train.py --config configs/final.yaml     # TODO: config path and flags

# 3. Evaluate, writes reports/results.csv and reports/figures/
python src/evaluate.py --weights models/final_model.h5    # TODO: flags
```

Training hardware and runtime: TODO.

### What runs today

```bash
pip install pandas numpy seaborn matplotlib missingno scipy scikit-learn tensorflow pillow jupyter
jupyter notebook notebooks/eda.ipynb
```

The notebook needs `data/raw/` to be populated. Most cells fail without it.

## Repository structure

What is in the repository now:

```
melanoma-classification/
├── .gitignore              # ignores data/, venv/, __pycache__/, notebook checkpoints
├── README.md               # this file
└── notebooks/
    └── eda.ipynb           # exploratory analysis: missingness, class balance, patient
                            # overlap, resolution spread, train vs test distributions
```

The directory scaffold below is committed on `dev` and `eda_m` (empty folders held open with `.gitkeep`). None of the files in it exist yet:

```
melanoma-classification/
├── configs/
│   └── final.yaml          # TODO: hyperparameters for the final run
├── data/
│   └── raw/                # git-ignored, downloaded from Kaggle
├── models/
│   └── final_model.h5      # TODO: trained weights, downloaded not committed
├── notebooks/
│   └── eda.ipynb
├── reports/
│   ├── results.csv         # TODO: one row per model, metrics as columns
│   ├── oof_predictions.csv # TODO: out-of-fold predictions for threshold picking
│   └── figures/            # TODO: ROC, PR, confusion matrix
├── src/
│   ├── dataset.py          # TODO: loading, patient-grouped splits, augmentation
│   ├── model.py            # TODO: architecture definition
│   ├── train.py            # TODO: training loop, writes weights
│   └── evaluate.py         # TODO: metrics and figures from saved weights
├── predict.py              # TODO: single-image inference
└── requirements.txt        # TODO: pinned dependencies
```

## Notes and limitations

**This is not a medical device and is not for clinical use.** It is a student project built for a Picsart Academy course. Do not use it to decide anything about a real mole on a real person. If something on your skin is changing, see a dermatologist.

The model, once trained, will only be valid on dermoscopic images: close-up photos taken through a dermatoscope against the skin. Phone camera snapshots are a different distribution and predictions on them mean nothing.

The ISIC archive skews heavily toward light skinned patients from a handful of clinics in the US, Australia, and Europe. Melanoma presents differently on darker skin, and is more often diagnosed late. A model trained on this data should be assumed to perform worse on skin tones the data barely contains, and this repository does not measure that gap because the metadata does not record skin tone.

At 1.76% positives, a high ROC-AUC can coexist with missing most of the melanomas at any threshold you would actually deploy. Read the sensitivity number, not the AUC.

The `diagnosis` column is mostly `unknown` for benign lesions, so the labels are coarser than they look. `benign_malignant` and `target` carry the same information, just in different formats.

Two patients have suspicious `age_approx` values (the zero age rows discussed above), and 527 training rows have no anatomical site. Whatever the model does with the metadata features has to tolerate that.

Known unknowns because the model is not built: generalization to other cameras, calibration of the output probabilities, behavior on lesion types absent from the training set. All TODO.

## Team

Final project, Picsart Academy.

- Ruzanna Barseghyan
- Anahit Tumasyan
- Mariam Petrosyan
- Harutyun Kesablyan
