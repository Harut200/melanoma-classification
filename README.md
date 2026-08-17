# Melanoma Classification

Binary classifier for dermoscopic skin lesion images. Feed it one JPEG, it returns the probability that the lesion is melanoma.

## Overview

Melanoma is the skin cancer that actually kills people. It is also rare compared to the benign moles it resembles, and the visual gap between an early melanoma and an ordinary nevus is small enough that trained dermatologists disagree with each other. A model that scores a dermoscopic photo is a triage aid. It sorts a queue, it does not make a diagnosis.

The SIIM-ISIC data is hard for reasons that have little to do with which network you pick. Only 1.76% of the training rows are malignant, so a model that answers "benign" every time scores 98.2% accuracy and is worthless. The images come from many different cameras and clinics, with resolutions running from 640x480 up to 6000x4000 in the same training set. Patients repeat, heavily: one patient contributes 115 images. Split the rows at random and the same person's skin lands on both sides of the split, which quietly inflates every validation number you look at.

This repository currently holds the exploratory analysis in [notebooks/eda.ipynb](notebooks/eda.ipynb). The findings live in the notebook and are not restated here. The training and inference code is not committed yet. Sections below marked TODO are the parts that are still missing, and they are marked rather than guessed at.

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

Images vary in resolution across the dataset, so everything gets resized to a fixed input size during preprocessing. Metadata has some missing values in `sex`, `age_approx`, and `anatom_site_general_challenge`. Many patients contribute more than one image, which drives the split strategy described under Approach.

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

**Read this first:** `predict.py` and the trained weights are not in the repository yet. The commands below are the intended interface and will fail today. What does work right now is creating the environment and opening the EDA notebook, which is the last block in this section.

Clone:

```bash
git clone https://github.com/Harut200/melanoma-classification.git
cd melanoma-classification
```

Create a virtual environment:

Python 3.12 is required — TensorFlow publishes no wheels for 3.13 or 3.14.

```bash
python3.12 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` pins every dependency: pandas, numpy, scipy, scikit-learn, matplotlib, seaborn, missingno, pillow, opencv-python, tensorflow, kaggle, jupyter, tqdm, and PyYAML. The versions in it were resolved and import-tested together on macOS arm64.

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

The directory scaffold below is committed on `main` and `dev` (empty folders held open with `.gitkeep`). None of the files in it exist yet:

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
└── requirements.txt        # pinned dependencies, needs Python 3.12
```

## Notes and limitations

**This is not a medical device and is not for clinical use.** It is a student project built for a Picsart Academy course. Do not use it to decide anything about a real mole on a real person. If something on your skin is changing, see a dermatologist.

The model, once trained, will only be valid on dermoscopic images: close-up photos taken through a dermatoscope against the skin. Phone camera snapshots are a different distribution and predictions on them mean nothing.

The ISIC archive skews heavily toward light skinned patients from a handful of clinics in the US, Australia, and Europe. Melanoma presents differently on darker skin, and is more often diagnosed late. A model trained on this data should be assumed to perform worse on skin tones the data barely contains, and this repository does not measure that gap because the metadata does not record skin tone.

At 1.76% positives, a high ROC-AUC can coexist with missing most of the melanomas at any threshold you would actually deploy. Read the sensitivity number, not the AUC.

The `diagnosis` column is mostly `unknown` for benign lesions, so the labels are coarser than they look. `benign_malignant` and `target` carry the same information, just in different formats.

The metadata has gaps and at least one implausible age value. Whatever the model does with the metadata features has to tolerate missing and wrong entries. See the notebook for the details.

Known unknowns because the model is not built: generalization to other cameras, calibration of the output probabilities, behavior on lesion types absent from the training set. All TODO.

## Team

Final project, Picsart Academy.

- Ruzanna Barseghyan
- Anahit Tumasyan
- Mariam Petrosyan
- Harutyun Kesablyan
