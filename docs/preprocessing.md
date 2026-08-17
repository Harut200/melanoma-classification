# Preprocessing runbook

Everything here runs on the CPU server (14 cores / 20 threads, 64 GB RAM, ~300 GB
free). No GPU is needed for any of it — the work is decode, resize, inpaint,
encode, which is embarrassingly parallel and I/O bound.

Model training does **not** happen here. The last step packages the output for
whoever runs the experiments, on a machine that has a GPU.

## 0. Environment

Python 3.12 is required — TensorFlow publishes no wheels for 3.13 or 3.14.

```bash
git clone https://github.com/Harut200/melanoma-classification.git
cd melanoma-classification
git checkout dp_h

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Kaggle authentication

The old `~/.kaggle/kaggle.json` flow is gone as of kaggle 2.2.

```bash
kaggle auth login                     # OAuth, opens a browser
# headless server instead:
export KAGGLE_API_TOKEN=<token from kaggle.com/settings/api>
```

Accept the competition rules on the website first, or every file returns 403.

## 2. Download

Check the real sizes before committing to anything:

```bash
python src/download_data.py --list
```

Then:

```bash
python src/download_data.py --csv       # metadata, a few MB
python src/download_data.py --images    # jpeg/, roughly 30 GB
```

`--images` refuses to start if free space minus a 5 GB margin is under what the
files need. The DICOM and TFRecord copies in the competition archive are not
downloaded — the JPEGs are the same pixels without the extra ~75 GB.

## 3. External data for the class imbalance

The malignant rate is 1.8%. The standard fix for this competition is folding in
malignant cases from earlier ISIC years, which roughly triples the positive count.

```bash
kaggle datasets list -s "isic 2019 jpeg"    # confirm the slug before pulling
python src/download_data.py --dataset <owner>/<slug>
```

Two rules, both enforced in code rather than left to discipline:

- External rows get `fold = -1` and never enter a validation fold. They are a
  different distribution from the competition test set, so validating on them
  breaks the correspondence between local CV and the leaderboard.
- External `patient_id`s are namespaced `EXT{n}_...` so they cannot collide with
  competition patients and silently merge two people into one group.

Before using any external set, deduplicate against the 2020 data — the ISIC
archives overlap between years, and a duplicate image spanning the train/val
boundary is leakage wearing a different hat.

## 4. Folds and metadata

```bash
python src/make_folds.py --external data/raw/<external>/train.csv
```

Writes `data/processed/folds.csv` and `metadata_processed.csv`, and refuses to
write either if any patient spans two folds. Expect roughly 411 patients and a
1.75–1.78% malignant rate per fold.

## 5. Images to TFRecords

```bash
python src/preprocess.py --split train --size 512 --workers 20
python src/preprocess.py --split test  --size 512 --workers 20 \
    --metadata data/raw/test.csv --image-dir data/raw/jpeg/test
```

Roughly 15–40 minutes for all 44k images on 20 threads, dominated by decoding the
6000x4000 originals. Output is ~5–6 GB.

Shards are written to a `.partial` name and renamed only on success, so an
interrupted run never leaves a truncated file that looks complete.

## 6. Verify before handing anything over

```bash
python src/preprocess.py --verify data/processed/train_512
python src/preprocess.py --verify data/processed/test_512
```

Decodes every example, asserts one consistent shape, and cross-checks the count
against `manifest.json`.

## 7. Hand off

```bash
python src/export_dataset.py --stage                        # inspect first
python src/export_dataset.py --create --user <kaggle-user>  # private dataset
```

Then on the dataset page: **Settings → Collaborators → add teammate**.

Staging hardlinks the shards, so the export folder costs no extra disk. The
upload is private by default. Later updates:

```bash
python src/export_dataset.py --version "added ISIC 2019 external malignant"
```

A `README.md` is generated into the upload with the fold rules, the TFRecord
schema, and the actual per-fold balance table, so the modelling work does not
have to come back and ask what the columns mean.

## Why the teammate gets a Kaggle dataset

The server has no GPU. Kaggle gives free GPU and TPU quota, reads TFRecords
natively, and versions the dataset so there is never ambiguity about which copy a
result came from. The alternatives — Drive, HuggingFace, a shared folder — all
work, but only this one puts a GPU next to the data at zero cost.

## Design decisions worth knowing

**Centre crop, not squash.** Resolutions run 640x480 to 6000x4000 across several
aspect ratios. Stretching to a square distorts lesion shape, and shape is one of
the things a dermoscopic model reads.

**Hair removal after resize, not before.** The DullRazor blackhat kernel is a
fixed pixel size. Applied to the original, it means something different for a
640x480 image than a 6000x4000 one. Applied after the resize, it is consistent.
Measured on test images it removes ~99% of hair-like pixels while changing mean
intensity by ~1%.

**`unknown` category, not modal fill.** The EDA found missingness in `sex` and
`age_approx` is correlated, so those values are not missing at random. Filling
with the mode would fabricate signal; an explicit category lets the model learn
that missingness itself carries information. `age_approx` uses the median because
it has no natural "unknown" bucket in a continuous feature.

**Folds frozen in a CSV.** With 33,126 images from 2,056 patients, a row-level
split leaks. Computing the split once and shipping it means no downstream
notebook can accidentally undo it.
