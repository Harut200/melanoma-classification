# Preprocessing - how to run it

Five scripts, run in order. Each one writes files that the next one reads, so if
you follow the numbers you cannot get lost.

| Script | What it does | Roughly how long |
| --- | --- | --- |
| `step1_download.py` | get the data from Kaggle | 1-2 hours (30 GB) |
| `step2_make_folds.py` | clean the metadata, build the folds | seconds |
| `step3_resize_images.py` | resize photos, remove hair | 15-40 min |
| `step4_add_external.py` | add ISIC 2019 to fix the imbalance | ~1 hour |
| `step5_package.py` | pack it up for the team | 10 min |

Every script prints `--help` if you run it with no options.

## Where this runs

The server: Rocky Linux 10, Intel i5-13500 (14 cores / 20 threads), 62 GB RAM,
~390 GB free. Everything here is CPU work, so no GPU is needed.

**Training the model does not happen here.** There is no GPU on this box.
Step 5 packages the data so it can go to a machine that has one.

## Setup, once

```bash
cd ~/melanoma-classification
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.12 specifically. TensorFlow does not publish builds for 3.13 or 3.14 yet.

## Step 1 - download

First log in to Kaggle:

```bash
kaggle auth login
```

On a server with no browser, get a token from
https://www.kaggle.com/settings/api and do this instead:

```bash
export KAGGLE_API_TOKEN=your_token_here
```

Then open the competition page and click **Join Competition**. Without that,
every download fails with a 403 error.

```bash
python src/step1_download.py --list     # see what is there and how big
python src/step1_download.py --csv      # small, do this first
python src/step1_download.py --images   # about 30 GB, slow
```

The photos take a long time. Start them in the background so the download
survives your SSH connection dropping:

```bash
nohup python src/step1_download.py --images > ~/download.log 2>&1 &
tail -f ~/download.log      # watch it; Ctrl-C stops watching, not the download
```

## Step 2 - clean the metadata and build the folds

```bash
python src/step2_make_folds.py
```

This is the step that prevents the biggest mistake in the whole project.

There are 33,126 photos from only 2,056 patients. If you split them randomly,
the same patient lands in training *and* validation, the model learns to
recognise the person rather than the cancer, and your validation score becomes a
lie. So all photos of one patient are kept in the same fold, and the answer is
saved to `folds.csv` so nobody has to remember to redo it.

The script refuses to save if any patient ended up in two folds. Expect about
411 patients and 1.75-1.78% cancer in each fold.

## Step 3 - resize the photos and remove hair

```bash
python src/step3_resize_images.py --split train --workers 20
python src/step3_resize_images.py --split test  --workers 20
```

Output is normal `.jpg` files, 512x512, in `data/processed/train_512/`. You can
open them in any image viewer and look at them.

Two things happen to each photo:

- **Square crop then resize.** Originals range from 640x480 to 6000x4000. We cut
  a square from the middle and shrink it, rather than squashing the photo into a
  square, because squashing changes the shape of the mole.
- **Hair removal.** Body hair lying across a mole is noise. We find the dark thin
  lines and paint over them using the surrounding colours.

Safe to stop and restart. Photos that are already done get skipped.

## Step 4 - add ISIC 2019

Full detail and the reasoning is in [external_data_roadmap.md](external_data_roadmap.md).
Short version:

```bash
python src/step4_add_external.py --download
python src/step4_add_external.py --prepare --which all
python src/step4_add_external.py --check-duplicates

python src/step3_resize_images.py --external \
    --input-folder data/raw/isic2019/ISIC_2019_Training_Input \
    --image-list data/processed/external_2019.csv \
    --output-folder data/processed/train_512

python src/step2_make_folds.py --external-csv data/processed/external_2019.csv
```

Run `--check-duplicates` **before** resizing the external photos. Removing a
duplicate afterwards leaves its resized copy behind, unreferenced by any csv.

Or skip all of this and run `./run_all.sh`, which does the whole pipeline from
nothing in the right order.

This takes the cancer rate from 1.76% to 8.73%.

Run `--check-duplicates` **before** the final `step2_make_folds.py`. The ISIC
archive is cumulative, so some 2019 photos also exist in our 2020 set, and a
duplicate spanning the train/validation line is leakage.

## Step 5 - pack it up and send it

```bash
python src/step5_package.py --archive
```

Builds `data/processed/handover/` containing the photos, the csv files, a
`README.md` explaining the rules to whoever trains the model, and
`checksums.sha256`.

Send it:

```bash
rsync -avP data/processed/handover/ someone@their-machine:~/melanoma-data/
```

`rsync` is worth it over `scp` because it resumes if the connection drops, which
it will on a multi-GB transfer.

If they cannot use rsync, `--archive` also builds a single `.tar` file. It is
`.tar` and not `.tar.gz` on purpose: jpg files are already compressed, so gzip
would take a long time to save almost nothing.

## If something goes wrong

**"403" or an empty file list from Kaggle.** You have not clicked "Join
Competition" on the website.

**Step 2 says it found an unexpected body site.** External data came in with
names step 2 does not know. Run step 4's `--prepare` first, which translates
them.

**Step 3 is slow.** Check you passed `--workers 20`. The default uses every
core, but if you are sharing the machine you may want fewer.

**Step 5 warns the photo count and the csv row count disagree.** Usually step 3
has not finished, or the external photos have not been resized yet. Do not send
the package until they match.
