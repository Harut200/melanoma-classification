# Data preprocessing phase, what we did and what went wrong

Author: Harutyun
Date: 17 August 2026
Branch: `dp_h`

## Short version

The preprocessing part is finished. All the photos are downloaded, cleaned up and
resized, the train/validation split is decided and frozen, and we added the ISIC
2019 dataset on top to help with the class imbalance. The output is a 2.62 GB
folder that can go straight to whoever trains the model.

The class imbalance went from 1 cancer photo in every 56, to 1 in every 9 in the
training data. More on that below, including the part that did not improve.

It took longer than expected, mostly because of problems that only showed up once
we ran things against the real Kaggle API and the real server instead of on test
data. There is a full list of those further down. I am writing them all out
because some of them will come back in the training phase.

## What exists now

Five scripts in `src/`, numbered so the order is obvious:

| Script | What it does |
| --- | --- |
| `step1_download.py` | Downloads the competition data from Kaggle |
| `step2_make_folds.py` | Cleans the metadata and decides the folds |
| `step3_resize_images.py` | Resizes photos to 512x512 and removes hair |
| `step4_add_external.py` | Adds ISIC 2019 for the imbalance |
| `step5_package.py` | Packs everything for the team |

Plus `run_all.sh` which runs all of it in the right order, and `status.sh` which
tells you how far along it is. Docs are in `docs/preprocessing.md` and
`docs/external_data_roadmap.md`.

Everything runs on the server (Rocky Linux, i5-13500, 14 cores, 62 GB RAM). No
GPU on that box, so training has to happen somewhere else. That is why step 5
exists.

## The output

`data/processed/handover/`, 2.62 GB:

```
train_512/          57,855 photos
test_512/           10,982 photos
folds.csv           image_name, patient_id, target, fold, is_external
metadata_clean.csv  same rows plus age, sex and body site as numbers
external_2019.csv
README.md           the rules for whoever trains the model
checksums.sha256
```

There is also `melanoma_512.tar` (2.93 GB) if we want to put it on Google Drive
instead of copying the folder directly.

The photos are ordinary jpg files. You can open them and look at them. We
considered TFRecords, which are faster to read during training, but nobody on the
team has used them before and they cannot be opened and inspected. Since we are
sending this to people who need to be able to poke at it, plain jpg won.

## What we did to the data

**Resizing.** The originals go from 640x480 up to 6000x4000. We cut a square out
of the middle and then shrink it to 512x512. We did not stretch the photo into a
square, because that changes the shape of the mole, and shape is one of the
things that tells melanoma apart from a normal mole.

**Hair removal.** A lot of the photos have body hair lying across the lesion. We
find the dark thin lines and paint over them using the colours around them (the
DullRazor method, which the EDA report already pointed at). Measured on test
images this removes about 99% of the hair pixels while only changing the average
brightness by about 1%, so it is not smearing the lesion itself.

One detail that matters: we remove hair after resizing, not before. The hair
detector looks for lines of a certain thickness in pixels. On a 6000x4000 photo a
hair is thick, on a 640x480 photo the same hair is thin, so running it on the
originals would behave differently for every photo. After resizing every photo is
512x512, so it behaves the same everywhere.

**Missing values.** Missing `sex` and `anatom_site_general_challenge` got their
own category called "unknown" instead of being filled with the most common value.
The EDA found that when `sex` is missing, `age_approx` is usually missing too, so
those rows are not missing by accident. Filling them with "male" would be making
up data. Missing `age_approx` got the median, because a number column has no
sensible "unknown" bucket.

**The folds.** This is the part that would have quietly ruined everything if we
got it wrong. There are 33,126 photos but only 2,056 patients, so the same person
shows up many times. If you split the rows randomly, photo 1 of a patient goes to
training and photo 2 of the same patient goes to validation. The model then
learns to recognise that person's skin instead of learning what cancer looks
like, and the validation score looks great while the real score is bad.

So we used a patient level split (StratifiedGroupKFold, grouped on `patient_id`,
stratified on `target`). All photos of one patient stay in the same fold. We do
it once, write the answer to `folds.csv`, and ship that file, so nobody
downstream has to remember to do it. The script refuses to save if any patient
ended up in two folds.

## The imbalance, before and after

|  | photos | cancer | rate | ratio |
| --- | ---: | ---: | ---: | ---: |
| Competition only | 33,126 | 584 | 1.76% | 1 : 56 |
| ISIC 2019 we added | 24,729 | 4,454 | 18.01% | 1 : 5 |
| Both together | 57,855 | 5,038 | 8.71% | 1 : 10 |

The number that actually matters is the training pool after one fold is held out
for validation:

```
train when val = fold 0    51,228 photos   4,921 cancer   9.61%   1 : 9
train when val = fold 1    51,230          4,922          9.61%   1 : 9
train when val = fold 2    51,234          4,922          9.61%   1 : 9
train when val = fold 3    51,227          4,921          9.61%   1 : 9
train when val = fold 4    51,230          4,920          9.60%   1 : 9
```

So each training run now sees 4,921 melanoma photos instead of 467. That is more
than ten times as many. All five folds came out the same, so no fold is a worse
deal than the others.

### The part that did not improve

Validation is still 1 : 56, about 117 melanoma photos per fold. That is on
purpose and it is not something to fix. The real Kaggle test set is that
imbalanced, so if we make validation more balanced our score stops predicting
what we will get on the leaderboard.

What it means in practice: 117 positives is a small number to measure on. A small
difference in PR-AUC between two models might just be noise. Whoever trains
should compare across all five folds, not one.

### It still needs class weights

1 : 9 is better but it is not balanced. Computed from the actual data, holding
out fold 0:

```python
class_weight = {0: 0.553, 1: 5.205}
```

And the metric should be PR-AUC, not accuracy. A model that always says "not
cancer" gets 98.2% accuracy on this data and is useless.

### This is a hypothesis, not a result

ISIC 2019 was taken with different cameras in different hospitals. Some of that
10x might not transfer. The first experiment in the training phase should be:
train fold 0 with the external data, train fold 0 without it, same seed, compare
PR-AUC. If it does not improve, the external data is adding noise and we should
talk about it before tuning anything else.

## About ISIC 2019 specifically

25,331 photos with 8 possible diagnoses. Only the MEL column becomes our
`target = 1`. BCC and SCC are also skin cancers but they are not melanoma, and
our target column means melanoma specifically, so they count as 0. Getting that
wrong would have silently corrupted every label.

Four things had to be handled:

1. The body site names do not match. 2020 has one category called "torso". 2019
   splits it into "anterior torso", "posterior torso" and "lateral torso". We map
   all three back to "torso".
2. ISIC 2019 has no `patient_id` at all. We generate one per row with an `EXT19_`
   prefix so it can never collide with a real patient from 2020.
3. ISIC 2019 already contains ISIC 2018 (the HAM10000 set) inside it. So we must
   not also add 2018 separately, it would be the same photos twice.
4. Some photos exist in both 2019 and 2020, because the ISIC archive is
   cumulative. We check for this and drop them, see below.

External photos are always `fold = -1`. They are used for training only and never
for validation, for the reason in the section above.

## Problems we hit, and what we did about them

This is the honest list. Most of these only appeared when the code met the real
world.

### 1. The EDA report was a PDF and could not be reviewed

The EDA write up was only a PDF, which cannot be diffed or read on GitHub without
downloading it. Converted it to Markdown and kept the PDF as well. Merged through
PR #2 into `dev`. No tools for reading PDFs were installed on the Mac, so we made
a throwaway virtualenv with pypdf just for the extraction.

### 2. Python 3.14 cannot run TensorFlow

The default `python3` on the Mac was 3.14, and TensorFlow does not publish builds
for it yet. `pip install tensorflow` just says "No matching distribution found",
which is confusing because the package obviously exists. Rebuilt everything on
Python 3.12, which was already installed. This is now written at the top of
`requirements.txt` so nobody loses an hour to it.

### 3. OpenCV would not import on the server

Every script that touches images died with:

```
ImportError: libxcb.so.1: cannot open shared object file
```

`opencv-python` is built against graphics libraries for showing windows on
screen. A server has none of that installed, and installing them needs root,
which we do not have on this box. The fix is `opencv-python-headless`, the same
library without the window drawing parts. We never open a window, so nothing is
lost. Updated `requirements.txt` on all four branches.

Worth saying: the check that was supposed to catch this did not, because it
treated exit code 1 as success, and that is exactly what a failed import returns.
The scripts were reported as fine while they were broken. Re-ran it properly by
actually importing each module, and that is when this showed up.

### 4. `main` and `dev` have no common ancestor

They were started as two separate root commits, so git thinks they are unrelated
repositories that happen to live in the same place. Merging `dev` into `main`
later will need `--allow-unrelated-histories` and will conflict on `README.md`
and `.gitignore`. Not fixed yet, flagged so it does not become a surprise on
release day.

### 5. Kaggle's login method changed, then it turned out it had not

kaggle 2.2.4 prints instructions telling you to use `kaggle auth login` or a
token file, and does not mention the old `~/.kaggle/kaggle.json` at all. So we
assumed the old way was removed. It is not, it still works fine, it is just no
longer advertised. Wasted a bit of time on that.

### 6. The Kaggle API returns an object, not a list

`competition_list_files()` looks like it gives you a list of files. It gives you a
response object, and iterating over it raises `TypeError`. The files are on
`.files`.

### 7. Listing all the files gets you rate limited

The same call returns 200 files per page with a token for the next page. This
competition has more than 44,000 files, which is over 220 requests, and partway
through Kaggle starts answering `429 Too Many Requests`.

We do not actually need the full list. Every photo name is already in `train.csv`
and `test.csv`. So `--list` now only asks for a few pages and prints the sizes
worth planning around, and the download does not use the listing at all.

### 8. Kaggle sends bigger files zipped without telling you

We asked for `train.csv` and got `train.csv.zip`. Step 2 would then have failed
saying `train.csv` does not exist, which points at the wrong problem entirely.
The download now unpacks these and checks all three files really exist before it
says it is done.

### 9. The photo downloader deadlocked

We downloaded the photos with 8 threads, sharing one Kaggle connection object
between them. That object is not built for being used by several threads at once.
The whole thing froze: 8 open sockets, 29 threads, zero photos downloaded for six
minutes, no error message. Had to kill it.

### 10. Then Kaggle blocked the account for downloads

While debugging problem 9 we made about 500 downloads plus 600 retries in a few
minutes. After that Kaggle answered `404 Not Found` to every single download
request, including `sample_submission.csv`, which had downloaded fine half an hour
earlier. Listing kept working the whole time. So it is a rate limit dressed up as
a "not found" error, which is not obvious at all when you are staring at it.

It cleared by itself after about 25 minutes.

The real lesson is that downloading 44,108 files one at a time was a bad design
from the start. We rewrote it to ask Kaggle for the whole competition as one zip
instead, and unpack only the jpg photos out of it. One request cannot be rate
limited the way tens of thousands can.

The cost is that the zip is 110 GB, because it also contains DICOM and TFRecord
copies of the same photos which we do not use. So we download 110 GB to keep
35 GB. It took about an hour at 28 to 35 MB/s, which is still much faster and much
safer than the alternative. The zip can be deleted afterwards.

### 11. The progress bar filled the log with garbage

The download progress used a carriage return to keep rewriting one line. That
works on a screen. In a log file a carriage return does nothing, so it produced
1.2 MB of `0.0% 0.0% 0.0%` in the first minute and made the log useless. Since the
log is the only way to check on a job that runs for hours, that matters. It now
prints once per whole percent with the GB count.

There is a second nasty side effect: because there are no newlines, one "line" in
the log can be several megabytes wide, and `tail` will happily print all of it at
you. `status.sh` strips the carriage returns and cuts lines to 160 characters
before showing them.

### 12. All the external photos were being silently skipped

`step3` filtered out every row where `is_external == 1`. That filter is correct
when resizing our own photos, because `metadata_clean.csv` holds both sets and
they live in different folders. But when we pointed it at the external csv it
removed all 25,331 rows and then reported success having done nothing at all.

This is the worst kind of bug because there is no error. We would have shipped a
handover where `folds.csv` claims 25,331 external photos and the folder contains
none of them. It now needs an explicit `--external` flag, and it refuses to run on
an empty list instead of pretending it worked.

### 13. A finished run looked like a failed one

Right at the end, `status.sh` reported 0 out of 58,457 photos resized, while the
log said FINISHED. It turned out the counter used `ls folder/*.jpg | wc -l`. The
shell expands that into one command line, and with 57,855 files it goes over the
maximum length and fails with "Argument list too long", which counts as zero.
Switched to `find`. The pipeline had been fine the whole time.

## Duplicate photos between 2019 and 2020

Because the ISIC archive is cumulative, some photos are in both years. If one copy
ends up in training and its twin ends up in validation, that is the same leakage
we built the patient level split to avoid.

We fingerprint every photo (shrink to an 8x8 grey square, compare each pixel to
the average, keep the 64 ones and zeros) and drop any 2019 photo whose
fingerprint already exists in 2020.

Before running it on the real data we tested it by planting three known
duplicates: an exact copy, the same photo saved at lower jpg quality, and the same
photo at half resolution. It caught all three, and did not wrongly flag any of the
20 unrelated photos.

On the real data it dropped **602 photos**, which is why the external set is
24,729 instead of 25,331. That number looks right. We expected somewhere under
500, and had decided in advance that anything above about 2,000 would mean the
fingerprint was matching photos that only look similar rather than real
duplicates. The full list is saved in `data/processed/duplicates_to_drop.txt` if
anyone wants to check it.

## Final checks

After everything finished we verified:

```
rows in folds.csv:            57,855
photos actually on disk:      57,855
in the csv but no photo:      0
photo with no csv row:        0
patients in more than 1 fold: 0
external rows not at fold -1: 0
```

Fold balance:

```
fold  photos  patients  cancer  cancer %
0     6627    412       117     1.77
1     6625    413       116     1.75
2     6621    407       116     1.75
3     6628    415       117     1.77
4     6625    409       118     1.78
```

## What is not included

These are not missing by mistake, they belong to the training phase:

- Augmentation (flipping, rotating, colour changes). This has to be different
  every epoch, so it cannot be baked into files.
- Pixel normalisation, because it depends on which model architecture is chosen.
- The data loading code itself.
- Labels for the test set. Kaggle keeps those, that is the leaderboard.

## Handing it over

Either copy the folder straight to their machine:

```bash
rsync -avP data/processed/handover/ someone@their-machine:~/melanoma-data/
```

Or put `melanoma_512.tar` on Google Drive and let them download it from there. If
we use Drive we should upload the tar file and not the folder, because Drive is
slow per file and the folder has 68,837 of them. The tar is 2.93 GB so it fits in
a free Drive account.

Whoever picks it up should read the `README.md` inside the package first. The two
things they must not get wrong are: use the `fold` column instead of making their
own split, and never validate on rows where `fold = -1`.

## Still open

- `main` and `dev` having unrelated git histories (problem 4).
- The 110 GB competition zip is still on the server and can be deleted to get the
  space back.
- The external data helping is still unproven. The with/without experiment should
  be the first thing the training phase runs.
