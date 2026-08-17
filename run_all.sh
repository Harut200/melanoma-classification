#!/bin/bash
#
# Run the whole preprocessing pipeline from nothing to a folder you can send
# to the rest of the team.
#
#   ./run_all.sh
#
# It takes a few hours, mostly downloading. Start it so it survives your SSH
# connection dropping, then watch the log:
#
#   nohup ./run_all.sh > ~/pipeline.log 2>&1 &
#   tail -f ~/pipeline.log
#
# Every step is safe to run again. Downloads skip files that are already there
# and resizing skips photos that are already done, so if this dies halfway you
# can just start it again and it will carry on from where it stopped.

# Stop immediately if any command fails, instead of carrying on with bad data.
set -e

cd "$(dirname "$0")"
PYTHON=.venv/bin/python

# Use every core for the CPU-heavy steps.
WORKERS=$(nproc)

say() {
    echo ""
    echo "==============================================================="
    echo "  $1"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "==============================================================="
}

say "STEP 1a - the three csv files"
$PYTHON src/step1_download.py --csv

say "STEP 1b - the competition photos (about 29 GB, the slow part)"
$PYTHON src/step1_download.py --images

say "STEP 4a - download ISIC 2019 (9.1 GB, plus unzipping)"
$PYTHON src/step4_add_external.py --download

say "STEP 4b - turn ISIC 2019 into our column layout"
$PYTHON src/step4_add_external.py --prepare --which all

# This has to happen BEFORE we resize the external photos and before we build
# the final folds. The ISIC archive is cumulative, so some 2019 photos are the
# same photo as one of ours. If a copy sits in training while its twin sits in
# validation, the validation score is a lie.
say "STEP 4c - find photos that exist in both 2019 and 2020, and drop them"
$PYTHON src/step4_add_external.py --check-duplicates --workers "$WORKERS"

say "STEP 3a - resize our training photos"
$PYTHON src/step3_resize_images.py --split train --workers "$WORKERS"

say "STEP 3b - resize the test photos"
$PYTHON src/step3_resize_images.py --split test --workers "$WORKERS"

# The external photos go into the SAME output folder as our own, because as far
# as training is concerned they are just more training photos. What keeps them
# separate is the fold number, not the folder.
say "STEP 3c - resize the ISIC 2019 photos"
$PYTHON src/step3_resize_images.py \
    --external \
    --input-folder data/raw/isic2019/ISIC_2019_Training_Input \
    --image-list data/processed/external_2019.csv \
    --output-folder data/processed/train_512 \
    --workers "$WORKERS"

say "STEP 2 - build the folds, now including the external rows"
$PYTHON src/step2_make_folds.py --external-csv data/processed/external_2019.csv

say "STEP 5 - pack everything up for the team"
$PYTHON src/step5_package.py --archive

say "FINISHED"
echo "Send it with:"
echo "   rsync -avP data/processed/handover/ someone@their-machine:~/melanoma-data/"
