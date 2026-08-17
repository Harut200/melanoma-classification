#!/bin/bash
#
# Upload the data to Google Drive so the team can download it.
#
#   ./upload_to_drive.sh                  everything (about 46 GB)
#   ./upload_to_drive.sh --essential-only  just what you need to train (2.7 GB)
#
# Before this works you need an rclone remote called "gdrive". Set it up once,
# the steps are in docs/preprocessing.md.
#
# What goes where on Drive:
#
#   melanoma-classification/
#     preprocessed-512/     what you actually train on, 2.7 GB
#     raw/                  the original photos, 43 GB, only needed if somebody
#                           wants to redo the preprocessing at a different size
#
# The photos go up as tar files rather than as loose files. Google Drive is slow
# per file and there are 68,837 of them in the processed set alone, so loose
# files would take most of a day while a tar takes minutes. The small csv files
# go up on their own so nobody has to download gigabytes to read the fold table.

set -e

cd "$(dirname "$0")"

RCLONE=~/bin/rclone
REMOTE=gdrive
BASE="melanoma-classification"

PROCESSED=data/processed
HANDOVER="$PROCESSED/handover"
ARCHIVES=data/archives

ESSENTIAL_ONLY="no"
if [ "$1" = "--essential-only" ]; then
    ESSENTIAL_ONLY="yes"
fi

# --- checks before we start -------------------------------------------------

if [ ! -x "$RCLONE" ]; then
    echo "rclone is not installed at $RCLONE"
    exit 1
fi

if ! "$RCLONE" listremotes 2>/dev/null | grep -q "^${REMOTE}:"; then
    echo ""
    echo "There is no rclone remote called '$REMOTE' yet."
    echo ""
    echo "Set it up once by running:   rclone config"
    echo "The answers are written down in docs/preprocessing.md"
    echo ""
    exit 1
fi

if [ ! -f "$PROCESSED/melanoma_512.tar" ]; then
    echo "Cannot find $PROCESSED/melanoma_512.tar"
    echo "Run this first:  python src/step5_package.py --archive"
    exit 1
fi

# Settings that make large uploads faster and less likely to stall.
FLAGS="--progress --transfers 4 --checkers 8 --drive-chunk-size 128M --retries 5 --low-level-retries 20"

# --- 1. the small files, so people can read them without downloading a tar ---

echo ""
echo "1/3  small files (a few MB)"
for f in README.md folds.csv metadata_clean.csv metadata_test_clean.csv \
         external_2019.csv sample_submission.csv original_train.csv \
         original_test.csv checksums.sha256; do
    if [ -f "$HANDOVER/$f" ]; then
        "$RCLONE" copy "$HANDOVER/$f" "$REMOTE:$BASE/preprocessed-512/" --retries 5
    fi
done
echo "     done"

# --- 2. the processed photos ------------------------------------------------

echo ""
echo "2/3  processed photos, melanoma_512.tar (2.7 GB)"
"$RCLONE" copy "$PROCESSED/melanoma_512.tar" "$REMOTE:$BASE/preprocessed-512/" $FLAGS

if [ "$ESSENTIAL_ONLY" = "yes" ]; then
    echo ""
    echo "3/3  skipping the raw photos because --essential-only was used"
else

# --- 3. the raw originals ---------------------------------------------------

echo ""
echo "3/3  raw original photos (43 GB, this is the long one)"
if [ -d "$ARCHIVES" ]; then
    "$RCLONE" copy "$ARCHIVES" "$REMOTE:$BASE/raw/" $FLAGS
else
    echo "     no $ARCHIVES folder, skipping"
fi

fi

# --- check everything really arrived ----------------------------------------

echo ""
echo "checking the uploads match what is on disk (comparing hashes, not sizes)"
FAILED=0

if ! "$RCLONE" check "$PROCESSED/melanoma_512.tar" "$REMOTE:$BASE/preprocessed-512/" --one-way; then
    FAILED=1
fi

if [ "$ESSENTIAL_ONLY" = "no" ] && [ -d "$ARCHIVES" ]; then
    if ! "$RCLONE" check "$ARCHIVES" "$REMOTE:$BASE/raw/" --one-way; then
        FAILED=1
    fi
fi

echo ""
echo "what is on Drive now:"
"$RCLONE" lsl "$REMOTE:$BASE/" | sort -k4

if [ "$FAILED" = "1" ]; then
    echo ""
    echo "WARNING: at least one file did not match. Run this script again."
    echo "rclone will only re-send what is missing or wrong."
    exit 1
fi

echo ""
echo "==============================================================="
echo " DONE, everything checked out"
echo "==============================================================="
echo ""
echo "Share it:"
echo "  open drive.google.com, find the '$BASE' folder,"
echo "  right click, Share, add your team members"
echo ""
echo "Tell them:"
echo "  1. read README.md first, it has the two rules about the folds"
echo "  2. download melanoma_512.tar from preprocessed-512/"
echo "  3. unpack it:  tar -xf melanoma_512.tar"
echo "  4. ignore the raw/ folder unless they want to redo the"
echo "     preprocessing at a different image size"
