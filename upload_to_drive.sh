#!/bin/bash
#
# Upload the finished data to Google Drive so the team can download it.
#
#   ./upload_to_drive.sh
#
# Before this works you need an rclone remote called "gdrive". Set it up once
# with:  rclone config      (see docs/preprocessing.md for the exact answers)
#
# What gets uploaded:
#
#   melanoma_512.tar     one 2.9 GB file with all 68,837 photos inside
#   README.md            so people can read the rules before downloading 3 GB
#   folds.csv            same reason
#   metadata_clean.csv
#   external_2019.csv
#   checksums.sha256
#
# We upload the photos as ONE tar file rather than 68,837 separate files.
# Google Drive is slow per file, so the loose folder would take most of a day
# while the tar takes a few minutes. The small csv files go up separately so
# nobody has to download 3 GB just to look at the fold table.

set -e

cd "$(dirname "$0")"

RCLONE=~/bin/rclone
REMOTE=gdrive
DRIVE_FOLDER="melanoma-classification/preprocessed-512"

PROCESSED=data/processed
HANDOVER="$PROCESSED/handover"
TAR="$PROCESSED/melanoma_512.tar"

# --- checks before we start -------------------------------------------------

if [ ! -x "$RCLONE" ]; then
    echo "rclone is not installed at $RCLONE"
    exit 1
fi

if ! "$RCLONE" listremotes | grep -q "^${REMOTE}:"; then
    echo ""
    echo "There is no rclone remote called '$REMOTE' yet."
    echo ""
    echo "Set it up once by running:   rclone config"
    echo "The answers are written down in docs/preprocessing.md"
    echo ""
    exit 1
fi

if [ ! -f "$TAR" ]; then
    echo "Cannot find $TAR"
    echo "Run this first:  python src/step5_package.py --archive"
    exit 1
fi

# --- upload -----------------------------------------------------------------

echo "uploading to Google Drive folder: $DRIVE_FOLDER"
echo ""

echo "1/2  the small files first, so they are readable straight away"
for f in README.md folds.csv metadata_clean.csv external_2019.csv checksums.sha256; do
    if [ -f "$HANDOVER/$f" ]; then
        "$RCLONE" copy "$HANDOVER/$f" "$REMOTE:$DRIVE_FOLDER/" --progress
    fi
done

echo ""
echo "2/2  the big tar file (2.9 GB, a few minutes)"
# --transfers 4 splits the file into parallel chunks.
# --drive-chunk-size 64M uses bigger chunks, which is faster for one large file.
"$RCLONE" copy "$TAR" "$REMOTE:$DRIVE_FOLDER/" \
    --progress \
    --transfers 4 \
    --drive-chunk-size 64M

# --- check it really arrived ------------------------------------------------

echo ""
echo "what is on Drive now:"
"$RCLONE" lsl "$REMOTE:$DRIVE_FOLDER/"

echo ""
echo "checking the tar file arrived intact (comparing hashes, not just sizes)"
if "$RCLONE" check "$TAR" "$REMOTE:$DRIVE_FOLDER/" --one-way; then
    echo "OK, the copy on Drive matches the copy here"
else
    echo "WARNING: the check failed. Run the script again, rclone will resume."
    exit 1
fi

echo ""
echo "==============================================================="
echo " DONE"
echo "==============================================================="
echo ""
echo "Now share it:"
echo "  1. open drive.google.com"
echo "  2. find the folder $DRIVE_FOLDER"
echo "  3. right click, Share, add your team members"
echo ""
echo "What they should do:"
echo "  1. read README.md on Drive first, it has the two rules"
echo "  2. download melanoma_512.tar"
echo "  3. unpack it:   tar -xf melanoma_512.tar"
