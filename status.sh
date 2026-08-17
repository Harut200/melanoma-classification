#!/bin/bash
#
# Show where the pipeline has got to and roughly how much is left.
#
#   ./status.sh
#
# Safe to run any time. It only reads things, it never changes anything.

cd "$(dirname "$0")"

LOG=~/pipeline.log
RAW=data/raw
PROCESSED=data/processed

# Totals we are working towards.
TOTAL_TRAIN=33126
TOTAL_TEST=10982
TOTAL_EXTERNAL=25331

count_jpg() {
    # $1 = folder. Prints 0 if the folder does not exist yet.
    ls "$1"/*.jpg 2>/dev/null | wc -l | tr -d ' '
}

bar() {
    # $1 = done, $2 = total. Draws [#####-----] 50%
    local done=$1 total=$2
    if [ "$total" -eq 0 ]; then return; fi
    local percent=$(( done * 100 / total ))
    local filled=$(( percent / 5 ))
    local i
    printf "["
    for ((i=0; i<20; i++)); do
        if [ $i -lt $filled ]; then printf "#"; else printf "-"; fi
    done
    printf "] %3d%%  %d / %d\n" "$percent" "$done" "$total"
}

echo "==============================================="
echo " MELANOMA PIPELINE STATUS   $(date '+%H:%M:%S')"
echo "==============================================="
echo ""

# --- is it alive -----------------------------------------------------------
if pgrep -f run_all.sh > /dev/null; then
    PID=$(pgrep -f run_all.sh | head -1)
    ELAPSED=$(ps -o etime= -p "$PID" | tr -d ' ')
    echo "STATE:    RUNNING   (running for $ELAPSED)"
else
    if grep -q "FINISHED" "$LOG" 2>/dev/null; then
        echo "STATE:    FINISHED"
    else
        echo "STATE:    NOT RUNNING  (stopped early - check the log)"
    fi
fi

# --- which step ------------------------------------------------------------
STEP=$(grep -E "^  STEP|^  FINISHED" "$LOG" 2>/dev/null | tail -1 | sed 's/^  //')
if [ -n "$STEP" ]; then
    echo "STEP:     $STEP"
fi
echo ""

# --- downloads -------------------------------------------------------------
echo "--- DOWNLOADS ---"

PART=$(ls -la "$RAW"/isic2019/*.part 2>/dev/null | awk '{print $5}')
if [ -n "$PART" ]; then
    echo -n "ISIC 2019 zip     "
    bar $(( PART / 1048576 )) 9318
elif [ -f "$RAW/isic2019/ISIC_2019_Training_Input.zip" ]; then
    echo "ISIC 2019 zip     downloaded"
fi

BIGZIP=$(ls -la "$RAW"/siim-isic-melanoma-classification.zip* 2>/dev/null | awk '{print $5}' | head -1)
if [ -n "$BIGZIP" ]; then
    echo -n "competition zip   "
    bar $(( BIGZIP / 1048576 )) 110000
fi

EXT_RAW=$(ls "$RAW/isic2019/ISIC_2019_Training_Input" 2>/dev/null | wc -l | tr -d ' ')
if [ "$EXT_RAW" -gt 0 ]; then
    echo -n "ISIC 2019 photos  "
    bar "$EXT_RAW" "$TOTAL_EXTERNAL"
fi

RAW_TRAIN=$(count_jpg "$RAW/jpeg/train")
RAW_TEST=$(count_jpg "$RAW/jpeg/test")
if [ "$RAW_TRAIN" -gt 0 ]; then echo -n "our train photos  "; bar "$RAW_TRAIN" "$TOTAL_TRAIN"; fi
if [ "$RAW_TEST" -gt 0 ];  then echo -n "our test photos   "; bar "$RAW_TEST"  "$TOTAL_TEST";  fi
echo ""

# --- resizing --------------------------------------------------------------
echo "--- RESIZING (512x512) ---"
DONE_TRAIN=$(count_jpg "$PROCESSED/train_512")
DONE_TEST=$(count_jpg "$PROCESSED/test_512")
echo -n "train + external  "; bar "$DONE_TRAIN" $(( TOTAL_TRAIN + TOTAL_EXTERNAL ))
echo -n "test              "; bar "$DONE_TEST" "$TOTAL_TEST"
echo ""

# --- outputs ---------------------------------------------------------------
echo "--- OUTPUT FILES ---"
for f in "$PROCESSED/folds.csv" "$PROCESSED/metadata_clean.csv" \
         "$PROCESSED/external_2019.csv" "$PROCESSED/handover/README.md"; do
    if [ -f "$f" ]; then echo "  yes  $f"; else echo "  --   $f"; fi
done
echo ""

# --- disk ------------------------------------------------------------------
echo "--- DISK ---"
df -h . | tail -1 | awk '{print "  used " $3 " of " $2 ", " $4 " free"}'
du -sh "$RAW" 2>/dev/null | awk '{print "  raw data:       " $1}'
du -sh "$PROCESSED" 2>/dev/null | awk '{print "  processed data: " $1}'
echo ""

# --- recent activity -------------------------------------------------------
echo "--- LAST FEW LOG LINES ---"
# cut -c1-160 matters: the old progress printing used carriage returns, which
# make no new lines in a file, so one "line" can be megabytes wide. Without the
# cut, tail happily prints the whole thing at you.
grep -vE "B/s|it/s" "$LOG" 2>/dev/null \
    | cut -c1-160 \
    | grep -vE "^ *([0-9.]+%? *)+$" \
    | grep -v "^$" \
    | tail -5

echo ""
echo "watch it live:   tail -f ~/pipeline.log"
echo "watch this:      watch -n 30 ./status.sh"
