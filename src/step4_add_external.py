"""
STEP 4 - Add the ISIC 2019 data to fix the class imbalance.

Our 2020 competition data has only 1.8% cancer photos. That is very unbalanced:
a model that always answers "not cancer" is right 98.2% of the time and is
completely useless. The usual fix is to bring in more cancer photos from an
earlier year of the same challenge.

Run the parts in order:

    python src/step4_add_external.py --download    # get ISIC 2019 (about 9 GB)
    python src/step4_add_external.py --prepare     # build the csv we can use
    python src/step4_add_external.py --check-duplicates   # slow but important

Then resize the external photos with the SAME script we used for our own:

    python src/step3_resize_images.py \\
        --input-folder data/raw/isic2019/ISIC_2019_Training_Input \\
        --image-list data/processed/external_2019.csv \\
        --output-folder data/processed/train_512

And finally rebuild the folds so they include the new rows:

    python src/step2_make_folds.py --external-csv data/processed/external_2019.csv


WHAT ISIC 2019 ACTUALLY IS
--------------------------
25,331 photos, labelled with 8 possible diagnoses, one-hot encoded:

    MEL   melanoma           4,522   <- this is our "cancer" class
    NV    mole (benign)     12,875
    BCC   basal cell ca.     3,323
    AK    actinic keratosis    867
    BKL   benign keratosis   2,624
    DF    dermatofibroma       239
    VASC  vascular lesion      253
    SCC   squamous cell ca.    628

Our 2020 target column means "is this MELANOMA, yes or no". So only the MEL
column becomes target = 1. BCC and SCC are also skin cancers, but they are NOT
melanoma, so for our task they count as 0.


FOUR THINGS THAT WILL BITE YOU IF YOU ARE NOT CAREFUL
-----------------------------------------------------
1. THE BODY SITE NAMES ARE DIFFERENT.
   2020 has one category called "torso".
   2019 splits it into "anterior torso", "posterior torso", "lateral torso".
   If we do not map them, step 2 will see categories it does not know.

2. THERE ARE NO PATIENTS IN 2019.
   The 2019 metadata has no patient_id column. We invent one per row, with an
   "EXT19_" prefix so it can never be confused with a real 2020 patient.

3. ISIC 2019 ALREADY CONTAINS ISIC 2018 (the HAM10000 set).
   So do NOT also add 2018 separately - you would be adding the same photos twice.

4. THE SAME PHOTO CAN EXIST IN BOTH YEARS.
   The ISIC archive is cumulative. A duplicate that lands in training while its
   twin is in validation is data leakage, exactly the problem we tried to avoid
   in step 2. That is what --check-duplicates is for.


THE RULE THAT MATTERS MOST
--------------------------
External photos are used for TRAINING ONLY. They get fold = -1 and never appear
in a validation fold.

Why: they were taken with different cameras, in different hospitals, from a
different mix of patients. Our score has to predict how well we do on the 2020
TEST set. If we validate on 2019 photos, our validation score stops telling us
anything about the 2020 test set.
"""

import argparse
import multiprocessing
import os
import sys
import urllib.request

import cv2
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

THIS_FILE = os.path.abspath(__file__)
SRC_FOLDER = os.path.dirname(THIS_FILE)
PROJECT_FOLDER = os.path.dirname(SRC_FOLDER)
RAW_FOLDER = os.path.join(PROJECT_FOLDER, "data", "raw")
PROCESSED_FOLDER = os.path.join(PROJECT_FOLDER, "data", "processed")
EXTERNAL_FOLDER = os.path.join(RAW_FOLDER, "isic2019")

# The official ISIC challenge files. These are the original source, so we do not
# depend on somebody's re-upload staying online.
ISIC_2019_FILES = {
    "ISIC_2019_Training_GroundTruth.csv":
        "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv",
    "ISIC_2019_Training_Metadata.csv":
        "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv",
    "ISIC_2019_Training_Input.zip":
        "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip",
}

# 2019 body site name  ->  2020 body site name.
# The three torso directions all collapse into the single 2020 "torso".
SITE_NAME_MAP = {
    "anterior torso": "torso",
    "posterior torso": "torso",
    "lateral torso": "torso",
    "upper extremity": "upper extremity",
    "lower extremity": "lower extremity",
    "head/neck": "head/neck",
    "palms/soles": "palms/soles",
    "oral/genital": "oral/genital",
}


# ---------------------------------------------------------------------------
# PART 1 - download
# ---------------------------------------------------------------------------

# Remembers the last percentage we printed, so we print each one only once.
last_percent_printed = [-1]


def show_progress(block_number, block_size, total_size):
    """
    Called repeatedly while downloading so we can show progress.

    We only print when the whole-number percentage changes. Printing on every
    chunk is fine on a screen, where "\r" rewrites the same line, but this
    script normally runs with its output redirected into a log file, where
    "\r" does nothing and you end up with a megabyte of "0.0% 0.0% 0.0%".
    """
    if total_size <= 0:
        return

    downloaded = block_number * block_size
    percent = int(downloaded * 100 / total_size)
    if percent > 100:
        percent = 100

    if percent != last_percent_printed[0]:
        last_percent_printed[0] = percent
        gb_done = downloaded / (1024 * 1024 * 1024)
        gb_total = total_size / (1024 * 1024 * 1024)
        print("   " + str(percent) + "%  (" + str(round(gb_done, 1)) + " of " +
              str(round(gb_total, 1)) + " GB)", flush=True)


def download_isic_2019(external_folder):
    """Download the three ISIC 2019 files, then unzip the photos."""
    if not os.path.exists(external_folder):
        os.makedirs(external_folder)

    for file_name in ISIC_2019_FILES:
        url = ISIC_2019_FILES[file_name]
        save_to = os.path.join(external_folder, file_name)

        if os.path.exists(save_to):
            print("already have " + file_name + ", skipping")
            continue

        print("downloading " + file_name)
        # Download to a temporary name so a half-finished file is never mistaken
        # for a complete one if the connection drops.
        temporary = save_to + ".part"
        urllib.request.urlretrieve(url, temporary, show_progress)
        os.rename(temporary, save_to)
        print("")

    # Unzip the photos.
    photos_folder = os.path.join(external_folder, "ISIC_2019_Training_Input")
    if os.path.exists(photos_folder):
        print("photos already unzipped")
    else:
        print("unzipping the photos, this takes a few minutes ...")
        import zipfile
        zip_path = os.path.join(external_folder, "ISIC_2019_Training_Input.zip")
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(external_folder)
        print("unzipped")

    print("")
    print("ISIC 2019 is in " + external_folder)


# ---------------------------------------------------------------------------
# PART 2 - turn ISIC 2019 into the same shape as our 2020 table
# ---------------------------------------------------------------------------

def prepare_external_csv(which_photos, external_folder, out_folder):
    """
    Read the two ISIC 2019 csv files and write one csv with OUR column names.

    `which_photos` is either "all" or "malignant-only". See the warning printed
    below for why "all" is the safer default.
    """
    ground_truth_path = os.path.join(external_folder, "ISIC_2019_Training_GroundTruth.csv")
    metadata_path = os.path.join(external_folder, "ISIC_2019_Training_Metadata.csv")

    for path in [ground_truth_path, metadata_path]:
        if not os.path.exists(path):
            print("Could not find " + path)
            print("Run this first:  python src/step4_add_external.py --download")
            sys.exit(1)

    labels = pd.read_csv(ground_truth_path)
    info = pd.read_csv(metadata_path)

    print("ISIC 2019 has " + str(len(labels)) + " photos")

    # --- join the two tables on the photo name -----------------------------
    combined = labels.merge(info, on="image", how="left")

    # --- build our columns --------------------------------------------------
    external = pd.DataFrame()
    external["image_name"] = combined["image"]

    # Our target means MELANOMA specifically. The MEL column is 1.0 or 0.0.
    external["target"] = combined["MEL"].astype(int)

    # 2019 has no patients, so we invent a unique id per photo. The EXT19_
    # prefix guarantees it can never collide with a real 2020 patient id.
    external["patient_id"] = ["EXT19_" + str(i) for i in range(len(combined))]

    external["sex"] = combined["sex"]
    external["age_approx"] = combined["age_approx"]

    # Translate the body site names into the 2020 spelling.
    sites_2019 = combined["anatom_site_general"]
    external["anatom_site_general_challenge"] = sites_2019.map(SITE_NAME_MAP)

    # Anything the map did not cover becomes empty, and step 2 will fill it
    # with "unknown". We print it so you can see if something unexpected showed up.
    unmapped = sites_2019[sites_2019.notna() & external["anatom_site_general_challenge"].isna()]
    if len(unmapped) > 0:
        print("NOTE: " + str(len(unmapped)) + " photos had a body site we did not "
              "recognise: " + str(sorted(unmapped.unique())))
        print("      They will become 'unknown', which is fine.")

    # Columns our 2020 table has that 2019 simply does not.
    external["diagnosis"] = "unknown"
    external["benign_malignant"] = np.where(external["target"] == 1, "malignant", "benign")
    external["is_external"] = 1

    # --- optionally keep only the cancer photos ----------------------------
    if which_photos == "malignant-only":
        print("")
        print("WARNING about --which malignant-only")
        print("If EVERY external photo is cancer, the model can cheat: it can learn")
        print("'this photo looks like a 2019 photo, so it must be cancer', instead of")
        print("learning what cancer looks like. The cheat works during training and")
        print("fails on the real test set. Using 'all' avoids this because the model")
        print("then sees 2019 photos in both classes.")
        print("")
        external = external[external["target"] == 1].copy()

    # --- report --------------------------------------------------------------
    cancer_count = int(external["target"].sum())
    print("")
    print("keeping " + str(len(external)) + " external photos, of which " +
          str(cancer_count) + " are melanoma")

    # Show what this does to the imbalance.
    train_csv = os.path.join(RAW_FOLDER, "train.csv")
    if os.path.exists(train_csv):
        ours = pd.read_csv(train_csv)
        our_cancer = int(ours["target"].sum())
        before = our_cancer / len(ours) * 100
        after = (our_cancer + cancer_count) / (len(ours) + len(external)) * 100
        print("")
        print("cancer rate BEFORE adding external: " + str(round(before, 2)) + "%")
        print("cancer rate AFTER  adding external: " + str(round(after, 2)) + "%")

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    save_to = os.path.join(out_folder, "external_2019.csv")
    external.to_csv(save_to, index=False)

    print("")
    print("saved " + save_to)
    print("")
    print("Next: resize these photos, then rebuild the folds. See the top of this file.")


# ---------------------------------------------------------------------------
# PART 3 - look for the same photo appearing in both years
# ---------------------------------------------------------------------------

def make_photo_fingerprint(image_path):
    """
    Make a short "fingerprint" of a photo so we can spot duplicates.

    This is called an average hash. The idea:
      1. Shrink the photo down to a tiny 8x8 grey square (64 pixels).
      2. Work out the average brightness of those 64 pixels.
      3. For each pixel write 1 if it is brighter than the average, else 0.

    Two copies of the same photo give the same 64 ones-and-zeros, even if one
    was saved at a different size or quality. Two different photos almost never do.
    """
    photo = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if photo is None:
        return None

    tiny = cv2.resize(photo, (8, 8), interpolation=cv2.INTER_AREA)
    average_brightness = tiny.mean()
    bits = (tiny > average_brightness).flatten()

    # Turn the 64 True/False values into one text string we can compare quickly.
    return "".join(["1" if bit else "0" for bit in bits])


def fingerprint_job(image_path):
    """Wrapper so multiprocessing can call it and we keep the file name."""
    return (os.path.basename(image_path), make_photo_fingerprint(image_path))


def fingerprint_a_folder(folder, name_for_printing, workers):
    """Make a fingerprint for every .jpg in a folder."""
    paths = []
    for file_name in sorted(os.listdir(folder)):
        if file_name.lower().endswith(".jpg"):
            paths.append(os.path.join(folder, file_name))

    print("fingerprinting " + str(len(paths)) + " photos in " + name_for_printing + " ...")

    results = {}
    pool = multiprocessing.Pool(workers)
    done = 0
    for file_name, fingerprint in pool.imap_unordered(fingerprint_job, paths, chunksize=32):
        done = done + 1
        if fingerprint is not None:
            # Several photos can share a fingerprint, so we keep a list.
            if fingerprint not in results:
                results[fingerprint] = []
            results[fingerprint].append(file_name)
        if done % 5000 == 0:
            print("   " + str(done) + " / " + str(len(paths)))
    pool.close()
    pool.join()

    return results


def check_duplicates(our_folder, external_folder, workers, out_folder):
    """Find photos that exist in BOTH our 2020 data and the 2019 data."""
    if not os.path.exists(our_folder):
        print("Could not find " + our_folder)
        sys.exit(1)
    if not os.path.exists(external_folder):
        print("Could not find " + external_folder)
        print("Run this first:  python src/step4_add_external.py --download")
        sys.exit(1)

    ours = fingerprint_a_folder(our_folder, "our 2020 data", workers)
    theirs = fingerprint_a_folder(external_folder, "ISIC 2019", workers)

    # A fingerprint that appears in both dictionaries is a duplicate photo.
    duplicate_external_names = []
    for fingerprint in theirs:
        if fingerprint in ours:
            for name in theirs[fingerprint]:
                duplicate_external_names.append(name)

    print("")
    print("found " + str(len(duplicate_external_names)) +
          " ISIC 2019 photos that also exist in our 2020 data")

    if len(duplicate_external_names) == 0:
        print("Nothing to remove.")
        return

    # Write the list out, and drop them from the external csv.
    list_path = os.path.join(out_folder, "duplicates_to_drop.txt")
    with open(list_path, "w") as handle:
        for name in sorted(duplicate_external_names):
            handle.write(name + "\n")
    print("wrote the list to " + list_path)

    external_csv = os.path.join(out_folder, "external_2019.csv")
    if os.path.exists(external_csv):
        external = pd.read_csv(external_csv)
        # The names in the list end with .jpg, our csv column does not.
        names_without_extension = set()
        for name in duplicate_external_names:
            names_without_extension.add(os.path.splitext(name)[0])

        before = len(external)
        external = external[~external["image_name"].isin(names_without_extension)]
        after = len(external)

        external.to_csv(external_csv, index=False)
        print("removed " + str(before - after) + " rows from external_2019.csv")
        print("external photos remaining: " + str(after))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Add ISIC 2019 as external training data.")
    parser.add_argument("--download", action="store_true",
                        help="download ISIC 2019 from the official ISIC servers")
    parser.add_argument("--prepare", action="store_true",
                        help="build external_2019.csv with our column names")
    parser.add_argument("--check-duplicates", action="store_true",
                        help="find photos that exist in both years and drop them")
    parser.add_argument("--which", choices=["all", "malignant-only"], default="all",
                        help="which 2019 photos to keep (default: all)")
    parser.add_argument("--our-photos",
                        default=os.path.join(RAW_FOLDER, "jpeg", "train"),
                        help="folder with our 2020 training photos")
    parser.add_argument("--external-photos",
                        default=os.path.join(EXTERNAL_FOLDER, "ISIC_2019_Training_Input"),
                        help="folder with the ISIC 2019 photos")
    parser.add_argument("--workers", type=int, default=0,
                        help="cpu cores for the duplicate check, 0 means all")
    parser.add_argument("--external-folder", default=EXTERNAL_FOLDER,
                        help="where the ISIC 2019 files live")
    parser.add_argument("--out-folder", default=PROCESSED_FOLDER,
                        help="where to write external_2019.csv")
    args = parser.parse_args()

    if not args.download and not args.prepare and not args.check_duplicates:
        parser.print_help()
        sys.exit(1)

    if args.download:
        download_isic_2019(args.external_folder)
    if args.prepare:
        prepare_external_csv(args.which, args.external_folder, args.out_folder)
    if args.check_duplicates:
        workers = args.workers
        if workers == 0:
            workers = multiprocessing.cpu_count()
        check_duplicates(args.our_photos, args.external_photos, workers, args.out_folder)


if __name__ == "__main__":
    main()
