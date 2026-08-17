"""
STEP 1 - Download the competition data from Kaggle.

Run this first. It puts the raw data into data/raw/.

What you can run:

    python src/step1_download.py --list       # just show what files exist and how big
    python src/step1_download.py --csv        # download the 3 small csv files
    python src/step1_download.py --images     # download all the photos (about 30 GB!)

Before this works you need to:

    1. Have a Kaggle account.
    2. Go to the competition page and click "Join Competition" to accept the rules.
       https://www.kaggle.com/competitions/siim-isic-melanoma-classification
    3. Log in from the terminal:  kaggle auth login
       (If you are on a server with no web browser, get a token from
        https://www.kaggle.com/settings/api and run:
            export KAGGLE_API_TOKEN=your_token_here )

If you skip step 2, every download will fail with a "403" error.
"""

import argparse
import os
import shutil
import sys


# ---------------------------------------------------------------------------
# Settings. Change these if you need to.
# ---------------------------------------------------------------------------

# The name of the competition on Kaggle (you can see it in the website address).
COMPETITION = "siim-isic-melanoma-classification"

# Where to save everything. We build the path from this file's location so that
# the script works no matter which folder you run it from.
THIS_FILE = os.path.abspath(__file__)
SRC_FOLDER = os.path.dirname(THIS_FILE)
PROJECT_FOLDER = os.path.dirname(SRC_FOLDER)
RAW_FOLDER = os.path.join(PROJECT_FOLDER, "data", "raw")

# The three small csv files with the labels and patient information.
CSV_FILES = ["train.csv", "test.csv", "sample_submission.csv"]

# We refuse to start a download if it would leave less than this much free space.
KEEP_FREE_GB = 5


# ---------------------------------------------------------------------------
# Small helper functions
# ---------------------------------------------------------------------------

def show_gb(number_of_bytes):
    """Turn a number of bytes into something readable, like '12.3 GB'."""
    gigabytes = number_of_bytes / (1024 * 1024 * 1024)
    return str(round(gigabytes, 1)) + " GB"


def free_space_bytes(folder):
    """How many free bytes are left on the disk that holds this folder."""
    disk = shutil.disk_usage(folder)
    return disk.free


def connect_to_kaggle():
    """Log in to Kaggle and give back the connection object we use to download."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception:
        print("")
        print("Could not log in to Kaggle.")
        print("")
        print("Fix it with ONE of these:")
        print("   kaggle auth login")
        print("   export KAGGLE_API_TOKEN=your_token_from_kaggle.com/settings/api")
        print("")
        sys.exit(1)
    except SystemExit:
        # The kaggle library sometimes quits by itself instead of raising an
        # error, so we catch that too and print our own message.
        print("")
        print("Could not log in to Kaggle. Run:  kaggle auth login")
        print("")
        sys.exit(1)
    return api


def check_there_is_enough_space(bytes_needed, what_we_are_downloading):
    """Stop the script if the download would fill up the disk."""
    free_now = free_space_bytes(RAW_FOLDER)
    keep_free = KEEP_FREE_GB * 1024 * 1024 * 1024

    if bytes_needed + keep_free > free_now:
        print("")
        print("NOT ENOUGH DISK SPACE for " + what_we_are_downloading)
        print("   needed:    " + show_gb(bytes_needed))
        print("   free now:  " + show_gb(free_now))
        print("   we also keep " + str(KEEP_FREE_GB) + " GB spare, just to be safe")
        print("")
        print("You can free up space, or use a smaller image size.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# The three things this script can do
# ---------------------------------------------------------------------------

def list_files(api):
    """Print every file in the competition with its size. Downloads nothing."""
    files = list(api.competition_list_files(COMPETITION))

    if len(files) == 0:
        print("")
        print("Kaggle gave us an empty file list.")
        print("This almost always means you have not accepted the competition")
        print("rules yet. Open the competition page and click 'Join Competition'.")
        sys.exit(1)

    # Sort the biggest files first so the important ones are easy to see.
    files.sort(key=lambda f: f.total_bytes, reverse=True)

    total_bytes = 0
    for one_file in files:
        total_bytes = total_bytes + one_file.total_bytes
        print(show_gb(one_file.total_bytes).rjust(10) + "   " + one_file.name)

    print("-" * 40)
    print(show_gb(total_bytes).rjust(10) + "   TOTAL (" + str(len(files)) + " files)")


def download_csv_files(api):
    """Download the 3 small csv files. This is fast and safe to run any time."""
    if not os.path.exists(RAW_FOLDER):
        os.makedirs(RAW_FOLDER)

    for file_name in CSV_FILES:
        print("downloading " + file_name + " ...")
        api.competition_download_file(COMPETITION, file_name, path=RAW_FOLDER)

    print("")
    print("Done. The csv files are in: " + RAW_FOLDER)


def download_images(api):
    """Download all the photos. This is big - about 30 GB - and slow."""
    all_files = list(api.competition_list_files(COMPETITION))

    # The competition also ships the same photos in DICOM and TFRecord format.
    # We only want the ordinary jpeg photos, which is why we filter here.
    image_files = []
    for one_file in all_files:
        if one_file.name.startswith("jpeg/"):
            image_files.append(one_file)

    if len(image_files) == 0:
        print("No files starting with 'jpeg/' were found. Did you join the competition?")
        sys.exit(1)

    total_bytes = 0
    for one_file in image_files:
        total_bytes = total_bytes + one_file.total_bytes

    print("The photos are " + str(len(image_files)) + " files, " + show_gb(total_bytes))
    check_there_is_enough_space(total_bytes, "the photos")

    if not os.path.exists(RAW_FOLDER):
        os.makedirs(RAW_FOLDER)

    # This loop takes a long time. We print every 500 files so you can see it
    # is still alive and roughly how far along it is.
    done = 0
    for one_file in image_files:
        api.competition_download_file(COMPETITION, one_file.name, path=RAW_FOLDER)
        done = done + 1
        if done % 500 == 0:
            print("  " + str(done) + " / " + str(len(image_files)) + " photos")

    print("")
    print("Done. The photos are in: " + RAW_FOLDER)


# ---------------------------------------------------------------------------
# This part reads what you typed on the command line and calls the right function
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download the melanoma competition data.")
    parser.add_argument("--list", action="store_true",
                        help="show the files and their sizes, download nothing")
    parser.add_argument("--csv", action="store_true",
                        help="download train.csv, test.csv and sample_submission.csv")
    parser.add_argument("--images", action="store_true",
                        help="download all the photos (about 30 GB)")
    args = parser.parse_args()

    # If you typed no option at all, show the help text instead of doing nothing.
    if not args.list and not args.csv and not args.images:
        parser.print_help()
        sys.exit(1)

    if not os.path.exists(RAW_FOLDER):
        os.makedirs(RAW_FOLDER)

    print("saving into: " + RAW_FOLDER)
    print("free space:  " + show_gb(free_space_bytes(RAW_FOLDER)))
    print("")

    api = connect_to_kaggle()

    if args.list:
        list_files(api)
    if args.csv:
        download_csv_files(api)
    if args.images:
        download_images(api)


# This line means: only run main() if somebody runs this file directly.
if __name__ == "__main__":
    main()
