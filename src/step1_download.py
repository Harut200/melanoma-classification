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
import time


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

# Kaggle sends the file list in pages. 200 is the biggest page it allows.
PAGE_SIZE = 200

# How many pages --list asks for before it stops. Kaggle answers "429 Too Many
# Requests" if we ask for hundreds of pages in a row.
LIST_PAGES = 5

# Rough sizes, used only for the disk-space check before we start.
# The zip is much bigger than the photos because it also carries DICOM and
# TFRecord copies of the very same images, which we do not use.
FULL_ZIP_BYTES = 110 * 1024 * 1024 * 1024
PHOTOS_BYTES = 35 * 1024 * 1024 * 1024


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

def get_some_files(api, how_many_pages):
    """
    Ask Kaggle for the list of files, a page at a time.

    Two things to know:

    1. Kaggle hands out the list in pages of at most 200, plus a token you give
       back to get the next page.
    2. This competition has over 44,000 files. Asking for all of them means
       hundreds of requests, and Kaggle starts answering "429 Too Many
       Requests". So we stop after `how_many_pages` on purpose.

    That is fine, because we never actually need the full list: the photo names
    are already in train.csv and test.csv.
    """
    files = []
    token = None
    hit_the_limit = False

    for page_number in range(how_many_pages):
        try:
            response = api.competition_list_files(COMPETITION, page_token=token,
                                                  page_size=PAGE_SIZE)
        except Exception as error:
            if "429" in str(error):
                print("   (Kaggle asked us to slow down, stopping the listing here)")
                hit_the_limit = True
                break
            raise

        files.extend(response.files)
        token = response.next_page_token
        if not token:
            # No token means that really was the last page.
            return files, False

    if token:
        hit_the_limit = True

    return files, hit_the_limit


def list_files(api):
    """Show a sample of the competition files and their sizes. Downloads nothing."""
    print("asking Kaggle for the file list ...")
    files, there_are_more = get_some_files(api, LIST_PAGES)

    if len(files) == 0:
        print("")
        print("Kaggle gave us an empty file list.")
        print("This almost always means you have not accepted the competition")
        print("rules yet. Open the competition page and click 'Join Competition'.")
        sys.exit(1)

    # Group by top folder so we print a handful of lines, not thousands.
    groups = {}
    for one_file in files:
        if "/" in one_file.name:
            group_name = one_file.name.split("/")[0] + "/"
        else:
            group_name = one_file.name
        if group_name not in groups:
            groups[group_name] = {"count": 0, "bytes": 0}
        groups[group_name]["count"] += 1
        groups[group_name]["bytes"] += one_file.total_bytes

    print("")
    for group_name in sorted(groups):
        info = groups[group_name]
        average = info["bytes"] / info["count"]
        print(show_gb(info["bytes"]).rjust(10) + "   " + group_name.ljust(22) +
              str(info["count"]).rjust(6) + " files seen, " +
              str(round(average / 1024)) + " KB each on average")

    if there_are_more:
        print("")
        print("This is only the first " + str(len(files)) + " files. There are over")
        print("44,000 in total and Kaggle rate-limits us if we page through them all.")
        print("")
        print("You do not need the full list. The sizes to plan around are:")
        print("   the 3 csv files   ~3 MB      --csv")
        print("   the photos        ~30 GB     --images")
        print("   DICOM + tfrecords ~78 GB     we do not download these")


def unzip_if_needed(folder, file_name):
    """
    Kaggle zips up anything above a certain size, so asking for "train.csv" can
    actually give you "train.csv.zip". If that happened, unpack it and throw the
    zip away so the rest of the pipeline just finds train.csv where it expects.
    """
    zip_path = os.path.join(folder, file_name + ".zip")
    if not os.path.exists(zip_path):
        return

    print("   Kaggle sent " + file_name + " zipped, unpacking it")
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(folder)
    os.remove(zip_path)


def download_csv_files(api):
    """Download the 3 small csv files. This is fast and safe to run any time."""
    if not os.path.exists(RAW_FOLDER):
        os.makedirs(RAW_FOLDER)

    for file_name in CSV_FILES:
        print("downloading " + file_name + " ...")
        api.competition_download_file(COMPETITION, file_name, path=RAW_FOLDER)
        unzip_if_needed(RAW_FOLDER, file_name)

    # Check we really ended up with all three, so a silent failure cannot slip
    # through to step 2.
    missing = []
    for file_name in CSV_FILES:
        if not os.path.exists(os.path.join(RAW_FOLDER, file_name)):
            missing.append(file_name)

    print("")
    if len(missing) > 0:
        print("WARNING: these files are still missing: " + str(missing))
        sys.exit(1)

    print("Done. The csv files are in: " + RAW_FOLDER)


def download_images(api):
    """
    Download the photos.

    IMPORTANT - why this asks Kaggle for ONE big zip instead of 44,108 photos:

    The obvious way is to loop over the photo names and download them one by
    one. We tried that. Kaggle allows a few hundred file downloads and then
    starts answering 404 to EVERY download request, including files that worked
    minutes earlier. It is a rate limit wearing a "not found" disguise, and once
    you trigger it you have to wait it out.

    So we make a single request instead. Kaggle sends the whole competition as
    one zip, we unpack only the jpeg photos out of it, and we throw the rest
    away. The zip also contains DICOM and TFRecord copies of the same photos,
    which we do not need - that is why the download is bigger than the photos.

    One request cannot be rate limited the way 44,108 requests can.
    """
    if not os.path.exists(RAW_FOLDER):
        os.makedirs(RAW_FOLDER)

    zip_path = os.path.join(RAW_FOLDER, COMPETITION + ".zip")

    if os.path.exists(zip_path):
        print("the zip is already downloaded, skipping straight to unpacking")
    else:
        # The zip holds the photos plus DICOM and TFRecord copies, so it is much
        # bigger than the photos alone. Make sure both it and the unpacked
        # photos will fit before we start.
        check_there_is_enough_space(FULL_ZIP_BYTES + PHOTOS_BYTES,
                                    "the competition download")

        print("downloading the whole competition as one zip")
        print("this is roughly " + show_gb(FULL_ZIP_BYTES) + " and takes a couple of hours")
        print("")
        started_at = time.time()
        api.competition_download_files(COMPETITION, path=RAW_FOLDER, quiet=False)
        minutes = (time.time() - started_at) / 60
        print("")
        print("downloaded in " + str(round(minutes, 1)) + " minutes")

    if not os.path.exists(zip_path):
        print("")
        print("The download did not produce " + zip_path)
        print("If you saw 404 errors, Kaggle has rate limited this account.")
        print("Wait an hour and run the same command again.")
        sys.exit(1)

    unpack_the_photos(zip_path)


def unpack_the_photos(zip_path):
    """Pull only the jpeg/ photos out of the big zip and ignore everything else."""
    import zipfile

    print("")
    print("unpacking the photos out of the zip")

    with zipfile.ZipFile(zip_path, "r") as archive:
        # Look at what is inside and keep only the ordinary photos.
        wanted = []
        for member in archive.namelist():
            if member.startswith("jpeg/") and member.lower().endswith(".jpg"):
                wanted.append(member)

        print("the zip holds " + str(len(archive.namelist())) + " files, " +
              str(len(wanted)) + " of them are photos we want")

        # If we found no photos, the zip is laid out differently than we expect
        # and silently carrying on would leave us with nothing to resize. Stop
        # here and show what is actually inside so it can be fixed quickly.
        if len(wanted) == 0:
            print("")
            print("PROBLEM: no files matching jpeg/*.jpg were found in the zip.")
            print("Here are the first 15 names it does contain:")
            for name in archive.namelist()[:15]:
                print("   " + name)
            print("")
            print("Fix the filter in unpack_the_photos() to match these names.")
            sys.exit(1)

        done = 0
        started_at = time.time()
        for member in wanted:
            target = os.path.join(RAW_FOLDER, member)
            # Skip anything we already unpacked, so this is restartable too.
            if not os.path.exists(target):
                archive.extract(member, RAW_FOLDER)
            done = done + 1
            if done % 5000 == 0:
                speed = done / (time.time() - started_at)
                print("   " + str(done) + " / " + str(len(wanted)) +
                      "   " + str(round(speed)) + " photos/second")

    print("")
    print("unpacked " + str(done) + " photos into " +
          os.path.join(RAW_FOLDER, "jpeg"))
    print("")
    print("The zip is no longer needed. Delete it to get "
          + show_gb(os.path.getsize(zip_path)) + " back:")
    print("   rm " + zip_path)


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
