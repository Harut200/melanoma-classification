"""
STEP 3 - Make every photo the same size and remove the hairs.

Run this after step 2. It reads the original photos from data/raw/jpeg/ and
writes clean, same-sized photos into data/processed/.

    python src/step3_resize_images.py --split train
    python src/step3_resize_images.py --split test

The output is ordinary .jpg files in a normal folder. You can open them and
look at them. Any deep learning library can read them.


WHY WE DO THIS
--------------
1. RESIZE. The EDA found photos from 640x480 all the way up to 6000x4000.
   A neural network needs every input to be the same size, so we make them all
   512x512.

   We first cut a SQUARE out of the middle of the photo, then shrink it.
   We do not simply squash the photo into a square, because squashing changes
   the shape of the mole - and the shape of a mole is one of the things a
   doctor (and the model) looks at.

2. REMOVE HAIRS. Many photos have body hair lying across the mole. The hair
   has nothing to do with cancer, so it is just noise. We find the dark thin
   lines and paint over them using the colours around them. This trick is
   called "DullRazor".

   We remove hair AFTER resizing, not before. The hair-finding tool looks for
   lines of a certain thickness in pixels. On a huge 6000x4000 photo a hair is
   many pixels thick; on a small 640x480 photo the same hair is thin. If we run
   it before resizing, it behaves differently for every photo. After resizing,
   every photo is 512x512, so it behaves the same for all of them.


SPEED
-----
There are about 44,000 photos. Doing them one at a time would take hours, so we
use several CPU cores at once ("multiprocessing"). On the 20-thread server this
takes roughly 15-40 minutes.
"""

import argparse
import multiprocessing
import os
import sys
import time

import cv2
import pandas as pd


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

THIS_FILE = os.path.abspath(__file__)
SRC_FOLDER = os.path.dirname(THIS_FILE)
PROJECT_FOLDER = os.path.dirname(SRC_FOLDER)
RAW_FOLDER = os.path.join(PROJECT_FOLDER, "data", "raw")
PROCESSED_FOLDER = os.path.join(PROJECT_FOLDER, "data", "processed")

# The size every photo becomes. 512 is a good balance between detail and speed.
DEFAULT_SIZE = 512

# How much to compress the saved jpg. 92 keeps the picture looking good while
# still making the file small. 100 would be huge, 70 would look blurry.
JPEG_QUALITY = 92

# Hair removal settings.
# The "kernel" is the little window the hair detector slides over the photo.
# 11 pixels works well on a 512x512 photo. If you change the size, we scale it.
HAIR_KERNEL_AT_512 = 11
# How dark a line has to be before we call it hair. Lower = removes more.
HAIR_DARKNESS_THRESHOLD = 10
# How far around a hair pixel we look for replacement colour.
INPAINT_RADIUS = 1


def work_out_kernel_size(image_size):
    """Scale the hair-detector window if you are not using 512x512."""
    kernel = round(HAIR_KERNEL_AT_512 * image_size / 512)
    if kernel < 3:
        kernel = 3
    # OpenCV wants an odd number here, so add 1 if it came out even.
    if kernel % 2 == 0:
        kernel = kernel + 1
    return kernel


# ---------------------------------------------------------------------------
# The image work
# ---------------------------------------------------------------------------

def cut_square_from_middle(photo):
    """
    Cut the biggest possible square out of the centre of the photo.

    A photo is stored as a grid of numbers, shape = (height, width, 3 colours).
    """
    height = photo.shape[0]
    width = photo.shape[1]

    # The square can only be as big as the shorter side.
    square_size = min(height, width)

    # Work out where to start cutting so the square is centred.
    start_row = (height - square_size) // 2
    start_column = (width - square_size) // 2

    return photo[start_row:start_row + square_size,
                 start_column:start_column + square_size]


def remove_hair(photo, kernel_size):
    """
    Find dark thin lines (hair) and paint over them.

    Three moves:
      1. Turn the photo grey, because hair is about darkness, not colour.
      2. "Blackhat" - an OpenCV operation that lights up small dark shapes on a
         lighter background. Hairs light up, the mole does not.
      3. Anything bright enough in that result is marked as hair, and inpaint()
         fills those pixels in using the colours around them.
    """
    grey = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)

    window = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    hair_highlighted = cv2.morphologyEx(grey, cv2.MORPH_BLACKHAT, window)

    # threshold() gives back two things; we only need the second one (the mask).
    _, hair_mask = cv2.threshold(hair_highlighted, HAIR_DARKNESS_THRESHOLD,
                                 255, cv2.THRESH_BINARY)

    return cv2.inpaint(photo, hair_mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)


def process_one_photo(job):
    """
    Do the whole job for a single photo: read, crop, resize, de-hair, save.

    This function runs inside a separate worker process, which is why all the
    information it needs is packed into one plain dictionary called `job`.
    It gives back a small dictionary saying what happened, so the main process
    can count successes and failures.
    """
    input_path = job["input_path"]
    output_path = job["output_path"]
    size = job["size"]
    do_hair_removal = job["do_hair_removal"]
    kernel_size = job["kernel_size"]

    # If we already did this photo in an earlier run, skip it. This makes the
    # script safe to stop and restart.
    if os.path.exists(output_path):
        return {"ok": True, "skipped": True, "message": ""}

    photo = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if photo is None:
        return {"ok": False, "skipped": False,
                "message": "could not read " + input_path}

    photo = cut_square_from_middle(photo)

    # INTER_AREA is the right choice when making a picture SMALLER.
    # INTER_CUBIC is the right choice when making it BIGGER.
    if photo.shape[0] > size:
        how_to_resize = cv2.INTER_AREA
    else:
        how_to_resize = cv2.INTER_CUBIC
    photo = cv2.resize(photo, (size, size), interpolation=how_to_resize)

    if do_hair_removal:
        photo = remove_hair(photo, kernel_size)

    # Turn the picture into jpg bytes in memory first.
    # (We do NOT use cv2.imwrite here. imwrite decides the file format from the
    # file extension, and our temporary file ends in ".tmp", which it does not
    # recognise. Encoding in memory lets us name the temporary file whatever
    # we like.)
    encoded_ok, jpg_bytes = cv2.imencode(".jpg", photo,
                                         [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not encoded_ok:
        return {"ok": False, "skipped": False,
                "message": "could not encode " + output_path}

    # Write to a temporary name, then rename. Renaming is instant, so if the
    # script is killed halfway we are left with a leftover ".tmp" file rather
    # than a half-written .jpg that looks finished.
    temporary_path = output_path + ".tmp"
    with open(temporary_path, "wb") as handle:
        handle.write(jpg_bytes.tobytes())
    os.rename(temporary_path, output_path)

    return {"ok": True, "skipped": False, "message": ""}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Resize photos and remove hair.")
    parser.add_argument("--split", choices=["train", "test"], default="train",
                        help="which set of photos to process")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE,
                        help="output width and height in pixels")
    parser.add_argument("--input-folder", default="",
                        help="where the original photos are (we guess if empty)")
    parser.add_argument("--image-list", default="",
                        help="csv with an image_name column (we guess if empty)")
    parser.add_argument("--output-folder", default="",
                        help="where to save (we guess if empty)")
    parser.add_argument("--workers", type=int, default=0,
                        help="how many CPU cores to use, 0 means all of them")
    parser.add_argument("--no-hair-removal", action="store_true",
                        help="skip the hair removal step")
    parser.add_argument("--external", action="store_true",
                        help="use this when the image list is the external ISIC 2019 csv")
    parser.add_argument("--limit", type=int, default=0,
                        help="only do the first N photos, useful for a quick test")
    args = parser.parse_args()

    # --- work out the folders ----------------------------------------------
    input_folder = args.input_folder
    if input_folder == "":
        input_folder = os.path.join(RAW_FOLDER, "jpeg", args.split)

    output_folder = args.output_folder
    if output_folder == "":
        output_folder = os.path.join(PROCESSED_FOLDER, args.split + "_" + str(args.size))

    image_list_path = args.image_list
    if image_list_path == "":
        if args.split == "train":
            # For training we use the cleaned list from step 2, so the photos
            # and the folds always match each other.
            image_list_path = os.path.join(PROCESSED_FOLDER, "metadata_clean.csv")
        else:
            image_list_path = os.path.join(RAW_FOLDER, "test.csv")

    if not os.path.exists(image_list_path):
        print("Could not find " + image_list_path)
        if args.split == "train":
            print("Run this first:  python src/step2_make_folds.py")
        else:
            print("Run this first:  python src/step1_download.py --csv")
        sys.exit(1)

    if not os.path.exists(input_folder):
        print("Could not find the photos in " + input_folder)
        print("Run this first:  python src/step1_download.py --images")
        sys.exit(1)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # --- build the list of jobs --------------------------------------------
    image_list = pd.read_csv(image_list_path)

    # metadata_clean.csv holds BOTH our photos and the external ones, but they
    # live in different folders on disk. So we keep only the half that matches
    # the folder we were pointed at.
    if "is_external" in image_list.columns:
        if args.external:
            image_list = image_list[image_list["is_external"] == 1]
        else:
            image_list = image_list[image_list["is_external"] == 0]

    if len(image_list) == 0:
        print("There are no photos to process in " + image_list_path + ".")
        print("If this is the external ISIC 2019 list, add the --external flag.")
        sys.exit(1)

    if args.limit > 0:
        image_list = image_list.head(args.limit)

    do_hair_removal = not args.no_hair_removal
    kernel_size = work_out_kernel_size(args.size)

    jobs = []
    for image_name in image_list["image_name"]:
        jobs.append({
            "input_path": os.path.join(input_folder, str(image_name) + ".jpg"),
            "output_path": os.path.join(output_folder, str(image_name) + ".jpg"),
            "size": args.size,
            "do_hair_removal": do_hair_removal,
            "kernel_size": kernel_size,
        })

    how_many_workers = args.workers
    if how_many_workers == 0:
        how_many_workers = multiprocessing.cpu_count()

    print("photos to do:  " + str(len(jobs)))
    print("reading from:  " + input_folder)
    print("writing to:    " + output_folder)
    print("size:          " + str(args.size) + "x" + str(args.size))
    if do_hair_removal:
        print("hair removal:  on (window " + str(kernel_size) + " pixels)")
    else:
        print("hair removal:  off")
    print("cpu cores:     " + str(how_many_workers))
    print("")

    # --- do the work --------------------------------------------------------
    started_at = time.time()
    done = 0
    skipped = 0
    problems = []

    # A Pool is a group of worker processes. imap_unordered hands each worker a
    # job and gives us back results as soon as they are ready, in any order.
    pool = multiprocessing.Pool(how_many_workers)
    for result in pool.imap_unordered(process_one_photo, jobs, chunksize=16):
        done = done + 1
        if result["skipped"]:
            skipped = skipped + 1
        if not result["ok"]:
            problems.append(result["message"])

        # Print progress every 1000 photos so you know it is alive.
        if done % 1000 == 0:
            seconds_so_far = time.time() - started_at
            speed = done / seconds_so_far
            photos_left = len(jobs) - done
            minutes_left = photos_left / speed / 60
            print("  " + str(done) + " / " + str(len(jobs)) +
                  "   " + str(round(speed)) + " photos/second" +
                  "   about " + str(round(minutes_left)) + " min left")

    pool.close()
    pool.join()

    total_seconds = time.time() - started_at

    print("")
    print("finished " + str(done) + " photos in " +
          str(round(total_seconds / 60, 1)) + " minutes")
    if skipped > 0:
        print("(" + str(skipped) + " were already done and were skipped)")
    print("saved into: " + output_folder)

    if len(problems) > 0:
        print("")
        print("WARNING: " + str(len(problems)) + " photos failed:")
        for message in problems[:10]:
            print("   " + message)
        if len(problems) > 10:
            print("   ... and " + str(len(problems) - 10) + " more")


if __name__ == "__main__":
    main()
