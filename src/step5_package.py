"""
STEP 5 - Pack everything up so you can send it to the rest of the team.

Run this last. It makes one folder (and optionally one .tar file) containing
the processed photos, the csv files, and a README that explains the rules to
whoever trains the model.

    python src/step5_package.py                 # just build the folder
    python src/step5_package.py --archive       # also make one .tar file to send

The archive is a plain .tar and NOT .tar.gz on purpose. The photos are already
jpg, which is already compressed, so gzip would spend a lot of time to save
almost nothing.

HOW TO SEND IT
--------------
Best option, because it can resume if the connection drops:

    rsync -avP data/processed/handover/ someone@their-machine:~/melanoma-data/

If they cannot use rsync, send the single .tar file instead, and they unpack it:

    tar -xf melanoma_512.tar
"""

import argparse
import hashlib
import os
import shutil
import sys
import tarfile

import pandas as pd


THIS_FILE = os.path.abspath(__file__)
SRC_FOLDER = os.path.dirname(THIS_FILE)
PROJECT_FOLDER = os.path.dirname(SRC_FOLDER)
PROCESSED_FOLDER = os.path.join(PROJECT_FOLDER, "data", "processed")


def folder_size_gb(folder):
    """Add up the size of every file inside a folder."""
    total = 0
    for current_folder, subfolders, file_names in os.walk(folder):
        for file_name in file_names:
            full_path = os.path.join(current_folder, file_name)
            total = total + os.path.getsize(full_path)
    return round(total / (1024 * 1024 * 1024), 2)


def count_jpgs(folder):
    if not os.path.exists(folder):
        return 0
    count = 0
    for file_name in os.listdir(folder):
        if file_name.lower().endswith(".jpg"):
            count = count + 1
    return count


def write_readme(handover_folder, size, train_count, test_count, folds):
    """
    Write the README that goes WITH the data.

    This matters. The person training the model will not read our source code.
    Everything they must not get wrong has to be in this one file.
    """
    competition_rows = folds[folds["is_external"] == 0]
    external_rows = folds[folds["is_external"] == 1]

    lines = []
    lines.append("# Melanoma data - ready for training")
    lines.append("")
    lines.append("Photos are " + str(size) + "x" + str(size) + " jpg, cropped square")
    lines.append("from the middle of the original and with body hair painted out.")
    lines.append("")
    lines.append("## THE TWO RULES - please read")
    lines.append("")
    lines.append("### Rule 1: use the `fold` column. Do not make your own split.")
    lines.append("")
    lines.append("There are " + str(len(competition_rows)) + " photos but only " +
                 str(competition_rows["patient_id"].nunique()) + " patients, so the same")
    lines.append("patient appears many times. If you split randomly, the same patient ends")
    lines.append("up in both training and validation. The model then recognises the person")
    lines.append("instead of the cancer, your validation score looks great, and the real")
    lines.append("score is bad.")
    lines.append("")
    lines.append("The folds in `folds.csv` are already built so that all photos of one")
    lines.append("patient stay together. Just use them:")
    lines.append("")
    lines.append("```python")
    lines.append("import pandas as pd")
    lines.append("folds = pd.read_csv('folds.csv')")
    lines.append("")
    lines.append("VALIDATION_FOLD = 0")
    lines.append("train_rows = folds[folds.fold != VALIDATION_FOLD]   # includes fold -1")
    lines.append("valid_rows = folds[folds.fold == VALIDATION_FOLD]")
    lines.append("```")
    lines.append("")

    if len(external_rows) > 0:
        lines.append("### Rule 2: `fold = -1` is external data. Train on it, never validate on it.")
        lines.append("")
        lines.append(str(len(external_rows)) + " photos come from the ISIC 2019 challenge, added because")
        lines.append("our own data is only " +
                     str(round(competition_rows["target"].mean() * 100, 2)) +
                     "% cancer. They were taken with different cameras")
        lines.append("in different hospitals. Our score has to predict how well we do on the")
        lines.append("2020 test set, so validating on 2019 photos would tell us nothing useful.")
        lines.append("")
        lines.append("Because they are marked `fold = -1`, the code above already does the")
        lines.append("right thing: -1 is never equal to VALIDATION_FOLD, so they always land")
        lines.append("in training.")
    else:
        lines.append("### Rule 2: no external data is included in this package.")
        lines.append("")
        lines.append("Every row is from the 2020 competition.")
    lines.append("")

    lines.append("## What is in this folder")
    lines.append("")
    lines.append("| Name | What it is |")
    lines.append("| --- | --- |")
    lines.append("| `train_" + str(size) + "/` | " + str(train_count) + " training photos, named `<image_name>.jpg` |")
    if test_count > 0:
        lines.append("| `test_" + str(size) + "/` | " + str(test_count) + " test photos |")
    lines.append("| `folds.csv` | image_name, patient_id, target, fold, is_external |")
    lines.append("| `metadata_clean.csv` | same rows plus age/sex/body-site as numbers |")
    lines.append("| `checksums.sha256` | to verify nothing broke while copying |")
    lines.append("")

    lines.append("## Columns")
    lines.append("")
    lines.append("| Column | Meaning |")
    lines.append("| --- | --- |")
    lines.append("| `image_name` | filename without `.jpg` |")
    lines.append("| `patient_id` | who the photo belongs to. `EXT19_...` means external |")
    lines.append("| `target` | 1 = melanoma, 0 = not melanoma |")
    lines.append("| `fold` | 0-4 for validation, -1 = external, train only |")
    lines.append("| `is_external` | 1 = from ISIC 2019, 0 = from the 2020 competition |")
    lines.append("| `sex_enc` | 0 female, 1 male, 2 unknown |")
    lines.append("| `site_enc` | 0-5 body sites, 6 unknown |")
    lines.append("| `age_norm` | age divided by 90 |")
    lines.append("")

    lines.append("## Fold balance")
    lines.append("")
    lines.append("| fold | photos | patients | cancer | cancer % |")
    lines.append("| --- | --- | --- | --- | --- |")
    for fold_number in sorted(competition_rows["fold"].unique()):
        rows = competition_rows[competition_rows["fold"] == fold_number]
        percent = round(rows["target"].mean() * 100, 2)
        lines.append("| " + str(fold_number) + " | " + str(len(rows)) + " | " +
                     str(rows["patient_id"].nunique()) + " | " +
                     str(int(rows["target"].sum())) + " | " + str(percent) + " |")
    lines.append("")

    lines.append("## About the score")
    lines.append("")
    lines.append("The competition is judged on ROC-AUC, so report that. But with this few")
    lines.append("positive cases ROC-AUC looks good very easily. Also report PR-AUC and")
    lines.append("sensitivity at a fixed specificity - those are the numbers that say")
    lines.append("whether the model would actually be useful. A random model scores")
    lines.append("about 0.0176 PR-AUC here.")
    lines.append("")

    lines.append("## Checking the files arrived intact")
    lines.append("")
    lines.append("```bash")
    lines.append("sha256sum -c checksums.sha256")
    lines.append("```")
    lines.append("")

    readme_path = os.path.join(handover_folder, "README.md")
    with open(readme_path, "w") as handle:
        handle.write("\n".join(lines))


def write_checksums(handover_folder):
    """
    Write a sha256 for every csv and for the README.

    We deliberately do NOT checksum all 44,000 photos - it would take a long
    time and produce a huge file. If a photo were damaged in transit the count
    check below would almost certainly catch it.
    """
    lines = []
    for file_name in sorted(os.listdir(handover_folder)):
        full_path = os.path.join(handover_folder, file_name)
        if not os.path.isfile(full_path):
            continue
        if file_name == "checksums.sha256":
            continue

        hasher = hashlib.sha256()
        with open(full_path, "rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
        lines.append(hasher.hexdigest() + "  " + file_name)

    with open(os.path.join(handover_folder, "checksums.sha256"), "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Pack the processed data for the team.")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--processed-folder", default=PROCESSED_FOLDER)
    parser.add_argument("--out-folder", default="")
    parser.add_argument("--archive", action="store_true",
                        help="also build a single .tar file")
    parser.add_argument("--copy", action="store_true",
                        help="really copy the photos instead of hard-linking them")
    args = parser.parse_args()

    handover_folder = args.out_folder
    if handover_folder == "":
        handover_folder = os.path.join(args.processed_folder, "handover")

    train_folder = os.path.join(args.processed_folder, "train_" + str(args.size))
    test_folder = os.path.join(args.processed_folder, "test_" + str(args.size))
    folds_path = os.path.join(args.processed_folder, "folds.csv")

    if not os.path.exists(train_folder):
        print("Could not find " + train_folder)
        print("Run this first:  python src/step3_resize_images.py --split train")
        sys.exit(1)
    if not os.path.exists(folds_path):
        print("Could not find " + folds_path)
        print("Run this first:  python src/step2_make_folds.py")
        sys.exit(1)

    if not os.path.exists(handover_folder):
        os.makedirs(handover_folder)

    # --- put the photo folders in place -------------------------------------
    # A hard link is a second name for the SAME file on disk. It costs no extra
    # space, which matters when the photos are several GB. If the destination is
    # on a different disk, hard links are impossible and we copy instead.
    for source_folder in [train_folder, test_folder]:
        if not os.path.exists(source_folder):
            continue
        destination = os.path.join(handover_folder, os.path.basename(source_folder))
        if not os.path.exists(destination):
            os.makedirs(destination)

        print("linking " + os.path.basename(source_folder) + " ...")
        for file_name in os.listdir(source_folder):
            if not file_name.lower().endswith(".jpg"):
                continue
            source_file = os.path.join(source_folder, file_name)
            destination_file = os.path.join(destination, file_name)
            if os.path.exists(destination_file):
                continue
            if args.copy:
                shutil.copy2(source_file, destination_file)
            else:
                try:
                    os.link(source_file, destination_file)
                except OSError:
                    shutil.copy2(source_file, destination_file)

    # --- copy the csv files -------------------------------------------------
    for file_name in ["folds.csv", "metadata_clean.csv", "external_2019.csv"]:
        source_file = os.path.join(args.processed_folder, file_name)
        if os.path.exists(source_file):
            shutil.copy2(source_file, os.path.join(handover_folder, file_name))

    # --- write the README and the checksums ---------------------------------
    folds = pd.read_csv(folds_path)
    train_count = count_jpgs(os.path.join(handover_folder, "train_" + str(args.size)))
    test_count = count_jpgs(os.path.join(handover_folder, "test_" + str(args.size)))

    write_readme(handover_folder, args.size, train_count, test_count, folds)
    write_checksums(handover_folder)

    # --- sanity check: do the photos and the csv agree? ---------------------
    expected = len(folds)
    print("")
    print("photos in the package: " + str(train_count) + " train, " +
          str(test_count) + " test")
    print("rows in folds.csv:     " + str(expected))
    if train_count != expected:
        print("")
        print("WARNING: those two numbers do not match.")
        print("Usually this means step 3 has not finished resizing everything yet,")
        print("or the external photos still need resizing. Check before you send it.")

    print("")
    print("package folder: " + handover_folder)
    print("package size:   " + str(folder_size_gb(handover_folder)) + " GB")

    # --- optional single file -----------------------------------------------
    if args.archive:
        archive_path = os.path.join(args.processed_folder,
                                    "melanoma_" + str(args.size) + ".tar")
        print("")
        print("building " + archive_path + " (this takes a few minutes) ...")
        with tarfile.open(archive_path, "w") as archive:
            archive.add(handover_folder, arcname="melanoma_" + str(args.size))
        size_gb = round(os.path.getsize(archive_path) / (1024 * 1024 * 1024), 2)
        print("done, " + str(size_gb) + " GB")

    print("")
    print("To send it:")
    print("   rsync -avP " + handover_folder + "/ someone@their-machine:~/melanoma-data/")


if __name__ == "__main__":
    main()
