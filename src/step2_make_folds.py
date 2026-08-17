"""
STEP 2 - Clean the metadata and decide the train/validation split.

Run this after step 1. It reads data/raw/train.csv and writes two files
into data/processed/:

    metadata_clean.csv   the same rows, but with missing values filled in
                         and the text columns turned into numbers
    folds.csv            which fold each image belongs to

    python src/step2_make_folds.py


WHY THIS STEP MATTERS MORE THAN IT LOOKS
----------------------------------------
The EDA found that the 33,126 photos come from only 2,056 patients. So the same
patient appears many times.

If we split the rows randomly, photo #1 of patient A can land in the training
set and photo #2 of the SAME patient can land in the validation set. The model
then recognises the patient's skin, not the cancer, and the validation score
looks great while the real performance is bad. This is called DATA LEAKAGE.

To stop that, we use "GroupKFold": all photos of one patient always stay
together in the same fold. We do it once here and save the answer to folds.csv,
so nobody has to remember to do it again later.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

THIS_FILE = os.path.abspath(__file__)
SRC_FOLDER = os.path.dirname(THIS_FILE)
PROJECT_FOLDER = os.path.dirname(SRC_FOLDER)
RAW_FOLDER = os.path.join(PROJECT_FOLDER, "data", "raw")
PROCESSED_FOLDER = os.path.join(PROJECT_FOLDER, "data", "processed")

# How many folds. 5 is the normal choice: train on 4, validate on 1, repeat.
NUMBER_OF_FOLDS = 5

# A fixed random seed so that everybody who runs this gets the SAME folds.
# Do not change this once the team has started training, or results stop
# being comparable between people.
RANDOM_SEED = 42

# The word we use when a value is missing.
UNKNOWN = "unknown"

# We list the categories by hand instead of letting pandas discover them.
# Reason: if one person's data is missing a category, pandas would give the
# remaining ones different numbers, and then two people's "sex = 1" would mean
# different things. Writing the list here keeps the numbers stable forever.
SEX_CATEGORIES = ["female", "male", UNKNOWN]

SITE_CATEGORIES = [
    "head/neck",
    "lower extremity",
    "oral/genital",
    "palms/soles",
    "torso",
    "upper extremity",
    UNKNOWN,
]


# ---------------------------------------------------------------------------
# Step 2a - fill in the missing values
# ---------------------------------------------------------------------------

def fill_missing_values(data, age_to_use_for_missing):
    """
    Fill the empty cells in sex, anatom_site_general_challenge and age_approx.

    For sex and body site we do NOT use the most common value. The EDA showed
    that when sex is missing, age is usually missing too - so the rows are not
    missing by accident. Writing "male" into those rows would be inventing
    information. Instead we give them their own category called "unknown", and
    the model is free to learn that "unknown" means something.

    Age has no sensible "unknown" bucket because it is a number, so we use the
    median (the middle value), which is the safe default for numbers.
    """
    data = data.copy()

    data["sex"] = data["sex"].fillna(UNKNOWN)
    data["sex"] = data["sex"].str.lower().str.strip()

    data["anatom_site_general_challenge"] = data["anatom_site_general_challenge"].fillna(UNKNOWN)
    data["anatom_site_general_challenge"] = data["anatom_site_general_challenge"].str.lower().str.strip()

    data["age_approx"] = data["age_approx"].fillna(age_to_use_for_missing)

    # Safety check: if some new spelling shows up that we did not plan for, we
    # want to know about it now, not silently get a wrong number later.
    for value in data["sex"].unique():
        if value not in SEX_CATEGORIES:
            print("ERROR: found an unexpected value in the sex column: " + str(value))
            sys.exit(1)

    for value in data["anatom_site_general_challenge"].unique():
        if value not in SITE_CATEGORIES:
            print("ERROR: found an unexpected body site: " + str(value))
            print("If this came from external data, map it in step4 first.")
            sys.exit(1)

    return data


# ---------------------------------------------------------------------------
# Step 2b - turn words into numbers
# ---------------------------------------------------------------------------

def turn_words_into_numbers(data):
    """
    A model cannot read the word "female", so we replace words with numbers.

    sex_enc:   0 = female, 1 = male, 2 = unknown
    site_enc:  0..5 = the six body sites, 6 = unknown
    age_norm:  age divided by 90, so it lands roughly between 0 and 1
    """
    data = data.copy()

    sex_as_category = pd.Categorical(data["sex"], categories=SEX_CATEGORIES)
    data["sex_enc"] = sex_as_category.codes

    site_as_category = pd.Categorical(
        data["anatom_site_general_challenge"], categories=SITE_CATEGORIES
    )
    data["site_enc"] = site_as_category.codes

    # Neural networks train better when the numbers going in are small and
    # similar in size. Dividing by 90 is enough here because ages stop at 90.
    data["age_norm"] = data["age_approx"] / 90.0

    return data


# ---------------------------------------------------------------------------
# Step 2c - give every row a fold number
# ---------------------------------------------------------------------------

def give_every_row_a_fold(data, number_of_folds, seed):
    """
    Split into folds so that:
      - all photos of one patient stay in the same fold  (this is "Group")
      - every fold has about the same % of cancer cases  (this is "Stratified")

    Rows that came from external data get fold = -1. See step 4 for why.
    """
    data = data.copy()
    data["fold"] = -1

    # Only the competition rows take part in the split.
    competition_rows = data[data["is_external"] == 0]

    if len(competition_rows) == 0:
        print("ERROR: there are no competition rows to split.")
        sys.exit(1)

    splitter = StratifiedGroupKFold(
        n_splits=number_of_folds,
        shuffle=True,
        random_state=seed,
    )

    # split() hands us, one fold at a time, the positions of the rows that
    # should be the VALIDATION set for that fold. We write that fold number
    # onto those rows.
    fold_number = 0
    for train_positions, validation_positions in splitter.split(
        competition_rows,
        y=competition_rows["target"],
        groups=competition_rows["patient_id"],
    ):
        rows_in_this_fold = competition_rows.index[validation_positions]
        data.loc[rows_in_this_fold, "fold"] = fold_number
        fold_number = fold_number + 1

    return data


# ---------------------------------------------------------------------------
# Step 2d - check we did not make a mistake
# ---------------------------------------------------------------------------

def check_the_folds_are_correct(data):
    """
    Stop the script if anything is wrong. It is much better to fail here than
    to hand a broken split to the person training the model.
    """
    competition_rows = data[data["is_external"] == 0]

    # 1. Every competition row must have a real fold number.
    rows_with_no_fold = (competition_rows["fold"] < 0).sum()
    if rows_with_no_fold > 0:
        print("ERROR: " + str(rows_with_no_fold) + " rows did not get a fold.")
        sys.exit(1)

    # 2. External rows must all still be -1.
    external_rows = data[data["is_external"] == 1]
    if len(external_rows) > 0:
        if (external_rows["fold"] != -1).any():
            print("ERROR: some external rows got a real fold number.")
            sys.exit(1)

    # 3. The important one: no patient is allowed to appear in two folds.
    folds_per_patient = competition_rows.groupby("patient_id")["fold"].nunique()
    patients_in_more_than_one_fold = folds_per_patient[folds_per_patient > 1]
    if len(patients_in_more_than_one_fold) > 0:
        print("ERROR: " + str(len(patients_in_more_than_one_fold)) +
              " patients ended up in more than one fold. That is data leakage.")
        sys.exit(1)

    # 4. No image should appear twice.
    if competition_rows["image_name"].duplicated().any():
        print("ERROR: the same image_name appears more than once.")
        sys.exit(1)

    print("")
    print("CHECK PASSED: " + str(competition_rows["patient_id"].nunique()) +
          " patients, and no patient is in more than one fold.")


def print_a_summary(data):
    """Print a small table so you can see the folds came out balanced."""
    competition_rows = data[data["is_external"] == 0]
    external_rows = data[data["is_external"] == 1]

    print("")
    print("rows in total:     " + str(len(data)))
    print("  from competition: " + str(len(competition_rows)))
    print("  from external:    " + str(len(external_rows)))

    print("")
    print("fold   photos   patients   cancer   cancer %")
    for fold_number in sorted(competition_rows["fold"].unique()):
        rows = competition_rows[competition_rows["fold"] == fold_number]
        photos = len(rows)
        patients = rows["patient_id"].nunique()
        cancer = int(rows["target"].sum())
        percent = round(cancer / photos * 100, 2)
        print(str(fold_number).ljust(6) +
              str(photos).ljust(9) +
              str(patients).ljust(11) +
              str(cancer).ljust(9) +
              str(percent))

    overall_percent = round(competition_rows["target"].mean() * 100, 2)
    print("")
    print("cancer rate in the competition data: " + str(overall_percent) + "%")

    if len(external_rows) > 0:
        extra_cancer = int(external_rows["target"].sum())
        all_cancer = int(competition_rows["target"].sum()) + extra_cancer
        combined_percent = round(all_cancer / len(data) * 100, 2)
        print("cancer rate once external data is added to training: " +
              str(combined_percent) + "%")
        print("  (external data added " + str(extra_cancer) + " more cancer photos)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Clean the metadata and build the folds.")
    parser.add_argument("--train-csv",
                        default=os.path.join(RAW_FOLDER, "train.csv"),
                        help="where train.csv is")
    parser.add_argument("--external-csv",
                        default="",
                        help="optional: the csv made by step 4 (external data)")
    parser.add_argument("--folds", type=int, default=NUMBER_OF_FOLDS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--out-folder", default=PROCESSED_FOLDER)
    args = parser.parse_args()

    if not os.path.exists(args.train_csv):
        print("Could not find " + args.train_csv)
        print("Run this first:  python src/step1_download.py --csv")
        sys.exit(1)

    # --- load the competition data -----------------------------------------
    competition = pd.read_csv(args.train_csv)
    competition["is_external"] = 0

    print("competition data: " + str(len(competition)) + " photos, " +
          str(competition["patient_id"].nunique()) + " patients, " +
          str(int(competition["target"].sum())) + " cancer cases")

    # --- optionally add the external data ----------------------------------
    everything = competition

    if args.external_csv != "":
        if not os.path.exists(args.external_csv):
            print("Could not find " + args.external_csv)
            print("Run this first:  python src/step4_add_external.py")
            sys.exit(1)

        external = pd.read_csv(args.external_csv)
        external["is_external"] = 1
        print("external data:    " + str(len(external)) + " photos, " +
              str(int(external["target"].sum())) + " cancer cases")

        # Both tables must have the same columns before we can stack them.
        for column_name in competition.columns:
            if column_name not in external.columns:
                external[column_name] = np.nan
        external["is_external"] = 1

        everything = pd.concat([competition, external[competition.columns]],
                               ignore_index=True)

    # --- clean, encode, split ----------------------------------------------

    # We take the median age from the COMPETITION rows only. If external data
    # has a different age profile, we do not want it changing how competition
    # rows are filled in.
    median_age = competition["age_approx"].median()
    print("median age used to fill missing ages: " + str(median_age))

    everything = fill_missing_values(everything, median_age)
    everything = turn_words_into_numbers(everything)
    everything = give_every_row_a_fold(everything, args.folds, args.seed)

    check_the_folds_are_correct(everything)
    print_a_summary(everything)

    # --- save ---------------------------------------------------------------
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    metadata_columns = [
        "image_name", "patient_id", "sex", "age_approx",
        "anatom_site_general_challenge",
        "sex_enc", "site_enc", "age_norm",
        "target", "fold", "is_external",
    ]
    metadata_path = os.path.join(args.out_folder, "metadata_clean.csv")
    everything[metadata_columns].to_csv(metadata_path, index=False)

    folds_columns = ["image_name", "patient_id", "target", "fold", "is_external"]
    folds_path = os.path.join(args.out_folder, "folds.csv")
    everything[folds_columns].to_csv(folds_path, index=False)

    print("")
    print("saved " + metadata_path)
    print("saved " + folds_path)


if __name__ == "__main__":
    main()
