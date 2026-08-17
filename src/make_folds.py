"""Build the metadata contract: imputed features and frozen patient-level folds.

Everything downstream reads folds.csv and nothing else decides splits. That is
deliberate. The EDA found 33,126 images belonging to only 2,056 patients, so a
random row-level split leaks a patient's skin tone and lesion appearance across
train and validation. Freezing the assignment here means no notebook can undo it.

    python src/make_folds.py                       # competition data only
    python src/make_folds.py --external <csv> ...  # add external ISIC rows

Outputs, both into data/processed/:
    metadata_processed.csv  imputed + encoded features, one row per image
    folds.csv               image_name, patient_id, target, fold, is_external

External rows always get fold = -1: they are train-only material for fixing the
1.8% malignant rate, and putting them in a validation fold would make local CV
stop tracking the competition metric.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

N_FOLDS = 5
SEED = 42

UNKNOWN = "unknown"

# Fixed category orders, so codes stay stable across runs and across whichever
# subset of the data a given run happens to see.
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


def impute(df: pd.DataFrame, age_median: float) -> pd.DataFrame:
    """Fill the three columns the EDA found missing, per its conclusion 1.

    sex and anatom_site get an explicit "unknown" category rather than the mode:
    the EDA showed missingness in sex and age_approx is correlated, so these are
    not missing at random and inventing a modal value would fabricate signal.
    """
    df = df.copy()
    df["sex"] = df["sex"].fillna(UNKNOWN).str.lower().str.strip()
    df["anatom_site_general_challenge"] = (
        df["anatom_site_general_challenge"].fillna(UNKNOWN).str.lower().str.strip()
    )
    df["age_approx"] = df["age_approx"].fillna(age_median)

    unseen_sex = set(df["sex"]) - set(SEX_CATEGORIES)
    unseen_site = set(df["anatom_site_general_challenge"]) - set(SITE_CATEGORIES)
    if unseen_sex:
        raise ValueError(f"unexpected sex values: {sorted(unseen_sex)}")
    if unseen_site:
        raise ValueError(f"unexpected anatom_site values: {sorted(unseen_site)}")
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Map categoricals to stable integer codes and normalise age."""
    df = df.copy()
    df["sex_enc"] = pd.Categorical(df["sex"], categories=SEX_CATEGORIES).codes
    df["site_enc"] = pd.Categorical(
        df["anatom_site_general_challenge"], categories=SITE_CATEGORIES
    ).codes
    # age_approx is binned to 5-year steps in the source data; /90 keeps it in
    # roughly [0, 1] without fitting a scaler that would have to be persisted.
    df["age_norm"] = df["age_approx"].astype("float32") / 90.0
    return df


def assign_folds(df: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    """Stratified on target, grouped by patient. No patient spans two folds."""
    df = df.copy()
    df["fold"] = -1

    internal = df["is_external"] == 0
    sub = df.loc[internal]
    if sub.empty:
        raise ValueError("no competition rows to fold; check the input CSVs")

    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(
        splitter.split(sub, y=sub["target"], groups=sub["patient_id"])
    ):
        df.loc[sub.index[val_idx], "fold"] = fold
    return df


def verify(df: pd.DataFrame, n_folds: int) -> None:
    """Fail loudly rather than hand a leaking split to the next person."""
    internal = df[df["is_external"] == 0]

    unassigned = (internal["fold"] < 0).sum()
    if unassigned:
        raise AssertionError(f"{unassigned} competition rows left unassigned")

    if (df.loc[df["is_external"] == 1, "fold"] != -1).any():
        raise AssertionError("external rows must stay at fold -1")

    # The check that matters: a patient must live in exactly one fold.
    spans = internal.groupby("patient_id")["fold"].nunique()
    leaking = spans[spans > 1]
    if not leaking.empty:
        raise AssertionError(
            f"{len(leaking)} patients span multiple folds, e.g. "
            f"{leaking.head().to_dict()}"
        )

    if internal["image_name"].duplicated().any():
        raise AssertionError("duplicate image_name in competition rows")

    print(f"\nverified: {internal['patient_id'].nunique()} patients across "
          f"{n_folds} folds, no patient in more than one")


def report(df: pd.DataFrame) -> None:
    internal = df[df["is_external"] == 0]
    print(f"\nrows: {len(df)}  (competition {len(internal)}, "
          f"external {len(df) - len(internal)})")

    summary = (
        internal.groupby("fold")
        .agg(images=("image_name", "size"),
             patients=("patient_id", "nunique"),
             malignant=("target", "sum"))
    )
    summary["malignant_rate"] = (summary["malignant"] / summary["images"] * 100).round(2)
    print("\nper-fold balance:")
    print(summary.to_string())

    overall = internal["target"].mean() * 100
    print(f"\noverall malignant rate: {overall:.2f}%")
    if len(df) > len(internal):
        ext_pos = df.loc[df["is_external"] == 1, "target"].sum()
        combined = (internal["target"].sum() + ext_pos) / len(df) * 100
        print(f"with external in train: {combined:.2f}%  "
              f"(+{ext_pos} malignant rows)")


def load_external(paths: list[str]) -> pd.DataFrame:
    """Load external ISIC rows, namespacing patient ids so they cannot collide."""
    frames = []
    for i, path in enumerate(paths):
        ext = pd.read_csv(path)
        missing = {"image_name", "target"} - set(ext.columns)
        if missing:
            sys.exit(f"{path} is missing required columns: {sorted(missing)}")

        # External sets often have no patient_id at all. A synthetic unique id per
        # row is correct here: these rows never enter a validation fold, so the
        # only job of the id is to stay out of the competition id namespace.
        source = Path(path).stem
        if "patient_id" in ext.columns:
            ext["patient_id"] = f"EXT{i}_" + ext["patient_id"].astype(str)
        else:
            ext["patient_id"] = [f"EXT{i}_{source}_{j}" for j in range(len(ext))]

        for col in ("sex", "age_approx", "anatom_site_general_challenge"):
            if col not in ext.columns:
                ext[col] = np.nan

        ext["is_external"] = 1
        frames.append(ext)
        print(f"external: {path} -> {len(ext)} rows, "
              f"{int(ext['target'].sum())} malignant")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--train-csv", default=str(RAW_DIR / "train.csv"))
    parser.add_argument("--external", nargs="*", default=[],
                        help="extra CSVs with at least image_name and target")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", default=str(PROCESSED_DIR))
    args = parser.parse_args()

    train_path = Path(args.train_csv)
    if not train_path.exists():
        sys.exit(f"{train_path} not found. Run: python src/download_data.py --csv")

    train = pd.read_csv(train_path)
    train["is_external"] = 0
    print(f"competition: {len(train)} rows, "
          f"{train['patient_id'].nunique()} patients, "
          f"{int(train['target'].sum())} malignant")

    df = train
    if args.external:
        df = pd.concat([train, load_external(args.external)], ignore_index=True)

    # Median from competition rows only, so an external set with a different age
    # profile cannot shift how competition rows are imputed.
    age_median = float(train["age_approx"].median())
    print(f"age_approx median for imputation: {age_median}")

    df = impute(df, age_median)
    df = encode(df)
    df = assign_folds(df, args.folds, args.seed)

    verify(df, args.folds)
    report(df)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_cols = ["image_name", "patient_id", "sex", "age_approx",
                 "anatom_site_general_challenge", "sex_enc", "site_enc",
                 "age_norm", "target", "fold", "is_external"]
    df[meta_cols].to_csv(out_dir / "metadata_processed.csv", index=False)

    fold_cols = ["image_name", "patient_id", "target", "fold", "is_external"]
    df[fold_cols].to_csv(out_dir / "folds.csv", index=False)

    print(f"\nwrote {out_dir / 'metadata_processed.csv'}")
    print(f"wrote {out_dir / 'folds.csv'}")


if __name__ == "__main__":
    main()
