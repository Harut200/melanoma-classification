"""Package the preprocessed data as a private Kaggle dataset for the modelling work.

The server has no GPU, so whoever runs model experiments needs the data somewhere
with one attached. A private Kaggle dataset is the shortest path: free GPU/TPU
quota, versioned, and TFRecords are what the TF input pipeline wants anyway.

    python src/export_dataset.py --stage                 # build the upload folder
    python src/export_dataset.py --create --user <name>  # first upload
    python src/export_dataset.py --version "note"        # every upload after that

Staging hardlinks the shards rather than copying them, so the export folder costs
no extra disk on the same filesystem.

Kaggle's uploader skips subdirectories by default, so the staged folder is flat.
Shard filenames already carry their split, so nothing is ambiguous once flattened.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
EXPORT_DIR = PROCESSED_DIR / "kaggle_export"

DEFAULT_SLUG = "siim-isic-melanoma-512-preprocessed"
DEFAULT_TITLE = "SIIM-ISIC Melanoma 512px preprocessed"


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_readme(export_dir: Path, folds: pd.DataFrame, manifests: dict) -> str:
    internal = folds[folds["is_external"] == 0]
    per_fold = (
        internal.groupby("fold")
        .agg(images=("image_name", "size"),
             patients=("patient_id", "nunique"),
             malignant=("target", "sum"))
    )
    per_fold["malignant_rate_%"] = (
        per_fold["malignant"] / per_fold["images"] * 100
    ).round(2)

    n_ext = int((folds["is_external"] == 1).sum())
    size = next(iter(manifests.values()))["size"] if manifests else "?"

    lines = [
        "# SIIM-ISIC Melanoma — preprocessed",
        "",
        f"Images centre-cropped to square and resized to {size}x{size}, hair artifacts",
        "removed, packed into TFRecords with their metadata.",
        "",
        "## Read this before you split anything",
        "",
        "**Use the `fold` column. Do not make your own split.**",
        "",
        "There are 33,126 images from only 2,056 patients, and one patient's images",
        "share skin tone and lesion appearance. A random row-level split puts the same",
        "patient in train and validation, which inflates validation scores by an amount",
        "you cannot measure. The folds here are patient-level (StratifiedGroupKFold,",
        "grouped on `patient_id`, stratified on `target`), so no patient spans two folds.",
        "",
        "**`fold = -1` means external data. Train on it, never validate on it.**",
        "",
        f"{n_ext} external rows are included to soften the 1.8% malignant rate. They come",
        "from a different distribution than the competition test set, so scoring against",
        "them would stop your CV tracking the leaderboard.",
        "",
        "```python",
        "train_ds = files_where(fold != VAL_FOLD)   # includes fold == -1",
        "val_ds   = files_where(fold == VAL_FOLD)   # never includes fold == -1",
        "```",
        "",
        "## Files",
        "",
        "| File | What |",
        "| --- | --- |",
        "| `train_*.tfrec` | training shards, one Example per image |",
        "| `test_*.tfrec` | competition test shards, `target = -1` |",
        "| `folds.csv` | `image_name, patient_id, target, fold, is_external` |",
        "| `metadata_processed.csv` | imputed and encoded features |",
        "| `manifest_*.json` | exact preprocessing settings used |",
        "",
        "## TFRecord schema",
        "",
        "```python",
        "FEATURES = {",
        '    "image":       tf.io.FixedLenFeature([], tf.string),   # JPEG bytes',
        '    "image_name":  tf.io.FixedLenFeature([], tf.string),',
        '    "patient_id":  tf.io.FixedLenFeature([], tf.string),',
        '    "sex_enc":     tf.io.FixedLenFeature([], tf.int64),    # 0 f, 1 m, 2 unknown',
        '    "site_enc":    tf.io.FixedLenFeature([], tf.int64),    # 0-5 sites, 6 unknown',
        '    "age_norm":    tf.io.FixedLenFeature([], tf.float32),  # age_approx / 90',
        '    "target":      tf.io.FixedLenFeature([], tf.int64),',
        '    "fold":        tf.io.FixedLenFeature([], tf.int64),',
        '    "is_external": tf.io.FixedLenFeature([], tf.int64),',
        "}",
        "```",
        "",
        "## Fold balance",
        "",
        "```",
        per_fold.to_string(),
        "```",
        "",
        "## What was done to the images",
        "",
        "- Centre crop to the shorter side, then resize. No aspect-ratio distortion.",
        "- Hair removal: morphological blackhat isolates dark thin structures, those",
        "  pixels are inpainted (DullRazor). Applied after resize so the kernel means",
        "  the same thing for a 640x480 and a 6000x4000 source.",
        "- Re-encoded as JPEG quality 92.",
        "",
        "## What was done to the metadata",
        "",
        "- `sex` and `anatom_site_general_challenge`: missing filled with `unknown` as its",
        "  own category. The EDA showed missingness in `sex` and `age_approx` is",
        "  correlated, so it is not missing at random and a modal fill would invent signal.",
        "- `age_approx`: filled with the competition-set median.",
        "",
        "## Metric note",
        "",
        "ROC-AUC is the competition metric but is generous at this positive rate. Report",
        "PR-AUC and sensitivity at fixed specificity too. Random-baseline PR-AUC is 0.0176.",
        "",
    ]
    return "\n".join(lines)


def stage(args) -> Path:
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    processed = Path(args.processed_dir)
    manifests, n_shards = {}, 0

    for split in ("train", "test"):
        shard_dir = processed / f"{split}_{args.size}"
        if not shard_dir.is_dir():
            print(f"skipping {split}: {shard_dir} not found")
            continue
        for shard in sorted(shard_dir.glob("*.tfrec")):
            link_or_copy(shard, export_dir / shard.name)
            n_shards += 1
        manifest = shard_dir / "manifest.json"
        if manifest.exists():
            manifests[split] = json.loads(manifest.read_text())
            shutil.copy2(manifest, export_dir / f"manifest_{split}.json")

    if not n_shards:
        sys.exit(f"no shards found under {processed}. Run src/preprocess.py first.")

    folds_path = processed / "folds.csv"
    if not folds_path.exists():
        sys.exit(f"{folds_path} not found. Run src/make_folds.py first.")
    for csv in ("folds.csv", "metadata_processed.csv"):
        if (processed / csv).exists():
            shutil.copy2(processed / csv, export_dir / csv)

    folds = pd.read_csv(folds_path)
    (export_dir / "README.md").write_text(build_readme(export_dir, folds, manifests))

    total = sum(p.stat().st_size for p in export_dir.iterdir() if p.is_file())
    print(f"staged {n_shards} shards + {len(list(export_dir.iterdir())) - n_shards} "
          f"support files -> {export_dir}")
    print(f"upload size: {total / 1024 ** 3:.2f} GB")
    return export_dir


def write_metadata(export_dir: Path, user: str, slug: str, title: str) -> None:
    meta = {
        "title": title,
        "id": f"{user}/{slug}",
        "licenses": [{"name": "other"}],
    }
    (export_dir / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"dataset id: {meta['id']}")


def get_api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    try:
        api.authenticate()
    except (Exception, SystemExit) as exc:
        sys.exit(f"\nKaggle authentication failed: {exc}\n"
                 "Run: kaggle auth login\n")
    return api


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", action="store_true",
                        help="build the flat upload folder, upload nothing")
    parser.add_argument("--create", action="store_true",
                        help="stage, then create the dataset (private)")
    parser.add_argument("--version", metavar="NOTES",
                        help="stage, then push a new version with these notes")
    parser.add_argument("--user", help="your Kaggle username, required to upload")
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--export-dir", default=str(EXPORT_DIR))
    args = parser.parse_args()

    if not any([args.stage, args.create, args.version]):
        parser.print_help()
        sys.exit(1)

    export_dir = stage(args)

    if not (args.create or args.version):
        print("\nstaged only. Upload with --create (first time) or --version 'notes'.")
        return

    if not args.user:
        sys.exit("--user is required to upload (your Kaggle username)")
    write_metadata(export_dir, args.user, args.slug, args.title)

    api = get_api()
    if args.create:
        # public=False: this is competition-derived data, keep it to the team.
        api.dataset_create_new(str(export_dir), public=False, dir_mode="skip")
        print(f"\ncreated https://www.kaggle.com/datasets/{args.user}/{args.slug}")
        print("Share it: dataset page -> Settings -> Collaborators -> add teammate")
    else:
        api.dataset_create_version(str(export_dir), args.version, dir_mode="skip")
        print(f"\nnew version pushed to {args.user}/{args.slug}")


if __name__ == "__main__":
    main()
