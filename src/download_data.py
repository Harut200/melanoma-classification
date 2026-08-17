"""Download the SIIM-ISIC Melanoma Classification data into data/raw/.

The competition archive is far larger than the metadata most preprocessing work
needs, so downloads are opt-in per group rather than all-or-nothing:

    python src/download_data.py --list             # file names and sizes, no download
    python src/download_data.py --csv              # train/test/sample_submission, a few MB
    python src/download_data.py --images           # jpeg/, tens of GB
    python src/download_data.py --dataset SLUG     # a Kaggle dataset (resized mirrors,
                                                   # external ISIC data for class balance)

Authentication (kaggle >= 2.2 dropped the old ~/.kaggle/kaggle.json flow):

    kaggle auth login                          # OAuth, credentials cached locally
    export KAGGLE_API_TOKEN=...                # or a token from kaggle.com/settings/api
    ~/.kaggle/access_token                     # or the same token in a file

Competition rules must be accepted on the website first, otherwise the API
returns 403 for every file.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

COMPETITION = "siim-isic-melanoma-classification"

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"

CSV_FILES = ["train.csv", "test.csv", "sample_submission.csv"]

# Safety margin so a download cannot fill the disk completely.
DISK_HEADROOM_GB = 5.0


def human_gb(num_bytes: float) -> str:
    return f"{num_bytes / 1024 ** 3:.1f} GB"


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def get_api():
    """Return an authenticated KaggleApi, or exit with the setup instructions."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    try:
        api.authenticate()
    except (Exception, SystemExit) as exc:  # kaggle calls sys.exit() on missing creds
        sys.exit(
            f"\nKaggle authentication failed: {exc}\n\n"
            "Fix it with one of:\n"
            "  kaggle auth login                 (OAuth, recommended)\n"
            "  export KAGGLE_API_TOKEN=<token>   (from kaggle.com/settings/api)\n"
            "  echo <token> > ~/.kaggle/access_token\n"
        )
    return api


def list_files(api) -> list:
    """Print every competition file with its size, and return the raw records."""
    files = list(api.competition_list_files(COMPETITION))
    if not files:
        sys.exit(
            f"No files listed for {COMPETITION}. The usual cause is not having "
            "accepted the competition rules on the website."
        )

    total = 0
    for f in sorted(files, key=lambda f: getattr(f, "total_bytes", 0), reverse=True):
        size = getattr(f, "total_bytes", 0)
        total += size
        print(f"{human_gb(size):>10}  {f.name}")
    print(f"{'-' * 40}\n{human_gb(total):>10}  TOTAL across {len(files)} files")
    return files


def check_disk(needed_bytes: int, label: str) -> None:
    free = free_bytes(RAW_DIR)
    headroom = int(DISK_HEADROOM_GB * 1024 ** 3)
    if needed_bytes + headroom > free:
        sys.exit(
            f"Not enough disk space for {label}.\n"
            f"  needed:    {human_gb(needed_bytes)} (+{DISK_HEADROOM_GB:.0f} GB headroom)\n"
            f"  available: {human_gb(free)}\n\n"
            "Options: free up space, download to an external volume, or use a\n"
            "resized mirror dataset with --dataset instead of the full images."
        )


def download_files(api, names: list[str]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        print(f"downloading {name} ...")
        api.competition_download_file(COMPETITION, name, path=str(RAW_DIR), force=False)
    print(f"done -> {RAW_DIR}")


def download_csv(api) -> None:
    download_files(api, CSV_FILES)


def download_images(api) -> None:
    files = [f for f in api.competition_list_files(COMPETITION)
             if f.name.startswith("jpeg/")]
    if not files:
        sys.exit("No jpeg/ entries found in the competition file list.")

    needed = sum(getattr(f, "total_bytes", 0) for f in files)
    print(f"jpeg/ is {len(files)} files, {human_gb(needed)}")
    check_disk(needed, "the jpeg image set")
    download_files(api, [f.name for f in files])


def download_dataset(api, slug: str) -> None:
    """Download a Kaggle dataset — resized image mirrors, external ISIC data, etc."""
    dest = RAW_DIR / slug.split("/")[-1]
    dest.mkdir(parents=True, exist_ok=True)

    files = list(api.dataset_list_files(slug).files)
    needed = sum(getattr(f, "total_bytes", 0) for f in files)
    print(f"{slug}: {len(files)} files, {human_gb(needed)}")
    check_disk(needed, slug)

    api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=False)
    print(f"done -> {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true",
                        help="print competition files and sizes, download nothing")
    parser.add_argument("--csv", action="store_true",
                        help="download train.csv, test.csv, sample_submission.csv")
    parser.add_argument("--images", action="store_true",
                        help="download the full jpeg/ image set (tens of GB)")
    parser.add_argument("--dataset", metavar="SLUG",
                        help="download a Kaggle dataset, e.g. owner/dataset-name")
    args = parser.parse_args()

    if not any([args.list, args.csv, args.images, args.dataset]):
        parser.print_help()
        sys.exit(1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"target: {RAW_DIR}  (free: {human_gb(free_bytes(RAW_DIR))})\n")

    api = get_api()

    if args.list:
        list_files(api)
    if args.csv:
        download_csv(api)
    if args.images:
        download_images(api)
    if args.dataset:
        download_dataset(api, args.dataset)


if __name__ == "__main__":
    main()
