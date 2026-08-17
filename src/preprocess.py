"""Turn the raw JPEGs into fixed-size TFRecord shards for model experiments.

The EDA found resolutions from 640x480 to 6000x4000, so a fixed input size is
mandatory. Images are centre-cropped to a square before resizing rather than
squashed, because distorting a lesion's aspect ratio changes the shape cues a
dermoscopic model relies on.

    python src/preprocess.py --split train --size 512
    python src/preprocess.py --split test  --size 512
    python src/preprocess.py --verify data/processed/train_512

Hair removal (on by default) follows the DullRazor approach the EDA cited: a
morphological blackhat isolates dark thin structures, and those pixels are
inpainted from their surroundings. It runs after the resize so the kernel size
means the same thing regardless of the source resolution.

Output layout, per split:
    data/processed/train_512/train_512-000-of-017.tfrec
    data/processed/train_512/manifest.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

SHARD_SIZE = 2000
JPEG_QUALITY = 92

# Blackhat kernel at the reference size DullRazor was tuned for. Scaled to the
# actual output size so hair strands stay the same relative thickness.
HAIR_KERNEL_AT_512 = 11
HAIR_THRESHOLD = 10
INPAINT_RADIUS = 1

# The teammate's training code imports this to parse the shards. Keeping it here
# means the writer and the reader can never disagree about the schema.
FEATURE_SPEC = {
    "image": "string",
    "image_name": "string",
    "patient_id": "string",
    "sex_enc": "int64",
    "site_enc": "int64",
    "age_norm": "float32",
    "target": "int64",
    "fold": "int64",
    "is_external": "int64",
}


def hair_kernel_size(size: int) -> int:
    k = max(3, round(HAIR_KERNEL_AT_512 * size / 512))
    return k if k % 2 == 1 else k + 1


def remove_hair(bgr, kernel_size: int):
    """DullRazor-style: blackhat to find dark thin structures, then inpaint."""
    import cv2

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, HAIR_THRESHOLD, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(bgr, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)


def center_square(bgr):
    h, w = bgr.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    return bgr[top:top + side, left:left + side]


def load_and_transform(path: Path, size: int, do_hair: bool, kernel: int) -> bytes:
    """Read one JPEG and return the processed image re-encoded as JPEG bytes."""
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"unreadable image: {path}")

    bgr = center_square(bgr)
    # INTER_AREA is the correct filter for downscaling; these images only shrink.
    interp = cv2.INTER_AREA if bgr.shape[0] > size else cv2.INTER_CUBIC
    bgr = cv2.resize(bgr, (size, size), interpolation=interp)

    if do_hair:
        bgr = remove_hair(bgr, kernel)

    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise OSError(f"failed to encode: {path}")
    return buf.tobytes()


def write_shard(task: dict) -> dict:
    """Process one shard end to end. Runs in its own process."""
    import tensorflow as tf

    def _bytes(v):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))

    def _int(v):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v)]))

    def _float(v):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[float(v)]))

    rows = task["rows"]
    size = task["size"]
    kernel = hair_kernel_size(size)
    image_dir = Path(task["image_dir"])
    out_path = Path(task["out_path"])
    tmp_path = out_path.with_suffix(".tfrec.partial")

    written, failures = 0, []
    with tf.io.TFRecordWriter(str(tmp_path)) as writer:
        for row in rows:
            src = image_dir / f"{row['image_name']}.jpg"
            try:
                image_bytes = load_and_transform(src, size, task["hair"], kernel)
            except OSError as exc:
                failures.append(str(exc))
                continue

            example = tf.train.Example(features=tf.train.Features(feature={
                "image": _bytes(image_bytes),
                "image_name": _bytes(row["image_name"].encode()),
                "patient_id": _bytes(str(row["patient_id"]).encode()),
                "sex_enc": _int(row["sex_enc"]),
                "site_enc": _int(row["site_enc"]),
                "age_norm": _float(row["age_norm"]),
                "target": _int(row["target"]),
                "fold": _int(row["fold"]),
                "is_external": _int(row["is_external"]),
            }))
            writer.write(example.SerializeToString())
            written += 1

    # Rename only on success, so an interrupted run never leaves a shard that
    # looks complete but is truncated.
    tmp_path.rename(out_path)
    return {"shard": out_path.name, "written": written, "failures": failures}


def build_tasks(df: pd.DataFrame, args, out_dir: Path) -> list[dict]:
    shards = int(np.ceil(len(df) / args.shard_size))
    # Explicit index slicing rather than np.array_split: under pandas 3 that
    # returns ndarrays and silently loses the column names.
    bounds = np.linspace(0, len(df), shards + 1).astype(int)
    tasks = []
    for i, (start, stop) in enumerate(zip(bounds[:-1], bounds[1:])):
        name = f"{out_dir.name}-{i:03d}-of-{shards:03d}.tfrec"
        tasks.append({
            "rows": df.iloc[start:stop].to_dict("records"),
            "size": args.size,
            "hair": not args.no_hair_removal,
            "image_dir": str(args.image_dir),
            "out_path": str(out_dir / name),
        })
    return tasks


def run(args) -> None:
    meta_path = Path(args.metadata)
    if not meta_path.exists():
        sys.exit(f"{meta_path} not found. Run: python src/make_folds.py")

    df = pd.read_csv(meta_path)

    if args.split == "test":
        # test.csv has no target/fold; synthesise the columns so one writer path
        # serves both splits and the schema stays identical.
        for col, val in (("target", -1), ("fold", -1), ("is_external", 0)):
            if col not in df.columns:
                df[col] = val
    if args.limit:
        df = df.head(args.limit)

    out_dir = Path(args.out_dir) / f"{args.split}_{args.size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(df, args, out_dir)
    workers = args.workers or min(mp.cpu_count(), len(tasks))

    print(f"{len(df)} images -> {len(tasks)} shards of ~{args.shard_size}")
    print(f"size {args.size}px, hair removal "
          f"{'off' if args.no_hair_removal else f'on (kernel {hair_kernel_size(args.size)})'}"
          f", {workers} workers\n")

    started = time.time()
    # spawn, not fork: TensorFlow starts threads on import and forking a process
    # that has already imported it deadlocks intermittently.
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        results = []
        for i, res in enumerate(pool.imap_unordered(write_shard, tasks), 1):
            results.append(res)
            done = sum(r["written"] for r in results)
            rate = done / max(time.time() - started, 1e-6)
            print(f"[{i}/{len(tasks)}] {res['shard']}  {done} images  "
                  f"{rate:.0f} img/s")

    elapsed = time.time() - started
    total = sum(r["written"] for r in results)
    failures = [f for r in results for f in r["failures"]]

    manifest = {
        "split": args.split,
        "size": args.size,
        "images": total,
        "shards": len(tasks),
        "hair_removal": not args.no_hair_removal,
        "jpeg_quality": JPEG_QUALITY,
        "crop": "center square then resize",
        "feature_spec": FEATURE_SPEC,
        "source_metadata": str(meta_path),
        "failures": failures,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n{total} images in {elapsed / 60:.1f} min "
          f"({total / max(elapsed, 1e-6):.0f} img/s) -> {out_dir}")
    if failures:
        print(f"WARNING: {len(failures)} images failed, listed in manifest.json")
        for f in failures[:5]:
            print(f"  {f}")


def verify(shard_dir: Path) -> None:
    """Read the shards back and confirm they decode and match the manifest."""
    import tensorflow as tf

    shards = sorted(str(p) for p in shard_dir.glob("*.tfrec"))
    if not shards:
        sys.exit(f"no .tfrec files in {shard_dir}")

    spec = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "patient_id": tf.io.FixedLenFeature([], tf.string),
        "sex_enc": tf.io.FixedLenFeature([], tf.int64),
        "site_enc": tf.io.FixedLenFeature([], tf.int64),
        "age_norm": tf.io.FixedLenFeature([], tf.float32),
        "target": tf.io.FixedLenFeature([], tf.int64),
        "fold": tf.io.FixedLenFeature([], tf.int64),
        "is_external": tf.io.FixedLenFeature([], tf.int64),
    }

    ds = tf.data.TFRecordDataset(shards, num_parallel_reads=tf.data.AUTOTUNE)
    count, positives, shapes = 0, 0, set()
    for raw in ds:
        ex = tf.io.parse_single_example(raw, spec)
        img = tf.io.decode_jpeg(ex["image"])
        shapes.add(tuple(img.shape.as_list()))
        positives += int(ex["target"].numpy() == 1)
        count += 1

    print(f"{len(shards)} shards, {count} examples decoded")
    print(f"shapes present: {sorted(shapes)}")
    print(f"malignant: {positives} ({positives / max(count, 1) * 100:.2f}%)")

    if len(shapes) != 1:
        sys.exit(f"FAIL: inconsistent image shapes {sorted(shapes)}")

    manifest_path = shard_dir / "manifest.json"
    if manifest_path.exists():
        expected = json.loads(manifest_path.read_text())["images"]
        if expected != count:
            sys.exit(f"FAIL: manifest says {expected} images, read {count}")
        print(f"matches manifest: {expected} images")
    print("OK")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--verify", metavar="SHARD_DIR",
                        help="read shards back and validate, then exit")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--metadata", default=str(PROCESSED_DIR / "metadata_processed.csv"))
    parser.add_argument("--image-dir", default=str(RAW_DIR / "jpeg" / "train"))
    parser.add_argument("--out-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    parser.add_argument("--workers", type=int, default=0,
                        help="0 uses every core")
    parser.add_argument("--no-hair-removal", action="store_true")
    parser.add_argument("--limit", type=int, default=0,
                        help="process only the first N rows, for smoke tests")
    args = parser.parse_args()

    if args.verify:
        verify(Path(args.verify))
        return
    run(args)


if __name__ == "__main__":
    main()
