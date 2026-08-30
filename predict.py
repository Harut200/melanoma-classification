#!/usr/bin/env python3
"""
Score a single skin photo with a trained melanoma model.

    python predict.py --image path/to/lesion.jpg

    # the model also takes the patient's details, if you have them
    python predict.py --image lesion.jpg --sex male --age 55 --site torso

    # a folder of weights averages the fold models together, which is
    # what the reported numbers are based on
    python predict.py --image lesion.jpg --weights models/

    # a folder of images writes a csv instead of printing
    python predict.py --image photos/ --csv predictions.csv

This loads saved weights and runs inference. It never trains anything.

The preprocessing here is the same as training, step for step: centre square
crop, resize to 512, DullRazor hair removal, resize to the model's input,
divide by 255, ImageNet normalisation. If those drift apart the model is being
shown a distribution it never saw and the output is worthless, so the two
shared steps are imported from the modules that training uses rather than
copied.
"""

import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
import torch

from src.checkpoints import find_weights, load_one_model
from src.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.step2_make_folds import SEX_CATEGORIES, SITE_CATEGORIES
from src.step3_resize_images import (cut_square_from_middle, remove_hair,
                                     work_out_kernel_size)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# What step 3 wrote to disk. Training never saw an image that had not been
# through this size first, so inference has to go through it too.
PREPROCESS_SIZE = 512

# Missing age is filled with the median age of the training set, which is what
# step 2 does. Hardcoded because predict.py must run without the dataset
# present; --age overrides it whenever the real age is known.
TRAIN_MEDIAN_AGE = 50.0

# Cutoff for the printed label. Tuned for best F1 on the pooled out-of-fold
# predictions, which is every one of the 33,126 competition photos scored once
# by a model that did not train on it. The per-fold thresholds ran from 0.447
# to 0.679, so no single fold's value would have been right; this one is fitted
# to all of them at once. The probability, not the label, is still the output
# to trust.
DEFAULT_THRESHOLD = 0.6245

IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')


# ---------------------------------------------------------------------------
# Preprocessing, identical to training
# ---------------------------------------------------------------------------

def preprocess_image(image_path, model_input_size):
    """
    Take a raw photo off disk and return the exact tensor training would have
    produced for it.

    Steps 1-3 are step3_resize_images.py, which is what built the training set
    on disk. Steps 4-6 are MelanomaDataset, which is what training did at load
    time. Both are imported rather than reimplemented.
    """
    photo = cv2.imread(image_path)
    if photo is None:
        raise SystemExit(
            f"Could not read {image_path}\n"
            "Not an image, or the file is corrupt. Supported: "
            + ', '.join(IMAGE_SUFFIXES)
        )

    original_shape = photo.shape[:2]

    # 1. biggest centred square, so the mole is not stretched out of shape
    photo = cut_square_from_middle(photo)

    # 2. down to 512, the size the hair remover is tuned for
    interpolation = cv2.INTER_AREA if photo.shape[0] > PREPROCESS_SIZE else cv2.INTER_LINEAR
    photo = cv2.resize(photo, (PREPROCESS_SIZE, PREPROCESS_SIZE), interpolation=interpolation)

    # 3. DullRazor: paint over the body hair lying across the lesion
    photo = remove_hair(photo, work_out_kernel_size(PREPROCESS_SIZE))

    # 4. BGR (OpenCV's order) to RGB (what the pretrained weights expect)
    photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)

    # 5. down to the model's input size
    if photo.shape[0] != model_input_size:
        interpolation = (cv2.INTER_AREA if photo.shape[0] > model_input_size
                         else cv2.INTER_LINEAR)
        photo = cv2.resize(photo, (model_input_size, model_input_size),
                           interpolation=interpolation)

    # 6. scale and normalise exactly as MelanomaDataset does
    photo = photo.astype(np.float32) / 255.0
    photo = (photo - IMAGENET_MEAN) / IMAGENET_STD

    tensor = torch.from_numpy(photo).permute(2, 0, 1).unsqueeze(0)
    return tensor, original_shape


def encode_metadata(sex, age, site):
    """
    Turn the patient's details into the three numbers the model's embedding
    layer expects. Same encoding as step2_make_folds.py, and anything missing
    becomes the 'unknown' category, which the model was trained to handle
    because a tenth of the real dataset is missing it too.
    """
    sex = (sex or 'unknown').lower().strip()
    if sex not in SEX_CATEGORIES:
        raise SystemExit(f"--sex must be one of {SEX_CATEGORIES}, got '{sex}'")

    site = (site or 'unknown').lower().strip()
    if site not in SITE_CATEGORIES:
        raise SystemExit(
            f"--site must be one of {SITE_CATEGORIES}, got '{site}'"
        )

    age_value = TRAIN_MEDIAN_AGE if age is None else float(age)

    return torch.tensor([[
        float(SEX_CATEGORIES.index(sex)),
        float(SITE_CATEGORIES.index(site)),
        age_value / 90.0,          # step 2 normalises age the same way
    ]], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_one(models, image_tensor, metadata, device, tta=4):
    """
    Melanoma probability for one image.

    Averaged over the fold models and over flipped copies of the image. A
    lesion has no natural up or down, so a flip is label-preserving here and
    the average over four views is steadier than any single one. Same TTA the
    validation numbers were measured with.
    """
    image_tensor = image_tensor.to(device)
    metadata = metadata.to(device)

    views = [image_tensor]
    if tta >= 2:
        views.append(torch.flip(image_tensor, dims=[3]))
    if tta >= 4:
        views.append(torch.flip(image_tensor, dims=[2]))
        views.append(torch.flip(image_tensor, dims=[2, 3]))

    per_model = []
    for model, spec in models:
        probs = []
        for view in views:
            logits = model(view, metadata) if spec['kind'] == 'final' else model(view)
            probs.append(torch.sigmoid(logits.float()).item())
        per_model.append(float(np.mean(probs)))

    return float(np.mean(per_model)), per_model


def risk_band(probability, threshold):
    """
    Plain words for the number, so nobody has to interpret a decimal.

    The bands are deliberately coarse. The model is a triage aid that sorts a
    queue; pretending to more precision than that would be misleading.
    """
    if probability >= threshold * 1.5:
        return "HIGH", "refer urgently"
    if probability >= threshold:
        return "ELEVATED", "worth a specialist look"
    if probability >= threshold * 0.5:
        return "LOW-MODERATE", "routine follow-up"
    return "LOW", "no action indicated by the model"


def format_result(image_path, probability, threshold, spec, n_models,
                  elapsed, per_model=None):
    label = "MALIGNANT (melanoma)" if probability >= threshold else "BENIGN"
    confidence = probability if probability >= threshold else 1.0 - probability
    band, advice = risk_band(probability, threshold)

    model_line = f"{spec['backbone']} @{spec['image_size']}px"
    if n_models > 1:
        model_line += f", {n_models}-fold ensemble"
    model_line += ", 4x flip TTA"

    lines = [
        "",
        f"  Image        {os.path.basename(image_path)}",
        f"  Model        {model_line}",
        "",
        f"  Prediction:  {label} | Confidence: {confidence * 100:.1f}%",
        "",
        f"  Melanoma probability   {probability:.4f}",
        f"  Decision threshold     {threshold:.4f}",
        f"  Risk band              {band} - {advice}",
    ]

    if per_model and len(per_model) > 1:
        spread = max(per_model) - min(per_model)
        lines.append(f"  Fold agreement         {spread:.4f} spread across "
                     f"{len(per_model)} models"
                     + ("  (they disagree; treat with caution)" if spread > 0.25 else ""))

    lines += [
        f"  Took                   {elapsed:.2f}s",
        "",
        "  Triage aid on dermoscopic images only. Not a diagnosis, not a",
        "  medical device. If a mole is changing, see a dermatologist.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------

def collect_images(image_arg):
    if os.path.isdir(image_arg):
        found = sorted(
            p for p in glob.glob(os.path.join(image_arg, '*'))
            if p.lower().endswith(IMAGE_SUFFIXES)
        )
        if not found:
            raise SystemExit(f"No images in {image_arg}")
        return found
    if not os.path.exists(image_arg):
        raise SystemExit(f"No such file: {image_arg}")
    if not image_arg.lower().endswith(IMAGE_SUFFIXES):
        # Fail here rather than after spending 20 seconds loading five models.
        raise SystemExit(
            f"{image_arg} is not an image file.\n"
            f"Supported: {', '.join(IMAGE_SUFFIXES)}"
        )
    return [image_arg]


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Predict melanoma probability for a skin photo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('This loads saved weights')[0].strip(),
    )
    parser.add_argument('--image', required=True,
                        help="one photo, or a folder of photos")
    parser.add_argument('--weights', default=os.path.join(BASE_DIR, 'models'),
                        help="a .pth file, a folder of them (averaged), or a glob "
                             "(default: models/)")
    parser.add_argument('--backbone', default=None,
                        help="override the architecture, if it cannot be read "
                             "from the filename or the sidecar json")
    parser.add_argument('--image_size', type=int, default=None,
                        help="override the model's input size")

    parser.add_argument('--sex', default=None, choices=SEX_CATEGORIES,
                        help="patient sex; omitted means 'unknown', which the "
                             "model was trained to handle")
    parser.add_argument('--age', type=float, default=None,
                        help=f"patient age; omitted uses the training median "
                             f"({TRAIN_MEDIAN_AGE:.0f})")
    parser.add_argument('--site', default=None, choices=SITE_CATEGORIES,
                        help="where the lesion is on the body")

    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help=f"cutoff for the malignant label (default "
                             f"{DEFAULT_THRESHOLD})")
    parser.add_argument('--tta', type=int, default=4, choices=[1, 2, 4],
                        help="flip averaging: 1 none, 2 horizontal, 4 all")
    parser.add_argument('--csv', default=None,
                        help="write results here instead of printing them")
    parser.add_argument('--device', default=None, choices=['cuda', 'mps', 'cpu'])
    return parser


def main():
    args = build_arg_parser().parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available()
                              else 'mps' if torch.backends.mps.is_available()
                              else 'cpu')

    weight_files = find_weights(args.weights)
    images = collect_images(args.image)

    print(f"loading {len(weight_files)} model(s) on {device.type} ...", flush=True)
    models = []
    for path in weight_files:
        model, spec = load_one_model(path, device, args.backbone, args.image_size)
        models.append((model, spec))
        print(f"  {os.path.basename(path)}  ->  {spec['backbone']} "
              f"@{spec['image_size']}px  ({spec['kind']}, from {spec['source']})")

    input_sizes = {spec['image_size'] for _, spec in models}
    if len(input_sizes) > 1:
        raise SystemExit(
            f"The checkpoints want different input sizes ({sorted(input_sizes)}), "
            "so they cannot be averaged. Point --weights at one fold set."
        )
    model_input_size = input_sizes.pop()

    # If the checkpoints carry their own tuned threshold, prefer it over the
    # placeholder default, unless the user asked for a specific one.
    thresholds = [spec['threshold'] for _, spec in models if spec['threshold']]
    threshold = args.threshold
    if thresholds and args.threshold == DEFAULT_THRESHOLD:
        threshold = float(np.mean(thresholds))

    metadata = encode_metadata(args.sex, args.age, args.site)
    rows = []

    for image_path in images:
        started = time.time()
        image_tensor, original_shape = preprocess_image(image_path, model_input_size)
        probability, per_model = predict_one(models, image_tensor, metadata,
                                             device, tta=args.tta)
        elapsed = time.time() - started

        if args.csv:
            rows.append({
                'image': os.path.basename(image_path),
                'melanoma_probability': round(probability, 6),
                'prediction': 'malignant' if probability >= threshold else 'benign',
                'threshold': threshold,
                'original_height': original_shape[0],
                'original_width': original_shape[1],
                'seconds': round(elapsed, 3),
            })
            print(f"  {os.path.basename(image_path):40s} {probability:.4f}", flush=True)
        else:
            print(format_result(image_path, probability, threshold,
                                models[0][1], len(models), elapsed, per_model))

    if args.csv:
        import pandas as pd
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"\nwrote {len(rows)} predictions to {args.csv}")


if __name__ == '__main__':
    main()
