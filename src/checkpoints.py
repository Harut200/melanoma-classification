"""
Load a saved .pth back into the architecture it belongs to.

Shared by predict.py and src/evaluate_test.py, so there is one place that knows
how to read a checkpoint and one place that knows how to reject a bad one. When
this logic lived in two files they drifted, and a checkpoint that one script
refused the other happily loaded.
"""

import glob
import json
import os
import re

import torch

# The repo layout, a rebuilt package, and Kaggle's flattened upload where every
# file sits side by side.
try:
    from src.models.final_model import SkinMelanomaFinalModel
    from src.models.baseline_models import get_model
except ImportError:
    try:
        from models.final_model import SkinMelanomaFinalModel
        from models.baseline_models import get_model
    except (ImportError, ModuleNotFoundError):
        from final_model import SkinMelanomaFinalModel
        from baseline_models import get_model


# Real activations after a convolution have a variance in the 0.1 to 1000
# range. Anything under this floor is a collapsed layer, not a small one.
DEGENERATE_VARIANCE = 1e-8

# What the final Kaggle run used. Only consulted when the checkpoint carries no
# sidecar json to say otherwise.
DEFAULT_FINAL_IMAGE_SIZE = 300
DEFAULT_BASELINE_IMAGE_SIZE = 224


def describe_checkpoint(weights_path, state_dict):
    """
    Work out which architecture a .pth file belongs to.

    Three sources, in order of trust:
      1. the sidecar json train_final.py writes next to its checkpoints
      2. the filename, which both training scripts generate to a fixed pattern
      3. the shape of the weights themselves

    Guessing wrong loads the file into the wrong class and either throws a wall
    of shape errors or, worse, loads partially and predicts nonsense. So this
    returns an explicit description and refuses rather than guessing blindly.

    The returned dict always carries `val_fold`, which is the fold the
    checkpoint was selected on. Scoring a model on the fold that chose it is
    the mistake this repo has already made once, so both callers check it.
    """
    info_path = weights_path.replace('.pth', '_info.json')
    if os.path.exists(info_path):
        with open(info_path) as handle:
            info = json.load(handle)
        return {
            'kind': 'final',
            'backbone': info['backbone'],
            'image_size': info.get('image_size', DEFAULT_FINAL_IMAGE_SIZE),
            'use_gem': info.get('use_gem', True),
            'proj_dim': info.get('proj_dim', 256),
            'use_metadata': info.get('meta_cols') is not None,
            'threshold': info.get('best_threshold'),
            'val_fold': info.get('val_fold'),
            'source': os.path.basename(info_path),
        }

    name = os.path.basename(weights_path)

    # run_final_kaggle.py writes final_<backbone>_fold<n>.pth, where <n> is the
    # fold it validated on.
    match = re.match(r'^final_(.+)_fold(\d+)\.pth$', name)
    if match:
        return {
            'kind': 'final',
            'backbone': match.group(1),
            'image_size': DEFAULT_FINAL_IMAGE_SIZE,
            'use_gem': True,
            'proj_dim': 256,
            'use_metadata': 'meta_embed.sex_embed.weight' in state_dict,
            'threshold': None,
            'val_fold': int(match.group(2)),
            'source': 'filename',
        }

    # train_baseline.py writes best_<model>.pth, sometimes with _fold<n>.
    match = re.match(r'^best_(.+?)(?:_fold(\d+))?\.pth$', name)
    if match:
        return {
            'kind': 'baseline',
            'backbone': match.group(1),
            'image_size': DEFAULT_BASELINE_IMAGE_SIZE,
            'use_gem': False,
            'proj_dim': None,
            'use_metadata': False,
            'threshold': None,
            'val_fold': int(match.group(2)) if match.group(2) else None,
            'source': 'filename',
        }

    # Last resort. The weights themselves say which family this is, but not
    # which backbone, and timm needs the exact name to rebuild it.
    is_final = any(k.startswith('backbone.') for k in state_dict)
    raise SystemExit(
        f"Cannot tell which model {name} is.\n"
        f"It looks like {'the final multi-modal model' if is_final else 'a baseline model'}, "
        f"but the backbone name is not in the filename and there is no sidecar json.\n"
        f"Pass it explicitly, for example:\n"
        f"  --backbone tf_efficientnet_b4 --image_size 300"
    )


def check_checkpoint_is_healthy(model, name):
    """
    Refuse a checkpoint that was trained on degenerate input.

    BatchNorm stores the running variance of its input. Real photos always give
    a positive variance in every channel, so an entire BatchNorm layer sitting
    at zero means every image that layer ever saw was identical.

    This is not hypothetical. An early version of the dataset class turned a
    missing file into a black square instead of raising, so a wrong image
    directory trained a model on 58,000 identical blank images. It ran to
    completion, saved weights, and reported metrics. The checkpoints from that
    run are still on disk and they load without complaint.

    A zero variance also makes the layer divide by sqrt(0 + eps), which
    multiplies activations by about 316 and pushes the output logit into the
    hundreds. Sigmoid then returns exactly 1.0 for almost any photo, which is a
    very confident answer that means nothing at all.
    """
    broken = []
    for module_name, module in model.named_modules():
        running_var = getattr(module, 'running_var', None)
        if running_var is not None and running_var.numel():
            if float(running_var.max()) < DEGENERATE_VARIANCE:
                broken.append(module_name)

    if broken:
        raise SystemExit(
            f"\n{name} is not a usable checkpoint.\n\n"
            f"  {len(broken)} BatchNorm layer(s) have a collapsed running variance "
            f"(at or below 1e-8), starting at '{broken[0]}'.\n"
            f"  That only happens when every training image was identical, which is what\n"
            f"  the black-square bug in the old dataset class did (see\n"
            f"  reports/Experiment_Report.md). This model learned a constant.\n\n"
            f"  Use the fold checkpoints from the fixed run instead."
        )


def load_one_model(weights_path, device, backbone_override=None,
                   image_size_override=None):
    """Rebuild the architecture the checkpoint belongs to and load it."""
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
    spec = describe_checkpoint(weights_path, state_dict)

    if backbone_override:
        spec['backbone'] = backbone_override
        spec['kind'] = ('final' if any(k.startswith('backbone.') for k in state_dict)
                        else 'baseline')
    if image_size_override:
        spec['image_size'] = image_size_override

    if spec['kind'] == 'final':
        model = SkinMelanomaFinalModel(
            backbone_name=spec['backbone'],
            pretrained=False,          # the checkpoint supplies every weight
            use_gem=spec['use_gem'],
            proj_dim=spec['proj_dim'],
            use_metadata=spec['use_metadata'],
        )
    else:
        model = get_model(spec['backbone'], pretrained=False)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        # Mismatched keys mean the architecture guess was wrong. Loading anyway
        # gives a model with randomly initialised layers that still returns a
        # confident-looking number, which is the worst possible failure here.
        raise SystemExit(
            f"{os.path.basename(weights_path)} does not fit a "
            f"{spec['backbone']} {spec['kind']} model.\n"
            f"  {len(missing)} weights missing, {len(unexpected)} unexpected.\n"
            f"  first missing:    {list(missing)[:3]}\n"
            f"  first unexpected: {list(unexpected)[:3]}\n"
            f"Pass the right --backbone."
        )

    check_checkpoint_is_healthy(model, os.path.basename(weights_path))

    model.to(device).eval()
    return model, spec


def find_weights(weights_arg):
    """Accept a file, a directory, or a glob, and return the .pth files."""
    if os.path.isdir(weights_arg):
        found = sorted(glob.glob(os.path.join(weights_arg, '*.pth')))
    elif any(ch in weights_arg for ch in '*?['):
        found = sorted(glob.glob(weights_arg))
    else:
        found = [weights_arg] if os.path.exists(weights_arg) else []

    if not found:
        raise SystemExit(
            f"No weights found at {weights_arg}\n"
            "Train a model first, or download the fold checkpoints from the "
            "Kaggle run into models/. See the README."
        )
    return found
