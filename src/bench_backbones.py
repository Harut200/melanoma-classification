"""
Measure real training throughput on this GPU, then work out what fits.

    python src/bench_backbones.py --n_train 51228 --folds 5 --epochs 12

Prints images/sec and the projected hours for a full run, so the backbone and
resolution get chosen from a measurement instead of a guess. Takes ~2 minutes.
"""

import argparse
import time

import torch

try:
    from src.models.final_model import SkinMelanomaFinalModel
except ImportError:
    from models.final_model import SkinMelanomaFinalModel

# Candidates worth considering, cheapest first. Each is (backbone, image_size).
CANDIDATES = [
    ('tf_efficientnet_b0', 224),
    ('tf_efficientnet_b0', 384),
    ('tf_efficientnet_b3', 300),
    ('tf_efficientnet_b4', 320),
    ('tf_efficientnet_b4', 380),
]


def bench_one(backbone, size, batch_size, device, iters=8):
    """One training step, timed. Includes the backward pass and the optimizer."""
    model = SkinMelanomaFinalModel(backbone_name=backbone, pretrained=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    crit = torch.nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    images = torch.randn(batch_size, 3, size, size, device=device)
    meta = torch.stack([
        torch.randint(0, 3, (batch_size,)).float(),
        torch.randint(0, 7, (batch_size,)).float(),
        torch.rand(batch_size),
    ], dim=1).to(device)
    targets = torch.rand(batch_size, 1, device=device).round()

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=True):
            loss = crit(model(images, meta), targets)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(3):
        step()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters

    peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
    torch.cuda.reset_peak_memory_stats()
    del model, opt
    torch.cuda.empty_cache()
    return batch_size / dt, peak_gb


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_train', type=int, default=51228, help="training photos per fold")
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--budget_h', type=float, default=10.0)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No GPU.")
    print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"projecting for {args.n_train} photos x {args.epochs} epochs x "
          f"{args.folds} folds, batch {args.batch_size}\n")

    print(f"{'backbone':26} {'px':>4} {'img/s':>7} {'min/epoch':>10} "
          f"{'h/fold':>7} {'h total':>8} {'GB':>5}  fits?")
    print("-" * 82)

    for backbone, size in CANDIDATES:
        try:
            ips, gb = bench_one(backbone, size, args.batch_size, torch.device('cuda'))
        except RuntimeError as e:
            print(f"{backbone:26} {size:4}  failed: {str(e)[:38]}")
            continue
        min_ep = args.n_train / ips / 60
        h_fold = min_ep * args.epochs / 60
        h_all = h_fold * args.folds
        fits = "yes" if h_all <= args.budget_h else ("2 sessions" if h_all <= args.budget_h * 2 else "NO")
        print(f"{backbone:26} {size:4} {ips:7.1f} {min_ep:10.1f} "
              f"{h_fold:7.2f} {h_all:8.2f} {gb:5.1f}  {fits}")

    print()
    print("h total is one session. Kaggle allows two concurrent GPU sessions,")
    print("so splitting the folds across two notebooks roughly halves wall clock.")


if __name__ == '__main__':
    main()
