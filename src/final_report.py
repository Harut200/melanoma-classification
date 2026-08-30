"""
Summarise the final-model runs and score the fold ensemble.

    python src/final_report.py --in_dir /kaggle/working/reports

Reads final_results.csv and final_oof.csv, prints per-fold and ensemble
numbers, and writes final_model_report.md next to them.

Not final_report.md: reports/Final_Report.md is the written-up version of this
and macOS filesystems are case-insensitive, so that name would overwrite it.
"""

import argparse
import os

import numpy as np
import pandas as pd

try:
    from src.metrics import evaluate_predictions, find_best_threshold, recall_at_specificity
except ImportError:
    from metrics import evaluate_predictions, find_best_threshold, recall_at_specificity

# The resnet34 baseline, for comparison. From reports/results.csv.
BASELINE = {'roc_auc': 0.8873, 'pr_auc': 0.2285, 'sens95': 0.5189}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in_dir', default='/kaggle/working/reports')
    args = p.parse_args()

    res_path = os.path.join(args.in_dir, 'final_results.csv')
    oof_path = os.path.join(args.in_dir, 'final_oof.csv')
    if not os.path.exists(res_path):
        raise SystemExit(f"{res_path} not found. Run src/run_final_kaggle.py first.")

    res = pd.read_csv(res_path).drop_duplicates(['experiment', 'fold'], keep='last')
    lines = ["# Final model results", ""]

    lines.append("## Per fold")
    lines.append("")
    lines.append("| experiment | fold | ROC-AUC | PR-AUC | Sens@95Spec | best epoch | min |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in res.sort_values(['experiment', 'fold']).itertuples():
        lines.append(f"| `{r.experiment}` | {r.fold} | {r.roc_auc:.4f} | {r.pr_auc:.4f} | "
                     f"{r.sens_at_95_spec:.4f} | {r.best_epoch} | {r.minutes:.0f} |")
    lines.append("")

    lines.append("## Averaged over folds")
    lines.append("")
    lines.append("| experiment | folds | ROC-AUC | PR-AUC | Sens@95Spec |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for exp, g in res.groupby('experiment'):
        sd_roc = 0.0 if len(g) < 2 else g.roc_auc.std()
        sd_pr = 0.0 if len(g) < 2 else g.pr_auc.std()
        lines.append(f"| `{exp}` | {len(g)} | {g.roc_auc.mean():.4f} ± {sd_roc:.4f} | "
                     f"{g.pr_auc.mean():.4f} ± {sd_pr:.4f} | {g.sens_at_95_spec.mean():.4f} |")
    lines.append(f"| _resnet34 baseline_ | 5 | {BASELINE['roc_auc']:.4f} | "
                 f"{BASELINE['pr_auc']:.4f} | {BASELINE['sens95']:.4f} |")
    lines.append("")

    if os.path.exists(oof_path):
        oof = pd.read_csv(oof_path).drop_duplicates(['experiment', 'image_name'], keep='last')

        lines.append("## Out of fold")
        lines.append("")
        lines.append("Every competition photo scored once, by the fold model that did not")
        lines.append("train on it. This is the number to quote.")
        lines.append("")
        lines.append("| experiment | photos | ROC-AUC | PR-AUC | Sens@95Spec | threshold |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")

        best_exp, best_roc = None, -1
        for exp, g in oof.groupby('experiment'):
            m = evaluate_predictions(g.target.values, g.prob.values)
            t, _ = find_best_threshold(g.target.values, g.prob.values)
            s95, _ = recall_at_specificity(g.target.values, g.prob.values)
            lines.append(f"| `{exp}` | {len(g)} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | "
                         f"{s95:.4f} | {t:.4f} |")
            if m['roc_auc'] > best_roc:
                best_exp, best_roc = exp, m['roc_auc']
        lines.append("")

        # Ensemble across experiments, if more than one was run. Averaging the
        # rank of each model's probabilities rather than the probabilities
        # themselves, because different models are calibrated differently.
        experiments = oof.experiment.unique()
        if len(experiments) > 1:
            wide = oof.pivot_table(index=['image_name', 'target'],
                                   columns='experiment', values='prob').dropna()
            if not wide.empty:
                ranked = wide.rank(pct=True).mean(axis=1)
                y = wide.index.get_level_values('target').values
                m = evaluate_predictions(y, ranked.values)
                s95, _ = recall_at_specificity(y, ranked.values)
                lines.append("## Ensemble of all experiments")
                lines.append("")
                lines.append(f"Rank averaged over {len(experiments)} models, "
                             f"{len(wide)} photos.")
                lines.append("")
                lines.append(f"- ROC-AUC **{m['roc_auc']:.4f}**  "
                             f"({m['roc_auc'] - best_roc:+.4f} vs the best single model)")
                lines.append(f"- PR-AUC **{m['pr_auc']:.4f}**")
                lines.append(f"- Sens@95Spec **{s95:.4f}**")
                lines.append("")

        lines.append("## Versus the baseline")
        lines.append("")
        g = oof[oof.experiment == best_exp]
        m = evaluate_predictions(g.target.values, g.prob.values)
        lines.append(f"| | ROC-AUC | PR-AUC |")
        lines.append(f"| --- | ---: | ---: |")
        lines.append(f"| resnet34 @224 baseline | {BASELINE['roc_auc']:.4f} | {BASELINE['pr_auc']:.4f} |")
        lines.append(f"| `{best_exp}` out of fold | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} |")
        lines.append(f"| **change** | **{m['roc_auc'] - BASELINE['roc_auc']:+.4f}** | "
                     f"**{m['pr_auc'] - BASELINE['pr_auc']:+.4f}** |")
        lines.append("")

    out = os.path.join(args.in_dir, 'final_model_report.md')
    with open(out, 'w') as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
