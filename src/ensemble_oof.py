"""
Ensemble out-of-fold predictions from several runs.

    python src/ensemble_oof.py final_oof.csv oof_predictions.csv

Each file needs image_name, target, prob and experiment columns. Rows whose
experiment name contains "noext" are dropped by default: those are the ablation
control, trained without the external data, and they are not meant to be part
of the final model.

Predictions are combined by averaging percentile ranks, not raw probabilities.
Models trained with different losses and thresholds are calibrated differently,
so averaging probabilities lets whichever model is more confident dominate.
ROC-AUC and PR-AUC only care about ordering, so averaging ranks is both fairer
and usually better.
"""

import argparse
import itertools
import os

import pandas as pd

try:
    from src.metrics import evaluate_predictions, find_best_threshold, recall_at_specificity
except ImportError:
    from metrics import evaluate_predictions, find_best_threshold, recall_at_specificity


def load(paths, keep_noext):
    frames = []
    for path in paths:
        if not os.path.exists(path):
            print(f"  skipping {path}, not found")
            continue
        df = pd.read_csv(path)
        missing = {'image_name', 'target', 'prob', 'experiment'} - set(df.columns)
        if missing:
            print(f"  skipping {path}, missing columns {sorted(missing)}")
            continue
        if not keep_noext:
            df = df[~df['experiment'].str.contains('noext', case=False, na=False)]
        # A resumed run can write the same fold twice.
        df = df.drop_duplicates(['experiment', 'image_name'], keep='last')
        for exp, g in df.groupby('experiment'):
            print(f"  {path}: {exp}, {len(g)} photos")
        frames.append(df)
    if not frames:
        raise SystemExit("nothing to ensemble")
    return pd.concat(frames, ignore_index=True)


def score(y_true, y_prob, label):
    m = evaluate_predictions(y_true, y_prob)
    s95, _ = recall_at_specificity(y_true, y_prob)
    t, _ = find_best_threshold(y_true, y_prob)
    return {'model': label, 'photos': len(y_true), 'roc_auc': m['roc_auc'],
            'pr_auc': m['pr_auc'], 'sens_at_95_spec': s95, 'threshold': t}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('paths', nargs='+', help="one or more out-of-fold csv files")
    p.add_argument('--out', default='ensemble_report.md')
    p.add_argument('--keep_noext', action='store_true')
    args = p.parse_args()

    print("loading:")
    oof = load(args.paths, args.keep_noext)
    experiments = sorted(oof['experiment'].unique())

    rows = []
    for exp in experiments:
        g = oof[oof['experiment'] == exp]
        rows.append(score(g['target'].values, g['prob'].values, exp))

    # Only photos every model scored can be ensembled, otherwise the comparison
    # is against a different set of photos and the numbers are not comparable.
    wide = oof.pivot_table(index=['image_name', 'target'],
                           columns='experiment', values='prob').dropna()
    print(f"\n{len(wide)} photos scored by all {len(experiments)} models")

    if len(experiments) > 1 and not wide.empty:
        y = wide.index.get_level_values('target').values
        # every combination, so we can see whether adding a model actually helps
        for size in range(2, len(experiments) + 1):
            for combo in itertools.combinations(wide.columns, size):
                ranked = wide[list(combo)].rank(pct=True).mean(axis=1)
                label = "ens(" + " + ".join(c.replace('final_', '') for c in combo) + ")"
                rows.append(score(y, ranked.values, label))

    table = pd.DataFrame(rows).sort_values('roc_auc', ascending=False)

    lines = ["# Ensemble report", ""]
    lines.append("Rank averaged. Single models scored on their own out-of-fold")
    lines.append(f"photos; ensembles on the {len(wide)} photos all models cover.")
    lines.append("")
    lines.append("| model | photos | ROC-AUC | PR-AUC | Sens@95Spec | threshold |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for r in table.itertuples():
        lines.append(f"| `{r.model}` | {r.photos} | {r.roc_auc:.4f} | {r.pr_auc:.4f} | "
                     f"{r.sens_at_95_spec:.4f} | {r.threshold:.4f} |")
    lines.append("")

    best = table.iloc[0]
    singles = table[~table['model'].str.startswith('ens(')]
    if not singles.empty:
        best_single = singles.iloc[0]
        lines.append(f"Best single model: `{best_single['model']}` at "
                     f"{best_single['roc_auc']:.4f} ROC-AUC.")
        lines.append(f"Best overall: `{best['model']}` at {best['roc_auc']:.4f}, "
                     f"{best['roc_auc'] - best_single['roc_auc']:+.4f}.")
        lines.append("")
        if best['roc_auc'] - best_single['roc_auc'] < 0.002:
            lines.append("The ensemble barely beats the best single model, so report the")
            lines.append("single one. A gain this small is inside the fold-to-fold noise and")
            lines.append("the extra complexity is not worth defending in a presentation.")
        lines.append("")

    with open(args.out, 'w') as fh:
        fh.write("\n".join(lines) + "\n")
    print()
    print("\n".join(lines))
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
