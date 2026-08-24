"""
Turn results.csv and oof_predictions.csv into a readable report.

    python src/report_results.py --in_dir reports

Writes reports/experiment_report.md plus figures.
"""

import argparse
import os

import numpy as np
import pandas as pd

from metrics import evaluate_predictions, find_best_threshold, recall_at_specificity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def summarise(results):
    """Mean and spread across folds, per experiment."""
    g = results.groupby('experiment')
    out = g.agg(
        folds=('fold', 'count'),
        pr_auc_mean=('pr_auc', 'mean'),
        pr_auc_std=('pr_auc', 'std'),
        roc_auc_mean=('roc_auc', 'mean'),
        roc_auc_std=('roc_auc', 'std'),
        sens95_mean=('sens_at_95_spec', 'mean'),
        minutes=('minutes', 'sum'),
    ).round(4).sort_values('pr_auc_mean', ascending=False)
    return out


def paired_ablation(results, with_name, without_name):
    """
    Compare two experiments fold by fold.

    Comparing the means alone hides whether the improvement is consistent. If
    the with-external run wins on 5 folds out of 5, that is a real effect. If it
    wins on 3 and loses on 2, the mean difference is probably noise, because
    each fold only has about 117 melanomas to measure on.
    """
    a = results[results['experiment'] == with_name].set_index('fold')
    b = results[results['experiment'] == without_name].set_index('fold')
    shared = sorted(set(a.index) & set(b.index))
    if not shared:
        return None

    rows = []
    for fold in shared:
        rows.append({
            'fold': fold,
            'with_external': a.loc[fold, 'pr_auc'],
            'without_external': b.loc[fold, 'pr_auc'],
            'difference': a.loc[fold, 'pr_auc'] - b.loc[fold, 'pr_auc'],
        })
    table = pd.DataFrame(rows)
    wins = int((table['difference'] > 0).sum())
    return table, wins, len(shared)


def ensemble_oof(oof, experiment_names):
    """Average the probabilities of several models on the same photos."""
    subset = oof[oof['experiment'].isin(experiment_names)]
    if subset.empty:
        return None
    wide = subset.pivot_table(index=['image_name', 'target'],
                              columns='experiment', values='prob')
    wide = wide.dropna()
    if wide.empty or len(wide.columns) < 2:
        return None
    y_true = wide.index.get_level_values('target').values
    y_prob = wide.mean(axis=1).values
    return evaluate_predictions(y_true, y_prob), len(wide)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default=os.path.join(BASE_DIR, 'reports'))
    args = parser.parse_args()

    results_path = os.path.join(args.in_dir, 'results.csv')
    oof_path = os.path.join(args.in_dir, 'oof_predictions.csv')

    if not os.path.exists(results_path):
        print(f"{results_path} not found. Run src/experiment_runner.py first.")
        return

    results = pd.read_csv(results_path)
    summary = summarise(results)

    lines = ["# Experiment results", ""]
    lines.append(f"{len(results)} runs across {results['experiment'].nunique()} "
                 f"experiments, {results['minutes'].sum() / 60:.1f} GPU-hours total.")
    lines.append("")
    lines.append("## Summary, averaged over folds")
    lines.append("")
    lines.append("| experiment | folds | PR-AUC | ROC-AUC | Sens@95Spec | minutes |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name, r in summary.iterrows():
        std = 0 if pd.isna(r['pr_auc_std']) else r['pr_auc_std']
        lines.append(f"| `{name}` | {int(r['folds'])} | "
                     f"{r['pr_auc_mean']:.4f} ± {std:.4f} | "
                     f"{r['roc_auc_mean']:.4f} | {r['sens95_mean']:.4f} | "
                     f"{r['minutes']:.0f} |")
    lines.append("")
    baseline = results['val_positives'].sum() / results['n_val'].sum()
    lines.append(f"A random model scores about **{baseline:.4f}** PR-AUC here, "
                 "so that is the number to beat.")
    lines.append("")

    # --- the ablation ------------------------------------------------------
    ab = paired_ablation(results, 'resnet34_224_ext', 'resnet34_224_noext')
    if ab:
        table, wins, n = ab
        lines.append("## Did ISIC 2019 actually help?")
        lines.append("")
        lines.append("Same folds, same seed, one variable changed.")
        lines.append("")
        lines.append("| fold | with external | without | difference |")
        lines.append("| ---: | ---: | ---: | ---: |")
        for r in table.itertuples():
            lines.append(f"| {r.fold} | {r.with_external:.4f} | "
                         f"{r.without_external:.4f} | {r.difference:+.4f} |")
        mean_diff = table['difference'].mean()
        lines.append(f"| **mean** | | | **{mean_diff:+.4f}** |")
        lines.append("")
        lines.append(f"External data wins on **{wins} of {n} folds**.")
        lines.append("")
        if wins == n and mean_diff > 0:
            lines.append("Consistent across every fold, so this is a real effect. "
                         "Keep the external data.")
        elif wins >= n - 1 and mean_diff > 0:
            lines.append("Wins nearly everywhere. Likely real, worth keeping.")
        elif mean_diff > 0:
            lines.append("The mean improves but it is not consistent across folds. "
                         "With only ~117 melanomas per fold this may be noise. "
                         "Worth another seed before trusting it.")
        else:
            lines.append("External data does **not** help here. Either the domain "
                         "shift is hurting, or the model is using source style as a "
                         "shortcut. Do not ship it without investigating.")
        lines.append("")

    # --- ensemble ----------------------------------------------------------
    if os.path.exists(oof_path):
        oof = pd.read_csv(oof_path)
        names = [n for n in results['experiment'].unique() if n.endswith('_ext')]
        ens = ensemble_oof(oof, names)
        if ens:
            metrics, n_photos = ens
            lines.append("## Ensemble of the with-external models")
            lines.append("")
            lines.append(f"Averaging predictions over {len(names)} models "
                         f"on {n_photos} photos:")
            lines.append("")
            lines.append(f"- PR-AUC  **{metrics['pr_auc']:.4f}**")
            lines.append(f"- ROC-AUC **{metrics['roc_auc']:.4f}**")
            best_single = summary['pr_auc_mean'].max()
            delta = metrics['pr_auc'] - best_single
            lines.append(f"- versus the best single model, {delta:+.4f} PR-AUC")
            lines.append("")

        # out-of-fold score per experiment, which uses every photo once
        lines.append("## Out-of-fold scores")
        lines.append("")
        lines.append("Every competition photo scored exactly once, by the model "
                     "that did not train on it. More reliable than a per-fold mean.")
        lines.append("")
        lines.append("| experiment | photos | PR-AUC | ROC-AUC | Sens@95Spec | best threshold |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for name in results['experiment'].unique():
            sub = oof[oof['experiment'] == name]
            if sub.empty:
                continue
            m = evaluate_predictions(sub['target'].values, sub['prob'].values)
            t, _ = find_best_threshold(sub['target'].values, sub['prob'].values)
            s95, _ = recall_at_specificity(sub['target'].values, sub['prob'].values)
            lines.append(f"| `{name}` | {len(sub)} | {m['pr_auc']:.4f} | "
                         f"{m['roc_auc']:.4f} | {s95:.4f} | {t:.4f} |")
        lines.append("")

    lines.append("## How to read these numbers")
    lines.append("")
    lines.append("- **PR-AUC** is the honest metric at a 1.8% positive rate. Compare "
                 "it against the random baseline above, not against 1.0.")
    lines.append("- **ROC-AUC** is the competition metric, but it flatters every "
                 "model on data this imbalanced.")
    lines.append("- **Sens@95Spec** is how many melanomas are caught while keeping "
                 "false alarms at 5%. This is the number a clinician would care about.")
    lines.append("- **Accuracy is not reported on purpose.** A model that always says "
                 "\"not cancer\" scores 98.2%.")
    lines.append("- Each fold has only ~117 melanomas, so differences smaller than "
                 "the fold-to-fold spread are not real.")

    out_path = os.path.join(args.in_dir, 'experiment_report.md')
    with open(out_path, 'w') as handle:
        handle.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nwrote {out_path}")


if __name__ == '__main__':
    main()
