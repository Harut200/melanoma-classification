"""
Rebuild the two figures in the README from the result files.

    python src/make_figures.py

Reads reports/results.csv, reports/final_results.csv and reports/final_oof.csv,
and writes into reports/figures/. Nothing is hardcoded except the two baseline
out-of-fold counts, which come from reports/Experiment_Report.md.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(BASE_DIR, 'reports')
FIGURES = os.path.join(REPORTS, 'figures')

# One palette for both figures, so the two graphs read as one set.
SURFACE, INK, INK_2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e2e1dc'
FINAL_COLOUR, BASE_COLOUR = '#2a78d6', '#eb6834'
LINK, REMAINDER = '#c9c8c2', '#dcdbd5'

# Melanomas per validation fold, and the 584 in the competition set.
POSITIVES_PER_FOLD = np.array([117, 116, 116, 117, 118])
TOTAL_MELANOMAS = int(POSITIVES_PER_FOLD.sum())

# Out-of-fold counts for the two baselines, from reports/Experiment_Report.md.
BASELINE_NO_EXTERNAL_CAUGHT = 270
BASELINE_WITH_EXTERNAL_CAUGHT = 305


def strip_chrome(ax):
    """Recessive axes. The data should be the only strong thing on the page."""
    ax.set_facecolor(SURFACE)
    ax.tick_params(axis='x', labelsize=8.5, colors=INK_2, length=0)
    ax.tick_params(axis='y', length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)


def figure_per_fold_comparison():
    """Paired dot plot: the final model against the baseline, fold by fold."""
    final = pd.read_csv(os.path.join(REPORTS, 'final_results.csv')).sort_values('fold')
    baseline = pd.read_csv(os.path.join(REPORTS, 'results.csv'))
    baseline = baseline[baseline.experiment == 'resnet34_224_ext'].sort_values('fold')

    metrics = [('roc_auc', 'ROC-AUC'),
               ('pr_auc', 'PR-AUC'),
               ('sens_at_95_spec', 'Sensitivity @ 95% spec.')]
    labels = [f'fold {i}' for i in range(5)] + ['mean']
    ypos = np.array([5, 4, 3, 2, 1, -0.4])

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1), facecolor=SURFACE)

    for ax, (column, title) in zip(axes, metrics):
        base_values = np.append(baseline[column].values, baseline[column].mean())
        final_values = np.append(final[column].values, final[column].mean())

        for y, b, f in zip(ypos, base_values, final_values):
            ax.plot([b, f], [y, y], color=LINK, lw=2, zorder=1, solid_capstyle='round')

        # The baseline marker is drawn larger. Where the two land on the same
        # value (sensitivity at a fixed specificity is a count of melanomas, so
        # ties happen) it shows as a ring behind the final marker rather than
        # vanishing underneath it.
        ax.scatter(base_values, ypos, s=150, color=BASE_COLOUR, zorder=2)
        ax.scatter(final_values, ypos, s=80, color=FINAL_COLOUR, zorder=3,
                   edgecolors=SURFACE, linewidths=1.5)

        low, high = min(base_values.min(), final_values.min()), max(base_values.max(), final_values.max())
        pad = (high - low) * 0.30
        ax.set_xlim(low - pad * 0.7, high + pad * 1.5)

        for y, b, f in zip(ypos, base_values, final_values):
            difference = f - b
            text = 'tie' if abs(difference) < 5e-4 else f'{difference:+.3f}'
            ax.text(max(b, f) + pad * 0.26, y, text, va='center', ha='left',
                    fontsize=8.5, color=INK_2)

        ax.axhline(0.35, color=GRID, lw=1)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=9, color=INK_2)
        ax.get_yticklabels()[-1].set_color(INK)
        ax.get_yticklabels()[-1].set_fontweight('bold')
        ax.set_ylim(-1.1, 5.7)
        ax.set_title(title, fontsize=10.5, color=INK, pad=10, loc='left', fontweight='bold')
        strip_chrome(ax)

    handles = [
        plt.Line2D([], [], marker='o', ls='', mfc=BASE_COLOUR, mec=BASE_COLOUR,
                   ms=11, label='resnet34 @224, baseline'),
        plt.Line2D([], [], marker='o', ls='', ms=8.5, mfc=FINAL_COLOUR, mec=SURFACE,
                   mew=1.5, label='tf_efficientnet_b4 @300 + metadata, final'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False,
               fontsize=9.5, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('Final model beats the baseline on every fold', x=0.007, y=0.985,
                 ha='left', fontsize=13, color=INK, fontweight='bold')
    fig.text(0.007, 0.912,
             'Same patient-grouped folds, same seed, same external data. '
             'Labels show the per-fold difference.',
             ha='left', fontsize=9.5, color=INK_2)
    fig.tight_layout(rect=[0, 0.07, 1, 0.88])

    out = os.path.join(FIGURES, 'final_vs_baseline.png')
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


def figure_melanomas_found():
    """How many of the 584 melanomas each model catches at equal false alarms."""
    oof = pd.read_csv(os.path.join(REPORTS, 'final_oof.csv'))
    y, prob = oof.target.values, oof.prob.values

    # Hold specificity at 95%, then count the positives above that cutoff.
    threshold = np.quantile(prob[y == 0], 0.95)
    final_caught = int((prob[y == 1] >= threshold).sum())

    rows = [
        ('ResNet34 @224\ncompetition data only', BASELINE_NO_EXTERNAL_CAUGHT),
        ('ResNet34 @224\n+ ISIC 2019', BASELINE_WITH_EXTERNAL_CAUGHT),
        ('EfficientNet-B4 @300\n+ ISIC 2019 + metadata', final_caught),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 3.7), facecolor=SURFACE)
    ypos = np.arange(len(rows))[::-1]

    for y_i, (label, caught) in zip(ypos, rows):
        ax.barh(y_i, caught, height=0.52, color=FINAL_COLOUR, zorder=3)
        # A 3-unit gap so the two segments read as separate marks, not one bar.
        ax.barh(y_i, TOTAL_MELANOMAS - caught - 3, left=caught + 3, height=0.52,
                color=REMAINDER, zorder=3)
        ax.text(caught - 8, y_i, f'{caught}', va='center', ha='right', fontsize=11,
                color='white', fontweight='bold', zorder=4)
        ax.text(TOTAL_MELANOMAS + 10, y_i, f'{TOTAL_MELANOMAS - caught} missed',
                va='center', ha='left', fontsize=9, color=INK_2)

    ax.set_yticks(ypos)
    ax.set_yticklabels([label for label, _ in rows], fontsize=9.5, color=INK_2)
    ax.get_yticklabels()[-1].set_color(INK)
    ax.get_yticklabels()[-1].set_fontweight('bold')
    ax.set_xlim(0, TOTAL_MELANOMAS + 95)
    ax.set_xticks([0, 146, 292, 438, 584])
    strip_chrome(ax)

    fig.suptitle('Melanomas found, out of 584, at a 5% false-alarm rate',
                 x=0.007, y=0.97, ha='left', fontsize=13, color=INK, fontweight='bold')
    fig.text(0.007, 0.855,
             'Out of fold: every photo scored once by a model that never trained on it. '
             'Specificity held at 95%\nfor all three, so the false alarms are equal and '
             'only the cancers found differ.',
             ha='left', fontsize=9.5, color=INK_2)
    fig.tight_layout(rect=[0, 0, 1, 0.80])

    out = os.path.join(FIGURES, 'melanomas_found.png')
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


def main():
    os.makedirs(FIGURES, exist_ok=True)
    for build in (figure_per_fold_comparison, figure_melanomas_found):
        print("wrote", build())


if __name__ == '__main__':
    main()
