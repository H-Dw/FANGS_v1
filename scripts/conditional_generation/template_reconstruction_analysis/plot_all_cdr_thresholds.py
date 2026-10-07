#!/usr/bin/env python3
"""Threshold plot using exactly the complete paired RMSD table of Figure 1."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from scipy.stats import binomtest

from analysis_common import CDRS, COLORS, INK, MODELS, holm, paired_rows, read_tsv, star, write_tsv
from analyze_and_plot import plot_style

THRESHOLDS = (1.5, 1.0)


def threshold_statistics(pairs, thresholds):
    source, summary, tests = [], [], []
    for target, pair in pairs.items():
        for model in MODELS:
            row = dict(target=target, model=model)
            for threshold in thresholds:
                key = f"all_CDR_below_{str(threshold).replace('.', 'p')}A"
                row[key] = int(all(float(pair[model][cdr]) < threshold for cdr in CDRS))
            source.append(row)
    source_map = {(row["target"], row["model"]): row for row in source}
    for threshold in thresholds:
        key = f"all_CDR_below_{str(threshold).replace('.', 'p')}A"
        a, b = [np.array([source_map[(target, model)][key] for target in pairs]) for model in MODELS]
        for model, flags in zip(MODELS, (a, b)):
            count = int(flags.sum())
            summary.append(dict(class_label=f"All CDR RMSDs < {threshold:.1f} Å", threshold_A=threshold,
                                model=model, count=count, n=len(pairs), percentage=100. * count / len(pairs)))
        a_only, b_only = int(np.sum((a == 1) & (b == 0))), int(np.sum((a == 0) & (b == 1)))
        discordant = a_only + b_only
        p = float(binomtest(min(a_only, b_only), n=discordant, p=.5, alternative="two-sided").pvalue) if discordant else 1.
        tests.append(dict(threshold_A=threshold, n=len(pairs), test="two-sided exact paired McNemar",
                          ESM3_only=a_only, AF3_only=b_only, both_pass=int(np.sum((a == 1) & (b == 1))),
                          neither_pass=int(np.sum((a == 0) & (b == 0))), p_raw=p))
    for item, p in zip(tests, holm([item["p_raw"] for item in tests])):
        item.update(p_holm_2_classes=p, symbol=star(p))
    return source, summary, tests


def draw(summary, tests, out):
    plot_style()
    fig, ax = plt.subplots(figsize=(14.5, 5.4))
    fig.subplots_adjust(left=.29, right=.785, bottom=.19, top=.93)
    centers, offsets = {1.5: 1., 1.0: 0.}, {MODELS[0]: .17, MODELS[1]: -.17}
    for row in summary:
        threshold, model, value = float(row["threshold_A"]), row["model"], float(row["percentage"])
        y = centers[threshold] + offsets[model]
        ax.barh(y, value, height=.29, color=COLORS[model], alpha=.88, edgecolor=INK, linewidth=1.)
        ax.text(value + 1.2, y, f"{value:.1f}%", ha="left", va="center", fontsize=18, fontweight="bold", color=INK)
    for item in tests:
        center = centers[float(item["threshold_A"])]
        ax.plot([114., 115.5, 115.5, 114.], [center - .17, center - .17, center + .17, center + .17],
                color=INK, linewidth=1., clip_on=False)
        ax.text(117., center, item["symbol"], ha="left", va="center", fontsize=21, fontweight="bold", color=INK)
    ax.set_yticks([1., 0.], ["All CDR RMSDs < 1.5 Å", "All CDR RMSDs < 1.0 Å"], fontsize=20)
    ax.set_ylabel("Class", fontsize=25, labelpad=13)
    ax.set_xlim(0., 127.5)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel("Proportion (%)", fontsize=25, labelpad=10)
    ax.tick_params(axis="x", labelsize=20, width=1., length=5)
    ax.tick_params(axis="y", width=0, length=0, pad=11)
    ax.set_ylim(-.62, 1.62)
    ax.grid(axis="x", color="#DCE3E6", alpha=.8, linewidth=.8)
    ax.set_axisbelow(True)
    ax.legend([Patch(facecolor=COLORS[model], edgecolor=INK, linewidth=.8) for model in MODELS],
              MODELS, loc="upper left", bbox_to_anchor=(1.02, .94), fontsize=18,
              ncol=1, labelspacing=.9, handlelength=1.1, handletextpad=.6)
    for extension in ("png", "svg"):
        fig.savefig(out / f"esm3_vs_af3_all_cdr_thresholds.{extension}", dpi=600 if extension == "png" else None,
                    facecolor="white", bbox_inches="tight", pad_inches=.08)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-metrics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-targets", type=int, default=841)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pairs = paired_rows(read_tsv(args.paired_metrics), args.expected_targets)
    source, summary, tests = threshold_statistics(pairs, THRESHOLDS)
    write_tsv(args.out / "all_cdr_threshold_source.tsv", source)
    write_tsv(args.out / "all_cdr_threshold_summary.tsv", summary)
    write_tsv(args.out / "all_cdr_threshold_tests.tsv", tests)
    # Keep the additional 0.5-A count for the earlier requested table, outside
    # the two-class statistical family displayed in this figure.
    _, extra, _ = threshold_statistics(pairs, (.5,))
    write_tsv(args.out / "cdr_thresholds.tsv", summary + extra)
    draw(summary, tests, args.out)
    report = dict(paired_targets=len(pairs), source=str(args.paired_metrics),
                  requested_cohort_exclusions=["7pklL", "8e99E"],
                  cohort="identical complete pairs to Figure 1; no separate CDR-only cohort",
                  threshold_rule="each of the three independently fitted CDR CA RMSDs strictly below threshold",
                  test="two-sided exact paired McNemar; Holm correction across two threshold classes",
                  significance_symbols="maximum three stars; *** denotes adjusted P < 0.001",
                  colors=COLORS, editable_svg_text=True)
    (args.out / "all_cdr_threshold_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("THRESHOLD_SUMMARY", json.dumps(summary))
    print("THRESHOLD_TESTS", json.dumps(tests))


if __name__ == "__main__":
    main()
