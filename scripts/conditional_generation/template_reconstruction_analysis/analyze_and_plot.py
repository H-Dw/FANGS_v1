#!/usr/bin/env python3
"""Score the AF3 observed-chain rerun against the unchanged ESM3 selection.

The exact observed sequence must match the template and both predictions.
Global uses their common C-alpha positions over the full observed input chain.
CDR RMSDs each use an independent proper Kabsch fit, with complete coverage.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, NullFormatter
import numpy as np
from scipy.stats import wilcoxon

from analysis_common import (COLORS, INK, MODELS, REGIONS, cdr_mappings,
                             describe, holm, paired_rows, read_chain, read_tsv,
                             reference_info, rmsd, star, write_tsv)


def score_target(config_root, target, selected, mappings, minimum_coverage):
    _, _, ref_path, ref_seq, ref_xyz, indices = reference_info(config_root, target, mappings)
    predictions = {}
    for model in MODELS:
        seq, xyz = read_chain(selected[model]["path"])
        if seq != ref_seq:
            raise ValueError(f"{model}: exact observed-chain sequence identity required")
        predictions[model] = xyz
    common = [i for i, xyz in enumerate(ref_xyz) if xyz is not None
              and all(predictions[model][i] is not None for model in MODELS)]
    coverage = len(common) / len(ref_seq)
    if len(common) < 3 or coverage < minimum_coverage:
        raise ValueError(f"Shared full-input-chain C-alpha coverage {coverage:.6f} is insufficient")
    for region, positions in indices.items():
        if any(i not in common for i in positions):
            raise ValueError(f"{region}: incomplete shared C-alpha coordinates")
    result = []
    for model in MODELS:
        xyz = predictions[model]
        values = {"global_CA_RMSD": rmsd([ref_xyz[i] for i in common], [xyz[i] for i in common])}
        for region, positions in indices.items():
            values[region + "_CA_RMSD"] = rmsd([ref_xyz[i] for i in positions], [xyz[i] for i in positions])
        result.append(dict(target=target, model=model, selected_pTM=float(selected[model]["ptm"]),
                           selected_sample=selected[model]["sample"],
                           selected_structure=selected[model]["path"], reference_structure=str(ref_path),
                           sequence_identity=1.0, observed_chain_length=len(ref_seq),
                           common_CA_count=len(common), common_CA_coverage=coverage,
                           common_CA_indices_0based=",".join(map(str, common)),
                           CDR1_CA_count=len(indices["CDR1"]), CDR2_CA_count=len(indices["CDR2"]),
                           CDR3_CA_count=len(indices["CDR3"]), **values))
    return result


def plot_style():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 1., "legend.frameon": False})


def draw(pairs, tests, out):
    plot_style()
    fig, ax = plt.subplots(figsize=(13.8, 6.3))
    fig.subplots_adjust(left=.13, right=.775, bottom=.17, top=.94)
    centers, offsets = np.arange(1, 5, dtype=float), (-.17, .17)
    all_arrays, whiskers = [], []
    for index, (region, metric) in enumerate(REGIONS):
        for model_index, model in enumerate(MODELS):
            values = np.array([float(pair[model][metric]) for pair in pairs.values()])
            all_arrays.append(values)
            boxes = ax.boxplot(values, positions=[centers[index] + offsets[model_index]],
                               widths=.27, patch_artist=True, whis=1.5, showfliers=False, manage_ticks=False,
                               boxprops={"linewidth": 1.15, "edgecolor": INK},
                               medianprops={"linewidth": 1.6, "color": INK},
                               whiskerprops={"linewidth": 1.05, "color": INK},
                               capprops={"linewidth": 1.05, "color": INK})
            boxes["boxes"][0].set_facecolor(COLORS[model])
            boxes["boxes"][0].set_alpha(.85)
            whiskers.append(max(boxes["whiskers"][1].get_ydata()))
    values = np.concatenate(all_arrays)
    if np.any(values <= 0):
        raise ValueError("Logarithmic RMSD axis requires positive RMSDs")
    ax.set_yscale("log")
    ax.set_xlim(.45, 4.55)
    ax.set_xticks(centers, [region for region, _ in REGIONS], fontsize=22)
    ax.set_xlabel("Region", fontsize=26, labelpad=12)
    ax.set_ylabel("RMSD (Å)", fontsize=26, labelpad=16)
    ax.tick_params(axis="y", labelsize=20, width=1., length=5)
    ax.tick_params(axis="x", width=1., length=0, pad=8)
    ax.grid(axis="y", color="#DAE1E3", alpha=.78, linewidth=.75)
    ax.set_axisbelow(True)
    fig.legend([Patch(facecolor=COLORS[model], edgecolor=INK, linewidth=.8) for model in MODELS],
               MODELS, loc="upper left", bbox_to_anchor=(.79, .86), fontsize=18, ncol=1,
               labelspacing=.9, handlelength=1.1, handletextpad=.6)
    upper = max(whiskers)
    for index, (region, _) in enumerate(REGIONS):
        x1, x2 = centers[index] + offsets[0], centers[index] + offsets[1]
        ax.plot([x1, x1, x2, x2], [upper * 1.14, upper * 1.21, upper * 1.21, upper * 1.14],
                color=INK, linewidth=1., clip_on=False)
        ax.text(centers[index], upper * 1.25, tests[region]["symbol"],
                ha="center", va="bottom", fontsize=22, fontweight="bold", color=INK)
    low, high = float(values.min()) / 1.45, upper * 1.55
    ax.set_ylim(low, high)
    ticks = [.001, .002, .005, .01, .02, .05, .1, .2, .5, 1., 2., 5., 10., 20., 50., 100.]
    ax.set_yticks([tick for tick in ticks if low <= tick <= high])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    for extension in ("png", "svg"):
        fig.savefig(out / f"esm3_vs_af3_paired_boxplots.{extension}",
                    dpi=600 if extension == "png" else None,
                    facecolor="white", bbox_inches="tight", pad_inches=.08)
    plt.close(fig)
    return dict(scale="logarithmic", lower_limit_A=low, upper_limit_A=high,
                ticks="three decimals", fliers="hidden only in plot; retained in all statistics")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-targets", type=int, default=841)
    parser.add_argument("--minimum-coverage", type=float, default=.9)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    selections = {}
    for model, filename in zip(MODELS, ("ESM3-Template_top1.tsv", "AF3_top1.tsv")):
        records = read_tsv(args.selection_dir / filename)
        selections[model] = {row["target"]: row for row in records}
        if len(selections[model]) != len(records):
            raise ValueError("Duplicate selected target")
    if set(selections[MODELS[0]]) != set(selections[MODELS[1]]):
        raise ValueError("Selected target sets differ")
    if {"7pklL", "8e99E"} & set(selections[MODELS[0]]):
        raise ValueError("A requested excluded target remains in the selected cohort")
    maps = cdr_mappings(args.config_root)
    rows, excluded = [], []
    for target in sorted(selections[MODELS[0]]):
        try:
            rows.extend(score_target(args.config_root, target,
                                     {model: selections[model][target] for model in MODELS},
                                     maps, args.minimum_coverage))
        except Exception as exc:
            excluded.append(dict(target=target, reason=str(exc)))
    write_tsv(args.out / "scoring_failures.tsv", excluded, ["target", "reason"])
    write_tsv(args.out / "paired_metrics.tsv", rows)
    pairs = paired_rows(rows, args.expected_targets)
    if excluded:
        raise ValueError("Excluded targets exist; do not silently shrink the cohort")
    summary, tests, outliers = [], [], []
    for region, metric in REGIONS:
        arrays = {model: np.array([float(pair[model][metric]) for pair in pairs.values()]) for model in MODELS}
        for model in MODELS:
            summary.append(dict(region=region, metric=metric, model=model, **describe(arrays[model])))
        delta = arrays[MODELS[1]] - arrays[MODELS[0]]
        p = float(wilcoxon(delta, alternative="two-sided", zero_method="wilcox", method="auto").pvalue) if np.any(delta) else 1.
        tests.append(dict(region=region, metric=metric, n=len(pairs),
                          test="paired two-sided Wilcoxon signed-rank", model_a=MODELS[0], model_b=MODELS[1],
                          median_delta_B_minus_A=float(np.median(delta)), A_lower=int(np.sum(delta > 0)),
                          B_lower=int(np.sum(delta < 0)), tied=int(np.sum(delta == 0)), p_raw=p))
    for item, p in zip(tests, holm([item["p_raw"] for item in tests])):
        item.update(p_holm_4_regions=p, symbol=star(p))
    for row in rows:
        for region, metric in REGIONS:
            if float(row[metric]) > 10:
                outliers.append(dict(target=row["target"], model=row["model"], region=region,
                                     RMSD_A=row[metric], selected_pTM=row["selected_pTM"],
                                     selected_structure=row["selected_structure"], reference_structure=row["reference_structure"]))
    write_tsv(args.out / "rmsd_summary.tsv", summary)
    write_tsv(args.out / "significance.tsv", tests)
    write_tsv(args.out / "outliers_above_10A.tsv", outliers,
              ["target", "model", "region", "RMSD_A", "selected_pTM", "selected_structure", "reference_structure"])
    axis = draw(pairs, {item["region"]: item for item in tests}, args.out)
    report = dict(paired_targets=len(pairs), rows=len(rows), excluded_targets=2,
                  scoring_failures=len(excluded),
                  requested_cohort_exclusions=["7pklL", "8e99E"],
                  requested_exclusion_count=2,
                  cohort="source-defined CDR conditioning positions consistent after the two requested exclusions",
                  sample_count_by_region={region: {model: len(pairs) for model in MODELS} for region, _ in REGIONS},
                  exact_observed_sequence_identity_targets=len(pairs),
                  minimum_shared_CA_coverage=min(float(row["common_CA_coverage"]) for row in rows),
                  reference_CA_incomplete_targets=sum(float(pair[MODELS[0]]["common_CA_coverage"]) < 1 for pair in pairs.values()),
                  observed_chain_length_range=[min(row["observed_chain_length"] for row in rows), max(row["observed_chain_length"] for row in rows)],
                  observed_chains_longer_than_200=sum(pair[MODELS[0]]["observed_chain_length"] > 200 for pair in pairs.values()),
                  Global="full experimentally observed input chain; identical shared CA positions for both models",
                  CDR="independent proper Kabsch fit per CDR; full CDR CA coverage",
                  comparison_to_legacy_Global="different region definition: legacy used the mapped INDI nanobody query domain",
                  selection="highest reported pTM, no confidence filter, no RMSD selection",
                  significance="paired two-sided Wilcoxon; Holm across four regions", axis=axis,
                  svg_text="editable", colors=COLORS,
                  outliers_above_10A={model: {region: sum(item['model'] == model and item['region'] == region for item in outliers)
                                              for region, _ in REGIONS} for model in MODELS})
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("RMSD_SUMMARY", json.dumps(summary))
    print("RMSD_TESTS", json.dumps(tests))


if __name__ == "__main__":
    main()
