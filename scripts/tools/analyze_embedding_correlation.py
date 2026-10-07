#!/usr/bin/env python3
"""Embedding-distance correlation and trend analysis for batch grafting.

Ported from ``notebook/correlation_distance_dupl_embe_250627.ipynb``.
Default analysis target is ``data/batch_graft_generation_id60_table``.

Tables are written to ``<input>/analysis/tables`` and figures to
``figures/batch_graft_generation_id60_table`` unless overridden.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.container import BarContainer
from scipy.stats import pearsonr, spearmanr

FINAL_VERSION_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = FINAL_VERSION_ROOT / "data" / "batch_graft_generation_id60_table"
DEFAULT_TABLE_DIR = DEFAULT_INPUT / "analysis" / "tables"
DEFAULT_FIG_DIR = FINAL_VERSION_ROOT / "figures" / "batch_graft_generation_id60_table"

REGION_LIST = ["all", "connector1", "connector2", "connector3"]
EMBE_LIST = ["raw_embeddings", "pre_q_embeddings", "ca_distance"]
STAGE_LIST = ["original", "grafted"]
ORIG_COLOR = "#4575D4"
GRAF_COLOR = "#6B9B00"


def p_to_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def safe_corr(x, y, method: str = "spearman") -> tuple[float, float]:
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "pearson":
            return pearsonr(x, y)
        return spearmanr(x, y)


def list_task_ids(generation_folder: Path, target_ids: Optional[Sequence[str]] = None) -> list[str]:
    if target_ids:
        return [str(x) for x in target_ids]
    ids = []
    for path in sorted(generation_folder.iterdir()):
        gen_tsv = path / "temp_generation" / "all_info" / "all_generation.tsv"
        if path.is_dir() and gen_tsv.exists():
            ids.append(path.name)
    return ids


def read_one_task(pdb_id: str, pdb_folder: Path, region_list: Sequence[str], embe_list: Sequence[str]) -> pd.DataFrame:
    generation_tsv = pdb_folder / "temp_generation" / "all_info" / "all_generation.tsv"
    generation_df = pd.read_csv(generation_tsv, sep="\t")
    for region in region_list:
        for embe in embe_list:
            orig_path = pdb_folder / "similarity" / f"{embe}_similarity_{region}.tsv"
            gen_path = pdb_folder / "generation_tokenized_structure" / f"{embe}_similarity_{region}.tsv"
            orig_df = pd.read_csv(orig_path, sep="\t").rename(
                columns={
                    "euclidean": f"original_euclidean_{region}_{embe}",
                    "cosine": f"original_cosine_{region}_{embe}",
                }
            )
            gen_df = pd.read_csv(gen_path, sep="\t").rename(
                columns={
                    "euclidean": f"grafted_euclidean_{region}_{embe}",
                    "cosine": f"grafted_cosine_{region}_{embe}",
                }
            )
            generation_df = generation_df.merge(
                orig_df, how="inner", left_on="PDB_ID", right_on="pdb_id"
            ).drop(columns=["pdb_id"])
            generation_df["pdb_id_stripped"] = generation_df["Filename"].str.replace(
                r"\.pdb$", "", regex=True
            )
            generation_df = generation_df.merge(
                gen_df, how="inner", left_on="pdb_id_stripped", right_on="pdb_id"
            ).drop(columns=["pdb_id_stripped", "pdb_id"])
    generation_df["FR_ID"] = generation_df["PDB_ID"].astype(str)
    generation_df["Donator"] = pdb_id
    generation_df["PDB_ID"] = generation_df["PDB_ID"].apply(lambda x: f"{pdb_id}_{x}")
    return generation_df


def _normalize_search_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]
    df["target"] = df["target"].astype(str).str.replace(r"\.pdb$", "", regex=True)
    return df.drop_duplicates(subset="target", keep="first")


def _query_length(foldseek_df: pd.DataFrame, pdb_id: str) -> int:
    self_hit = foldseek_df[foldseek_df["target"] == pdb_id]
    row = self_hit.iloc[0] if len(self_hit) else foldseek_df.iloc[0]
    return max(len(str(row["tseq"])), 1)


def maybe_merge_similarity(generation_df: pd.DataFrame, pdb_id: str, pdb_folder: Path) -> pd.DataFrame:
    foldseek_tsv = pdb_folder / "similarity" / f"{pdb_id}_foldseek_output.tsv"
    mmseqs_tsv = pdb_folder / "similarity" / f"{pdb_id}_mmseqs_output.tsv"
    if not foldseek_tsv.exists() or not mmseqs_tsv.exists():
        return generation_df

    foldseek_df = _normalize_search_target(pd.read_csv(foldseek_tsv, sep="\t")).rename(
        columns={"bits": "structure_bits", "nident": "structure_nident"}
    )
    query_len = _query_length(foldseek_df, pdb_id)
    foldseek_df["structure_qident"] = np.sqrt(foldseek_df["structure_nident"] / query_len).round(3)

    mmseqs_df = _normalize_search_target(pd.read_csv(mmseqs_tsv, sep="\t")).rename(
        columns={"bits": "sequence_bits", "nident": "sequence_nident"}
    )
    mmseqs_df["sequence_qident"] = np.sqrt(mmseqs_df["sequence_nident"] / query_len).round(3)

    # Left-merge so missing hits do not drop grafting rows used by other plots.
    merged = generation_df.merge(
        foldseek_df[["target", "structure_qident", "structure_bits", "qtmscore"]],
        how="left",
        left_on="FR_ID",
        right_on="target",
    ).drop(columns=["target"])
    merged = merged.merge(
        mmseqs_df[["target", "sequence_qident", "sequence_bits"]],
        how="left",
        left_on="FR_ID",
        right_on="target",
    ).drop(columns=["target"])
    return merged


def read_generations(
    generation_folder: Path,
    target_ids: Optional[Sequence[str]],
    region_list: Sequence[str],
    embe_list: Sequence[str],
) -> pd.DataFrame:
    all_results = []
    success = fail = 0
    for pdb_id in list_task_ids(generation_folder, target_ids):
        pdb_folder = generation_folder / pdb_id
        gen_tsv = pdb_folder / "temp_generation" / "all_info" / "all_generation.tsv"
        if not gen_tsv.exists():
            print(f"[WARNING] Skipped {pdb_id}: missing {gen_tsv}")
            fail += 1
            continue
        try:
            generation_df = read_one_task(pdb_id, pdb_folder, region_list, embe_list)
            generation_df = maybe_merge_similarity(generation_df, pdb_id, pdb_folder)
            all_results.append(generation_df)
            success += 1
        except Exception as exc:
            print(f"[ERROR] extract {pdb_id} fail: {exc}")
            fail += 1
    if not all_results:
        raise RuntimeError(f"No valid generation data found in {generation_folder}")
    print(f"Passed files: {success}\t Failed files: {fail}")
    return pd.concat(all_results, ignore_index=True)


def add_norm_and_change(
    df: pd.DataFrame,
    stage_list: Sequence[str],
    region_list: Sequence[str],
    embe_list: Sequence[str],
) -> pd.DataFrame:
    df = df.copy()
    for stage in stage_list:
        for region in region_list:
            for embe in embe_list:
                length = 12 if region == "all" else 4
                col = f"{stage}_euclidean_{region}_{embe}"
                df[f"{col}_norm"] = df[col] / np.sqrt(length)
    for region in region_list:
        for embe in embe_list:
            df[f"change_euclidean_{region}_{embe}"] = (
                df[f"original_euclidean_{region}_{embe}_norm"]
                - df[f"grafted_euclidean_{region}_{embe}_norm"]
            ).abs()
    return df


def collect_results(
    df: pd.DataFrame,
    stage: str,
    source_label: str,
    region_list: Sequence[str],
    embe_list: Sequence[str],
    target: str,
    method: str,
) -> pd.DataFrame:
    rows = []
    for region in region_list:
        for embe in embe_list:
            col = f"{stage}_euclidean_{region}_{embe}_norm"
            r, p = safe_corr(df[col], df[target], method=method)
            rows.append(
                {
                    "method": method,
                    "region": region,
                    "embedding": embe,
                    "label": f"{region}_{embe}",
                    "correlation": r,
                    "p_value": p,
                    "source": source_label,
                    "target": target,
                    "n": int(df[col].notna().sum()),
                }
            )
    return pd.DataFrame(rows)


def savefig(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def plot_crmsd_ptm_scatter(df: pd.DataFrame, fig_dir: Path, dpi: int) -> pd.DataFrame:
    x = df["cRMSD"]
    y = df["pTM"]
    r, p_val = safe_corr(x, y, method="spearman")
    r2 = r**2 if pd.notna(r) else float("nan")

    sns.set(style="white")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=x, y=y, s=50, alpha=0.7, ax=ax)
    sns.regplot(x=x, y=y, scatter=False, ci=95, line_kws={"color": "red", "lw": 1.5}, ax=ax)
    ax.axhline(0.8, color="blue", linestyle="--", lw=1, label="pTM = 0.8")
    ax.axvline(1.5, color="green", linestyle="--", lw=1, label="cRMSD = 1.5")
    ax.set_title(f"$r={r:.2f},\\ \\;R^2={r2:.2f},\\ \\;p={p_val:.2g}$", loc="center", fontsize=16, pad=10, fontstyle="italic")
    ax.set_xlabel("cRMSD", fontsize=16)
    ax.set_ylabel("pTM", fontsize=16)
    ax.legend(loc="lower left", fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    savefig(fig, fig_dir / "scatter_cRMSD_pTM_spearmanr.png", dpi)
    return pd.DataFrame([{"target_x": "cRMSD", "target_y": "pTM", "method": "spearman", "correlation": r, "p_value": p_val, "r2": r2, "n": int(len(df))}])


def plot_global_similarity(df: pd.DataFrame, fig_dir: Path, dpi: int) -> Optional[pd.DataFrame]:
    labels = ["structure_qident", "sequence_qident"]
    if any(col not in df.columns for col in labels):
        print("[INFO] Skip global similarity plot: foldseek/mmseqs qident columns are absent (table mode).")
        return None

    plot_df = df.dropna(subset=labels + ["cRMSD", "pTM"]).copy()
    if len(plot_df) < 3:
        print("[INFO] Skip global similarity plot: too few rows with Foldseek/MMseqs qident.")
        return None
    print(f"[INFO] Global qident plot uses {len(plot_df)} / {len(df)} rows with Foldseek and MMseqs hits.")

    rows = []
    for lab in labels:
        for tgt in ["cRMSD", "pTM"]:
            corr, pval = safe_corr(plot_df[lab], plot_df[tgt], method="spearman")
            if tgt == "pTM" and pd.notna(corr):
                corr = -corr
            rows.append({"label": lab, "correlation": corr, "p_value": pval, "source": tgt})
    res_df = pd.DataFrame(rows)
    xticks = ["Structure\nQident", "Sequence\nQident"]

    sns.set(style="white")
    fig, ax = plt.subplots(figsize=(6, 4))
    palette = {"cRMSD": sns.color_palette("deep")[0], "pTM": sns.color_palette("deep")[1]}
    sns.barplot(data=res_df, x="label", y="correlation", hue="source", palette=palette, dodge=True, ax=ax)
    ax.set_xticks(range(len(xticks)))
    ax.set_xticklabels(xticks, rotation=0, ha="center", fontsize=14)
    for container in ax.containers:
        if isinstance(container, BarContainer):
            heights = [patch.get_height() for patch in container]
            ax.bar_label(container, labels=[f"{h:.2f}" for h in heights], label_type="center", padding=0, fontsize=12, color="black")
    for bar, p in zip(ax.patches, res_df["p_value"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), p_to_stars(p), ha="center", va="bottom", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("Correlation", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    leg = ax.legend(title=None, loc="upper left", bbox_to_anchor=(0.8, 1.1), fontsize=16, frameon=False)
    for text in leg.get_texts():
        if text.get_text() == "pTM":
            text.set_text("-pTM (inverted)")
    ymax = max(float(res_df["correlation"].max()), 0.0)
    ax.set_ylim(0, max(ymax * 1.25, 0.05))
    fig.tight_layout()
    savefig(fig, fig_dir / "global_similarity_spearmanr.png", dpi)
    return res_df


def plot_connector_correlation(
    res_df: pd.DataFrame,
    target: str,
    fig_path: Path,
    dpi: int,
    xticks: Optional[Sequence[str]] = None,
    figsize=(8, 5),
    x_rotation: Optional[float] = None,
    invert: bool = False,
    fontsize: int = 14,
    legend_loc: float = 0.8,
) -> None:
    sns.set(style="white")
    fig, ax = plt.subplots(figsize=figsize)
    palette = {"Original connector": ORIG_COLOR, "Grafted connector": GRAF_COLOR}
    sns.barplot(data=res_df, x="label", y="correlation", hue="source", palette=palette, dodge=True, ax=ax)
    if xticks:
        ax.set_xticks(range(len(xticks)))
        ax.set_xticklabels(xticks, fontsize=fontsize)
    if x_rotation:
        plt.xticks(rotation=x_rotation, ha="center", fontsize=fontsize)

    r_orig = res_df[res_df["source"] == "Original connector"]["correlation"].values
    r_graft = res_df[res_df["source"] == "Grafted connector"]["correlation"].values
    for container, r_vals in zip(ax.containers, [r_orig, r_graft]):
        ax.bar_label(container, labels=[f"{val:.2f}" for val in r_vals], label_type="center", fontsize=fontsize - 2, color="black")
    for bar, p in zip(ax.patches, res_df["p_value"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), p_to_stars(p), ha="center", va="bottom", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel(f"{target} correlation", fontsize=fontsize + 2)
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.legend(title=None, loc="upper left", bbox_to_anchor=(legend_loc, 1.05), borderaxespad=0, fontsize=fontsize + 2, frameon=False)
    if invert:
        ax.invert_yaxis()
    fig.tight_layout()
    savefig(fig, fig_path, dpi)


def plot_change_histograms(df: pd.DataFrame, fig_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["change_euclidean_all_raw_embeddings"], kde=True, label="Raw distance", color="blue", ax=ax)
    sns.histplot(df["change_euclidean_all_pre_q_embeddings"], kde=True, label="Pre-VQ distance", color="orange", ax=ax)
    sns.histplot(df["change_euclidean_all_ca_distance"], kde=True, label="CA distance", color="red", ax=ax)
    ax.legend(frameon=False)
    ax.set_xlabel("Distance change", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    xmax = ax.get_xlim()[1]
    ax.set_xticks(np.arange(0, np.ceil(xmax) + 5, 5))
    ax.set_xlim(left=0)
    sns.despine(ax=ax, top=True, right=True)
    fig.tight_layout()
    savefig(fig, fig_dir / "change_distance.png", dpi)

    cols = [
        "cRMSD",
        "change_euclidean_all_raw_embeddings",
        "change_euclidean_all_pre_q_embeddings",
        "change_euclidean_all_ca_distance",
    ]
    z_cols = {}
    for col in cols:
        series = df[col].dropna()
        z_cols[col] = (df[col] - series.mean()) / series.std(ddof=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(z_cols[cols[3]], kde=True, label="CA distance", color="green", ax=ax)
    sns.histplot(z_cols[cols[2]], kde=True, label="Pre-VQ distance", color="orange", ax=ax)
    sns.histplot(z_cols[cols[1]], kde=True, label="Raw distance", color="blue", ax=ax)
    sns.histplot(z_cols[cols[0]], kde=True, label="cRMSD", color="red", ax=ax)
    ax.legend(frameon=False)
    ax.set_xlabel("Z-score of distance change", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    xmin, xmax = ax.get_xlim()
    ax.set_xticks(np.arange(np.floor(xmin), np.ceil(xmax) + 0.5, 5))
    sns.despine(ax=ax, top=True, right=True)
    fig.tight_layout()
    savefig(fig, fig_dir / "change_distance_z-score.png", dpi)


def plot_threshold_percent(df: pd.DataFrame, fig_dir: Path, dpi: int) -> pd.DataFrame:
    embes = [("raw_embeddings", sns.color_palette("deep")[0]), ("pre_q_embeddings", sns.color_palette("deep")[1])]
    thresholds = np.arange(0, 31, 1)
    step = 5
    n_total = len(df)
    rows = []
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(thresholds))
    width = 0.36
    offsets = [-width / 2, width / 2]
    for idx, (embe, color) in enumerate(embes):
        change_col = f"change_euclidean_all_{embe}"
        cum_pct = [100.0 * (df[change_col] <= th).sum() / n_total for th in thresholds]
        ax.bar(x + offsets[idx], cum_pct, width, label=embe, color=color, edgecolor="white", linewidth=0.2)
        for th, pct in zip(thresholds, cum_pct):
            rows.append({"embedding": embe, "threshold": int(th), "cumulative_percent": pct, "n_kept": int((df[change_col] <= th).sum())})
    ax.set_xlabel("Threshold on distance change", fontsize=16)
    ax.set_ylabel("Cumulative sample (%)", fontsize=16)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(thresholds[::step])
    ax.set_ylim(0, 105)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(["Raw distance", "Pre-VQ distance"], bbox_to_anchor=(0.95, 1), loc="upper left", fontsize=16, frameon=False)
    fig.tight_layout()
    savefig(fig, fig_dir / "threshold_on_distance_change_percent.png", dpi)
    return pd.DataFrame(rows)


def plot_threshold_correlation(
    df: pd.DataFrame,
    target: str,
    fig_path: Path,
    dpi: int,
    invert: bool = False,
    include_ca_legend: bool = False,
) -> pd.DataFrame:
    embes = [("raw_embeddings", "-"), ("pre_q_embeddings", "--")]
    thresholds = np.arange(0, 30, 1)
    step = 5
    rows = []
    fig, ax = plt.subplots(figsize=(8, 5))
    for embe, linestyle in embes:
        change_col = f"change_euclidean_all_{embe}"
        orig_col = f"original_euclidean_all_{embe}_norm"
        graf_col = f"grafted_euclidean_all_{embe}_norm"
        corr_orig, corr_graf = [], []
        for th in thresholds:
            df_filt = df[df[change_col] <= th]
            r_orig, p_orig = safe_corr(df_filt[orig_col], df_filt[target])
            r_graf, p_graf = safe_corr(df_filt[graf_col], df_filt[target])
            corr_orig.append(r_orig)
            corr_graf.append(r_graf)
            rows.append({"embedding": embe, "stage": "original", "threshold": int(th), "target": target, "correlation": r_orig, "p_value": p_orig, "n": len(df_filt)})
            rows.append({"embedding": embe, "stage": "grafted", "threshold": int(th), "target": target, "correlation": r_graf, "p_value": p_graf, "n": len(df_filt)})
        ax.plot(thresholds, corr_orig, linestyle=linestyle, color=ORIG_COLOR, marker="o", markevery=step, markersize=5)
        ax.plot(thresholds, corr_graf, linestyle=linestyle, color=GRAF_COLOR, marker="o", markevery=step, markersize=5)
    ax.set_xlabel("Threshold on distance change", fontsize=16)
    ax.set_ylabel(f"{target} correlation", fontsize=16)
    ax.set_xticks(thresholds[::step])
    ax.tick_params(labelsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend_entries = [
        "Original Raw distance",
        "Grafted Raw distance",
        "Original Pre-VQ distance",
        "Grafted Pre-VQ distance",
    ]
    if include_ca_legend:
        legend_entries += ["Original CA distance", "Grafted CA distance"]
    ax.legend(legend_entries, bbox_to_anchor=(0.80, 1.05), loc="upper left", fontsize=16, frameon=False)
    if invert:
        ax.invert_yaxis()
    fig.tight_layout()
    savefig(fig, fig_path, dpi)
    return pd.DataFrame(rows)


def plot_within_donator_original_rank(
    df: pd.DataFrame,
    fig_dir: Path,
    dpi: int,
    change_th: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    orig_col = "original_euclidean_all_raw_embeddings_norm"
    graf_col = "grafted_euclidean_all_raw_embeddings_norm"
    change_col = "change_euclidean_all_raw_embeddings"
    rank_col = "rank_pct_original_raw_within_donator"

    filtered = work[work[change_col] <= change_th].copy()
    filtered[rank_col] = filtered.groupby("Donator")[orig_col].rank(pct=True, ascending=True) * 100
    thresholds = np.arange(5, 55, 5)
    rows = []
    corr_crmsd, corr_ptm = [], []
    for th in thresholds:
        df_filt = filtered[filtered[rank_col] <= th]
        r_c, pv_c = safe_corr(df_filt[graf_col], df_filt["cRMSD"])
        r_p, pv_p = safe_corr(df_filt[graf_col], -df_filt["pTM"])
        corr_crmsd.append(r_c)
        corr_ptm.append(r_p)
        rows.append({"threshold_pct": int(th), "target": "cRMSD", "correlation": r_c, "p_value": pv_c, "n": len(df_filt)})
        rows.append({"threshold_pct": int(th), "target": "-pTM", "correlation": r_p, "p_value": pv_p, "n": len(df_filt)})

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, corr_crmsd, color=sns.color_palette("deep")[0], linestyle="-", marker="o", markersize=5, label="Grafted raw distance → cRMSD")
    ax.plot(thresholds, corr_ptm, color=sns.color_palette("deep")[1], linestyle="--", marker="s", markersize=5, label="Grafted raw distance → −pTM (negated)")
    ax.set_xlabel("Within-Donator top % by original raw embedding score", fontsize=15)
    ax.set_ylabel("Spearman correlation", fontsize=15)
    ax.set_xticks(thresholds)
    ax.set_xticklabels([f"{int(t)}%" for t in thresholds])
    ax.tick_params(labelsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", fontsize=12, frameon=False)
    fig.tight_layout()
    savefig(fig, fig_dir / "threshold_original_raw_pct_within_donator_crmsd_ptm.png", dpi)

    work[rank_col] = work.groupby("Donator")[orig_col].rank(pct=True, ascending=True) * 100
    thresholds_full = np.arange(5, 100, 5)
    rows_ptm = []
    corr_orig, corr_graf = [], []
    for th in thresholds_full:
        df_filt = work[work[rank_col] <= th]
        r_o, pv_o = safe_corr(df_filt[orig_col], df_filt["pTM"])
        r_g, pv_g = safe_corr(df_filt[graf_col], df_filt["pTM"])
        corr_orig.append(r_o)
        corr_graf.append(r_g)
        rows_ptm.append({"threshold_pct": int(th), "stage": "original", "correlation": r_o, "p_value": pv_o, "n": len(df_filt)})
        rows_ptm.append({"threshold_pct": int(th), "stage": "grafted", "correlation": r_g, "p_value": pv_g, "n": len(df_filt)})
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds_full, corr_orig, color=ORIG_COLOR, linestyle="-", marker="o", markersize=5, label="Original raw distance → pTM")
    ax.plot(thresholds_full, corr_graf, color=GRAF_COLOR, linestyle="-", marker="o", markersize=5, label="Grafted raw distance → pTM")
    ax.set_xlabel("Within-Donator top % by original raw (ascending rank)", fontsize=12)
    ax.set_ylabel("pTM correlation (Spearman)", fontsize=14)
    ax.set_xticks(thresholds_full)
    ax.set_xticklabels([f"{int(t)}%" for t in thresholds_full])
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", fontsize=12, frameon=False)
    fig.tight_layout()
    savefig(fig, fig_dir / "threshold_original_raw_pct_within_donator_pTM.png", dpi)
    return pd.DataFrame(rows), pd.DataFrame(rows_ptm)


def plot_change_rank_correlation(df: pd.DataFrame, target: str, fig_path: Path, dpi: int, invert: bool = False) -> pd.DataFrame:
    embes = [("raw_embeddings", "-"), ("pre_q_embeddings", "--")]
    thresholds = np.arange(5, 55, 5)
    work = df.copy()
    rows = []
    fig, ax = plt.subplots(figsize=(8, 5))
    for embe, linestyle in embes:
        change_col = f"change_euclidean_all_{embe}"
        orig_col = f"original_euclidean_all_{embe}_norm"
        graf_col = f"grafted_euclidean_all_{embe}_norm"
        rank_col = f"rank_pct_{embe}"
        work[rank_col] = work.groupby("Donator")[change_col].rank(pct=True, ascending=True) * 100
        corr_orig, corr_graf = [], []
        for th in thresholds:
            df_filt = work[work[rank_col] <= th]
            r_orig, p_orig = safe_corr(df_filt[orig_col], df_filt[target])
            r_graf, p_graf = safe_corr(df_filt[graf_col], df_filt[target])
            corr_orig.append(r_orig)
            corr_graf.append(r_graf)
            rows.append({"embedding": embe, "stage": "original", "threshold_pct": int(th), "target": target, "correlation": r_orig, "p_value": p_orig, "n": len(df_filt)})
            rows.append({"embedding": embe, "stage": "grafted", "threshold_pct": int(th), "target": target, "correlation": r_graf, "p_value": p_graf, "n": len(df_filt)})
        ax.plot(thresholds, corr_orig, linestyle=linestyle, color=ORIG_COLOR, marker="o", markersize=5)
        ax.plot(thresholds, corr_graf, linestyle=linestyle, color=GRAF_COLOR, marker="o", markersize=5)
    ax.set_xlabel("Percentage threshold on distance change", fontsize=12)
    ax.set_ylabel(f"{target} correlation", fontsize=14)
    ax.set_xticks(thresholds)
    ax.set_xticklabels([f"{int(t)}%" for t in thresholds])
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        ["Original Raw distance", "Grafted Raw distance", "Original Pre-VQ distance", "Grafted Pre-VQ distance"],
        bbox_to_anchor=(0.80, 1),
        loc="upper left",
        fontsize=12,
        frameon=False,
    )
    if invert:
        ax.invert_yaxis()
    fig.tight_layout()
    savefig(fig, fig_path, dpi)
    return pd.DataFrame(rows)


def plot_diff_from_top1(df: pd.DataFrame, target: str, fig_path: Path, dpi: int, invert: bool = False) -> pd.DataFrame:
    embes = [("raw_embeddings", "-"), ("pre_q_embeddings", "--")]
    thresholds = np.arange(5, 55, 5)
    work = df.copy()
    total_samples = len(work)
    rows = []
    fig, ax = plt.subplots(figsize=(8, 5))
    for embe, linestyle in embes:
        change_col = f"change_euclidean_all_{embe}"
        orig_col = f"original_euclidean_all_{embe}_norm"
        graf_col = f"grafted_euclidean_all_{embe}_norm"
        min_change_col = work.groupby("Donator")[change_col].transform("min")
        top10pct_boundary_col = work.groupby("Donator")[change_col].transform(lambda x: x.quantile(0.1))
        diff_pct_col = f"diff_pct_from_top1_{embe}"
        work[diff_pct_col] = (work[change_col] - min_change_col) / (top10pct_boundary_col - min_change_col + 1e-8) * 100
        corr_orig, corr_graf = [], []
        for th in thresholds:
            df_filt = work[work[diff_pct_col] <= th]
            r_orig, p_orig = safe_corr(df_filt[orig_col], df_filt[target])
            r_graf, p_graf = safe_corr(df_filt[graf_col], df_filt[target])
            corr_orig.append(r_orig)
            corr_graf.append(r_graf)
            rows.append(
                {
                    "embedding": embe,
                    "stage": "original",
                    "threshold_pct": int(th),
                    "target": target,
                    "correlation": r_orig,
                    "p_value": p_orig,
                    "n": len(df_filt),
                    "remain_percent": 100.0 * len(df_filt) / total_samples,
                }
            )
            rows.append(
                {
                    "embedding": embe,
                    "stage": "grafted",
                    "threshold_pct": int(th),
                    "target": target,
                    "correlation": r_graf,
                    "p_value": p_graf,
                    "n": len(df_filt),
                    "remain_percent": 100.0 * len(df_filt) / total_samples,
                }
            )
        ax.plot(thresholds, corr_orig, linestyle=linestyle, color=ORIG_COLOR, marker="o", markersize=5)
        ax.plot(thresholds, corr_graf, linestyle=linestyle, color=GRAF_COLOR, marker="o", markersize=5)
    ax.set_xlabel("Difference from Top-1 (%)", fontsize=12)
    ax.set_ylabel(f"{target} correlation", fontsize=14)
    ax.set_xticks(thresholds)
    ax.set_xticklabels([f"{int(t)}%" for t in thresholds])
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        ["Original Raw distance", "Grafted Raw distance", "Original Pre-VQ distance", "Grafted Pre-VQ distance"],
        bbox_to_anchor=(0.90, 1.1),
        loc="upper left",
        fontsize=12,
        frameon=False,
    )
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    ax.set_ylim(ylim[0], ylim[1] + y_range * 0.1)
    if invert:
        ax.invert_yaxis()
    fig.tight_layout()
    savefig(fig, fig_path, dpi)
    return pd.DataFrame(rows)


def write_tsv(df: Optional[pd.DataFrame], path: Path) -> None:
    if df is None or df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    print(f"Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze embedding-distance correlation and change trends for batch grafting."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Batch grafting output folder")
    parser.add_argument("--table-dir", default=str(DEFAULT_TABLE_DIR), help="Output folder for analysis tables")
    parser.add_argument("--fig-dir", default=str(DEFAULT_FIG_DIR), help="Output folder for figures")
    parser.add_argument("--dpi", type=int, default=600, help="Figure DPI")
    parser.add_argument("--target-list", default=None, help="Optional TSV with a pdb_id column")
    parser.add_argument("--embe-list", nargs="+", default=list(EMBE_LIST))
    parser.add_argument("--region-list", nargs="+", default=list(REGION_LIST))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input).resolve()
    table_dir = Path(args.table_dir).resolve()
    fig_dir = Path(args.fig_dir).resolve()
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    target_ids = None
    if args.target_list:
        target_df = pd.read_csv(args.target_list, sep="\t")
        if "pdb_id" not in target_df.columns:
            raise KeyError(f"{args.target_list} must contain a pdb_id column")
        target_ids = target_df["pdb_id"].astype(str).tolist()

    connector_df = read_generations(input_dir, target_ids, args.region_list, args.embe_list)
    connector_df = add_norm_and_change(connector_df, STAGE_LIST, args.region_list, args.embe_list)
    write_tsv(connector_df, table_dir / "merged_generations.tsv")

    scatter_stats = plot_crmsd_ptm_scatter(connector_df, fig_dir, args.dpi)
    write_tsv(scatter_stats, table_dir / "scatter_cRMSD_pTM_spearmanr.tsv")

    global_df = plot_global_similarity(connector_df, fig_dir, args.dpi)
    write_tsv(global_df, table_dir / "global_similarity_spearmanr.tsv")

    plot_change_histograms(connector_df, fig_dir, args.dpi)

    corr_frames = []
    xticks_all = ["Raw\ndistance", "Pre-VQ\ndistance", "CA\ndistance"]
    for method in ("spearman", "pearson"):
        for target, invert in (("cRMSD", False), ("pTM", True)):
            orig = collect_results(connector_df, "original", "Original connector", ["all"], args.embe_list, target, method)
            graf = collect_results(connector_df, "grafted", "Grafted connector", ["all"], args.embe_list, target, method)
            res_df = pd.concat([orig, graf], ignore_index=True)
            corr_frames.append(res_df)
            plot_connector_correlation(
                res_df,
                target=target,
                fig_path=fig_dir / f"connector_{target}_{method}r.png",
                dpi=args.dpi,
                xticks=xticks_all,
                invert=invert,
                fontsize=16 if method == "spearman" else 14,
                legend_loc=0.8 if method == "spearman" else 1,
            )

    xticks_each = [
        f"{region}\n{embe}"
        for region in ["Connector1", "Connector2", "Connector3"]
        for embe in ["Raw distance", "Pre-VQ distance", "CA distance"]
    ]
    for target, invert in (("cRMSD", False), ("pTM", True)):
        orig = collect_results(
            connector_df, "original", "Original connector",
            ["connector1", "connector2", "connector3"], args.embe_list, target, "spearman",
        )
        graf = collect_results(
            connector_df, "grafted", "Grafted connector",
            ["connector1", "connector2", "connector3"], args.embe_list, target, "spearman",
        )
        res_df = pd.concat([orig, graf], ignore_index=True)
        corr_frames.append(res_df)
        plot_connector_correlation(
            res_df,
            target=target,
            fig_path=fig_dir / f"each_connector_{target}_spearmanr.png",
            dpi=args.dpi,
            xticks=xticks_each,
            x_rotation=45,
            figsize=(12, 4),
            invert=invert,
            fontsize=10,
            legend_loc=1,
        )
    write_tsv(pd.concat(corr_frames, ignore_index=True), table_dir / "connector_correlations.tsv")

    thresh_pct = plot_threshold_percent(connector_df, fig_dir, args.dpi)
    write_tsv(thresh_pct, table_dir / "threshold_distance_change_percent.tsv")

    crmsd_th = plot_threshold_correlation(
        connector_df, "cRMSD", fig_dir / "threshold_on_distance_change_cRMSD.png", args.dpi
    )
    ptm_th = plot_threshold_correlation(
        connector_df, "pTM", fig_dir / "threshold_on_distance_change_pTM.png", args.dpi,
        invert=True, include_ca_legend=True,
    )
    write_tsv(pd.concat([crmsd_th, ptm_th], ignore_index=True), table_dir / "threshold_distance_change_correlations.tsv")

    rank_crmsd_ptm, rank_ptm = plot_within_donator_original_rank(connector_df, fig_dir, args.dpi)
    write_tsv(rank_crmsd_ptm, table_dir / "threshold_original_raw_pct_within_donator_crmsd_ptm.tsv")
    write_tsv(rank_ptm, table_dir / "threshold_original_raw_pct_within_donator_pTM.tsv")

    change_rank_crmsd = plot_change_rank_correlation(
        connector_df, "cRMSD", fig_dir / "threshold_on_distance_change_percentage_cRMSD.png", args.dpi
    )
    change_rank_ptm = plot_change_rank_correlation(
        connector_df, "pTM", fig_dir / "threshold_on_distance_change_percentage_pTM.png", args.dpi, invert=True
    )
    write_tsv(
        pd.concat([change_rank_crmsd, change_rank_ptm], ignore_index=True),
        table_dir / "threshold_change_rank_correlations.tsv",
    )

    top1_crmsd = plot_diff_from_top1(
        connector_df, "cRMSD", fig_dir / "threshold_diff_percentage_from_top1_cRMSD.png", args.dpi
    )
    top1_ptm = plot_diff_from_top1(
        connector_df, "pTM", fig_dir / "threshold_diff_percentage_from_top1_pTM.png", args.dpi, invert=True
    )
    write_tsv(pd.concat([top1_crmsd, top1_ptm], ignore_index=True), table_dir / "threshold_diff_from_top1_correlations.tsv")

    print(f"Merged rows: {len(connector_df)}")
    print(f"Tables: {table_dir}")
    print(f"Figures: {fig_dir}")
    return 0


if __name__ == "__main__":
    os.chdir(FINAL_VERSION_ROOT)
    raise SystemExit(main())
