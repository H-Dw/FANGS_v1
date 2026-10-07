#!/usr/bin/env python3
"""Plot completed pairwise-structure TSV results without touching worker state.

Published runs live below ``OUTPUT/.versions/<run-id>``.  ``OUTPUT/CURRENT`` is
atomically replaced only after every artifact has been written successfully.
Consumers should resolve the relative path stored in ``CURRENT``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:  # Optional: numpy/pandas provide deterministic fallbacks below.
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover - exercised only in scipy-free installs
    scipy_stats = None

DEFAULT_X_COLUMN = "ConnectorDistance"
DEFAULT_METRIC = "usalign_global_rmsd"
METRIC_SUFFIXES = ("_ca_rmsd", "_backbone_rmsd")
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(name: str) -> str:
    cleaned = SAFE_NAME_RE.sub("_", str(name)).strip("._")
    return cleaned or "unnamed"


def _normalise_formats(formats: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(formats, str):
        values = re.split(r"[\s,]+", formats)
    else:
        values = list(formats)
    result: list[str] = []
    for value in values:
        fmt = str(value).lower().lstrip(".").strip()
        if fmt and fmt not in result:
            result.append(fmt)
    if not result:
        raise ValueError("At least one output format is required")
    return tuple(result)


def _normalise_metrics(metrics: None | str | Sequence[str], columns: Iterable[str]) -> list[str]:
    column_list = list(columns)
    if metrics is None or metrics == "":
        result = []
        if DEFAULT_METRIC in column_list:
            result.append(DEFAULT_METRIC)
        for column in column_list:
            if column.endswith(METRIC_SUFFIXES) and column not in result:
                result.append(column)
        return result
    if isinstance(metrics, str):
        values = re.split(r"[\s,]+", metrics)
    else:
        values = list(metrics)
    result = []
    for value in values:
        metric = str(value).strip()
        if metric and metric not in result:
            result.append(metric)
    return result


def _critical_value(ci: float, degrees_freedom: int) -> float:
    alpha = 1.0 - ci
    if scipy_stats is not None:
        return float(scipy_stats.t.ppf(1.0 - alpha / 2.0, degrees_freedom))
    return float(NormalDist().inv_cdf(1.0 - alpha / 2.0))


def _correlations(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float | None, float | None]:
    if scipy_stats is not None:
        pearson = scipy_stats.pearsonr(x, y)
        spearman = scipy_stats.spearmanr(x, y)
        return float(pearson.statistic), float(spearman.statistic), float(pearson.pvalue), float(spearman.pvalue)
    pearson_r = float(np.corrcoef(x, y)[0, 1])
    x_rank = pd.Series(x).rank(method="average").to_numpy()
    y_rank = pd.Series(y).rank(method="average").to_numpy()
    spearman_r = float(np.corrcoef(x_rank, y_rank)[0, 1])
    return pearson_r, spearman_r, None, None


def _fit_with_mean_ci(
    x: np.ndarray, y: np.ndarray, ci: float
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    slope, intercept = np.polyfit(x, y, 1)
    grid = np.linspace(float(x.min()), float(x.max()), 200)
    fitted = intercept + slope * grid
    fitted_observed = intercept + slope * x
    residual_sse = float(np.sum((y - fitted_observed) ** 2))
    residual_variance = max(residual_sse / (len(x) - 2), 0.0)
    x_mean = float(np.mean(x))
    sxx = float(np.sum((x - x_mean) ** 2))
    standard_error = np.sqrt(
        residual_variance * (1.0 / len(x) + ((grid - x_mean) ** 2) / sxx)
    )
    margin = _critical_value(ci, len(x) - 2) * standard_error
    return float(slope), float(intercept), grid, fitted - margin, fitted + margin


@contextmanager
def _plot_lock(output_dir: Path, timeout: float):
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".plot.lock"
    deadline = time.monotonic() + max(0.0, timeout)
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Plot lock is already held: {lock_path}")
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    try:
        payload = f"pid={os.getpid()}\ncreated={datetime.now(timezone.utc).isoformat()}\n"
        os.write(fd, payload.encode("utf-8"))
        os.close(fd)
        fd = None
        yield
    finally:
        if fd is not None:
            os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _save_figure(fig: plt.Figure, staging: Path, stem: str, formats: Sequence[str], dpi: int) -> list[str]:
    files = []
    for fmt in formats:
        filename = f"{_safe_name(stem)}.{fmt}"
        fig.savefig(staging / filename, dpi=dpi, bbox_inches="tight", format=fmt)
        files.append(filename)
    plt.close(fig)
    return files


def _blank_axis(ax: plt.Axes, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def _analyse_metric(
    df: pd.DataFrame,
    metric: str,
    x_column: str,
    rmsd_max: float | None,
    ci: float,
    staging: Path,
    formats: Sequence[str],
    dpi: int,
) -> tuple[dict[str, Any], list[str], np.ndarray]:
    row: dict[str, Any] = {
        "analysis_type": "metric",
        "metric": metric,
        "x_column": x_column,
        "rows_input": len(df),
        "rows_numeric": 0,
        "rows_after_threshold": 0,
        "rows_filtered_non_numeric": len(df),
        "rows_filtered_threshold": 0,
        "pearson_r": None,
        "pearson_p": None,
        "spearman_r": None,
        "spearman_p": None,
        "slope": None,
        "intercept": None,
        "status": "failed",
        "reason": "",
    }
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    numeric_y = np.array([], dtype=float)
    if metric not in df.columns:
        row["reason"] = "missing_metric_column"
        _blank_axis(ax, f"Missing metric column: {metric}")
    elif x_column not in df.columns:
        row["reason"] = "missing_x_column"
        _blank_axis(ax, f"Missing x column: {x_column}")
    else:
        x_values = pd.to_numeric(df[x_column], errors="coerce")
        y_values = pd.to_numeric(df[metric], errors="coerce")
        finite = np.isfinite(x_values.to_numpy(dtype=float)) & np.isfinite(y_values.to_numpy(dtype=float))
        numeric_x = x_values.to_numpy(dtype=float)[finite]
        numeric_y = y_values.to_numpy(dtype=float)[finite]
        row["rows_numeric"] = int(finite.sum())
        row["rows_filtered_non_numeric"] = int(len(df) - finite.sum())
        if rmsd_max is not None:
            threshold_mask = numeric_y <= rmsd_max
            row["rows_filtered_threshold"] = int((~threshold_mask).sum())
            numeric_x = numeric_x[threshold_mask]
            numeric_y = numeric_y[threshold_mask]
        row["rows_after_threshold"] = len(numeric_x)
        if len(numeric_x):
            ax.scatter(numeric_x, numeric_y, alpha=0.72, edgecolor="none")
        ax.set_xlabel(x_column)
        ax.set_ylabel(metric)
        ax.grid(alpha=0.2)
        if len(numeric_x) < 3:
            row["reason"] = "fewer_than_3_observations"
        elif float(np.var(numeric_x)) == 0.0:
            row["reason"] = "zero_x_variance"
        elif float(np.var(numeric_y)) == 0.0:
            row["reason"] = "zero_metric_variance"
        else:
            pearson_r, spearman_r, pearson_p, spearman_p = _correlations(numeric_x, numeric_y)
            slope, intercept, grid, lower, upper = _fit_with_mean_ci(numeric_x, numeric_y, ci)
            ax.plot(grid, intercept + slope * grid, color="tab:red", linewidth=2, label="linear fit")
            ax.fill_between(grid, lower, upper, color="tab:red", alpha=0.2, label=f"{ci:.0%} mean CI")
            ax.legend()
            row.update(
                pearson_r=pearson_r,
                pearson_p=pearson_p,
                spearman_r=spearman_r,
                spearman_p=spearman_p,
                slope=slope,
                intercept=intercept,
                status="success",
                reason="",
            )
        if row["status"] != "success":
            ax.text(0.02, 0.98, f"Statistics unavailable: {row['reason']}", va="top", transform=ax.transAxes)
    ax.set_title(f"{metric} vs {x_column}")
    files = _save_figure(fig, staging, f"scatter_{metric}_vs_{x_column}", formats, dpi)
    return row, files, numeric_y


def _distribution_plot(
    metric_values: dict[str, np.ndarray], staging: Path, formats: Sequence[str], dpi: int
) -> list[str]:
    metrics = list(metric_values)
    fig, axes = plt.subplots(max(1, len(metrics)), 1, figsize=(7.0, max(4.0, 3.2 * len(metrics))), squeeze=False)
    if not metrics:
        _blank_axis(axes[0, 0], "No RMSD metrics available")
        axes[0, 0].set_title("RMSD distributions")
    for ax, metric in zip(axes[:, 0], metrics):
        values = metric_values[metric]
        if len(values):
            bins = min(40, max(5, int(math.ceil(math.sqrt(len(values))))))
            ax.hist(values, bins=bins, color="tab:blue", alpha=0.8)
            ax.set_ylabel("count")
            ax.set_xlabel(metric)
        else:
            _blank_axis(ax, "No numeric values after filtering")
        ax.set_title(f"{metric} distribution (n={len(values)})")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("RMSD distributions")
    return _save_figure(fig, staging, "rmsd_distributions", formats, dpi)


def _status_columns(columns: Iterable[str]) -> tuple[list[str], list[str]]:
    pair, region = [], []
    for column in columns:
        normalised = re.sub(r"[^a-z0-9]+", "_", column.lower()).strip("_")
        if "status" not in normalised:
            continue
        if "pair" in normalised:
            pair.append(column)
        if "region" in normalised:
            region.append(column)
    return pair, region


def _status_plot(
    df: pd.DataFrame, staging: Path, formats: Sequence[str], dpi: int
) -> tuple[list[dict[str, Any]], list[str]]:
    pair_columns, region_columns = _status_columns(df.columns)
    selected = [("pair_status", col) for col in pair_columns] + [("region_status", col) for col in region_columns]
    rows: list[dict[str, Any]] = []
    fig, axes = plt.subplots(max(1, len(selected)), 1, figsize=(8.0, max(4.0, 3.4 * len(selected))), squeeze=False)
    if not selected:
        _blank_axis(axes[0, 0], "No pair/region status columns found")
        axes[0, 0].set_title("Pair/region status QC")
        rows.append({"analysis_type": "status_qc", "metric": "", "x_column": "", "status": "unavailable", "reason": "no_status_columns", "rows_input": len(df)})
    for ax, (kind, column) in zip(axes[:, 0], selected):
        counts = df[column].fillna("<missing>").astype(str).value_counts(dropna=False)
        if len(counts):
            labels = [str(label) for label in counts.index]
            ax.bar(range(len(counts)), counts.to_numpy(), color="tab:green", alpha=0.8)
            ax.set_xticks(range(len(counts)), labels, rotation=30, ha="right")
            ax.set_ylabel("count")
        else:
            _blank_axis(ax, "No status rows")
        ax.set_title(f"{column} ({kind})")
        for label, count in counts.items():
            rows.append({
                "analysis_type": "status_qc",
                "metric": column,
                "x_column": "",
                "status": str(label),
                "reason": "",
                "rows_input": len(df),
                "status_count": int(count),
            })
    fig.suptitle("Pair/region status QC")
    return rows, _save_figure(fig, staging, "pair_region_status_qc", formats, dpi)


def _write_tables_and_metadata(
    staging: Path,
    stats_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(staging / "stats.tsv", sep="\t", index=False)
    records = [{key: _json_value(value) for key, value in row.items()} for row in stats_rows]
    (staging / "stats.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (staging / "plot_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=_json_value) + "\n", encoding="utf-8"
    )


def plot_results(
    input_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    metrics: None | str | Sequence[str] = None,
    x_column: str = DEFAULT_X_COLUMN,
    rmsd_max: float | None = None,
    formats: str | Sequence[str] = ("png",),
    dpi: int = 150,
    ci: float = 0.95,
    plot_lock_timeout: float = 0.0,
) -> Path:
    """Plot one explicit TSV snapshot and atomically publish a complete version.

    Returns the published version directory.  The caller's input DataFrame cannot
    be mutated because this API accepts a path and reads an isolated DataFrame.
    """
    input_file = Path(input_path).expanduser().resolve(strict=True)
    destination = Path(output_dir).expanduser().resolve()
    if not input_file.is_file():
        raise ValueError(f"Input is not a file: {input_file}")
    if input_file.suffix.lower() not in {".tsv", ".txt"}:
        raise ValueError("Input must be an explicit TSV snapshot")
    if rmsd_max is not None and (not math.isfinite(rmsd_max) or rmsd_max < 0):
        raise ValueError("rmsd_max must be a finite non-negative number")
    if not (0.0 < ci < 1.0):
        raise ValueError("ci must be between 0 and 1")
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    output_formats = _normalise_formats(formats)
    input_digest = _sha256(input_file)

    with _plot_lock(destination, plot_lock_timeout):
        df = pd.read_csv(input_file, sep="\t", low_memory=False)
        selected_metrics = _normalise_metrics(metrics, df.columns)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"-{uuid.uuid4().hex[:12]}"
        staging = destination.parent / f".{destination.name}.plot-staging-{run_id}"
        staging.mkdir(mode=0o755, parents=False, exist_ok=False)
        artifact_files: list[str] = []
        stats_rows: list[dict[str, Any]] = []
        metric_values: dict[str, np.ndarray] = {}
        try:
            for metric in selected_metrics:
                row, files, values = _analyse_metric(
                    df, metric, x_column, rmsd_max, ci, staging, output_formats, dpi
                )
                stats_rows.append(row)
                artifact_files.extend(files)
                if metric in df.columns:
                    metric_values[metric] = values
            if not selected_metrics:
                stats_rows.append({
                    "analysis_type": "metric",
                    "metric": "",
                    "x_column": x_column,
                    "rows_input": len(df),
                    "status": "unavailable",
                    "reason": "no_metrics_detected",
                })
            artifact_files.extend(_distribution_plot(metric_values, staging, output_formats, dpi))
            status_rows, status_files = _status_plot(df, staging, output_formats, dpi)
            stats_rows.extend(status_rows)
            artifact_files.extend(status_files)
            metric_rows = [row for row in stats_rows if row.get("analysis_type") == "metric" and row.get("metric")]
            success_count = sum(row.get("status") == "success" for row in metric_rows)
            failure_count = len(metric_rows) - success_count
            metadata = {
                "schema_version": 1,
                "run_id": run_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "input": str(input_file),
                "input_sha256": input_digest,
                "parameters": {
                    "metrics": selected_metrics,
                    "metrics_requested": None if metrics is None else list(_normalise_metrics(metrics, df.columns)),
                    "x_column": x_column,
                    "rmsd_max": rmsd_max,
                    "formats": list(output_formats),
                    "dpi": dpi,
                    "ci": ci,
                    "plot_lock_timeout": plot_lock_timeout,
                },
                "counts": {
                    "input_rows": len(df),
                    "metrics_selected": len(selected_metrics),
                    "metric_successes": success_count,
                    "metric_failures": failure_count,
                    "rows_numeric_by_metric": {row["metric"]: row.get("rows_numeric", 0) for row in metric_rows},
                    "rows_after_filter_by_metric": {row["metric"]: row.get("rows_after_threshold", 0) for row in metric_rows},
                },
                "artifacts": sorted(artifact_files + ["stats.tsv", "stats.json", "plot_metadata.json"]),
                "scipy_available": scipy_stats is not None,
            }
            _write_tables_and_metadata(staging, stats_rows, metadata)
            versions = destination / ".versions"
            versions.mkdir(parents=True, exist_ok=True)
            published = versions / run_id
            os.replace(staging, published)
            current_temp = destination / f".CURRENT.{run_id}.tmp"
            current_temp.write_text(f".versions/{run_id}\n", encoding="utf-8")
            os.replace(current_temp, destination / "CURRENT")
            return published
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Completed final TSV or explicit TSV snapshot")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metrics", nargs="*", default=None, help="Metric columns; comma-separated values are accepted")
    parser.add_argument("--x-column", default=DEFAULT_X_COLUMN)
    parser.add_argument("--rmsd-max", type=float, default=None)
    parser.add_argument("--formats", nargs="+", default=["png"])
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--plot-lock-timeout", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    metrics = None if args.metrics is None else [part for item in args.metrics for part in item.split(",") if part]
    formats = [part for item in args.formats for part in item.split(",") if part]
    published = plot_results(
        input_path=args.input,
        output_dir=args.output_dir,
        metrics=metrics,
        x_column=args.x_column,
        rmsd_max=args.rmsd_max,
        formats=formats,
        dpi=args.dpi,
        ci=args.ci,
        plot_lock_timeout=args.plot_lock_timeout,
    )
    print(published)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
