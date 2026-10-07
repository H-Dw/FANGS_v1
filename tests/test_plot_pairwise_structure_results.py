import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "plot_pairwise_structure_results.py"
SPEC = importlib.util.spec_from_file_location("plot_pairwise_structure_results", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_tsv(path: Path, **columns) -> pd.DataFrame:
    frame = pd.DataFrame(columns)
    frame.to_csv(path, sep="\t", index=False)
    return frame


def _current(output: Path) -> Path:
    return output / (output / "CURRENT").read_text().strip()


def test_statistics_auto_metrics_and_input_unchanged(tmp_path):
    source = tmp_path / "final.tsv"
    original = _write_tsv(
        source,
        ConnectorDistance=[1, 2, 3, 4, 5],
        usalign_global_rmsd=[2, 4, 6, 8, 10],
        loop_ca_rmsd=[5, 4, 3, 2, 1],
        loop_backbone_rmsd=[1, 1.5, 2, 2.5, 3],
        unrelated=[9, 8, 7, 6, 5],
        pair_status=["ok", "ok", "failed", "ok", "ok"],
        region_status=["complete", "complete", "partial", "complete", "complete"],
    )
    before = source.read_bytes()

    published = MODULE.plot_results(source, tmp_path / "plots", formats=["png"], dpi=60)

    assert source.read_bytes() == before
    assert pd.read_csv(source, sep="\t").equals(original)
    stats = pd.read_csv(published / "stats.tsv", sep="\t")
    metric_stats = stats[stats.analysis_type == "metric"].set_index("metric")
    assert set(metric_stats.index) == {
        "usalign_global_rmsd",
        "loop_ca_rmsd",
        "loop_backbone_rmsd",
    }
    assert metric_stats.loc["usalign_global_rmsd", "pearson_r"] == pytest.approx(1.0)
    assert metric_stats.loc["loop_ca_rmsd", "spearman_r"] == pytest.approx(-1.0)
    assert (published / "rmsd_distributions.png").is_file()
    assert (published / "pair_region_status_qc.png").is_file()
    metadata = json.loads((published / "plot_metadata.json").read_text())
    assert metadata["input_sha256"] == hashlib.sha256(before).hexdigest()
    assert metadata["counts"]["metric_successes"] == 3
    assert metadata["counts"]["metric_failures"] == 0


def test_threshold_filters_metric_values(tmp_path):
    source = tmp_path / "snapshot.tsv"
    _write_tsv(
        source,
        ConnectorDistance=[1, 2, 3, 4, 5, "bad"],
        usalign_global_rmsd=[1, 2, 3, 50, 60, 2],
    )
    published = MODULE.plot_results(source, tmp_path / "plots", rmsd_max=10, dpi=60)
    stats = pd.read_csv(published / "stats.tsv", sep="\t")
    row = stats[(stats.analysis_type == "metric") & (stats.metric == "usalign_global_rmsd")].iloc[0]
    assert row.rows_input == 6
    assert row.rows_numeric == 5
    assert row.rows_filtered_non_numeric == 1
    assert row.rows_filtered_threshold == 2
    assert row.rows_after_threshold == 3
    assert row.status == "success"


@pytest.mark.parametrize(
    ("x_values", "y_values", "reason"),
    [
        ([1, 2], [2, 3], "fewer_than_3_observations"),
        ([1, 1, 1], [2, 3, 4], "zero_x_variance"),
        ([1, 2, 3], [4, 4, 4], "zero_metric_variance"),
    ],
)
def test_small_and_constant_samples_record_reason(tmp_path, x_values, y_values, reason):
    source = tmp_path / f"{reason}.tsv"
    _write_tsv(source, ConnectorDistance=x_values, usalign_global_rmsd=y_values)
    published = MODULE.plot_results(source, tmp_path / f"plots-{reason}", dpi=60)
    stats = pd.read_csv(published / "stats.tsv", sep="\t")
    row = stats[(stats.analysis_type == "metric") & (stats.metric == "usalign_global_rmsd")].iloc[0]
    assert row.status == "failed"
    assert row.reason == reason
    assert (published / "scatter_usalign_global_rmsd_vs_ConnectorDistance.png").is_file()


def test_empty_input_and_missing_status_do_not_crash(tmp_path):
    source = tmp_path / "empty.tsv"
    _write_tsv(source, ConnectorDistance=pd.Series(dtype=float), usalign_global_rmsd=pd.Series(dtype=float))
    published = MODULE.plot_results(source, tmp_path / "plots", dpi=60)
    stats = json.loads((published / "stats.json").read_text())
    assert any(row.get("reason") == "fewer_than_3_observations" for row in stats)
    assert any(row.get("reason") == "no_status_columns" for row in stats)


def test_atomic_current_publish_retains_old_versions(tmp_path):
    source = tmp_path / "final.tsv"
    _write_tsv(source, ConnectorDistance=[1, 2, 3], usalign_global_rmsd=[2, 3, 5])
    output = tmp_path / "plots"
    marker = output / "user-owned.txt"
    output.mkdir()
    marker.write_text("keep")

    first = MODULE.plot_results(source, output, dpi=60)
    first_current = _current(output)
    _write_tsv(source, ConnectorDistance=[1, 2, 3, 4], usalign_global_rmsd=[2, 3, 5, 8])
    second = MODULE.plot_results(source, output, dpi=60)

    assert first == first_current
    assert first.is_dir() and second.is_dir() and first != second
    assert _current(output) == second
    assert marker.read_text() == "keep"
    assert len(list((output / ".versions").iterdir())) == 2
    assert not list(tmp_path.glob(".plots.plot-staging-*"))


def test_lock_conflict_leaves_current_unchanged(tmp_path):
    source = tmp_path / "final.tsv"
    _write_tsv(source, ConnectorDistance=[1, 2, 3], usalign_global_rmsd=[1, 2, 4])
    output = tmp_path / "plots"
    first = MODULE.plot_results(source, output, dpi=60)
    current_before = (output / "CURRENT").read_bytes()
    lock = output / ".plot.lock"
    lock.write_text("held")

    with pytest.raises(TimeoutError, match="already held"):
        MODULE.plot_results(source, output, dpi=60, plot_lock_timeout=0)

    assert (output / "CURRENT").read_bytes() == current_before
    assert _current(output) == first
    assert lock.read_text() == "held"
