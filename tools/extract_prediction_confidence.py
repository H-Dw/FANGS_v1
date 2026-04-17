#!/usr/bin/env python3
"""
Collect ipTM, pTM, pLDDT from structure-prediction JSON outputs and write:
  - Per-sample TSV (all predictions)
  - Per-target TSV (best value per metric across samples)

Built-in presets: protenix, rf3. Custom layout via --config JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator


@dataclass(frozen=True)
class ModelPreset:
    """How to find JSON files under root/<target>/... and which JSON keys to read."""

    name: str
    # Given root, yield (json_path, meta) where meta includes at least target, seed, sample
    discover: Callable[[Path], Iterator[tuple[Path, dict[str, Any]]]]
    json_keys: dict[str, str]  # canonical -> key in JSON (iptm, ptm, plddt)
    plddt_scale: float = 1.0  # multiply raw pLDDT after read (e.g. 100.0 if JSON is 0–1)


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _discover_protenix(root: Path) -> Iterator[tuple[Path, dict[str, Any]]]:
    """
    .../<target>/seed_<seed>/predictions/<target>_summary_confidence_sample_<n>.json
    """
    pat = re.compile(r"^(?P<target>.+)_summary_confidence_sample_(?P<sample>\d+)\.json$")
    seed_pat = re.compile(r"seed_(?P<seed>\d+)")
    if not root.is_dir():
        return
    for target_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        target = target_dir.name
        for pred in target_dir.glob("seed_*/predictions"):
            m_seed = seed_pat.search(pred.parts[-2])
            seed = m_seed.group("seed") if m_seed else ""
            for jf in sorted(pred.glob(f"{re.escape(target)}_summary_confidence_sample_*.json")):
                m = pat.match(jf.name)
                if not m or m.group("target") != target:
                    continue
                yield jf, {
                    "target": target,
                    "seed": seed,
                    "sample": m.group("sample"),
                }


def _discover_rf3(root: Path) -> Iterator[tuple[Path, dict[str, Any]]]:
    """
    .../<target>/seed-<s>_sample-<n>/<target>_seed-<s>_sample-<n>_summary_confidences.json
    Skip aggregate files directly under <target>/ if present.
    """
    dir_pat = re.compile(r"^seed-(?P<seed>\d+)_sample-(?P<sample>\d+)$")
    file_pat = re.compile(
        r"^(?P<target>.+)_seed-(?P<seed>\d+)_sample-(?P<sample>\d+)_summary_confidences\.json$"
    )
    if not root.is_dir():
        return
    for target_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        target = target_dir.name
        for sub in sorted(target_dir.iterdir()):
            if not sub.is_dir():
                continue
            md = dir_pat.match(sub.name)
            if not md:
                continue
            for jf in sub.glob(f"{re.escape(target)}_seed-*_sample-*_summary_confidences.json"):
                mf = file_pat.match(jf.name)
                if not mf or mf.group("target") != target:
                    continue
                yield jf, {
                    "target": target,
                    "seed": mf.group("seed"),
                    "sample": mf.group("sample"),
                }


PRESETS: dict[str, ModelPreset] = {
    "protenix": ModelPreset(
        name="protenix",
        discover=_discover_protenix,
        json_keys={"iptm": "iptm", "ptm": "ptm", "plddt": "plddt"},
        plddt_scale=1.0,
    ),
    "rf3": ModelPreset(
        name="rf3",
        discover=_discover_rf3,
        json_keys={"iptm": "iptm", "ptm": "ptm", "plddt": "overall_plddt"},
        plddt_scale=1.0,
    ),
}


def load_custom_config(path: Path) -> ModelPreset:
    """
    Load preset from JSON.

    Either set \"base_preset\": \"protenix\" | \"rf3\" and optionally override
    \"name\", \"json_keys\", \"plddt_scale\"; or set \"discover_kind\" + \"name\" + \"json_keys\".
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    base = data.get("base_preset")
    if base:
        if base not in PRESETS:
            raise ValueError(f"Unknown base_preset {base!r}, expected one of {sorted(PRESETS)}")
        p0 = PRESETS[base]
        name = data.get("name", p0.name)
        json_keys = {**p0.json_keys, **data.get("json_keys", {})}
        plddt_scale = float(data.get("plddt_scale", p0.plddt_scale))
        discover = p0.discover
        return ModelPreset(name=name, discover=discover, json_keys=json_keys, plddt_scale=plddt_scale)

    name = data["name"]
    json_keys = data["json_keys"]
    plddt_scale = float(data.get("plddt_scale", 1.0))
    kind = data.get("discover_kind", "protenix")
    if kind == "protenix":
        discover = _discover_protenix
    elif kind == "rf3":
        discover = _discover_rf3
    else:
        raise ValueError(f"Unknown discover_kind: {kind}")
    return ModelPreset(name=name, discover=discover, json_keys=json_keys, plddt_scale=plddt_scale)


def read_metrics(path: Path, preset: ModelPreset) -> dict[str, float | None]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, float | None] = {}
    for canon, jkey in preset.json_keys.items():
        out[canon] = _safe_float(obj.get(jkey))
    if out.get("plddt") is not None:
        out["plddt"] = out["plddt"] * preset.plddt_scale
    return out


def collect_rows(root: Path, preset: ModelPreset) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for jpath, meta in preset.discover(root):
        m = read_metrics(jpath, preset)
        try:
            rel = jpath.relative_to(root)
        except ValueError:
            rel = jpath
        rows.append(
            {
                **meta,
                "iptm": m.get("iptm"),
                "ptm": m.get("ptm"),
                "plddt": m.get("plddt"),
                "json_path": str(rel),
            }
        )
    rows.sort(key=lambda r: (r["target"], int(r["seed"]) if str(r["seed"]).isdigit() else r["seed"], int(r["sample"]) if str(r["sample"]).isdigit() else r["sample"]))
    return rows


def best_per_target(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For each target, take max iptm, ptm, plddt independently; record argmax sample/seed."""

    def key_num(x: Any) -> float:
        if x is None:
            return float("-inf")
        try:
            return float(x)
        except (TypeError, ValueError):
            return float("-inf")

    by_target: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_target.setdefault(r["target"], []).append(r)

    bests: list[dict[str, Any]] = []
    for target in sorted(by_target):
        rs = by_target[target]
        best: dict[str, Any] = {"target": target}

        for metric in ("iptm", "ptm", "plddt"):
            top = max(rs, key=lambda r: key_num(r.get(metric)))
            v = top.get(metric)
            best[f"best_{metric}"] = v if v is not None else ""
            best[f"best_{metric}_seed"] = top.get("seed", "")
            best[f"best_{metric}_sample"] = top.get("sample", "")
            best[f"best_{metric}_json_path"] = top.get("json_path", "")
        bests.append(best)
    return bests


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r[k]) for k in fieldnames})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Model output root (contains one subfolder per target, e.g. protenix_output/)",
    )
    ap.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        help="Built-in discovery + JSON key mapping (protenix or rf3)",
    )
    ap.add_argument(
        "--config",
        type=Path,
        help="JSON config: use base_preset (protenix|rf3) + optional json_keys, or full discover_kind + name + json_keys",
    )
    ap.add_argument(
        "--plddt-scale",
        type=float,
        default=None,
        help="Override preset: multiply pLDDT after reading (e.g. 100 if JSON is 0–1)",
    )
    ap.add_argument(
        "-o",
        "--out-tsv",
        type=Path,
        required=True,
        help="Output TSV path for all samples",
    )
    ap.add_argument(
        "--out-best-tsv",
        type=Path,
        required=True,
        help="Output TSV path for per-target bests",
    )
    args = ap.parse_args()

    if args.preset and args.config:
        ap.error("Use either --preset or --config, not both")
    if not args.preset and not args.config:
        ap.error("Specify --preset or --config")

    if args.preset:
        preset = PRESETS[args.preset]
    else:
        preset = load_custom_config(args.config)

    if args.plddt_scale is not None:
        preset = ModelPreset(
            name=preset.name,
            discover=preset.discover,
            json_keys=preset.json_keys,
            plddt_scale=args.plddt_scale,
        )

    rows = collect_rows(args.root, preset)
    if not rows:
        print(f"No JSON files found under {args.root} (preset={preset.name})", file=sys.stderr)

    sample_fields = ["target", "seed", "sample", "iptm", "ptm", "plddt", "json_path"]
    write_tsv(args.out_tsv, sample_fields, rows)

    best_rows = best_per_target(rows)
    best_fields = [
        "target",
        "best_iptm",
        "best_iptm_seed",
        "best_iptm_sample",
        "best_iptm_json_path",
        "best_ptm",
        "best_ptm_seed",
        "best_ptm_sample",
        "best_ptm_json_path",
        "best_plddt",
        "best_plddt_seed",
        "best_plddt_sample",
        "best_plddt_json_path",
    ]
    write_tsv(args.out_best_tsv, best_fields, best_rows)

    print(f"Wrote {len(rows)} rows -> {args.out_tsv}")
    print(f"Wrote {len(best_rows)} targets -> {args.out_best_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
