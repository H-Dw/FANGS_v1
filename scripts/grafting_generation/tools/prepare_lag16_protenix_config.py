#!/usr/bin/env python3
"""Prepare the six archived LaG16–GFP Protenix jobs with current MSA paths."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

TARGETS = ["6lr7B", "6lr7B_7nowA", "6lr7B_4dkaA", "6lr7B_8taoC", "6lr7B_3k1kC", "6lr7B_1zmyA"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    root = args.repository_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name in TARGETS:
        source = args.config_dir/f"{name}-update-msa.json"
        payload = json.loads(source.read_text())
        if len(payload) != 1 or payload[0]["name"] != name:
            raise ValueError(f"Unexpected target in {source}")
        for record in payload:
            for entity in record["sequences"]:
                chain = entity["proteinChain"]
                for key in ["pairedMsaPath", "unpairedMsaPath"]:
                    old = chain.get(key, "")
                    if not old:
                        continue
                    if "/final_version/" in old:
                        relative = old.split("/final_version/", 1)[1]
                        path = root/relative
                    else:
                        path = Path(old)
                        if not path.is_absolute(): path = root/path
                    path = path.resolve()
                    if not path.is_relative_to(root) or not path.is_file():
                        raise FileNotFoundError(path)
                    chain[key] = str(path)
        output = args.output_dir/f"{name}.json"
        output.write_text(json.dumps(payload, indent=2)+"\n")
        records.append({"target": name, "input": str(source),
                        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "output": str(output), "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
    (args.output_dir/"preparation_manifest.tsv").write_text(
        "target\tinput\tinput_sha256\toutput\toutput_sha256\n"+
        "".join("\t".join(x[k] for k in ["target", "input", "input_sha256", "output", "output_sha256"])+"\n" for x in records)
    )
    print(f"Prepared {len(records)} LaG16–GFP Protenix jobs in {args.output_dir}")


if __name__ == "__main__":
    main()
