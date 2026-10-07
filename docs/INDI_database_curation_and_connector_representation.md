# INDI Database Curation and Connector Representation

Source data, preprocessing commands, connector encoding, and the resolved-structure analysis used in Fig. 1f of the current manuscript.

## 1. Computational requirements

All paths are relative to the repository root; execute commands from that directory. Connector encoding uses the `fangs` Conda environment and the repository-local `esm/` package. Pairwise analysis requires Python 3.10 or later, NumPy, pandas, SciPy, Matplotlib, and `USalign` on the shell search path.

## 2. Source data

`PDBChain` identifies the PDB accession and nanobody chain. The archived resources are:

| Resource | Repository-relative path | Content |
| --- | --- | --- |
| PDB accession list | `data/INDI_database/INDI_info_full_pdb_id.txt` | 1,587 unique accessions |
| Full CDR annotations | `data/INDI_database/INDI_info_full_cdr_info.tsv` | 3,022 records; sequence, CDR1–CDR3, and `PDBChain` |
| Deduplicated annotations | `data/INDI_database/INDI_info_full_cdr_info_dedup.tsv` | 946 records |
| Curated annotations | `data/INDI_database/INDI_info_full_cdr_info_dedup_clean.tsv` | 841 representative chains |
| Curated ID manifest | `data/INDI_database/dupl_INDI_passed_cleaned.tsv` | 841 identifiers in `pdb_id` |
| Chain-specific coordinate input | `data/INDI_database/pdb_model0/` | 2,767 PDB files |
| Encoded native structures | `data/INDI_database/passed_pdbs/` | 2,767 PDB files used in pairwise analysis |

The full structural library and the curated representative cohort have different membership and file counts.

## 3. INDI curation and connector encoding

### 3.1. CIF retrieval

**Script:** [scripts/preprocess/CIFdownload.sh](../scripts/preprocess/CIFdownload.sh). It downloads the listed CIF structures using `axel`; supply a destination directory as the second argument.

```bash
conda activate fangs
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
bash "scripts/preprocess/CIFdownload.sh" \
  "data/INDI_database/INDI_info_full_pdb_id.txt" \
  <Path to store cif files>
```

**Archived CIF collection:** `data/INDI_database/cif.tar.gz`. The chain-specific PDB input is the existing `data/INDI_database/pdb_model0/` library; CIF-to-chain conversion is a separate preprocessing step.

### 3.2. Annotation deduplication

**Script:** [scripts/preprocess/dedup_INDI_info.py](../scripts/preprocess/dedup_INDI_info.py). It reads CSV input, groups records by `(Sequence, CDR1, CDR2, CDR3)`, and writes a TSV with the first representative and a comma-separated `Member` field.

**Output:** `data/INDI_database/INDI_info_full_cdr_info_dedup.tsv`. The curated table `data/INDI_database/INDI_info_full_cdr_info_dedup_clean.tsv` follows selection by the frozen ID manifest.

### 3.3. Connector encoding

**Tokenizer:** [scripts/grafting_generation/tokenizer_INDI.py](../scripts/grafting_generation/tokenizer_INDI.py).  
**Custom method source:** [scripts/esm3_custom.py](../scripts/esm3_custom.py), containing `ESM3.structure_encode_full`.  
**Structure encoder source:** [scripts/vqvae_custom.py](../scripts/vqvae_custom.py), containing `StructureTokenEncoder.encode_full`.

The two custom source files are exact copies of `esm/models/esm3.py` and `esm/models/vqvae.py`. Runtime imports require the repository-local `esm/` package, selected through `PYTHONPATH` below.

```bash
conda activate fangs
```

```bash
PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" python "scripts/grafting_generation/tokenizer_INDI.py" \
  "data/INDI_database/INDI_info_full_cdr_info.tsv" \
  "data/INDI_database/pdb_model0/" \
  "data/INDI_database/tokenized_INDI_full_250612_2C" \
  --connector_len 2
```

The tokenizer encodes the observed chain and extracts two framework residues on each side of each CDR. A complete three-CDR connector vector contains 12 residue representations in CDR1–CDR2–CDR3 order. The standalone entry point loads `esm3_sm_open_v1` on the CPU.

**Manuscript representation:** `data/INDI_database/raw_embeddings.tsv`, with an `ID` column and 12,288 RAW feature columns (12 residues × 1,024 channels). Regenerated outputs are written to `data/INDI_database/tokenized_INDI_full_250612_2C/`.

## 4. Resolved structure pairs with identical CDR sequences

### 4.1. Data and figure

This analysis supplies **Fig. 1f**: connector distance versus framework-aligned combined/individual CDR RMSD and globally aligned whole-chain RMSD. The current archive contains 37 identical-CDR groups, 122 candidate pairs, 118 successful upstream calculations, and **115 final pairs from 34 groups**.

| Analysis input | Repository-relative path |
| --- | --- |
| Selected annotations | `data/INDI_database/analysis/INDI_info_selected_cdr_full_info.tsv` |
| Identical-CDR groups | `data/INDI_database/analysis/cdr_grouped_output.tsv` |
| Detailed pair table | `data/INDI_database/analysis/cdr_region_pair_comparison.tsv` |

The native coordinates and connector vectors are `data/INDI_database/passed_pdbs/` and `data/INDI_database/raw_embeddings.tsv`, respectively.

### 4.2. Selected annotations and CDR grouping

**Script:** [scripts/preprocess/group_resolved_indi_cdrs.py](../scripts/preprocess/group_resolved_indi_cdrs.py). This reproduction entry point reconstructs the stored annotations and groups. It retains one chain per PDB accession and groups by identical CDR1, CDR2, and CDR3 sequences, retaining groups with at least two members.

```bash
python "scripts/preprocess/group_resolved_indi_cdrs.py" \
  --annotations "data/INDI_database/INDI_info_full_cdr_info.tsv" \
  --manifest "data/INDI_database/dupl_INDI_passed_cleaned.tsv" \
  --selected-out "data/INDI_database/analysis/INDI_info_selected_cdr_full_info.tsv" \
  --groups-out "data/INDI_database/analysis/cdr_grouped_output.tsv"
```

**Outputs:** the selected-annotation and group tables listed in Section 4.1. The stored cohort comprises 841 selected chains and 779 accession representatives.

### 4.3. Pairwise structural comparison

**Script:** [scripts/analyze_pairwise_structures.py](../scripts/analyze_pairwise_structures.py). It enumerates within-group pairs, excludes identical full-chain sequences, and calculates whole-chain US-align and regional CDR metrics. Connector distance uses the full RAW vector, normalized by `sqrt(12)`; `--embedding-duplicate-policy last` resolves repeated representation IDs.

```bash
command -v USalign

python "scripts/analyze_pairwise_structures.py" \
  --groups "data/INDI_database/analysis/cdr_grouped_output.tsv" \
  --metadata-tsv "data/INDI_database/analysis/INDI_info_selected_cdr_full_info.tsv" \
  --metadata-id-col PDBChain \
  --target-region-cols CDR1 CDR2 CDR3 \
  --pdb-dir "data/INDI_database/passed_pdbs/" \
  --usalign-path "$(command -v USalign)" \
  --emb "data/INDI_database/raw_embeddings.tsv" \
  --embedding-id-col ID \
  --embedding-duplicate-policy last \
  --connector 12 \
  --alignment-mode whole-chain-usalign \
  --out "data/INDI_database/analysis/cdr_region_pair_comparison.tsv" \
  --cache-dir "data/INDI_database/analysis/pairwise_cache/" \
  --run-id cdr-region-analysis \
  --workers 8 \
  --max-in-flight 8
```

**Outputs:** `data/INDI_database/analysis/cdr_region_pair_comparison.tsv` and alignment/checkpoint records in `data/INDI_database/analysis/pairwise_cache/`.

### 4.4. Framework alignment and Fig. 1f statistics

**Script:** [analysis/revision_20260831/analyze_resolved_cdr_variation.py](../analysis/revision_20260831/analyze_resolved_cdr_variation.py).

The script fits a Kabsch transform to sequence-aligned framework Cα atoms and applies it to the CDR coordinates. Final correlations use the 115 retained pairs, with 10,000 group-bootstrap resamples and group-level sign-flip randomizations, followed by Benjamini–Hochberg correction across six measures.

```bash
python "analysis/revision_20260831/analyze_resolved_cdr_variation.py" \
  --pairs "data/INDI_database/analysis/cdr_region_pair_comparison.tsv" \
  --groups "data/INDI_database/analysis/cdr_grouped_output.tsv" \
  --output-dir "data/INDI_database/analysis/" \
  --min-fr-coverage 0.80 \
  --n-resamples 10000 \
  --seed 20260831
```

### 4.5. Results

| Result | Existing repository-relative path |
| --- | --- |
| Per-pair FR-aligned measurements and QC | `data/INDI_database/analysis/resolved_cdr_pairwise_fr_aligned.tsv` |
| Correlation statistics | `data/INDI_database/analysis/resolved_cdr_correlation_stats.tsv` |
| Group-level ranges | `data/INDI_database/analysis/resolved_cdr_group_ranges.tsv` |
| Analysis parameters and cohort counts | `data/INDI_database/analysis/resolved_cdr_analysis_metadata.json` |
| Archived Fig. 1f exports | `figures/revision_20260831/resolved/Fig1d_resolved_CDR_variation_spearman.png`, `figures/revision_20260831/resolved/Fig1d_resolved_CDR_variation_spearman.svg`, and `figures/revision_20260831/resolved/Fig1d_resolved_CDR_variation_spearman.pdf` |

The figure basename retains the historical `Fig1d` numbering and corresponds to Fig. 1f in the current manuscript. The command in Section 4.4 writes regenerated tables and image exports to `data/INDI_database/analysis/`.

## 5. Cohort accounting

The available annotation archive contains 946 deduplicated records and 841 curated representatives; the manuscript reports 947 representatives and 843 benchmark structures. These counts should remain distinguished. The paired 841-target ESM3–AF3 benchmark is documented separately in [ESM3 and AF3 template-conditioned reconstruction](ESM3_and_AF3_template_conditioned_reconstruction.md).
