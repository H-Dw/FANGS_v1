# Nanobody grafting and binding experiments

This directory is the file set for one Zenodo record. It contains nine archives (22,188,979,971 bytes; 22.19 GB). Together with this README, the record has 10 files and remains within the default Zenodo limits of 50 GB and 100 files per record.

Upload the archives as separate files of a single record. A further outer archive is unnecessary: these files are already compressed, and separate files allow each experiment to be downloaded on its own. The 149.8 GB full `batch_graft_generation_id70_table.tar.gz` is intentionally omitted. The reduced table in `batch_graft_generation_id70_table_simple.tar.zst` is the deposited form of that experiment.

## Contents

| Archive | Size | Experiment |
|---|---:|---|
| `6lr7B_grafting_connector_all_temp07_s50.tar.gz` | 11.80 GB | Full connector grafting on the 6lr7 chain B framework |
| `batch_graft_generation_id70_table_simple.tar.zst` | 6.74 GB | Grafting across 185 frameworks from the 70% identity set |
| `samples_evaluation.tar.gz` | 891 MB | Sampling-depth comparison for 6lr7B connector candidates |
| `humanization.tar.gz` | 795 MB | Humanization of the 5m2j VHH on human IGHV3 genes |
| `INDI_database.tar.gz` | 1.21 GB | Processed INDI nanobody reference set |
| `ESM3-Template_validation.tar.gz` | 399 MB | ESM3 versus AlphaFold 3 template comparison |
| `connector_screening.tar.gz` | 175 MB | CDR3 and CDR1/CDR2 connector screening |
| `selected_candidates_lag16.tar.gz` | 170 MB | Selected LaG16 designs and complex predictions |
| `BLI.tar.gz` | 3.4 MB | Processed biolayer interferometry binding curves |

## How to extract the archives

GNU tar and zstd are required. From this directory:

```bash
tar -xzf ARCHIVE.tar.gz
```

The identity-70 table uses Zstandard rather than gzip:

```bash
tar -I zstd -xf batch_graft_generation_id70_table_simple.tar.zst
```

If `tar -I zstd` is unavailable:

```bash
zstd -dc batch_graft_generation_id70_table_simple.tar.zst | tar -x
```

Two archives contain a second compressed member. Those inner archives are described under the corresponding experiments below.

## Experiments

### `6lr7B_grafting_connector_all_temp07_s50.tar.gz`

Full connector-grafting generation for PDB 6lr7, chain B, at sampling temperature 0.7 with 50 samples. Extraction creates `6lr7B_grafting_connector_all_temp07_s50_250904/`.

The archive contains the generated coordinates and sequences (`temp_generation`), side-chain models, tokenized structural embeddings, Cα-distance and embedding-similarity tables, and sequence searches against the INDI set (MMseqs2 and Foldseek). `properties/` reports hydrophobicity, charge, and isoelectric point, including values at pH 7.4. `selected_candidates/` contains the shortlisted grafted models (1zmyA, 3k1kC, 4dkaA, 7nowA, and 8taoC) and their sequence alignment. `analysis/` contains the LaG16 framework ranking, sampling-depth comparison, normalized confidence scores, affinity–yield summary, and sequence-difference matrix used for the corresponding figures.

```bash
tar -xzf 6lr7B_grafting_connector_all_temp07_s50.tar.gz
```

### `batch_graft_generation_id70_table_simple.tar.zst`

Reduced results of the same grafting procedure applied to 185 representative nanobody structures from the 70% sequence-identity clustering. The archive root is `batch_graft_generation_id70_table_simple/`.

For each acceptor, named by PDB code and chain (for example `1i3vA`), the archive keeps:

- the acceptor coordinate file `{id}.pdb` and its annotation table `{id}.tsv`;
- `extract_distance/`, including the per-target generation-distribution figure and the filtered and full distance and embedding tables;
- `similarity/raw_embeddings_similarity_*.tsv` and `generation_tokenized_structure/raw_embeddings_similarity_*.tsv`;
- FASTA and TSV files in `temp_generation/`, including `all_info/all_generation.tsv`;
- generated backbone coordinates for all designs, stored as `temp_generation/all_info/pdb.tar.zst` rather than as individual PDB files.

`analysis/tables/` contains the merged generation table and the cohort-level summary tables.

Raw embedding matrices, query embeddings, Cα-distance matrices, rejected designs, and run logs are not included.

```bash
tar -I zstd -xf batch_graft_generation_id70_table_simple.tar.zst
```

Restore the generated PDB files for one acceptor, here `1i3vA`:

```bash
cd batch_graft_generation_id70_table_simple/1i3vA/temp_generation/all_info
mkdir -p pdb
zstd -dc pdb.tar.zst | tar -x -C pdb
```

Repeat the last two commands in `temp_generation/all_info` of any other acceptor. Each `pdb.tar.zst` expands to on the order of 9,000 PDB files.

### `samples_evaluation.tar.gz`

Sampling-depth series for connector-candidate grafting on 6lr7 chain B at temperature 0.7. Independent runs use 5, 10, 30, 50, 80, 100, and 500 samples. Each run directory contains the generation output, similarity tables, distance tables, and side-chain models for that sample size. `carrier_structures/` contains the acceptor PDB files used in the comparison, with `source_manifest.json` recording their origin. Per-run logs are at the archive root.

```bash
tar -xzf samples_evaluation.tar.gz
```

### `humanization.tar.gz`

Humanization of the 5m2j VHH. The archive contains the parental VHH sequence, human IGHV3 amino-acid alleles, CDR3–FR4 grafts onto those alleles (Kabat annotation and the 74A/94R substitutions), AbNatiV humanization output, and Boltz-2 confidence scores for the grafted sequences. `5m2jD_grafting_74A_94R_temp07_s50_250917/` is the corresponding grafting run at temperature 0.7 with 50 samples, including generated coordinates, embeddings, distance tables, and analysis.

```bash
tar -xzf humanization.tar.gz
```

### `INDI_database.tar.gz`

Processed reference set derived from the INDI nanobody collection. It contains CDR annotations (`INDI_info_full_cdr_info.tsv` and its deduplicated tables), sequences (`pdb2seq.fasta`), structural embeddings (`raw_embeddings.tsv`, `q_embeddings.tsv`, `pre_q_embeddings.tsv`), structure tokens, and Cα distances. `mmseqs_cluster_structure/` contains MMseqs2 clusters at 40%, 50%, 60%, 70%, 80%, and 90% sequence identity. `pdb_model0/` contains cleaned coordinate models, and `analysis/` contains the resolved-CDR comparison.

mmCIF coordinates are stored as a nested archive, `INDI_database/cif.tar.gz`.

```bash
tar -xzf INDI_database.tar.gz
tar -xzf INDI_database/cif.tar.gz -C INDI_database
```

### `ESM3-Template_validation.tar.gz`

Comparison of ESM3 and AlphaFold 3 structural templates on a matched cohort of 841 observed chains. `esm3_vs_af3_observed_matched841_20261002/` contains the paired RMSD and CDR-threshold statistics, significance tests, cohort-exclusion tables, and the corresponding figures. `af3_template/` contains the AlphaFold 3 template mmCIF files used for chains in the observed set. `all_info/` contains generation scores and coordinate models, and `reproduction_20261007/` records the reproduction run.

```bash
tar -xzf ESM3-Template_validation.tar.gz
```

### `connector_screening.tar.gz`

Connector screening on two acceptors. `8v13HL_CDR3_s50/` is a CDR3 grafting run on the 8v13 heavy–light framework at 50 samples, and includes the input PDB files `8v13H.pdb`, `8v13L.pdb`, and `8v13HL.pdb`. `carrier_structure_3eakA_cdr12/` contains 3eak chain A carriers with alternative CDR1 and CDR2 connector sequences. `carrier_structure/` also contains 3eakA and 3dwtA. CDR boundary tables are at the archive root.

```bash
tar -xzf connector_screening.tar.gz
```

### `selected_candidates_lag16.tar.gz`

Selected LaG16 grafts and their complex models. The grafted coordinates are `1zmyA_generated_27257.pdb`, `3k1kC_generated_101418.pdb`, `4dkaB_generated_22638.pdb`, `7nowA_generated_70840.pdb`, and `8taoC_generated_27577.pdb`, together with the parental `6lr7.pdb` and the design sequences. `affinity_lag16.tsv` is the binding summary for these designs.

`rf3_output/` and `protenix_output/` contain RoseTTAFold3 and Protenix complex predictions for the 6lr7B fusions, with per-sample confidence files and the aggregated tables `rf3_best.tsv`, `rf3_all_samples.tsv`, `protenix_best.tsv`, and `protenix_all_samples.tsv`. `complex_candidates_colabfold_msa/` contains the ColabFold alignments used for those predictions. `6lr7B_7nowA_mutations.pse` is a PyMOL session and opens in PyMOL; the rendered mutation figures are also provided as PNG files.

```bash
tar -xzf selected_candidates_lag16.tar.gz
```

### `BLI.tar.gz`

Processed biolayer interferometry measurements, as 125 spreadsheet files (`.xls`). `GFP_processed_data/` covers the GFP-binding series for 1zmyA, 3k1kC, 4dkaB, 5immB, 6lr7B, 7nowA, and 8taoC. `TNF_processed_data/` covers the TNF-α series, including 5m2jD and the numbered clones, each as a twofold dilution from 1000 nM. `SNX_processed_data/` covers the SNX-binding series for B03, D09, and the humanized variants B03H3L3 and D09H13.

```bash
tar -xzf BLI.tar.gz
```
