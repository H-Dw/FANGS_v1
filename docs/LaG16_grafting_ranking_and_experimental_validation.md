# LaG16 Data Sources, Execution Commands, and Output Locations

This document records the data sources, execution commands, and output locations for library-wide LaG16 grafting and framework ranking, carrier sequence comparison, expression yield and GFP BLI affinity, sampling depth, and Protenix confidence–affinity comparison.

All paths are relative to the repository root. Execute commands from that root in the `fangs` Conda environment. MAFFT is installed with `environment.fangs.yml` and is available on the executable search path after activation. Protenix requires a separate installation with the appropriate model parameters. Regenerating MSAs additionally requires user-installed ColabFold and MMseqs2, together with the UniRef30 and ColabFold environmental databases.

The experimental identifiers are parental LaG16 (`6lr7B`) and grafts on `7nowA`, `4dkaB`, `8taoC`, `3k1kC`, and `1zmyA`. The selected-sequence and Protenix records use `4dkaA` for the construct identified as `4dkaB` in the measurement tables.

All derived tables, figures, and newly generated workflow outputs are assigned to `data/6lr7B_grafting_connector_all_temp07_s50_250904/analysis/`.

```bash
conda activate fangs
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
LAG=data/6lr7B_grafting_connector_all_temp07_s50_250904
OUT="$LAG/analysis"
mkdir -p "$OUT"
```

## 1. Library-wide LaG16 grafting and framework ranking

### Data sources

| Repository-relative path | Data content |
|---|---|
| `example/6lr7B.pdb` | Parental LaG16 coordinates, PDB 6LR7 chain B. |
| `example/6lr7B_CDR.tsv` | Donor CDR annotations. |
| `data/INDI_database/INDI_info_full_cdr_info.tsv` | INDI carrier annotations. |
| `data/INDI_database/pdb_model0/` | Native carrier coordinates. |
| `data/INDI_database/pdb2seq.fasta` | Native carrier sequences. |
| `data/INDI_database/` | RAW, pre-VQ, and Cα connector representation tables. |
| `data/6lr7B_grafting_connector_all_temp07_s50_250904/process_file.tsv` | Generation-workflow record used for score extraction. |

The following source files are relative to `data/6lr7B_grafting_connector_all_temp07_s50_250904/`.

| Archive-relative path | Data content |
|---|---|
| `temp_generation/temp_input.tsv` | Carrier-specific generation prompts. |
| `temp_generation/all_info/all_generation.tsv` | Generated sequences, pTM, cRMSD, and sample filenames. |
| `temp_generation/all_info/pdb/` | Generated coordinates. |
| `tokenized_input/`, `similarity/`, and `generation_tokenized_structure/` | Donor representations, native comparisons, and generated-graft representations. |
| `extract_distance/full/grafted_raw_embeddings.tsv` | All generated donor–graft score records. |
| `extract_distance/filter/change_raw_embeddings.tsv` | Retained ΔConnector records. |
| `extract_distance/filter/grafted_raw_embeddings.tsv` | Retained donor–graft scores and sample filenames. |
| `extract_distance/filter_best/grafted_raw_embeddings.tsv` | Minimum whole-connector score per retained design. |
| `extract_distance/full_avge/original_raw_embeddings.tsv` | Native donor–carrier scores used for framework ranking. |
| `temp_generation.log` | Generation execution record. |

The archive contains 135,150 samples from 2,703 generated designs. The manuscript count, **43,170 of 135,150 samples (31.9%) passed**, corresponds to the connector-filtered sample table, comprising 1,561 retained designs.

### Commands

The generation command uses RAW representations (`--type 1`), the full carrier set (`--top_n 0`), 50 samples per design, a maximum prompt length of 180 residues, temperature 0.7, and a ΔConnector threshold of 10. A regenerated run is written beneath the consolidated output directory.

```bash
python scripts/grafting_generation/graft_CDR_connector.py \
  -qp example/6lr7B.pdb \
  -qc example/6lr7B_CDR.tsv \
  -tp data/INDI_database/pdb_model0 \
  -tc data/INDI_database/INDI_info_full_cdr_info.tsv \
  -o "$OUT/full_library/generation" \
  --seqs_file data/INDI_database/pdb2seq.fasta \
  --tokenized_folder data/INDI_database \
  --ss 50 \
  --len_limit 180 \
  --gpu 0 \
  --mode connector \
  --type 1 \
  --top_n 0 \
  --temperature 0.7 \
  --distance_limit 10
```

The following commands extract scores and summarize framework rankings from the archived run.

```bash
python scripts/grafting_generation/extract_generation.py \
  "$LAG/process_file.tsv" \
  "$LAG" \
  "$OUT/full_library/extract_distance" \
  --limitation 10

python scripts/grafting_generation/tools/summarize_lag16_framework_ranking.py \
  --run-dir "$LAG" \
  --output-dir "$OUT/framework_ranking"
```

### Output locations

- `data/6lr7B_grafting_connector_all_temp07_s50_250904/analysis/full_library/generation/`: regenerated sequences, coordinates, representations, and workflow records.
- `data/6lr7B_grafting_connector_all_temp07_s50_250904/analysis/full_library/extract_distance/`: extracted `full`, `filter`, `full_avge`, `filter_avge`, `full_best`, and `filter_best` tables.
- `data/6lr7B_grafting_connector_all_temp07_s50_250904/analysis/framework_ranking/`: `LaG16_ranked_native_frameworks.tsv`, `LaG16_ranked_grafted_frameworks.tsv`, `LaG16_selected_carrier_ranks.tsv`, `LaG16_filter_counts.json`, and `LaG16_framework_ranking.{pdf,svg,png}`.

## 2. Sequence comparison of the five LaG16 carriers

### Data sources

The source FASTA is `data/6lr7B_grafting_connector_all_temp07_s50_250904/selected_candidates/candidate_seqs.fa`. Its display order is specified in `selected_candidates/order.txt` within the same archive. Parental LaG16 and the five graft sequences are included.

| Experimental carrier | Coordinate file in `selected_candidates/` |
|---|---|
| `7nowA` | `7nowA_generated_70840.pdb` |
| `4dkaB` | `4dkaA_generated_47772.pdb` |
| `8taoC` | `8taoC_generated_27577.pdb` |
| `3k1kC` | `3k1kC_generated_101418.pdb` |
| `1zmyA` | `1zmyA_generated_27257.pdb` |

### Command

`scripts/grafting_generation/tools/calc_seq_diff.py` uses the MAFFT executable supplied by `fangs` to align the source sequences and exports the sequence-difference matrix and its display.

```bash
mkdir -p "$OUT/sequence_comparison"
python scripts/grafting_generation/tools/calc_seq_diff.py \
  -i "$LAG/selected_candidates/candidate_seqs.fa" \
  -o "$OUT/sequence_comparison/diff" \
  --order "$LAG/selected_candidates/order.txt" \
  --mafft mafft \
  --simplify \
  --cell-size 0.8 \
  --format both
```

### Output location

`data/6lr7B_grafting_connector_all_temp07_s50_250904/analysis/sequence_comparison/`: `diff_aligned.fasta`, `diff_matrix.csv`, and `diff_heatmap.{png,svg}`.

## 3. LaG16 expression yield and GFP BLI affinity

### Data sources

| Repository-relative path | Data content |
|---|---|
| `scripts/grafting_generation/tools/Lag16_affinity_yield.txt` | Current measurement table containing `PDB_ID`, `Affinity` (GFP-binding equilibrium K_D, nM), and `Yield` (purified protein mass, mg). |
| `data/BLI/GFP_processed_data/` | Archived GFP BLI measurement exports. |
| `data/selected_candidates_lag16/affinity_lag16.tsv` | Affinity-only table used for computational comparisons. |

### Command

`scripts/grafting_generation/tools/plot_lag16_affinity_yield_barplot.py` reads the current measurement table, retains the construct order, and exports the affinity and yield summary with editable SVG text.

```bash
python scripts/grafting_generation/tools/plot_lag16_affinity_yield_barplot.py \
  --input scripts/grafting_generation/tools/Lag16_affinity_yield.txt \
  --output "$OUT/affinity_yield/LaG16_affinity_yield_barplot.svg"
```

### Output location

`data/6lr7B_grafting_connector_all_temp07_s50_250904/analysis/affinity_yield/LaG16_affinity_yield_barplot.svg`.

## 4. LaG16 sampling-depth experiment

### Data sources

The ten native carrier structures are stored in `data/samples_evaluation/carrier_structures/`: `1zmyA`, `3k1kC`, `4dkaB`, `5immB`, `6gwqB`, `6jb8A`, `7aqyD`, `7nowA`, `7ubyC`, and `8taoC`. Their source paths and checksums are recorded in `source_manifest.json`.

Each archived directory below is relative to `data/samples_evaluation/`. The score input within each directory is `extract_distance/filter_best/grafted_raw_embeddings.tsv`.

| Samples per design | Archived directory |
|---:|---|
| 5 | `6lr7B_grafting_connector_candidates_temp07_s5_250831` |
| 10 | `6lr7B_grafting_connector_candidates_temp07_s10_250901` |
| 30 | `6lr7B_grafting_connector_candidates_temp07_s30_250901` |
| 50 | `6lr7B_grafting_connector_candidates_temp07_s50_250901` |
| 80 | `6lr7B_grafting_connector_candidates_temp07_s80_250901` |
| 100 | `6lr7B_grafting_connector_candidates_temp07_s100_250831` |
| 500 | `6lr7B_grafting_connector_candidates_temp07_s500_250901` |

### Commands

The seven generation settings share temperature 0.7, a maximum prompt length of 180 residues, RAW connector mode, and random seed 42.

```bash
for N in 5 10 30 50 80 100 500; do
  case "$N" in
    5|100) STAMP=250831 ;;
    *) STAMP=250901 ;;
  esac

  python scripts/grafting_generation/graft_CDR_connector.py \
    -qp example/6lr7B.pdb \
    -qc example/6lr7B_CDR.tsv \
    -tp data/samples_evaluation/carrier_structures \
    -tc data/INDI_database/INDI_info_full_cdr_info.tsv \
    -o "$OUT/sampling_depth/generation/6lr7B_grafting_connector_candidates_temp07_s${N}_${STAMP}" \
    --ss "$N" \
    --len_limit 180 \
    --gpu 0 \
    --mode connector \
    --type 1 \
    --top_n 0 \
    --temperature 0.7 \
    --distance_limit 10 \
    --seed 42
done
```

The existing Python entry point exports the sampling-depth display and supporting tables from the archived runs.

```bash
python scripts/grafting_generation/tools/plot_lag16_sampling_affinity.py \
  --base-path data/samples_evaluation \
  --output-dir "$OUT/sampling_depth"
```

To read regenerated runs, set `--base-path "$OUT/sampling_depth/generation"`.

### Output location

`data/6lr7B_grafting_connector_all_temp07_s50_250904/analysis/sampling_depth/`: `Fig2f_lag16_sampling_affinity_source.tsv`, `Fig2f_lag16_sampling_affinity_stats.tsv`, `Fig2f_lag16_sampling_affinity_metadata.json`, `Fig2f_lag16_sampling_affinity.{pdf,svg,png}`, and the retained `LaG16_sampling_pearson_statistics.tsv`. Regenerated runs are written to its `generation/` subdirectory.

## 5. Protenix global confidence and GFP-affinity comparison

### Data sources

The source directory is `data/selected_candidates_lag16/`. The six target identifiers are `6lr7B`, `6lr7B_7nowA`, `6lr7B_4dkaA`, `6lr7B_8taoC`, `6lr7B_3k1kC`, and `6lr7B_1zmyA`.

| Path relative to the source directory | Data content |
|---|---|
| `candidates_sequences.fa` | Nanobody sequences. |
| `complex_candidates_sequences.fa` | GFP–nanobody complex sequences. |
| `complex_candidates_sequences_ind.fa` | Individual GFP and nanobody sequences. |
| `protenix_config/` | Archived target configurations, including one `*-update-msa.json` per target. |
| `complex_candidates_colabfold_msa/` | MSA search archive. |
| `protenix_output/<target>/msa/0/` and `msa/1/` | Prediction-ready `pairing.a3m` and `non_pairing.a3m` files. |
| `protenix_output/<target>/seed_101/predictions/` | Archived predictions and confidence JSON files. |
| `protenix_pred.log` | Prediction execution record. |
| `affinity_lag16.tsv` | GFP affinity measurements. |

The prediction archive contains 50 samples per target at seed 101. The connector-score input for the five grafts is `data/samples_evaluation/6lr7B_grafting_connector_candidates_temp07_s50_250901/extract_distance/filter_best/grafted_raw_embeddings.tsv`.

### Configuration preparation and prediction commands

`scripts/grafting_generation/tools/prepare_lag16_protenix_config.py` prepares the six target jobs using the archived prediction-ready MSAs and records their checksums in `preparation_manifest.tsv`.

```bash
python scripts/grafting_generation/tools/prepare_lag16_protenix_config.py \
  --config-dir data/selected_candidates_lag16/protenix_config \
  --output-dir "$OUT/global_confidence/protenix_config"
```

Run prediction in the user-installed Protenix environment.

```bash
CUDA_VISIBLE_DEVICES=1 protenix pred \
  -i "$OUT/global_confidence/protenix_config" \
  -o "$OUT/global_confidence/protenix_output" \
  --sample 50 \
  --seeds 101 \
  --model_name protenix_base_default_v1.0.0 \
  --use_msa true \
  --use_default_params true
```

### Confidence extraction and summary commands

`scripts/grafting_generation/tools/extract_prediction_confidence.py` exports per-sample confidence records and target-level metric summaries. `scripts/grafting_generation/tools/plot_fangs_protenix_vs_affinity.py` combines those summaries with the 50-sample connector-score inputs and the experimental construct labels.

```bash
python scripts/grafting_generation/tools/extract_prediction_confidence.py \
  --root data/selected_candidates_lag16/protenix_output \
  --preset protenix \
  -o "$OUT/global_confidence/protenix_all_samples.tsv" \
  --out-best-tsv "$OUT/global_confidence/protenix_best.tsv"

python scripts/grafting_generation/tools/plot_fangs_protenix_vs_affinity.py \
  --base-path data/samples_evaluation \
  --protenix "$OUT/global_confidence/protenix_best.tsv" \
  --output-dir "$OUT/global_confidence"
```

For newly generated predictions, set the extraction command's `--root` to `"$OUT/global_confidence/protenix_output"`.

### Output location

`data/6lr7B_grafting_connector_all_temp07_s50_250904/analysis/global_confidence/`: `protenix_all_samples.tsv`, `protenix_best.tsv`, `Fig2g_lag16_normalized_scores_source.tsv`, `Fig2g_lag16_normalized_scores_stats.tsv`, `Fig2g_lag16_normalized_scores_metadata.json`, and `Fig2g_lag16_normalized_scores.{pdf,svg,png}`. Prepared configurations and regenerated predictions are assigned to its `protenix_config/` and `protenix_output/` subdirectories.
