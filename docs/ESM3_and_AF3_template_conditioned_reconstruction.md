# ESM3-Template and AF3-Template Structure Reconstruction

**Scope:** input provenance, template-conditioning mechanisms, inference commands, model-selection records, and paired reconstruction analysis.  

## 1. Path convention and execution environments

Commands assume the project root is the current working directory. AF3 preparation, inference, and paired scoring scripts are located in `scripts/conditional_generation/`. Historical ESM3 resources beginning with `../` are relative to the same working directory.

ESM3 inference uses the unified `fangs` Conda environment and the `esm3_sm_open_v1` model. AF3 inference requires the installed AlphaFold 3 runtime, its environment activation script, model weights, and four available GPUs. Structural extraction and paired analysis also use `fangs`, including Gemmi, NumPy, SciPy, pandas, and Matplotlib. The current extraction and summary utilities require Python 3.10 or later.

The existing ESM3 outputs are stored beneath `data/ESM3-Template_validation/all_info/`; AF3 predictions are stored beneath `data/ESM3-Template_validation/af3_template/af3_observed_predictions_20260929/`; the final paired analysis is stored beneath `data/ESM3-Template_validation/esm3_vs_af3_observed_matched841_20261002/`. The commands identify these original destinations. Generation and analysis commands use these existing result destinations.

## 2. Experimental definition and input provenance

This experiment evaluates reconstruction of an experimentally observed protein chain from its sequence and native CDR structural information. Both methods receive the observed-chain sequence and conditioning information for CDR1, CDR2, and CDR3. ESM3 supplies CDR structure tokens derived from encoding the native chain; AF3 supplies template coordinates through explicitly mapped CDR residues. These mechanisms should be described separately because they impose different forms of structural conditioning.

The AF3 canonical input bundle contains **843 targets**. Each target has one observed-chain query, one CDR template, and one recorded model seed. The final paired comparison contains **841 targets**, after excluding two targets with inconsistent conditioning-position assignments between the methods. For each included target, one previously selected highest-pTM structure from each method is compared with the same experimentally observed reference chain.

| Resource | Project path | Role |
| --- | --- | --- |
| Native coordinate library | `data/INDI_database/passed_pdbs/` | Chain-specific PDB inputs to ESM3 and the structural source of the observed-chain AF3 templates |
| INDI annotation tables | `data/INDI_database/INDI_info_full_cdr_info.tsv`, `data/INDI_database/INDI_info_full_cdr_info_dedup.tsv`, and `data/INDI_database/INDI_info_full_cdr_info_dedup_clean.tsv` | Sequence and CDR annotations; their membership is not interchangeable with the frozen reconstruction cohort |
| AF3 canonical configurations | `data/ESM3-Template_validation/af3_template/af3_observed_chain_canonical_20260929/configs/` | 843 target-specific JSON inputs |
| Observed-chain reference templates | `data/ESM3-Template_validation/af3_template/af3_observed_chain_canonical_20260929/templates/` | Canonical single-chain CIF references identified by each configuration |
| Configuration manifest | `data/ESM3-Template_validation/af3_template/af3_observed_chain_canonical_20260929/generation_manifest.tsv` | Source PDB and chain, observed-chain length, annotation source, excluded modified residues, model seed, and configuration/template locations |
| CDR residue mapping | `data/ESM3-Template_validation/af3_template/af3_observed_chain_canonical_20260929/cdr_mapping_report.tsv` | 2,529 records defining the three CDR spans for 843 targets |
| Configuration validation | `data/ESM3-Template_validation/af3_template/af3_observed_chain_canonical_20260929/af3_config_validation.tsv` and `data/ESM3-Template_validation/af3_template/af3_observed_chain_canonical_20260929/summary.json` | Archived input-validation and configuration-generation accounting |

The canonical query sequence represents the selected chain's experimentally observed standard amino-acid residues. It can differ from the full sequence recorded by INDI: the archived input report identifies 562 such differences among 843 targets. Modified residues were excluded during canonical input preparation, as recorded in the manifest. The longest observed chain is 879 residues. Consequently, the final Global metric refers to the full observed input chain, including any additional observed sequence, rather than automatically to an isolated nanobody domain.

The frozen AF3 configurations and mapping records establish the 843-target reconstruction cohort. The ESM3 historical run started from the 946-row deduplicated annotation table and retained predictions for 843 distinct targets. Substituting the current 841-record clean INDI table would change this input population. The AF3 canonical-configuration generator is `scripts/conditional_generation/generate_af3_observed_chain_config.py`. The existing validated configuration bundle is the verified input to AF3 inference.

## 3. ESM3 template-conditioned reconstruction

### 3.1. Verified annotation processing and historical inputs

**Annotation-processing script:** `scripts/preprocess/dedup_INDI_info.py`.  
**Inference script:** `scripts/conditional_generation/temp_generation_eval.py`.  
**Retained historical inference source:** `../code/temp_generation_eval_0211.py`.

The actual preprocessing script groups records by `Sequence`, `CDR1`, `CDR2`, and `CDR3`, retains the first representative `PDBChain`, and records group membership in `Member`. Its input is a comma-separated file. The retained upstream source is `../INDI_241125/INDI_info_full_cdr_info.csv`; the similarly named project file `data/INDI_database/INDI_info_full_cdr_info.tsv` is tab-separated and is not a direct input to this script's default CSV reader.

The command below invokes the existing preprocessing implementation and writes to the existing annotation-table location. Verification using this script and the retained upstream CSV reproduced all 946 records in `data/INDI_database/INDI_info_full_cdr_info_dedup.tsv`, including their order and column values.

```bash
conda activate fangs
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python "scripts/preprocess/dedup_INDI_info.py" \
  "../INDI_241125/INDI_info_full_cdr_info.csv" \
  "data/INDI_database/INDI_info_full_cdr_info_dedup.tsv"
```

The historical ESM3 log is retained at `../connector_correlation/temp_generation_eval_250211.log`. It records 946 input rows, representing 944 distinct chain identifiers. Their identities and order match the current deduplicated table exactly. For all 844 rows with three printed CDR motifs, the logged motifs also match the retained annotations. The historical run produced structures for 843 distinct targets; this successful generation cohort subsequently defined the AF3 reconstruction cohort.

The original generation table at `../connector_correlation/temp_generation_eval_250211/all_info/all_generation.tsv` is byte-identical to `data/ESM3-Template_validation/all_info/all_generation.tsv`. For all 843 successful targets, the original native PDB in `../INDI_241125/pdb/` is byte-identical to the migrated counterpart in `data/INDI_database/passed_pdbs/`. The migrated inference script differs from `../code/temp_generation_eval_0211.py` only in comments.

The inference TSV requires `PDBChain`, `CDR1`, `CDR2`, and `CDR3`. The existing `Sequence` column records annotation provenance; inference reads the actual chain sequence from the native coordinates. Section 3.2 invokes this verified table directly through the current script interface.

### 3.2. Conditioning mechanism and inference command

The current implementation locates CDR motifs by their first exact substring occurrence in the coordinate-derived chain sequence. It processes motifs in CDR3, CDR2, and CDR1 order. With `--linker_len 0`, the full observed-chain amino-acid sequence remains specified, while the structure track is masked outside the CDRs. Structure tokens at the CDR positions are copied from the encoded native chain. The generation step fills masked structure tokens and decodes the resulting structure.

The code constructs a coordinate prompt internally, but the actual ESM3 prompt is initialized from the sequence and populated with CDR structure tokens. It therefore does not directly impose the native CDR atom coordinates as an exact coordinate constraint. The reported CDR reconstruction errors are evaluated on decoded structures.

The following command is reconstructed from the retained input table, the current inference interface, and the historical log, which records five structure samples, five stored samples, and zero linker length. It uses the migrated data and original result destination; the complete original shell invocation is not retained in the log.

```bash
conda activate fangs
python "scripts/conditional_generation/temp_generation_eval.py" \
  "data/INDI_database/passed_pdbs/" \
  "data/INDI_database/INDI_info_full_cdr_info_dedup.tsv" \
  "data/ESM3-Template_validation/" \
  --gpu 0 \
  --structure_sample 5 \
  --sample_to_store 5 \
  --linker_len 0 \
  --cpu_num 8
```

The structure-sampling temperature is fixed at **0.7** in the current code. The number of generation steps is the number of masked structure tokens. The standalone entry point exposes no random-seed argument and does not establish a fixed seed, so newly generated samples should not be expected to reproduce the archived structures exactly. The parsed `--sample_to_store` value is logged but is not used to truncate or rank saved designs; the script writes all successful generated designs. Highest-pTM selection is a subsequent analysis operation.

The documented reconstruction setting uses no linker and no antigen. The optional linker and antigen interfaces are outside this benchmark. Missing motif matches cause a target to be skipped; the later paired analysis requires the specified cohort size and does not silently accept missing targets.

### 3.3. ESM3 outputs and retained predictions

| Output | Project path | Content |
| --- | --- | --- |
| Generation table | `data/ESM3-Template_validation/all_info/all_generation.tsv` | 4,220 records for 843 targets; target ID, sequence, reported pTM, generation-time CDR cRMSDs, and structure filename |
| Predicted coordinates | `data/ESM3-Template_validation/all_info/pdb/` | 4,215 decoded PDB structures |
| Final ESM3 selection manifest | `data/ESM3-Template_validation/esm3_vs_af3_observed_matched841_20261002/ESM3-Template_top1.tsv` | One selected structure per target for the final 841-target comparison |

The archived table contains ten rows for `8jbhE` and five rows for each of the other 842 targets. The five `8jbhE` filenames each occur twice, whereas the coordinate archive contains one file for each filename. Thus, 4,220 table rows do not represent 4,220 independent retained coordinate files. The final comparison uses the frozen selection manifest and recomputes RMSD from the selected coordinates, rather than treating repeated historical rows as independent samples.

## 4. AF3 CDR-template reconstruction

### 4.1. Preparation scripts and inputs

| Operation | Script | Role |
| --- | --- | --- |
| Configuration generation | `scripts/conditional_generation/generate_af3_observed_chain_config.py` | Extract observed-chain sequences, locate CDRs, write CIF templates and JSON inputs, retain model seeds, and distribute GPU shards |
| Template conversion | `scripts/conditional_generation/generate_af3_template_config.py` | Shared chain-selection and CIF metadata functions imported by the generator |
| Input validation | `scripts/conditional_generation/validate_af3_observed_configs.py` | Validate AF3 parsing, sequences, template indices, coordinates, and seeds |
| Environment activation | `scripts/conditional_generation/activate_af3.sh` | Activate the installed AF3 environment and configure its runtime |
| Inference | `scripts/conditional_generation/run_af3_4gpu.sh` | Execute four independent GPU workers |
| Observed-cohort entry point | `scripts/conditional_generation/run_af3_observed_4gpu.sh` | Supply the archived cohort, checkpoint directory, result directory, and inference settings |

Configuration generation used `data/INDI_database/INDI_info_full_cdr_info_dedup_clean.tsv`, the fallback table `data/INDI_database/INDI_info_full_cdr_info_dedup.tsv`, and native coordinates in `data/INDI_database/passed_pdbs/`. The retained preparation result is `data/ESM3-Template_validation/af3_template/af3_observed_chain_canonical_20260929/`, containing 843 configurations and templates, the generation manifest, CDR mappings, validation records, and four GPU shards.

The generator accepts `--tsv_path`, `--pdb_dir`, and `--output_dir`, with optional cohort, fallback, CDR-reference, and seed-configuration inputs. The original cohort and seed-reference inputs are partly unavailable; inference uses the retained canonical bundle.

### 4.2. Conditioning and inference parameters

Each AlphaFold 3 JSON input uses dialect version 4 and specifies one observed-chain protein sequence, empty paired and unpaired MSAs, one native-chain CIF template, and a target-specific model seed. The template contains the observed chain; `queryIndices` and `templateIndices` restrict structural conditioning to CDR1, CDR2, and CDR3.

| Parameter | Value |
| --- | --- |
| Targets | 843 |
| Configurations per GPU shard | 206, 213, 212, and 212 |
| Physical GPUs | 0, 1, 2, and 3 |
| Diffusion samples per target | 5 |
| Recycles | 10 |
| Attention implementation | Triton |
| Data pipeline | Disabled |

### 4.3. Inference commands and outputs

The observed-cohort entry point activates AF3 and uses the server's installed runtime and checkpoints. `AF3_HOME`, `AF3_CONDA_ENV`, and `AF3_MODEL_DIR` can override these defaults. Validate the launcher without running predictions:

```bash
bash "scripts/conditional_generation/run_af3_observed_4gpu.sh" --dry_run
```

Run inference with the archived settings:

```bash
bash "scripts/conditional_generation/run_af3_observed_4gpu.sh" \
  > "data/ESM3-Template_validation/af3_template/af3_observed_predictions_20260929/launcher.log" 2>&1
```

Predictions are retained in `data/ESM3-Template_validation/af3_template/af3_observed_predictions_20260929/`, under `gpu0/` through `gpu3/`. This directory also contains worker logs in `logs/`, PID records in `pids/`, compilation caches in `jax_cache/`, and the combined `launcher.log`. The archived launcher log records successful completion of all four workers.

## 5. Per-model extraction and diagnostic summaries

### 5.1. AF3 sample selection

**Extraction script:** `scripts/conditional_generation/extract_af3_results.py`.

The extractor reads prediction CIFs and confidence JSON files, checks canonical sequences and template mappings, and selects samples by descending pTM. Each CDR is scored by an independent proper Kabsch fit; its global metric requires complete sequence identity and coordinates. It requires a new output directory; no standalone historical extraction bundle is retained.

The final AF3 selection is `data/ESM3-Template_validation/esm3_vs_af3_observed_matched841_20261002/AF3_top1.tsv`. It records one selected CIF and confidence record per target. Section 6 uses this manifest directly to recompute the paired metrics.

### 5.2. ESM3 generation-table summary

**Script:** `scripts/conditional_generation/calc_temp_generation_cRMSD.py`.

```bash
python "scripts/conditional_generation/calc_temp_generation_cRMSD.py" \
  "data/ESM3-Template_validation/all_info/all_generation.tsv" \
  "data/ESM3-Template_validation/all_info/" \
  --template-pdb-dir "data/INDI_database/passed_pdbs/" \
  --generated-pdb-dir "data/ESM3-Template_validation/all_info/pdb/" \
  --top-k-per-pdb 1 \
  --exclude-protein-ids 7pklL,8jbhE \
  > "data/ESM3-Template_validation/all_info/calc_generation.log" 2>&1
```

This command is reconstructed from the current interface and the existing diagnostic log. The log records exclusion of `7pklL` and `8jbhE`, and the archived statistics specify top-1 selection with no minimum-pTM threshold. The utility summarizes rounded CDR cRMSDs from the generation table and computes whole-protein metrics using PDB chain/residue identifiers. Its existing outputs are `data/ESM3-Template_validation/all_info/ptm_filtered.tsv`, `data/ESM3-Template_validation/all_info/statistics.txt`, and diagnostic figures in the same directory. The legacy copy at `data/ESM3-Template_validation/all_info/processed_data.csv` is tab-separated despite its filename extension. The recorded execution log is `data/ESM3-Template_validation/all_info/calc_generation.log`.

This diagnostic utility does not implement the final common observed-chain residue correspondence. In particular, the existing summary at `data/ESM3-Template_validation/all_info/statistics.txt` excludes `7pklL` and `8jbhE` and reports whole-protein metrics for only 197 structures. It is a different analysis from the final 841-target comparison and should not supply the manuscript's Global or CDR statistics. The final statistics are recomputed from the retained coordinates with Section 6.

## 6. Final paired analysis and figure generation

### 6.1. Cohort selection and conditioning-position audit

The final selection manifests each contain 841 unique targets, with matching target membership and one highest-reported-pTM structure per model. No additional confidence threshold or RMSD-based model selection is applied. All selected coordinate files and canonical reference templates were verified to exist at their recorded locations.

Two targets are excluded because repeated CDR2 motifs lead to different conditioning spans in the ESM3 first-occurrence mapping and the AF3 ordered mapping:

| Target | CDR2 motif | ESM3 positions in the observed sequence | AF3 positions in the observed sequence |
| --- | --- | --- | --- |
| `7pklL` | `SAS` | 133–135 | 171–173 |
| `8e99E` | `AAAAAAA` | 25–31 | 45–51 |

Positions in this table are one-based. The source-defined exclusion record is `data/ESM3-Template_validation/esm3_vs_af3_observed_matched841_20261002/cohort_exclusions.tsv`; the retained 2,523 CDR mappings are audited in `data/ESM3-Template_validation/esm3_vs_af3_observed_matched841_20261002/conditioning_mapping_audit.tsv`. The exclusion of `8e99E` in this comparison is distinct from the absence of `8jbhE` in the current clean INDI annotation table.

### 6.2. Geometry and statistical definitions

**Analysis scripts:** `scripts/conditional_generation/template_reconstruction_analysis/analyze_and_plot.py`, `scripts/conditional_generation/template_reconstruction_analysis/plot_all_cdr_thresholds.py`, and their shared geometry implementation at `scripts/conditional_generation/template_reconstruction_analysis/analysis_common.py`.

The scoring code requires exact observed-chain sequence identity between the reference and both selected predictions. Global Cα RMSD uses the same intersection of available reference and prediction Cα positions for both models over the full observed input chain. The minimum permitted shared coverage is 0.90. Each CDR requires complete shared Cα coverage and receives its own proper Kabsch superposition; the CDR RMSD is not evaluated after a framework-only fit.

The four regional RMSD distributions are compared using two-sided paired Wilcoxon signed-rank tests, with Holm correction across Global, CDR1, CDR2, and CDR3. A target passes a joint threshold only if all three independently fitted CDR RMSDs are **strictly below** that threshold. The 1.5 Å and 1.0 Å passing proportions are compared using two-sided exact paired McNemar tests, with Holm correction across those two classes. A supplementary 0.5 Å count is exported separately from the two-class statistical family.

### 6.3. Verified direct analysis commands

The retained top-1 manifests provide a complete analysis entry point. The following commands recompute metrics and figures from existing coordinate files without repeating inference:

```bash

python "scripts/conditional_generation/template_reconstruction_analysis/analyze_and_plot.py" \
  --config-root "data/ESM3-Template_validation/af3_template/af3_observed_chain_canonical_20260929/" \
  --selection-dir "data/ESM3-Template_validation/esm3_vs_af3_observed_matched841_20261002/" \
  --out "data/ESM3-Template_validation/esm3_vs_af3_observed_matched841_20261002/" \
  --expected-targets 841 \
  --minimum-coverage 0.90

python "scripts/conditional_generation/template_reconstruction_analysis/plot_all_cdr_thresholds.py" \
  --paired-metrics "data/ESM3-Template_validation/esm3_vs_af3_observed_matched841_20261002/paired_metrics.tsv" \
  --out "data/ESM3-Template_validation/esm3_vs_af3_observed_matched841_20261002/" \
  --expected-targets 841
```

The retained analysis contains 841 complete pairs, zero scoring failures, and a minimum shared Cα coverage of 0.98374. Verification reproduced the regional and joint-threshold summary and test tables exactly.

### 6.4. Existing analysis outputs and figures

The existing manuscript result bundle and the output destination of the direct analysis commands are `data/ESM3-Template_validation/esm3_vs_af3_observed_matched841_20261002/`.

| Archived result | Project path | Interpretation |
| --- | --- | --- |
| Per-target paired metrics | `data/ESM3-Template_validation/esm3_vs_af3_observed_matched841_20261002/paired_metrics.tsv` | RMSD, selected pTM, source structures, common residue indices, and coverage for 841 × 2 model records |
| Regional summary | `data/ESM3-Template_validation/esm3_vs_af3_observed_matched841_20261002/rmsd_summary.tsv` | Mean, standard deviation, median, quartiles, minimum, and maximum |
| Regional tests | `data/ESM3-Template_validation/esm3_vs_af3_observed_matched841_20261002/significance.tsv` | Paired Wilcoxon results and Holm-adjusted P values |
| Joint-threshold counts | `data/ESM3-Template_validation/esm3_vs_af3_observed_matched841_20261002/all_cdr_threshold_summary.tsv` | Counts and percentages for all three CDRs below 1.5 Å or 1.0 Å |
| Joint-threshold tests | `data/ESM3-Template_validation/esm3_vs_af3_observed_matched841_20261002/all_cdr_threshold_tests.tsv` | Exact paired McNemar results and Holm-adjusted P values |
| Cohort accounting and QA | `data/ESM3-Template_validation/esm3_vs_af3_observed_matched841_20261002/cohort_accounting.tsv` and `data/ESM3-Template_validation/esm3_vs_af3_observed_matched841_20261002/QA_report.json` | Original-to-final sample accounting and archived cross-checks |
| RMSD comparison figure | `data/ESM3-Template_validation/esm3_vs_af3_observed_matched841_20261002/esm3_vs_af3_paired_boxplots.png` and `data/ESM3-Template_validation/esm3_vs_af3_observed_matched841_20261002/esm3_vs_af3_paired_boxplots.svg` | Global/CDR comparison; 600-dpi PNG and editable SVG |
| Joint-threshold figure | `data/ESM3-Template_validation/esm3_vs_af3_observed_matched841_20261002/esm3_vs_af3_all_cdr_thresholds.png` and `data/ESM3-Template_validation/esm3_vs_af3_observed_matched841_20261002/esm3_vs_af3_all_cdr_thresholds.svg` | Joint CDR passing proportions; 600-dpi PNG and editable SVG |

The regional boxplot uses a logarithmic RMSD axis. Extreme observations are hidden as plotting fliers but retained in all statistical calculations.

| Region | ESM3-Template median RMSD (Å) | AF3-Template median RMSD (Å) | Targets per method |
| --- | ---: | ---: | ---: |
| Global | 0.758 | 1.128 | 841 |
| CDR1 | 0.175 | 0.401 | 841 |
| CDR2 | 0.144 | 0.248 | 841 |
| CDR3 | 0.244 | 0.427 | 841 |

| Joint criterion | ESM3-Template | AF3-Template |
| --- | ---: | ---: |
| All three CDR RMSDs < 1.5 Å | 837/841 (99.52%) | 527/841 (62.66%) |
| All three CDR RMSDs < 1.0 Å | 832/841 (98.93%) | 450/841 (53.51%) |

## 7. Reproducibility boundaries

The retained canonical inputs, selected coordinates, manifests, and paired-analysis scripts reproduce the 841-target statistics. ESM3's original shell invocation and random-state record are unavailable. Rebuilding the earlier AF3 preparation and selection-audit stages requires the missing target manifest, reference mappings, seed configurations, and 843-target source bundle. Relocating the data also requires updating paths embedded in JSON configurations and selection TSVs.
