# FANGS

## 1. Project overview

FANGS is a structure-guided workflow for nanobody framework selection and complementarity-determining region (CDR) grafting. It encodes the framework residues flanking each CDR as connector representations, compares donor and carrier structures, generates CDR-conditioned grafts with ESM3, and ranks candidate designs by connector compatibility. The repository supports library-wide grafting, humanization, and dual-motif window grafting.

- **Inputs:** a donor structure and CDR annotation table; carrier structures and corresponding CDR annotations; and, when available, precomputed carrier connector representations and sequences.
- **Outputs:** grafted sequences and structures, per-sample confidence and CDR reconstruction metrics, original and grafting connector scores, filtered candidate tables, and ranked designs. Experiment-specific summaries and figures are described in the protocols below.

## 2. Repository layout

| Directory | Contents |
|---|---|
| `data/` | Archived annotations, native structures, connector representations, generated samples, experimental measurements, and experiment-specific analysis outputs. |
| `scripts/` | Workflow entry points grouped under `preprocess/`, `conditional_generation/`, `batch_generation/`, `grafting_generation/`, `humanization/`, and `connector_screening/`; shared utilities are in `tools/`. |
| `docs/` | Six experiment protocols specifying data sources, execution commands, and result locations. |
| `example/` | Donor structures, sequences, and CDR annotation tables for the LaG16, VHH2, and B03 workflows. |
| `esm/` | Project-local ESM3 implementation containing the custom structure-encoding methods. |
| `results/` | Standalone score tables and outputs from independent reruns, including the quick-start example. |
| `figures/` | Archived figure exports; additional exports are stored in the experiment-specific analysis directories. |
| `analysis/` | Revision-stage analysis code and provenance records, including the resolved-structure framework-alignment workflow. |

The principal dataset directories are `data/INDI_database/`, `data/ESM3-Template_validation/`, `data/batch_graft_generation_id70_table/`, `data/6lr7B_grafting_connector_all_temp07_s50_250904/`, `data/samples_evaluation/`, `data/selected_candidates_lag16/`, `data/BLI/`, `data/humanization/`, and `data/connector_screening/`. The protocols identify the relevant inputs and outputs within each directory.

## 3. Runtime requirements

Commands use a Linux Bash shell with the repository root as the working directory. Data, script, and result paths are relative to that root. The unified `fangs` Conda environment provides Python 3.10.14, PyTorch 2.2.0 with CUDA 11.8, NumPy 1.26.4, pandas 2.2.3, SciPy 1.15.2, Matplotlib 3.10.0, Biopython 1.84, and Gemmi 0.7.5. Biotite 0.41.2 is retained for compatibility with the project-local ESM3 structure encoder.

Create the environment using [environment.fangs.yml](environment.fangs.yml) and its pinned Python dependencies in [requirements-fangs.txt](requirements-fangs.txt). The requirements select the CUDA 11.8 PyTorch wheels from the official PyTorch index.

```bash
conda env create --file environment.fangs.yml
conda activate fangs
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

The module search path selects the repository-local `esm/` implementation, which provides the custom connector-encoding and generation methods. ESM3 inference uses the `esm3_sm_open_v1` checkpoint together with its structure encoder, structure decoder, and associated model data. Make the complete model data available through `esm/data/` or the Hugging Face cache used by the project loader.

| Step | Environment | Principal requirements |
|---|---|---|
| Connector encoding, ESM3 template reconstruction, and graft generation | `fangs` | Project-local ESM3, complete ESM3 model data, PyTorch 2.2.0, and CUDA 11.8; PIPPack for side-chain completion. |
| Human-framework sequence preparation | `fangs` | Biopython, ANARCI, and HMMER 3.3.2 with Kabat numbering support. |
| Archived-table summaries, statistics, sequence comparison, and paired structural analysis | `fangs` | NumPy, pandas, SciPy, Matplotlib, Biopython, Gemmi, and MAFFT 7.525. |
| AF3 template reconstruction | `alphafold3` | AlphaFold 3, model parameters, and CUDA; the archived runner distributes inference across four GPUs. Activate through `scripts/conditional_generation/activate_af3.sh`. |
| Humanized carrier structure prediction | `boltz` | Boltz-2, model parameters, PyTorch, and CUDA. |
| AbNatiV sequence scoring | `abnativ` | AbNatiV and its VH and VHH model parameters. The comparison protocol reads the archived AbNatiV scores and HuDiff sequences. |
| LaG16 global confidence prediction | `protenix` | Protenix, the `protenix_base_default_v1.0.0` model, CUDA, and the archived prediction-ready MSAs. |

ANARCI, HMMER, and MAFFT are installed with `environment.fangs.yml`. Additional external tools are **axel** for CIF retrieval, **USalign** for structural comparison, **MMseqs2** for sequence clustering and search, and **Foldseek** for global structure search. Make these executables available on `PATH` or supply their locations through the protocol arguments. MSA regeneration for Protenix additionally uses ColabFold, MMseqs2, and the UniRef30 and ColabFold environmental databases. Model parameters and database locations are configured in the corresponding runtime scripts and arguments.

Validate the scientific libraries, structural file processing, ANARCI, MAFFT, CUDA execution, and a single ESM3 encoding–generation–decoding cycle with:

```bash
python scripts/tools/check_fangs_environment.py \
  --output-dir results/fangs_environment_check \
  --device cuda:0 --load-model
```

## 4. Quick start

Use the archived B03 connector-score tables to retain the lowest original-score decile and rank those candidates by ascending grafting score. This example requires Python and pandas and runs on the CPU.

```bash
conda activate fangs
python scripts/tools/filter_top_embeddings.py \
  --input data/connector_screening/8v13HL_CDR3_s50/filter_best_selected_scores \
  --output results/quick_start/B03_top10pct_candidates.tsv \
  --orig_col selected_euclidean_average \
  --graf_col selected_euclidean_average
```

The output is `results/quick_start/B03_top10pct_candidates.tsv`, containing four candidates from the 40 archived window pairs. The first-ranked candidate is `8v13HL_3eakA_CDR1-SYST_CDR2-SMGG`, with original and grafting connector scores of 40.34 and 36.12, respectively. The full workflow is documented in the B03 protocol.

## 5. Experiment protocols

1. [INDI database curation and connector representation](docs/INDI_database_curation_and_connector_representation.md): source annotations, structural preprocessing, connector encoding, and resolved-structure pairs with identical CDR sequences.
2. [ESM3 and AF3 template-conditioned reconstruction](docs/ESM3_and_AF3_template_conditioned_reconstruction.md): conditioning inputs, inference, selected-model manifests, and the paired reconstruction benchmark.
3. [id70 virtual grafting and global controls](docs/id70_virtual_grafting_and_global_controls.md): batch grafting, global metrics, the ΔConnector threshold scan, and cumulative donor-relative neighborhoods.
4. [LaG16 grafting, ranking, and experimental validation](docs/LaG16_grafting_ranking_and_experimental_validation.md): library-wide grafting, carrier sequence comparison, expression yield and GFP affinity, sampling depth, and Protenix confidence.
5. [VHH2 humanization and AbNatiV–HuDiff comparison](docs/Humanization_VHH2_library_ranking_and_AbNatiV_HuDiff_comparison.md): human-framework preparation, structural ranking, and the 15-candidate humanization comparison.
6. [B03 dual-motif window grafting](docs/B03_dual_motif_window_grafting.md): paired-window enumeration, graft generation, and original-score top-decile selection followed by grafting-score ranking.

## 6. Data and result conventions

**Archived data.** The datasets and completed runs under `data/` define the recorded input cohorts and manuscript results. Compressed `*.tar.gz` files contain archived dataset bundles; commands use their extracted directories. Cohort membership is specified by the experiment's ID tables, configuration bundles, or selection manifests.

**Recomputed outputs.** The protocols specify experiment-specific result directories, frequently under a dataset's `analysis/` subtree. For an independent rerun, assign a new named or dated output directory under `results/` or the relevant `analysis/` directory, and set the corresponding `OUT`, `--output-dir`, or `-o` argument before execution. Within generation runs, `temp_generation/all_info/` contains sample records and coordinates; `extract_distance/full/`, `filter/`, and `filter_best/` contain all scored samples, threshold-retained samples, and retained representative samples, respectively. The protocols define the score columns and selection rules for each experiment.

**Identifier mapping.** Native chains use `PDBChain` identifiers such as `6lr7B`; representation tables use `ID`, and connector-score tables use `PDB_ID`. Generation records additionally identify individual samples by filename. Preserve chain case, donor prefixes, allele labels, and window suffixes when joining tables. Deduplicated annotation tables retain source membership in `Member`. For the selected LaG16 construct, the experimental label `4dkaB` corresponds to `4dkaA` in the selected-sequence and Protenix archives; apply this alias to that construct. The protocols record further experiment-specific mappings. When relocating data, update paths embedded in JSON configurations, selection TSVs, and runtime scripts.

**Historical figure labels.** Archived basenames retain panel identifiers assigned during manuscript development. Interpret them using the current manuscript panel assignments documented in the protocols. For example, `Fig1d_resolved_CDR_variation_spearman` corresponds to current Fig. 1f, and `Fig3d_abnativ_vh_vhh` corresponds to current Fig. 3b. The current id70 exports use Fig. 1g–i. Preserve archived filenames when updating manuscript captions and cross-references.
