# id70 Virtual Grafting Data and Commands

Scope: Fig. 1g–i and their corresponding Results and Methods sections. Run all commands from the repository root. All dataset and script paths are relative to that root.

```bash
conda activate fangs
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
ANALYSIS=data/batch_graft_generation_id70_table/analysis
```

## 1. Source data

| Dataset | Repository-relative path |
|---|---|
| id70 representative annotations | `data/INDI_database/mmseqs_cluster_structure/id70/cdr_info.tsv` |
| Native representative structures | `data/INDI_database/mmseqs_cluster_structure/id70/cluster_rep_structures/` |
| Representative sequences | `data/INDI_database/mmseqs_cluster_structure/id70/cluster_rep_seq.fasta` |
| Donor-level generation records | `data/batch_graft_generation_id70_table/<donor>/temp_generation/all_info/all_generation.tsv` |
| Merged generated-sample records | `data/batch_graft_generation_id70_table/analysis/tables/merged_generations.tsv` |
| Selected donor–carrier designs | `data/batch_graft_generation_id70_table/analysis/tables/id70_selected_best_sample_per_design.tsv` |
| Native global controls | `data/batch_graft_generation_id70_table/analysis/tables/id70_native_pairwise_global_controls.tsv` |

`<donor>` is a `PDBChain` identifier in the id70 annotation table; for example, `data/batch_graft_generation_id70_table/1i3vA/temp_generation/all_info/all_generation.tsv`.

### Graft generation

```bash
python scripts/batch_generation/batch_graft.py \
  <(awk -F '\t' 'BEGIN {print "pdb_id"} NR>1 {print $5}' \
    data/INDI_database/mmseqs_cluster_structure/id70/cdr_info.tsv) \
  data/INDI_database/mmseqs_cluster_structure/id70/cluster_rep_structures \
  data/INDI_database/mmseqs_cluster_structure/id70/cluster_rep_seq.fasta \
  data/INDI_database/mmseqs_cluster_structure/id70/cdr_info.tsv \
  data/batch_graft_generation_id70_table \
  --mode table --gpus 0,1,2,3 --bf16 \
  --ss_batch 50 --ss 50 --top 0 --len_limit -1 --cpu 32
```

Output: `data/batch_graft_generation_id70_table/<donor>/`.

## 2. Virtual grafting and global metrics Fig. 1g

- **Data:** `data/batch_graft_generation_id70_table/analysis/tables/merged_generations.tsv`, `id70_selected_best_sample_per_design.tsv`, and `id70_native_pairwise_global_controls.tsv` in the same directory.
- **Final statistics:** `data/batch_graft_generation_id70_table/analysis/main/id70_main_correlation_stats.tsv`.
- **Calculation script:** `scripts/batch_generation/tools/analyze_id70_virtual_grafting.py`.

```bash
# Please replace the actual USalign path
USALIGN=../../../USalign/USalign
test -x "$USALIGN"

python scripts/batch_generation/tools/analyze_id70_virtual_grafting.py \
  --merged-generations "$ANALYSIS/tables/merged_generations.tsv" \
  --cdr-info data/INDI_database/mmseqs_cluster_structure/id70/cdr_info.tsv \
  --structure-dir data/INDI_database/mmseqs_cluster_structure/id70/cluster_rep_structures \
  --usalign "$USALIGN" \
  --tables-dir "$ANALYSIS/tables" \
  --output-dir "$ANALYSIS/main" \
  --figure-dir "$ANALYSIS/figures/main" \
  --workers 24 --n-resamples 10000 --seed 20260831 \
  --manuscript-only
```

Outputs: selected designs and native controls in `data/batch_graft_generation_id70_table/analysis/tables/`; statistics and cohort metadata in `data/batch_graft_generation_id70_table/analysis/main/`; figures in `data/batch_graft_generation_id70_table/analysis/figures/main/`.

### Figure export

```bash
python scripts/batch_generation/tools/plot_id70_main_correlation_stats.py \
  --stats-table "$ANALYSIS/main/id70_main_correlation_stats.tsv" \
  --output-dir "$ANALYSIS/figures/main"
```

Final figure: `data/batch_graft_generation_id70_table/analysis/figures/main/Fig1g_id70_virtual_grafting_comparison.{pdf,svg,png}`. The adjacent `_source.tsv` and `_metadata.json` files accompany the figure.

## 3. ΔConnector threshold scan Fig. 1h

- **Input:** `data/batch_graft_generation_id70_table/analysis/sensitivity/id70_selected_manuscript_designs.tsv`.
- **Final statistics:** `data/batch_graft_generation_id70_table/analysis/sensitivity/id70_threshold_curves.tsv` and `id70_threshold_anchor_statistics.tsv` in the same directory.
- **Calculation script:** `scripts/batch_generation/tools/analyze_id70_virtual_grafting_sensitivity.py`.

The following command calculates the statistics for both Fig. 1h and Fig. 1i.

```bash
python scripts/batch_generation/tools/analyze_id70_virtual_grafting_sensitivity.py \
  --selected-complete-table \
    "$ANALYSIS/sensitivity/id70_selected_manuscript_designs.tsv" \
  --output-dir "$ANALYSIS/sensitivity" \
  --figure-dir "$ANALYSIS/figures/main" \
  --threshold-min 0 --threshold-max 30 --threshold-step 1 \
  --anchors 5,8,10,12,15,20,30 \
  --within-donor-percentages 5,10,15,20,25,30,35,40,45,50 \
  --n-resamples 10000 --seed 20260831 \
  --manuscript-only
```

Outputs: statistics and input records in `data/batch_graft_generation_id70_table/analysis/sensitivity/`; figures in `data/batch_graft_generation_id70_table/analysis/figures/main/`.

## 4. Cumulative donor-relative neighborhoods Fig. 1i

- **Input:** `data/batch_graft_generation_id70_table/analysis/sensitivity/id70_selected_manuscript_designs.tsv`.
- **Final statistics:** `data/batch_graft_generation_id70_table/analysis/sensitivity/id70_within_donor_cumulative_stability.tsv`.
- **Calculation script and command:** `scripts/batch_generation/tools/analyze_id70_virtual_grafting_sensitivity.py`, using the command in Section 3.
- **Calculation output:** `data/batch_graft_generation_id70_table/analysis/sensitivity/`.

### Figure exports for Fig. 1h and Fig. 1i

```bash
python scripts/batch_generation/tools/replot_id70_deformation_proxy_panels.py \
  --table-dir "$ANALYSIS/sensitivity" \
  --output-dir "$ANALYSIS/figures/main"
```

| Manuscript panel | Final figure |
|---|---|
| Fig. 1h | `data/batch_graft_generation_id70_table/analysis/figures/main/Fig1h_id70_delta_connector_threshold_scan.{pdf,svg,png}` |
| Fig. 1i | `data/batch_graft_generation_id70_table/analysis/figures/main/Fig1i_id70_donor_cumulative_neighborhood.{pdf,svg,png}` |

The adjacent `_source.tsv` and `_metadata.json` files accompany both panels; Fig. 1h additionally has an `_anchor_statistics.tsv` file.
