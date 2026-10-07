# Humanization: VHH2 Library Construction and AbNatiV–HuDiff Comparison

This document lists the existing data, workflow commands, and result locations for VHH2 humanization. The selected cohort contains **15 FANGS candidates**, with **9 passing both AbNatiV thresholds** (VH ≥0.80 and VHH ≥0.80).

All paths are relative to the repository root. Commands use the existing sequence files, structural outputs, and AbNatiV scores.

```bash
conda activate fangs
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
HUM=data/humanization
RUN="$HUM/5m2jD_grafting_74A_94R_temp07_s50_250917"
OUT="$RUN/analysis"
mkdir -p "$OUT"
```

## 1. VHH2 humanization library and structural ranking

### Data sources

- **VHH2 donor:** `example/5m2jD.pdb`, `example/5m2jD_CDR.tsv`, and `data/humanization/5m2j_VHH2_sequence.fasta`.
- **Human IGHV3 library:** `data/humanization/IGHV3_aa_alleles.fasta`.
- **CDR3–FR4 segment:** `example/5M2J_CDR3_FR4.fasta`, containing the Q108L substitution.
- **Framework substitutions:** `data/humanization/mutation_74A_94R.csv` (Kabat 74A and 94R).
- **Prepared sequences and numbering:** `data/humanization/IGHV3_5M2J_CDR3_FR4.fasta`, `IGHV3_5M2J_CDR3_FR4_kabat.csv`, and `IGHV3_5M2J_CDR3_FR4_74A_94R.fasta`.
- **Completed Boltz carrier models:** `data/humanization/boltz2_output_IGHV3_5M2J_CDR3_FR4_74A_94R_extracted/pdb/` and `json/`. These files are used directly as the carrier structure and confidence inputs.
- **Completed FANGS run:** `data/humanization/5m2jD_grafting_74A_94R_temp07_s50_250917/`.

### Sequence preparation commands

```bash
conda activate fangs
python scripts/humanization/fusion_cdr3fr4.py \
  --db "$HUM/IGHV3_aa_alleles.fasta" \
  --target example/5M2J_CDR3_FR4.fasta \
  --output "$HUM/IGHV3_5M2J_CDR3_FR4.fasta"

python scripts/humanization/renumber.py \
  --input "$HUM/IGHV3_5M2J_CDR3_FR4.fasta" \
  --output "$HUM/IGHV3_5M2J_CDR3_FR4_kabat.csv" \
  --processes 8

python scripts/humanization/mutation.py \
  --target "$HUM/IGHV3_5M2J_CDR3_FR4_kabat.csv" \
  --mutation "$HUM/mutation_74A_94R.csv" \
  --output "$HUM/IGHV3_5M2J_CDR3_FR4_74A_94R.fasta"
```

The prepared sequence files and numbering table are stored directly under `data/humanization/`.

### Grafting command and completed outputs

```bash
conda activate fangs
PYTHONPATH=. python scripts/grafting_generation/graft_CDR_connector.py \
  -qp example/5m2jD.pdb \
  -qc example/5m2jD_CDR.tsv \
  -tp "$HUM/boltz2_output_IGHV3_5M2J_CDR3_FR4_74A_94R_extracted/pdb" \
  -o "$RUN" \
  --ss 50 \
  --len_limit 180 \
  --gpu 2 \
  --mode connector \
  --type 1 \
  --top_n 0 \
  --temperature 0.7 \
  --distance_limit 10 \
  --seed 42
```

The completed results under `data/humanization/5m2jD_grafting_74A_94R_temp07_s50_250917/` include:

- `temp_generation/all_info/all_generation.tsv` and `pdb/`: generated sequences, metrics, and structures.
- `generation_tokenized_structure/`: generated connector representations and distance tables.
- `extract_distance/filter/`: sample-level scores after ΔConnector ≤10 filtering.
- `extract_distance/filter_best/`: per-carrier minimum distance summaries.
- `extract_distance/filter_best/original_raw_embeddings.tsv` and `grafted_raw_embeddings.tsv`: the ordered score tables used directly by the plotting workflow.

### Framework ranking plot

The Python plotting entry point preserves the original framework-pyramid plotting logic and reads the completed grafted score table.

```bash
conda activate fangs
python scripts/humanization/plot_vhh2_humanization_ranking.py \
  --scores-dir "$RUN/extract_distance/filter_best" \
  --n-selected 15 \
  --output-dir "$OUT" \
  --highlight 'IGHV3-23*04' 'IGHV3-23*01' 'IGHV3-20*04'
```

Outputs are stored in `data/humanization/5m2jD_grafting_74A_94R_temp07_s50_250917/analysis/`:

| File | Contents |
|---|---|
| `VHH2_pyramid.pdf`, `.svg`, `.png` | Framework ranking plot |
| `VHH2_grafted_framework_ranking.tsv` | Grafted-distance ranking and allele labels |
| `VHH2_plot_selected_candidates.tsv` | Candidates selected directly within the plotting workflow |
| `VHH2_plot_metadata.json` | Plot inputs and selection rule |

## 2. AbNatiV and HuDiff humanization comparison

### Data sources

- **Sequence input:** `data/humanization/abnativ_humanization/5m2jD_humanization_comparasion.fasta`.
- **Completed VH scores:** `data/humanization/abnativ_humanization/5m2jD_VH_abnativ_seq_scores.csv`.
- **Completed VHH scores:** `data/humanization/abnativ_humanization/5m2jD_VHH_abnativ_seq_scores.csv`.
- **Residue scores:** `data/humanization/abnativ_humanization/5m2jD_VH_abnativ_res_scores.csv` and `5m2jD_VHH_abnativ_res_scores.csv`.
- **Candidate-selection inputs:** `data/humanization/5m2jD_grafting_74A_94R_temp07_s50_250917/extract_distance/filter_best/original_raw_embeddings.tsv` and `grafted_raw_embeddings.tsv`.

The comparison uses 15 selected FANGS candidates, seven archived HuDiff sequences (`HuDiff_VHv_nano_0`–`HuDiff_VHv_nano_6`), parental VHH2 (`Original_5m2jD`), and TNF30 (`TNF30_8z8mD`). The plotting code reads the completed VH/VHH scores and selects candidates directly from the ordered connector-score tables. It retains the 15 lowest-distance original-score rows (`--n-selected 15`) and joins them to the grafted-score table, preserving grafted-rank order. The archived library contains 142 carriers.

### Candidate selection and comparison plot

```bash
conda activate fangs
python scripts/humanization/plot_abnativ_comparison.py \
  --vh "$HUM/abnativ_humanization/5m2jD_VH_abnativ_seq_scores.csv" \
  --vhh "$HUM/abnativ_humanization/5m2jD_VHH_abnativ_seq_scores.csv" \
  --scores-dir "$RUN/extract_distance/filter_best" \
  --n-selected 15 \
  --output-dir "$OUT/abnativ_comparison" \
  --n-resamples 10000 \
  --seed 20260831
```

Comparison outputs are stored in `data/humanization/5m2jD_grafting_74A_94R_temp07_s50_250917/analysis/abnativ_comparison/`:

| File | Contents |
|---|---|
| `Fig3d_abnativ_candidates.tsv` | The 15 candidates selected by the plotting code |
| `Fig3d_abnativ_source.tsv` | VH/VHH scores, group labels, and dual-threshold status |
| `Fig3d_abnativ_stats.tsv` | Group means, mean differences, and bootstrap intervals |
| `Fig3d_abnativ_metadata.json` | Thresholds, 9/15 result, resampling count, and seed |
| `Fig3d_abnativ_vh_vhh.pdf`, `.svg`, `.png` | Humanization comparison panel (manuscript Fig. 3b) |
