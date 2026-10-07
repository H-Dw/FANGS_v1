# B03 Dual-Motif Window Grafting

The B03 HCDR3 motif `GDYD` and LCDR3 motif `WDNN` were grafted into four-residue windows of carrier CDR1 and CDR2, respectively, using h-cAbBCII-10 (PDB 3EAK, chain A). The 40 paired designs were evaluated by original and grafting connector scores. The final candidate was selected from the **top 10% of candidates ranked by original connector score** by choosing the lowest grafting connector score within that subset, yielding **SYST–SMGG**.

All paths are relative to the repository root. Activate the `fangs` environment and execute commands from that root. `BASE` identifies the existing input directory, `RUN` the completed generation run, and `OUT` its connector-score directory.

```bash
conda activate fangs
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
BASE=data/connector_screening
RUN="$BASE/8v13HL_CDR3_s50"
OUT="$RUN/filter_best_selected_scores"
```

## 1. Data sources

- **Donor motif coordinates and annotations:** `data/connector_screening/8v13HL.pdb` and `data/connector_screening/8v13HL_cdr3.tsv`. The annotation maps `GDYD` and `WDNN` to the program’s `CDR1` and `CDR2` slots.
- **Carrier coordinates and CDR annotations:** `data/connector_screening/carrier_structure/3eakA.pdb` and `data/connector_screening/3eakA_cdr.tsv`.
- **Paired-window inputs:** `data/connector_screening/3eakA_cdr12_4region.tsv` and `data/connector_screening/carrier_structure_3eakA_cdr12/`.
- **Generation and connector scores:** `data/connector_screening/8v13HL_CDR3_s50/`.

## 2. Window enumeration and graft generation

Enumerate all four-residue windows in CDR1 and CDR2 and combine them into 40 paired candidates. The argument `AAAA` specifies the window length; the donor annotation supplies the replacement motifs.

```bash
python scripts/connector_screening/generate_cdr_candidates.py \
  "$BASE/3eakA_cdr.tsv" \
  AAAA \
  CDR1,CDR2 \
  -o "$BASE/3eakA_cdr12_4region.tsv" \
  --pdb "$BASE/carrier_structure/3eakA.pdb" \
  --pdb_dir "$BASE/carrier_structure_3eakA_cdr12" \
  --mode cleaned

PYTHONPATH=. python scripts/grafting_generation/graft_CDR_connector.py \
  -qp "$BASE/8v13HL.pdb" \
  -qc "$BASE/8v13HL_cdr3.tsv" \
  -tp "$BASE/carrier_structure_3eakA_cdr12" \
  -tc "$BASE/3eakA_cdr12_4region.tsv" \
  -o "$RUN" \
  --ss 50 \
  --gpu 2 \
  --cpu_num 8 \
  --mode connector \
  --type 1 \
  --top_n 0 \
  --len_limit -1 \
  --temperature 0.7 \
  --distance_limit 10 \
  --seed 42
```

**Outputs:** `data/connector_screening/3eakA_cdr12_4region.tsv`, the carrier inputs under `data/connector_screening/carrier_structure_3eakA_cdr12/`, and the completed generation results under `data/connector_screening/8v13HL_CDR3_s50/`.

## 3. Connector scoring, candidate ranking, and visualization

Calculate the mean score for connectors 1 and 2. The bar-plot script reads the original and grafting score tables directly, assigns percentile groups using the original scores of all 40 candidates, and sorts candidates within each group by ascending grafting score. The `--top50` option additionally displays the top 50% of candidates by grafting score, with bar lengths representing grafting scores and colors representing original-score groups.

```bash
python scripts/connector_screening/extract_scores.py \
  --input "$RUN/extract_distance/filter_best" \
  --output "$OUT" \
  --index 1,2

python scripts/connector_screening/plot_insertion_bar.py \
  --cdr_info "$BASE/3eakA_cdr.tsv" \
  --values_color "$OUT/original_raw_embeddings.tsv" \
  --values_size "$OUT/grafted_raw_embeddings.tsv" \
  --out_dir "$RUN/insertion" \
  --cdr_indices 1,2 \
  --top50
```

The first candidate in the original-score top-decile group is `8v13HL_3eakA_CDR1-SYST_CDR2-SMGG`, which has the lowest grafting score within that group: `GDYD` replaces the carrier CDR1 window `SYST`, and `WDNN` replaces its CDR2 window `SMGG`.

## 4. Result locations

- **Archived generated sequences:** `data/connector_screening/8v13HL_CDR3_s50/temp_generation/all_info/all_generation.tsv`.
- **Archived connector-score inputs:** `data/connector_screening/8v13HL_CDR3_s50/extract_distance/filter_best/`.
- **Archived original and grafting scores:** `data/connector_screening/8v13HL_CDR3_s50/filter_best_selected_scores/original_raw_embeddings.tsv` and `grafted_raw_embeddings.tsv`.
- **Ranked plotting data:** `data/connector_screening/8v13HL_CDR3_s50/insertion/CDR1-CDR2_insertion_barplot.tsv` and `CDR1-CDR2_insertion_barplot_top50.tsv`.
- **Candidate-ranking figures:** `data/connector_screening/8v13HL_CDR3_s50/insertion/CDR1-CDR2_insertion_barplot.png` and `CDR1-CDR2_insertion_barplot_top50.png`.
