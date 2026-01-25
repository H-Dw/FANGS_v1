# FANGS


## Batch run grafting for INDI dataset

```bash
nohup python scripts/batch_graft.py \
  data/dupl_INDI_passed_cleaned.tsv \
  data/INDI_database/tokenized_INDI_full_250612_2C/pdb_model0/ \
  data/INDI_database/tokenized_INDI_full_250612_2C/pdb2seq.fasta \
  data/INDI_database/INDI_info_full_cdr_info.tsv \
  data/INDI_database/batch_graft_generation/ \
  --tokenized_folder data/INDI_database/ \
  --gpu 3 \
  --ss 1 \
  --top 100 \
  --cpu 16 \
  > ./batch_graft_dupl_INDI_passed_1s_top100_all_connector.log &
```

| Parameter                                                           | Annotation |
| ------------------------------------------------------------------- | ---------- |
| `dupl_INDI_passed_cleaned.tsv`                                   | Input ID table (filtered INDI data set) |
| `tokenized_INDI_full_250612_2C/pdb_model0/`         | Folder storing structure PDB files |
| `tokenized_INDI_full_250612_2C/pdb2seq.fasta`       | Sequence FASTA file |
| `INDI_info_full_cdr_info.tsv`                              | Nanobody CDR table |
| `batch_graft_generation/`   | Output path |
| `--tokenized_folder`                                               | Tokenized data path |
| `--gpu 3`                                                          | GPU ID |
| `--ss 1`                                                           | Structure sample number (`1` means each designed sequence generates one structure) |
| `--top 100`                                                        | Top N results ranked by sequence and structure similarity |
| `--cpu 16`                                                         | Number of CPUs used |
| `> ...log &`                                                       | Log file (redirect output and run in background) |

**NOTE: Env set in 'batch_graft.py' need to be check !!**

**Results visualization can be found in 'notebook/INDI_batch_graft.ipynb'**

## Lag16 run grafting for INDI dataset

## Humanization of anti-TNFα nanobody

### 1. Download IMGT IGHV3 database (stored as IGHV3_aa_alleles.fasta);
### 2. Constructed fusion protein with IGHV3 gene and VHH2’s CDR3 and FR4 (mutated Q108L) [removed sequence including “*”, which used to mark a translation STOP-CODON within the V-region sequence, and would not be translated as full protein.]
```bash
python fusion_cdr3fr4.py --db IGHV3_aa_alleles.fasta --target 5M2J_CDR3_FR4.fasta --output IGHV3_5M2J_CDR3_FR4.fasta
```
### 3. General back-mutation based on Kabat numbering
```bash
# Renumbering
# Renumber the target sequence to Kabat numbering
python renumber.py --input 5m2j_VHH2_sequence.fasta --output 5m2j_VHH2_sequence_kabat.csv
# Renumber the fusion sequence to Kabat numbering
python renumber.py --input IGHV3_5M2J_CDR3_FR4.fasta --output IGHV3_5M2J_CDR3_FR4_kabat.csv

# Mutation
python mutation.py --target IGHV3_5M2J_CDR3_FR4_kabat.csv --mutation mutation_74A_94R.csv --output IGHV3_5M2J_CDR3_FR4_74A_94R.fasta
```

### 4. Structure prediction of the fusion protein
Predicted the structure of IGHV3_CDR3_FR4 fusion protein using Boltz-2
```bash
# Data preparation
python ../split_fasta.py -i IGHV3_5M2J_CDR3_FR4_74A_94R.fasta -o input_IGHV3_5M2J_CDR3_FR4_74A_94R/

# Run Boltz-2
nohup bash ../run_boltz.sh input_IGHV3_5M2J_CDR3_FR4_74A_94R/ boltz2_output_IGHV3_5M2J_CDR3_FR4_74A_94R/ 2 > boltz2_prediction_IGHV3_5M2J_CDR3_FR4_74A_94R.log &

python ../../extract_boltz_predictions.py ./boltz_results_input_IGHV3_5M2J_CDR3_FR4_74A_94R/predictions/ ./boltz2_output_IGHV3_5M2J_CDR3_FR4_74A_94R_extracted/
```

### 5. Grafted CDR123 (Kabat) to the fusion protein
```bash
# Grafting
nohup python ../code/graft_CDR_connector_0711.py -qp ./5m2jD.pdb -qc ./5m2jD_CDR.tsv -tp ./boltz2_output_IGHV3_5M2J_CDR3_FR4_74A_94R_extracted/pdb/ -o ./5m2jD_grafting_74A_94R_temp07_s50_250917/ --ss 50 --len_limit 180 --gpu 2 --mode connector --type 1 --temperature 0.7 --seed 42 > 5m2jD_grafting_74A_94R_temp07_s80_250902.log &
```