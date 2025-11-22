# FANGS


### Batch run grafting for INDI dataset

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