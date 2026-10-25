# FANGS

## 1. Project overview

FANGS is a structure-guided workflow for nanobody framework selection and complementarity-determining region (CDR) grafting. It encodes the framework residues flanking each CDR as connector representations, compares donor and carrier structures, generates CDR-conditioned grafts with ESM3, and ranks candidate designs by connector compatibility. The repository supports library-wide grafting, humanization, and dual-motif window grafting.

- **Inputs:** a donor structure and CDR annotation table; carrier structures and corresponding CDR annotations; and, when available, precomputed carrier connector representations and sequences.
- **Outputs:** grafted sequences and structures, per-sample confidence and CDR reconstruction metrics, original and grafting connector scores, filtered candidate tables, and ranked designs. Experiment-specific summaries and figures are described in the protocols below.

## 2. Repository layout

| Directory | Contents |
|---|---|
| `data/` | Archived experiment datasets. Each protocol names the subdirectory it uses. Curated experiment data are deposited on Zenodo `https://doi.org/10.5281/zenodo.23535874`|
| `scripts/` | Workflow entry points grouped under `preprocess/`, `conditional_generation/`, `batch_generation/`, `grafting_generation/`, `humanization/`, and `connector_screening/`; shared utilities are in `tools/`. |
| `docs/` | Six experiment protocols specifying data sources, execution commands, and result locations. |
| `example/` | Donor and carrier inputs for the LaG16, VHH2, and B03 workflows. The quick-start grafting outputs are written to `example/outputs/`. |
| `esm/` | Project-local ESM3 implementation containing the custom structure-encoding methods. |
| `checkpoints/` | Local ESM3 weight files. |

## 3. Runtime requirements

Commands use a Linux Bash shell with the repository root as the working directory. Data, script, and result paths are relative to that root. The unified `fangs` Conda environment provides Python 3.10.14, PyTorch 2.2.0 with CUDA 11.8, NumPy 1.26.4, pandas 2.2.3, SciPy 1.15.2, Matplotlib 3.10.0, Biopython 1.84, and Gemmi 0.7.5. Biotite 0.41.2 is retained for compatibility with the project-local ESM3 structure encoder.

Create the environment using [environment.fangs.yml](environment.fangs.yml) and its pinned Python dependencies in [requirements-fangs.txt](requirements-fangs.txt). The requirements select the CUDA 11.8 PyTorch wheels from the official PyTorch index.

```bash
conda env create --file environment.fangs.yml
conda activate fangs
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

The module search path selects the repository-local `esm/` implementation, which provides the custom connector-encoding and generation methods. ESM3 code is at [evolutionaryscale/esm](https://github.com/evolutionaryscale/esm). Download `esm3_sm_open_v1` from [EvolutionaryScale/esm3-sm-open-v1](https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1) and place these weights in `checkpoints/` before running inference:

- `checkpoints/esm3_sm_open_v1.pth`
- `checkpoints/esm3_structure_encoder_v0.pth`
- `checkpoints/esm3_structure_decoder_v0.pth`
- `checkpoints/esm3_function_decoder_v0.pth`

Pass `--checkpoint ./checkpoints/esm3_sm_open_v1.pth` to load the main weights from that file. The structure encoder, structure decoder, and function decoder weights are read from the same directory. Tokenizer data is loaded from `esm/data/` or the Hugging Face cache used by the project loader.

| Step | Environment | Principal requirements |
|---|---|---|
| Connector encoding, ESM3 template reconstruction, and graft generation | `fangs` | Project-local ESM3, complete ESM3 model data, PyTorch 2.2.0, and CUDA 11.8; PIPPack for side-chain completion. |
| Human-framework sequence preparation | `fangs` | Biopython, ANARCI, and HMMER 3.3.2 with Kabat numbering support. |
| Archived-table summaries, statistics, sequence comparison, and paired structural analysis | `fangs` | NumPy, pandas, SciPy, Matplotlib, Biopython, Gemmi, and MAFFT 7.525. |
| AF3 template reconstruction | `alphafold3` | [AlphaFold 3](https://github.com/google-deepmind/alphafold3) |
| Humanized carrier structure prediction | `boltz` | Boltz-2, model parameters, PyTorch, and CUDA. |
| AbNatiV sequence scoring | `abnativ` | AbNatiV and its VH and VHH model parameters. The comparison protocol reads the archived AbNatiV scores and HuDiff sequences. |
| LaG16 global confidence prediction | `protenix` | [Protenix](https://github.com/bytedance/Protenix) |

ANARCI, HMMER, and MAFFT are installed with `environment.fangs.yml`. Additional external tools are **axel** for CIF retrieval, **USalign** for structural comparison, **MMseqs2** for sequence clustering and search, and **Foldseek** for global structure search. Make these executables available on `PATH` or supply their locations through the protocol arguments.

Foldseek, PIPPack, AlphaFold 3, and Boltz live outside the `fangs` environment. Point the variables below at the corresponding directories on the machine that runs those steps. Grafting scripts also accept `--foldseek-bin`, `--pippack-bin`, and `--pippack-home`. `scripts/tools/backfill_global_search.py` accepts `--env` for the Foldseek/MMseqs directory. `FOLDSEEK_BIN` can be omitted when `foldseek` and `mmseqs` are already on `PATH`. `BOLTZ_CACHE` can be omitted; `scripts/humanization/run_boltz.sh` then leaves the cache location to Boltz. `PIPPACK_HOME` and `AF3_HOME` have no command-name fallback.

| Variable | Used for |
|---|---|
| `FOLDSEEK_BIN` | Directory containing the `foldseek` and `mmseqs` executables |
| `PIPPACK_BIN` | Directory containing the PIPPack `python` executable |
| `PIPPACK_HOME` | PIPPack source directory |
| `AF3_HOME` | Directory that contains the `alphafold3` checkout. `AF3_ROOT` may name that checkout directly. `AF3_CONDA_ENV` selects the environment name (default `alphafold3`). |
| `BOLTZ_CACHE` | Optional Boltz parameter cache passed to `scripts/humanization/run_boltz.sh` |

The following exports are an example from one workstation:

```bash
export FOLDSEEK_BIN=/data1/dhuang/miniconda3/envs/foldseek/bin
export PIPPACK_BIN=/data1/dhuang/miniconda3/envs/pippack/bin
export PIPPACK_HOME=/data1/dhuang/PIPPack
export AF3_HOME=/data1/dhuang/af3
export BOLTZ_CACHE=/s0/dhuang/boltz/cache
```

Validate the scientific libraries, structural file processing, ANARCI, MAFFT, CUDA execution, and a single ESM3 encoding–generation–decoding cycle with:

```bash
python scripts/tools/check_fangs_environment.py \
  --output-dir results/fangs_environment_check \
  --device cuda:0 --load-model
```

## 4. Quick start

Graft the LaG16 CDRs from `example/6lr7B.pdb` onto the five carrier structures in `example/carrier_structures/`. Run the commands from the repository root in the `fangs` environment, after placing the ESM3 weights in `checkpoints/` as described above. Export `PIPPACK_BIN` and `PIPPACK_HOME` before the command so side-chain packing can find PIPPack. This connector-mode example does not run Foldseek. The carrier annotations are `example/carrier_structures/INDI_cdr_info.tsv`. `--ss 5` draws five structure samples for each carrier at temperature 0.7.

```bash
conda activate fangs
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python scripts/grafting_generation/graft_CDR_connector.py \
  -qp example/6lr7B.pdb \
  -qc example/6lr7B_CDR.tsv \
  -tp example/carrier_structures \
  -tc example/carrier_structures/INDI_cdr_info.tsv \
  -o example/outputs \
  --ss 5 \
  --len_limit 180 \
  --gpu 0 \
  --mode connector \
  --type 1 \
  --top_n 0 \
  --temperature 0.7 \
  --distance_limit 10 \
  --seed 42 \
  --checkpoint ./checkpoints/esm3_sm_open_v1.pth
```

The run writes 25 designs, five for each of `1zmyA`, `3k1kC`, `4dkaB`, `7nowA`, and `8taoC`. Sample records and coordinates are in `example/outputs/temp_generation/all_info/`. Connector distances are in `example/outputs/extract_distance/`. Side-chain models for the ten retained designs are in `example/outputs/side_chain_generation/`. The library-wide LaG16 workflow is documented in [LaG16 grafting, ranking, and experimental validation](docs/LaG16_grafting_ranking_and_experimental_validation.md).

## 5. Experiment protocols

1. [INDI database curation and connector representation](docs/INDI_database_curation_and_connector_representation.md): source annotations, structural preprocessing, connector encoding, and resolved-structure pairs with identical CDR sequences.
2. [ESM3 and AF3 template-conditioned reconstruction](docs/ESM3_and_AF3_template_conditioned_reconstruction.md): conditioning inputs, inference, selected-model manifests, and the paired reconstruction benchmark.
3. [id70 virtual grafting and global controls](docs/id70_virtual_grafting_and_global_controls.md): batch grafting, global metrics, the ΔConnector threshold scan, and cumulative donor-relative neighborhoods.
4. [LaG16 grafting, ranking, and experimental validation](docs/LaG16_grafting_ranking_and_experimental_validation.md): library-wide grafting, carrier sequence comparison, expression yield and GFP affinity, sampling depth, and Protenix confidence.
5. [VHH2 humanization and AbNatiV–HuDiff comparison](docs/Humanization_VHH2_library_ranking_and_AbNatiV_HuDiff_comparison.md): human-framework preparation, structural ranking, and the 15-candidate humanization comparison.
6. [B03 dual-motif window grafting](docs/B03_dual_motif_window_grafting.md): paired-window enumeration, graft generation, and original-score top-decile selection followed by grafting-score ranking.
