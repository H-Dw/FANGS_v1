# Need to install Foldseek, ESM3 and mafft.
# Enable grafting based on defined CDR, rather than used structure alignment.
# Do not process pdbs in each grafting, used prepared database directly.
# NOTE 'FR_pdb_folder' should be same with 'seqs_file', or used pdb2seq program generated.
# Add extract distance program

import os, subprocess, sys, time, shutil, argparse, torch, glob
from scripts.batch_generation import similarity_search
from scripts.batch_generation import batch_extract_seqs
from scripts.batch_generation import transformat_generation_input

import scripts.grafting_generation.temp_based_prediction as temp_based_prediction
import pandas as pd
from packaging import version

from scripts.grafting_generation.tokenizer_INDI import main as tokenizer
from scripts.grafting_generation.tokenizer_grafting import main as tokenizer_grafted
from scripts.grafting_generation.calc_embed_distance_connector import main as calc_distance
from scripts.grafting_generation.extract_generation import extract_distance

from scripts.grafting_generation.define_random import set_global_seed, init_seed

SIDE_CHAIN_MAX_DEFAULT = 10


class GraftPipelineError(Exception):
    """Raised when the graft pipeline fails in batch mode."""


def _fail(message, raise_on_error=False):
    print(message)
    if raise_on_error:
        raise GraftPipelineError(message)
    sys.exit(1)


def _extract_chain_id(pdb_chain):
    if pdb_chain is None:
        return 'A'
    pdb_chain = str(pdb_chain)
    return pdb_chain[4:] if len(pdb_chain) > 4 else 'A'


def resolve_side_chain_script(code_path):
    candidates = [
        os.path.join(code_path, 'side_chain_generation.py'),
        os.path.abspath(os.path.join(code_path, '..', '..', '..', 'code', 'side_chain_generation.py')),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]

def set_cpu(cpu_num):
    os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

def input_selection(Nb_pdb_path, Nb_CDR_file, similarity_path, FR_pdb_folder, cdr_db_file, select_mode, select_distance, top_n, generation_path, generation_input_file, dupl_pdb=True):
    foldseek_df = None
    mmseqs_df = None

    if select_mode in ('foldseek', 'all'):
        similarity_file = os.path.join(similarity_path, 'similarity_foldseek.tsv')
        temp_generation_input_file = os.path.join(generation_path, "temp_input_structure.tsv")
        foldseek_df = transformat_generation_input.main(
            similarity_file, cdr_db_file, FR_pdb_folder, top_n, temp_generation_input_file
        )

    if select_mode in ('mmseqs', 'all'):
        similarity_file = os.path.join(similarity_path, 'similarity_mmseqs.tsv')
        temp_generation_input_file = os.path.join(generation_path, "temp_input_sequence.tsv")
        mmseqs_df = transformat_generation_input.main(
            similarity_file, cdr_db_file, FR_pdb_folder, top_n, temp_generation_input_file
        )

    if select_mode == 'all':
        parts = [df for df in (foldseek_df, mmseqs_df) if df is not None]
        if not parts:
            raise ValueError("No Foldseek or MMseqs candidates found for select_mode='all'")
        all_transformat_df = pd.concat(parts, axis=0, ignore_index=True)
        all_transformat_df = all_transformat_df.drop_duplicates(subset='pdb_path', keep='first')
    elif select_mode == 'foldseek':
        all_transformat_df = foldseek_df
    elif select_mode == 'mmseqs':
        all_transformat_df = mmseqs_df
    elif select_mode == 'connector':
        similarity_file = os.path.join(similarity_path, f'{select_distance}_similarity_all.tsv')
        if not os.path.exists(similarity_file):
            raise FileNotFoundError(
                f"Select_mode: {select_mode}, similarity file not exists: {similarity_file}"
            )
        temp_generation_input_file = os.path.join(generation_path, "temp_input_connector.tsv")
        similarity_df = pd.read_csv(similarity_file, sep='\t')
        merged_top_df = transformat_generation_input.extract_top(
            similarity_df, cdr_db_file, top_n, dupl_pdb=dupl_pdb
        )
        all_transformat_df = transformat_generation_input.transformat(
            merged_top_df, FR_pdb_folder, temp_generation_input_file
        )
    elif select_mode == 'table':
        table_df = pd.read_csv(cdr_db_file, sep='\t', dtype=str).fillna('')
        required = {'target', 'cdr1_seq', 'cdr2_seq', 'cdr3_seq'}
        missing = required - set(table_df.columns)
        if missing:
            raise KeyError(f"Missing table candidate columns: {sorted(missing)}")
        temp_generation_input_file = os.path.join(generation_path, "temp_input_table.tsv")
        all_transformat_df = transformat_generation_input.transformat(
            table_df, FR_pdb_folder, temp_generation_input_file
        )
    else:
        raise ValueError(f"Unsupported select_mode: {select_mode}")

    cdr_df = pd.read_csv(Nb_CDR_file, sep='\t', dtype=str).fillna('')
    for col in ['PDBChain', 'CDR1', 'CDR2', 'CDR3']:
        if col not in cdr_df.columns:
            raise KeyError(f"Column '{col}' not found in {Nb_CDR_file}")

    cdr1 = cdr_df.loc[0, 'CDR1']
    cdr2 = cdr_df.loc[0, 'CDR2']
    cdr3 = cdr_df.loc[0, 'CDR3']
    chain = _extract_chain_id(cdr_df.loc[0, 'PDBChain'])
    cdr3_length = len(str(cdr3))
    query_path = os.path.abspath(Nb_pdb_path)
    donor_id = str(cdr_df.loc[0, 'PDBChain'])

    temp_df = pd.DataFrame({
        'pdb_path': query_path,
        'chain': chain,
        'cdr1': cdr1,
        'cdr2': cdr2,
        'cdr3': cdr3,
        'cdr3_length': cdr3_length
    }, index=[0])

    for col in all_transformat_df.columns:
        if col not in temp_df.columns:
            temp_df[col] = pd.NA
    temp_df = temp_df[all_transformat_df.columns]

    print(f'temp_df:\n{temp_df}')

    all_transformat_df = all_transformat_df.drop_duplicates(subset='pdb_path', keep='first')
    candidate_ids = all_transformat_df['pdb_path'].map(
        lambda path: os.path.splitext(os.path.basename(str(path)))[0]
    )
    all_transformat_df = all_transformat_df[candidate_ids != donor_id]
    if select_mode == 'table' and top_n > 0:
        all_transformat_df = all_transformat_df.iloc[:top_n]
    all_transformat_df = pd.concat([temp_df, all_transformat_df], ignore_index=True)

    os.makedirs(os.path.dirname(generation_input_file), exist_ok=True)
    all_transformat_df.to_csv(generation_input_file, sep='\t', index=False)
    print(f"Count of final inputs: {len(all_transformat_df)}")

def grafting_generation(generation_input_file, temp_generation_path, output_path, device, cpu_num, ag_pdb_path, structure_sample, len_limit, s_temperature, model=None, model_lock=None, output_stream=None, structure_batch_size=1, bf16=False):
    sample_to_store=10
    pytorch_version = torch.__version__
    if version.parse(pytorch_version) <= version.parse("2.0.0"):
        print(f'Torch version {pytorch_version} unsatisfied with ESM3 requirement, used CPU to generate')
        esm_device = 'cpu'
    else:
        esm_device = device

    output_stream = output_stream or sys.stdout
    try:
        temp_based_prediction.main(
            graft_info_path=generation_input_file,
            output_path=temp_generation_path,
            structure_sample=structure_sample,
            sample_to_store=sample_to_store,
            sequence_sample=0,
            linker_len=0,
            linker_design='',
            device=esm_device,
            cpu_num=cpu_num,
            ag=ag_pdb_path,
            len_limit=len_limit,
            model=model,
            model_lock=model_lock,
            s_temperature=s_temperature,
            output_stream=output_stream,
            structure_batch_size=structure_batch_size,
            bf16=bf16,
        )
    except Exception as e:
        print(f"[ERROR] Fail to generate grafting: {e}", file=output_stream)
        raise

def sele_generation(generation_folder, select_n, side_chain_max=SIDE_CHAIN_MAX_DEFAULT):
    generation_df = pd.read_csv(os.path.join(generation_folder, 'all_generation.tsv'), sep='\t')

    if generation_df.empty:
        print('None acceptable generations')
        return generation_df

    print(f'Obtained {generation_df.shape[0]} acceptable generations')

    if select_n <= 0:
        select_n = generation_df.shape[0]
    select_n = min(select_n, side_chain_max, generation_df.shape[0])
    selected_data = generation_df.iloc[:int(select_n)]
    return selected_data

def side_chain(passed_generation_path, selected_data, ag_pdb_path, output_path, pippack_env, pippack_path, code_path, device):
    use_cuda = isinstance(device, torch.device) and device.type == 'cuda'
    gpu_id = 0 if use_cuda else None
    side_chain_script = resolve_side_chain_script(code_path)
    
    side_chain_path = os.path.join(output_path, 'side_chain_generation/')
    if os.path.exists(side_chain_path):
        print(f"side_chain path exists. Deleting old side_chain at: {side_chain_path}")
        shutil.rmtree(side_chain_path)
    os.makedirs(side_chain_path)

    # Write the data
    selected_data.to_csv(os.path.join(side_chain_path, 'dupl_selected_data.tsv'), sep='\t')
    
    log_output = []
    # for pdb in selected_data['Filename']:
    for pdb in selected_data.iloc[:, 5]:
        pdb_path = os.path.join(passed_generation_path, pdb)
        command = [
            f'{pippack_env}python',
            side_chain_script,
            pippack_path,
            pdb_path,
            side_chain_path
        ]
        if ag_pdb_path:
            command.extend(['--ag', ag_pdb_path])
        if gpu_id is not None:
            command.extend(['--gpu', str(gpu_id)])
        
        try:
            # Capture standard output and standard error from the side-chain packing command.
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            log_output.append(result.stdout)
        except subprocess.CalledProcessError as e:
            log_output.append(f"ERROR occurred: {e.stderr}")
        
    # Write the captured command output to the side-chain generation log.
    side_chain_log = os.path.join(output_path, "side_chain_generation.log")
    with open(side_chain_log, "w") as file:
        for log in log_output:
            file.write(log + "\n")

def convert(cdr_tsv, generation_path, output_path=None):
    # Paths
    gen_tsv = os.path.join(generation_path, 'all_info', 'all_generation.tsv')
    print(f'Convert based on {gen_tsv} and {cdr_tsv}')

    # Output file
    if output_path is None:
        output_path = os.path.join(generation_path, 'temp_output.tsv')

    # Read generation info
    gen_df = pd.read_csv(gen_tsv, sep='\t')
    # Read CDR info
    cdr_df = pd.read_csv(cdr_tsv, sep='\t')

    # Ensure required columns
    for col in ['CDR1', 'CDR2', 'CDR3']:
        if col not in cdr_df.columns:
            raise KeyError(f"Column '{col}' not found in {cdr_tsv}")

    # Take first row or where appropriate (assuming one set of CDRs per folder)
    cdr1 = cdr_df.loc[0, 'CDR1']
    cdr2 = cdr_df.loc[0, 'CDR2']
    cdr3 = cdr_df.loc[0, 'CDR3']

    cdr3_length = len(str(cdr3))

    # Build target DataFrame
    out_df = pd.DataFrame({
        'pdb_path': gen_df['Filename'].apply(
            lambda fn: os.path.join(generation_path, 'all_info', 'pdb', fn)
        ),
        'chain': 'A',
        'cdr1': cdr1,
        'cdr2': cdr2,
        'cdr3': cdr3,
        'cdr3_length': cdr3_length
    })

    # Write output
    out_df.to_csv(output_path, sep='\t', index=False)

def main(
    Nb_pdb_path, Nb_CDR_file, 
    FR_pdb_folder, FR_CDR_file, output_path, 
    ag_pdb_path, structure_sample, gpu, cpu_num, 
    non_generation, select_mode, s_type, top_n, len_limit, 
    seqs_file=None, tokenized_folder=None, 
    global_search=True, alignment=False, dupl_pdb=True, s_temperature=0.7,
    distance_limit=10.0, seed=None,
    model=None, model_lock=None, gpu_managed=False, output_stream=None,
    tmp_subdir=None, raise_on_error=False, side_chain_max=SIDE_CHAIN_MAX_DEFAULT, structure_batch_size=1, bf16=False):
    # Record the pipeline start time for subsequent runtime reporting.
    start_time = time.time()

    if seed is not None:
        set_global_seed(seed)
    else:
        init_seed(seed)

    print(f'Process: {Nb_pdb_path}. NOTE: The first row in the {Nb_CDR_file} was selected as target.')

    code_path = os.path.dirname(os.path.abspath(__file__)) + "/"
    if model is not None:
        device = next(model.parameters()).device
        search_gpu_id = 0 if device.type == 'cuda' else ""
    elif gpu == '':
        print("Program run in CPU...")
        device = torch.device("cpu")
        search_gpu_id = ""
    elif gpu_managed:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        search_gpu_id = 0 if device.type == 'cuda' else ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        gpu = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        search_gpu_id = gpu if device.type == 'cuda' else ""
    set_cpu(cpu_num)

    os.environ['MKL_THREADING_LAYER'] = 'GNU'   # For PIPPack

    # Check path !!!
    foldseek_env = "/data1/dhuang/miniconda3/envs/foldseek/bin/"
    pippack_env = "/data1/dhuang/miniconda3/envs/pippack/bin/"
    pippack_path = "/data1/dhuang/PIPPack/"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Searched similar proteins using Foldseek and MMSeqs2, and detected the alignment at connector
    similarity_path = os.path.join(output_path, "similarity/")
    if not os.path.exists(similarity_path):
        os.makedirs(similarity_path)

    search_tmp_dir = tmp_subdir or os.path.join(similarity_path, "search_tmp")
    os.makedirs(search_tmp_dir, exist_ok=True)

    # Run Foldseek and MMSeqs2
    if global_search:
        extract = True if alignment else False
        structure_similarity_result = similarity_search.main(
            foldseek_env, Nb_pdb_path, Nb_CDR_file, FR_pdb_folder, similarity_path,
            prefix="foldseek", cpu_num=cpu_num, gpu_id=search_gpu_id, extract=extract,
            tmp_dir=search_tmp_dir
        )
        print(structure_similarity_result)

        if seqs_file == '':
            seqs_file = os.path.join(similarity_path, 'pdb2seq.fasta')
            batch_extract_seqs.main(FR_pdb_folder, seqs_file)
        sequence_similarity_result = similarity_search.main(
            foldseek_env, Nb_pdb_path, Nb_CDR_file, seqs_file, similarity_path,
            prefix="mmseqs", cpu_num=cpu_num, gpu_id=search_gpu_id, extract=extract,
            tmp_dir=search_tmp_dir
        )
        print(sequence_similarity_result)

    if tokenized_folder is None:
        # If do not has pre tokenized files
        FR_CDR_file = os.path.join(similarity_path, f"foldseek_target_pdb_cdr_info.tsv") if alignment else FR_CDR_file  # extraFR_site output
        if select_mode in ('connector', 'table') and (FR_CDR_file is None or not os.path.exists(FR_CDR_file)):
            print(f"[ERROR] FR_CDR_file ({FR_CDR_file}) does not exist; {select_mode} mode needs FR_CDR_file")
        try:
            tokenized_folder = os.path.join(output_path, 'tokenized_db/')
            tokenizer(
                FR_CDR_file, FR_pdb_folder, tokenized_folder,
                connector_len=2, model=model, model_lock=model_lock
            )
        except Exception as e:
            _fail(f"[ERROR] Fail to tokenize using {FR_CDR_file} and {FR_pdb_folder}: {e}", raise_on_error)
    
    input_tokenized_folder = os.path.join(output_path, 'tokenized_input/')
    tokenizer(
        Nb_CDR_file, os.path.dirname(Nb_pdb_path), input_tokenized_folder,
        connector_len=2, model=model, model_lock=model_lock
    )

    # Calculate connector distance
    connector_distance_list = ['raw_embeddings', 'pre_q_embeddings', 'ca_distance']
    if s_type < 1 or s_type > len(connector_distance_list):
        _fail(f"[ERROR] --type need a value in [1-3]", raise_on_error)
    try:
        for table in connector_distance_list:
            output_file_path = os.path.join(similarity_path, f'{table}_similarity.tsv')
            calc_distance(
                query_table_path=os.path.join(input_tokenized_folder, f'{table}.tsv'),
                target_table_path=os.path.join(tokenized_folder, f'{table}.tsv'),
                output_similarity_tsv=output_file_path
            )
    except Exception as e:
        _fail(f"[ERROR] Fail to process similarity calculation for input template: {e}", raise_on_error)

    select_distance = connector_distance_list[s_type - 1]
    print(f"Used calculation s_type {s_type} ({select_distance}) for grafting selection")


    if not non_generation:
        # Use foldseek_tsv_filename directly, only consider global structure similariry
        generation_path = os.path.join(output_path, "temp_generation/")
        generation_input_file = os.path.join(generation_path, "temp_input.tsv")
        if not os.path.exists(generation_path):
            os.makedirs(generation_path)

        # Selected the input FR
        if not os.path.exists(FR_CDR_file):
            print(f"{FR_CDR_file} do not exist")
        fr_df = pd.read_csv(FR_CDR_file, sep='\t')
        fr_df = fr_df.rename(columns={
            'PDBChain': 'target',
            'CDR1':     'cdr1_seq',
            'CDR2':     'cdr2_seq',
            'CDR3':     'cdr3_seq'
        })
        cdr_db_file = os.path.join(similarity_path, os.path.basename(FR_CDR_file))
        fr_df.to_csv(cdr_db_file, sep='\t', index=False, encoding='utf-8')

        try:
            input_selection(
                Nb_pdb_path, Nb_CDR_file, similarity_path, FR_pdb_folder, cdr_db_file,
                select_mode, select_distance, top_n, generation_path, generation_input_file, dupl_pdb
            )
        except Exception as e:
            _fail(f"[ERROR] processing 'input_selection': {e}", raise_on_error)

        grafting_generation(
            generation_input_file, generation_path, output_path, device, cpu_num,
            ag_pdb_path, structure_sample, len_limit, s_temperature,
            model=model, model_lock=model_lock, output_stream=output_stream,
            structure_batch_size=structure_batch_size, bf16=bf16
        )

        # Collected embeddings and distance for generations
        output_table_path = os.path.join(generation_path, 'temp_output.tsv')
        try:
            convert(Nb_CDR_file, generation_path, output_table_path)
        except Exception as e:
            _fail(f"[ERROR] convert to temp_output '{generation_path}': {e}", raise_on_error)
        
        try:
            generation_embeddings_path = os.path.join(output_path, 'generation_tokenized_structure')
            tokenizer_grafted(
                output_table_path, generation_embeddings_path,
                connector_len=2, model=model, model_lock=model_lock
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            _fail(f"[ERROR] processing 'tokenizer_grafted': {e}", raise_on_error)

        # Calculate connector distance for generations
        try:
            for table in connector_distance_list:
                output_file_path = os.path.join(generation_embeddings_path, f'{table}_similarity.tsv')
                calc_distance(query_table_path=os.path.join(input_tokenized_folder, f'{table}.tsv'), target_table_path=os.path.join(generation_embeddings_path, f'{table}.tsv'), output_similarity_tsv=output_file_path)
        except Exception as e:
            _fail(f"[ERROR] Fail to process similarity calculation for grafted designs: {e}", raise_on_error)

        # Write process list
        process_file_path = os.path.join(output_path, 'process_file.tsv')
        with open(process_file_path, "w") as f:
            f.write(f'pdb_id\n{os.path.splitext(os.path.basename(Nb_pdb_path))[0]}')

        try:
            extract_distance(
                target_list_path=process_file_path,
                generation_folder=output_path,
                output_path=os.path.join(output_path, 'extract_distance/'),
                limitation=distance_limit
            )
        except Exception as e:
            print(f"[ERROR] Fail to process extract_distance for generations: {e}")

        # Generate side chain
        # Selected the generations which were sorted by cRMSD
        generation_folder = os.path.join(generation_path, 'all_info/')
        selected_data = sele_generation(generation_folder, top_n, side_chain_max=side_chain_max)

        if not selected_data.empty:
            generation_path = os.path.join(generation_folder, 'pdb/')
            side_chain(
                generation_path, selected_data, ag_pdb_path, output_path,
                pippack_env, pippack_path, code_path, device
            )

    else:
        print("Generation is disabled.")

    end_time = time.time()
    runtime = end_time - start_time
    
    return f'Finish\nTotal runtime: {runtime} seconds'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graft CDR")
    parser.add_argument("-qp", "--Nb_pdb_path", required=True, type=str, help="Path to the target grafting nanobody file path")
    parser.add_argument("-qc", "--Nb_CDR_file", required=True, type=str, help="Path to the target grafting region file path")
    parser.add_argument("-tp", "--FR_pdb_folder", required=True, type=str, help="Path to the framework database path")
    parser.add_argument("-o", "--output_path", required=True, type=str, help="Path to the output directory")
    parser.add_argument("-tc", "--FR_CDR_file", type=str, default=None, help="Path to the CDR region file extracted from framework database")
    parser.add_argument("--ag", type=str, default='', help="Path to the antigen PDB path")
    parser.add_argument("--ss", type=int, default=5, help="Number of structure_sample")
    parser.add_argument("--ss_batch", type=int, default=1, help="GPU mini-batch size for same-prompt structure samples")
    parser.add_argument("--gpu", type=str, default='', help="gpu_id")
    parser.add_argument("--cpu_num", type=int, default=8, help="CPU used in program")
    parser.add_argument(
        "--mode", type=str, default='connector',
        choices=['connector', 'table', 'all', 'foldseek', 'mmseqs'],
        help="candidate mode: connector, table (use FR_CDR_file rows directly), all, foldseek, mmseqs"
    )
    parser.add_argument("--type", type=int, default=1, help="[Only work when select connector as mode] type: 1 (raw_embeddings), 2 (pre_q_embeddings), 3 (ca_distance)")
    parser.add_argument("--top_n", type=int, default=0, help="top_n selected for generation (default: select all)")
    parser.add_argument("--len_limit", type=int, default=-1, help="len_limit")
    parser.add_argument("--seqs_file", type=str, default='', help="sequences for database")
    parser.add_argument("--tokenized_folder", type=str, default=None, help="[Only work when select connector as mode] tokenized files folder for database")
    parser.add_argument("--distance_limit", type=float, default=10.0, help="Limitation of connector change (recommand: 10-13)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for program (recommand: 42)")
    parser.add_argument("--bf16", action="store_true", help="Cast ESM3 to bfloat16 for faster GPU generation")
    
    # NOTE: follow argument do not activate before 9 July, and previous analysis used 0.7 (might too random). 
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature used for structure generation (recommand: 0.5-0.7)")

    parser.add_argument(
        "--global_search", 
        action="store_true", 
        help="Run global search"
    )
    parser.add_argument(
        "--non_generation", 
        action="store_true", 
        help="Disable grafting generation"
    )
    parser.add_argument(
        "--alignment", 
        action="store_true", 
        help="Using alignment method"
    )
    # NOTE: follow argument do not activate before 9 July, and previous analysis used automatic duplication. 
    parser.add_argument(
        "--dupl_pdb", 
        action="store_true", 
        help="Duplicate PDB file before generated structure based on four char at head."
    )

    args = parser.parse_args()

    # Auto active global search and alignment process
    if args.mode in ['all', 'foldseek', 'mmseqs'] or args.FR_CDR_file is None or args.alignment:
        args.global_search = True
        args.alignment = True
    
    if args.alignment:
        args.tokenized_folder = None

    print(f"Input: {args}")

    output = main(
        args.Nb_pdb_path, 
        args.Nb_CDR_file,
        args.FR_pdb_folder, 
        args.FR_CDR_file, 
        args.output_path, 
        args.ag, 
        args.ss, 
        args.gpu, 
        args.cpu_num, 
        args.non_generation, 
        args.mode, 
        args.type,
        args.top_n, 
        args.len_limit,
        args.seqs_file, 
        args.tokenized_folder, 
        args.global_search,
        args.alignment,
        args.dupl_pdb,
        args.temperature,
        args.distance_limit,
        args.seed,
        structure_batch_size=args.ss_batch,
        bf16=args.bf16
        )
    print(output)
