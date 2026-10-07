import argparse
import csv
import os
import shutil
import subprocess
import numpy as np
import pandas as pd
from esm.sdk.api import ESMProtein
from esm.models.esm3 import ESM3
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils import residue_constants


import torch

def find_cdr_start(pep_chain, cdr):
    print(f'PDB: {pep_chain.id}, CDR: {cdr}')
    sequence = pep_chain.sequence
    cdr_start = cdr_end = -1
    try:
        # Locate the CDR substring within the full chain sequence.
        cdr_start = sequence.find(cdr)
        cdr_end = cdr_start + len(cdr)
        # Verify that the CDR occurs in the full chain sequence.
        if cdr is None or cdr_start == -1:
            print(f"WARNNING: CDR sequence '{cdr}' not found in the {pep_chain.id}.")
        else:
            print(f'Find CDR: {sequence[cdr_start:cdr_end]}')
    except Exception as error:
        print(f"{pep_chain.id}: {str(error)}\n")

    return cdr_start, cdr_end  # Return -1 when the CDR cannot be located, so downstream steps can detect the failure.

def transformation(embeddings):
    assert embeddings.size(0) == 1, f"Batch size must be 1 to squeeze, but got {embeddings.size(0)}"
    matrix: torch.Tensor = embeddings.squeeze(0)  # Remove the batch dimension to obtain shape (L, C).
    return matrix

def tokenization(model, pep_chain):
    structure_raw_embeddings = None
    structure_pre_q_embeddings = None
    structure_q_embeddings = None
    structure_tokens = None

    protein = ESMProtein.from_protein_chain(pep_chain)

    # Tokenization
    structure_raw_embeddings, structure_pre_q_embeddings, structure_q_embeddings, structure_tokens = model.structure_encode_full(protein)

    print(f"Length of protein: {len(protein.sequence)}\nShape of embeddings:\nRaw: {structure_raw_embeddings.size()}\nPre-q: {structure_pre_q_embeddings.size()}\nTokens: {structure_tokens.size()}")

    return transformation(structure_raw_embeddings), transformation(structure_pre_q_embeddings), transformation(structure_q_embeddings), transformation(structure_tokens)

def calc_distance(coords: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances among the three-dimensional coordinates in coords.

    The calculation is vectorized over all coordinate pairs.
    """
    M = coords.shape[0]
    # Broadcast coordinate differences to shape (M, M, 3).
    diff = coords[:, None, :] - coords[None, :, :]  # shape = (M, M, 3)
    # Pairwise Euclidean distance matrix of shape (M, M).
    dist_mat = np.linalg.norm(diff, axis=-1)        # shape = (M, M)
    # Retain the strict upper triangle (i < j).
    i, j = np.triu_indices(M, k=1)
    return dist_mat[i, j]

def collect_connector_atom(pep_chain, cdr_pos_set, connector_len):
    atom_name = "CA"
    indices = residue_constants.atom_order[atom_name]

    full_connector = []
    for i in range(len(cdr_pos_set)):
        cdr_start, cdr_end = cdr_pos_set[i]
        head_connector = pep_chain.atom37_positions[cdr_start - connector_len : cdr_start, indices, :]
        tail_connector = pep_chain.atom37_positions[cdr_end : cdr_end + connector_len, indices, :]

        # Concatenate the N- and C-terminal flanks to shape (2 * connector_len, 3).
        connector = np.concatenate([head_connector, tail_connector], axis=0)
        # Pairwise-distance vector of length (2 * connector_len) * (2 * connector_len - 1) / 2.
        distance = calc_distance(connector) # Calculate distance between each residues
        full_connector.append(distance)

    return np.concatenate(full_connector, axis=0)   # → shape (N, D), N = len(cdr_pos_set)

def collect_connector(encoded_structure, cdr_pos_set, connector_len):
    # Connector of tokens
    full_connector = []
    for i in range(len(cdr_pos_set)):
        cdr_start, cdr_end = cdr_pos_set[i]

        # When using self-defined structrue encoder, output would not contain <EOS> and <BOS>
        # Auto process like encoded_structure[cdr_start - connector_len : cdr_start, :]
        head_connector = encoded_structure[cdr_start - connector_len : cdr_start]
        tail_connector = encoded_structure[cdr_end : cdr_end + connector_len]

        h_flat = head_connector.flatten()   # shape == (connector_len * C, )
        t_flat = tail_connector.flatten()   # shape == (connector_len * C, )
        
        full_connector.append(torch.cat([h_flat, t_flat], dim=0))  # (2*connector_len, 3)

    return torch.cat(full_connector, dim=0)

def output_connetor(ids_list, full_connector_structures, output_filename):
    # Output as DataFrame
    df = pd.DataFrame(full_connector_structures)
    df.insert(0, "ID", ids_list)
    df.to_csv(output_filename, sep='\t', index=False)

def main(table_path, pdb_path, output_path, connector_len, model=None):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable Hugging Face tokenizer parallelism.
    print(f"table_path: {table_path}\npdb_path: {pdb_path}\noutput_path: {output_path}\nconnector_len:{connector_len}\n")

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    passed_pdbs_folder = os.path.join(output_path, 'passed_pdbs/')
    os.makedirs(passed_pdbs_folder, exist_ok=True)


    atom_file = os.path.join(output_path, 'ca_distance.tsv')
    raw_embeddings_file = os.path.join(output_path, 'raw_embeddings.tsv')
    pre_q_embeddings_file = os.path.join(output_path, 'pre_q_embeddings.tsv')
    q_embeddings_file = os.path.join(output_path, 'q_embeddings.tsv')
    tokens_file = os.path.join(output_path, 'structure_tokens.tsv')

    model = ESM3.from_pretrained("esm3_sm_open_v1", device=torch.device("cpu")) if model is None else model

    with open(table_path, mode='r', encoding='utf-8') as file:
        reader = list(csv.DictReader(file, delimiter='\t'))  # Load the complete TSV table into memory.
    
    # Cache PDB filenames present on disk to avoid repeated filesystem checks.
    existing_pdb_files = set(os.listdir(pdb_path))

    cdr_keys = ['Sequence', 'CDR1', 'CDR2', 'CDR3', 'PDBChain']
    ids_list = []
    connector_atom_distances_list = []
    connector_raw_embeddings_list = []
    connector_pre_q_embeddings_list = []
    connector_q_embeddings_list = []
    connector_tokens_list = []

    for row in reader:
        for key in cdr_keys:
            if row[key] == '':
                row[key] = None
        pdb_file_name = row['PDBChain']
        if not pdb_file_name.endswith(".pdb"):
            pdb_file_name = f"{pdb_file_name}.pdb"

        hchain = row['PDBChain'][4:]
        hcdr1, hcdr2, hcdr3 = row['CDR1'], row['CDR2'], row['CDR3']
        hcdr_set = [hcdr1, hcdr2, hcdr3]

        # Filiter
        if pdb_file_name not in existing_pdb_files:
            print(f"ERROR: {pdb_file_name} didn't exist in {pdb_path}")
            continue

        # print(f"Process PDB: {pdb_file_name}")
        pdb_file_path = os.path.join(pdb_path, pdb_file_name)
        pep_chain = ProteinChain.from_pdb(pdb_file_path, id=os.path.basename(pdb_file_path))
        cdr_miss = False

        # Check CDR structure
        try:
            cdr_pos_set = []
            for cdr in hcdr_set:
                # When input part of CDRs, we only generate embeddings for provided regions (need )
                if cdr is None:
                    continue
                cdr_start, cdr_end = find_cdr_start(pep_chain, cdr)
                if cdr_start != -1 and cdr_end != -1:
                    cdr_pos_set.append((cdr_start, cdr_end))
                else:
                    cdr_miss = True
        except ValueError as e:
            print(f"ERROR: {pdb_file_path} -- {e}")
            continue

        if cdr_miss:
            print(f"ERROR: {pdb_file_name} -- miss CDRs")
            continue
        
        ids_list.append(pdb_file_name.replace(".pdb", ""))

        connector_atom_distances_list.append(collect_connector_atom(pep_chain, cdr_pos_set, connector_len))

        structure_raw_embeddings, structure_pre_q_embeddings, structure_q_embeddings, structure_tokens = tokenization(model, pep_chain)
        
        connector_raw_embeddings = collect_connector(structure_raw_embeddings, cdr_pos_set, connector_len)
        connector_pre_q_embeddings = collect_connector(structure_pre_q_embeddings, cdr_pos_set, connector_len)
        connector_q_embeddings = collect_connector(structure_q_embeddings, cdr_pos_set, connector_len)
        connector_tokens = collect_connector(structure_tokens, cdr_pos_set, connector_len)

        connector_raw_embeddings_list.append(connector_raw_embeddings.detach().cpu().numpy())
        connector_pre_q_embeddings_list.append(connector_pre_q_embeddings.detach().cpu().numpy())
        connector_q_embeddings_list.append(connector_q_embeddings.detach().cpu().numpy())
        connector_tokens_list.append(connector_tokens.detach().cpu().numpy())

        # Copy passed files
        shutil.copy2(pdb_file_path, passed_pdbs_folder)

    output_connetor(ids_list, connector_atom_distances_list, atom_file)
    output_connetor(ids_list, connector_raw_embeddings_list, raw_embeddings_file)
    output_connetor(ids_list, connector_pre_q_embeddings_list, pre_q_embeddings_file)
    output_connetor(ids_list, connector_q_embeddings_list, q_embeddings_file)
    output_connetor(ids_list, connector_tokens_list, tokens_file)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Tokenizer")
    parser.add_argument("table_path", type=str, help="Path to the nanobody data file")
    parser.add_argument("pdb_path", type=str, help="Path to the nanobody PDB data file")
    parser.add_argument("output_path", type=str, help="Path to the output directory")
    # parser.add_argument("extracted_pdb_path", type=str, help="Path to the extract selected pdb output directory")

    # Optional arguments.
    # parser.add_argument("--gpu", type=int, default=0, help="ID of the GPU to use (default: 0)")
    parser.add_argument("--connector_len", type=int, default=2, help="Connector length used in analysis (default: 2)")

    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

    main(args.table_path, args.pdb_path, args.output_path, args.connector_len)

