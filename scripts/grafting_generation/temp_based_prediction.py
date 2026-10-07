import re
import argparse
import sys, os, time, torch
import pandas as pd
import numpy as np
from esm.utils.structure.protein_chain import ProteinChain
from esm.models.esm3 import ESM3
from esm.sdk.api import (
    ESMProtein,
    ESMProteinTensor,
    GenerationConfig,
)
from biotite.structure.io.pdb import PDBFile
from tqdm import tqdm
from scripts.grafting_generation.extract_fasta import tsv_to_fasta

def set_cpu(cpu_num):
    os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

def find_cdr_start(pep_chain, cdr):
    # print(f'CDR: {cdr}')
    full_seq = pep_chain.sequence
    start = None
    end = None
    # print(f'Find CDR {cdr}')
    # Confirm that the CDR subsequence occurs in the full chain sequence.
    if cdr == None or cdr not in full_seq:
        print(f"WARNNING: CDR sequence '{cdr}' not found!") 
        # NOTE sometime foldseek's aligned sequence miss some residues, which will cause the CDR could not be found in esm3 program 
        return start, end
    # When the CDR is present, record its start index in the chain sequence.
    start = full_seq.index(cdr)
    end = start + len(cdr)   # +1
    return start, end

def check_structure_info(regions: list, dipep_chain: ProteinChain, template: list):
    # Check the motifs
    regions_sequence = []
    regions_atom37_positions = []
    regions_inds = np.arange(0)
    regions_start_end_pos = []
    selected_regions = []
    # For Nb, it need three regions
    # Used tempalte regions as limitation, when tempalte region was None, skipped correspond region
    for i in range(len(template)):
        if template[i] is None:
            continue

        if regions[i] is None:  # Do not has correspond grafting region
            return selected_regions, regions_start_end_pos, regions_sequence, regions_inds, regions_atom37_positions
        else:
            selected_regions.append(regions[i])

        # Obtain start pos and end pos of target region
        region_start, region_end = find_cdr_start(dipep_chain, regions[i])
        if region_start is None:    # Could not find full CDR
            regions_start_end_pos = []  # Initiated
            return selected_regions, regions_start_end_pos, regions_sequence, regions_inds, regions_atom37_positions

        regions_start_end_pos.append((region_start, region_end))
        # Construct numpy array of target region
        new_region_inds = np.arange(region_start, region_end)
        # Add new region array
        regions_inds = np.concatenate((regions_inds, new_region_inds),axis=0)
        # Obtain target sequence and atom37_positions
        # `ProteinChain` objects can be indexed like numpy arrays to extract the sequence and atomic coordinates of a subset of residues
        region_sequence = dipep_chain[new_region_inds].sequence
        region_atom37_positions = dipep_chain[new_region_inds].atom37_positions
        print("Region sequence: ", region_sequence)
        # print("Region atom37_positions shape: ", region_atom37_positions.shape)
        # Add seq and atom37 position to list
        regions_sequence.append(region_sequence)
        regions_atom37_positions.append(region_atom37_positions)

    # print(f'{len(regions_sequence)} regions are selected')
    # Return: List of sequence, numpy array and list of atom37 position contain (three) target region
    return selected_regions, regions_start_end_pos, regions_sequence, regions_inds, regions_atom37_positions

def prompt_design(model, linker_len, target_chain: ProteinChain, template_chain: ProteinChain, template_pos: list, target_pos: list, motifs_sequence: list, motifs_atom37_positions: list, graft_regions: list, target_regions: list, linker_design: str):
    target_sequence=target_chain.sequence
    # Add target FR sequence to fill the vacancy, but hold three GS-linker (GGS) in the terminal of target sequence
    # linker = 'GGS'
    linker = '_'*linker_len
    # linker_len = len(linker)

    # NOTE: Three region in target_regions (including start and end positions)
    print("Length of target sequence: ", len(target_sequence))
    print("Count of motif sequences need to graft into target: ", len(motifs_sequence))

    # Used in evaluation of prompt length
    template_regions_total_len = sum(len(item) for item in graft_regions)
    target_regions_total_len = sum(len(item) for item in target_regions)
    generated_prompt_len = len(target_sequence) - target_regions_total_len + template_regions_total_len + linker_len*len(target_pos)*2    # Two linker for each region

    # Design prompts including three region to be grafted with motifs, from CDR3 region to CDR1 region    
    pre_sequence_prompt = None
    generated_pos = []
    diff_len = None
    for region in range(len(graft_regions)):
        # Update the start pos in the next epoch
        diff_len = len(graft_regions[region]) - len(target_regions[region]) + linker_len*2    # Length difference after grafting
        generated_pos = [item + diff_len for item in generated_pos] if region != 0 else generated_pos

        # Original region need to be replaced in forward steps
        target_start = target_pos[region][0]
        target_end = target_pos[region][1]
        replace_len = target_end - target_start
        # print(f"Insert region start at: {target_start}, length will be replaced: {replace_len}")

        insert_start = target_start
        motif_sequence = motifs_sequence[region]
        print("Length of motif sequences: ", len(motif_sequence))
        insert_end = insert_start + linker_len + len(motif_sequence) # +1

        # Constract prompts with three region replaced
        # Read the unchanged seq and store at frist
        if pre_sequence_prompt == None: # The first grafting
            # Calculate the prompt for generation (after grafting)
            prompt_length = len(target_sequence) - replace_len + len(motif_sequence) + linker_len*2
            # print("Length of sequence prompt: ", prompt_length)
            sequence_prompt = ["_"]*prompt_length
            sequence_prompt[:insert_start] = list(target_sequence[:insert_start])

            # Insert end pos need to skip the linker region
            sequence_prompt[insert_end+linker_len*2:] = list(target_sequence[insert_start+replace_len:])

            # Insert linker seq
            sequence_prompt[insert_start+1:insert_start+1+linker_len] = list(linker)
            sequence_prompt[insert_end+linker_len:insert_end+linker_len*2] = list(linker)

        else:
            # Update grafted prompt, method same as previous steps
            # Build new sequence prompt including previous setting
            prompt_length = len(sequence_prompt) - replace_len + len(motif_sequence) + linker_len*2
            # print("Length of sequence prompt: ", prompt_length)
            sequence_prompt = ["_"]*prompt_length
            sequence_prompt[:insert_start] = list(pre_sequence_prompt[:insert_start])
            sequence_prompt[insert_end+linker_len*2:] = list(pre_sequence_prompt[insert_start+replace_len:])
        
        # Insert motif
        sequence_prompt[insert_start+linker_len:insert_end+linker_len] = list(motif_sequence)
        if linker_len != 0:
            sequence_prompt[insert_start:insert_start+linker_len] = list(linker)
            sequence_prompt[insert_end:insert_end+linker_len] = list(linker)
        sequence_prompt = "".join(sequence_prompt)
        pre_sequence_prompt = sequence_prompt
        print(f'Sequence_prompt: {sequence_prompt}')

        # Log the start pos in generated prompt
        generated_start_pos = insert_start + linker_len
        generated_pos.append(generated_start_pos)

    if linker_design != '':
        sequence_prompt = sequence_prompt.replace(linker, linker_design)  # linker = '_'*linker_len
        print(f"Sequence_prompt: {sequence_prompt}")
    if generated_prompt_len != len(sequence_prompt):
        print(f"ERROR: sequence prompt missed ({generated_prompt_len}!= {len(sequence_prompt)})")
    if len(generated_pos) != len(motifs_sequence):
        print(f"Start position miss ({len(generated_pos)} != {len(motifs_sequence)})")

    # Build coordinates prompt
    # Initiate coordinates prompt
    coordinates_prompt = torch.full((len(sequence_prompt), 37, 3), np.nan)
    for region in range(len(generated_pos)):
        insert_start = generated_pos[region]
        motif_atom37_positions = motifs_atom37_positions[region]

        # Insert template coordinates prompt
        coordinates_prompt[insert_start+linker_len:insert_start+linker_len+len(motif_atom37_positions)] = torch.tensor(motif_atom37_positions)

    # Initiate structure prompt
    # prompt = model.encode(ESMProtein(sequence=sequence_prompt, coordinates=coordinates_prompt))
    prompt = model.encode(ESMProtein(sequence=sequence_prompt))
    prompt.structure = torch.full_like(prompt.sequence, 4096)
    prompt.structure[0] = 4098
    prompt.structure[-1] = 4097

    template = ESMProtein.from_protein_chain(template_chain)
    template_tokens = model.encode(template)
    # print(f"structure tokens: {template_tokens.structure}")

    # Build structure and motif_inds, which will be used in comparation of crmsd
    motif_inds_in_generation = np.arange(0)
    for region in range(len(generated_pos)):
        # Attention: Embedding has additional signal at the head of token
        insert_start = generated_pos[region]
        motif_sequence = motifs_sequence[region]
        insert_end = insert_start + len(motif_sequence) # +1

        template_start = template_pos[region][0]
        template_end = template_pos[region][1]

        print(f'insert ({insert_start}:{insert_end}), template ({template_start}:{template_end})')
        if insert_end > insert_start and template_end > template_start and (template_end-template_start == insert_end-insert_start):
            template_to_graft = template_tokens.structure[template_start+1:template_end+1].clone()
            template_to_graft = template_to_graft.to(prompt.structure.device)
            # Insert template structure prompt
            # print(f"insert: {prompt.structure[insert_start:insert_end].shape}, template: {template_to_graft.shape}")
            prompt.structure[insert_start+1:insert_end+1].copy_(template_to_graft)

        else:
            print(f"Skipping invalid range: insert ({insert_start}:{insert_end}), template ({template_start}:{template_end})")

        motif_inds = np.arange(insert_start, insert_end)
        motif_inds_in_generation = np.concatenate((motif_inds_in_generation, motif_inds),axis=0)

    print(f"Row 1: Sequence prompt; Row 2: Masked sequence token; Row 3: Masked structure token.")
    print(f"_{sequence_prompt}_")
    print("".join(["#" if st == 32 else "_" for st in prompt.sequence]))
    # BUG: the first residue in template, will not less than 4096, so it fail to add the structure of first residue
    print("".join(["#" if st < 4096 else "_" for st in prompt.structure]))

    return prompt, motif_inds_in_generation, sequence_prompt

def merge_prompt(model, protein_sequence:str, encoded_protein:ESMProteinTensor, complex_sequence:str, complex_structure:torch.tensor):
        curr_length = len(complex_sequence) # Encoded length = curr_length + 2
        complex_sequence = f'{complex_sequence}|{protein_sequence}'

        # Initiate encoded tokens based on merged sequence
        encoded_complex: ESMProteinTensor = model.encode(ESMProtein(sequence=complex_sequence))
        encoded_complex.structure = torch.full_like(encoded_complex.sequence, 4096)

        # Set as '<cls>...<break>...<eos>'
        encoded_complex.structure[curr_length + 1] = 5000 # Insert the chainbreak token in the previous '<eos>' site

        # Insert the new structure tokens at the end of break
        encoded_complex.structure[:curr_length + 1].copy_(complex_structure[:-1])    # Remove the '<eos>', but retain the '<cls>'
        current_encoded_protein = encoded_protein.structure[1:].clone() # Remove the '<cls>'
        encoded_complex.structure[curr_length + 2:].copy_(current_encoded_protein)
        
        return complex_sequence, encoded_complex

def complex_generation(model, designed_prompt:ESMProteinTensor, designed_sequence:str, ag_path):
    # Generated original antigen prompt
    atom_array = PDBFile.read(ag_path).get_structure(
                model=1, extra_fields=["b_factor"]
            )
    antigen_chains: list = atom_array.chain_id
    antigen_chains = list(set(antigen_chains))
    antigen_sequence = None

    for chain in antigen_chains:
        protein = ProteinChain.from_pdb(ag_path, chain)
        encoded_protein: ESMProteinTensor = model.encode(ESMProtein.from_protein_chain(protein))
        if antigen_sequence != None:
            # Add the protein.sequence and encoded_protein to the tail of antigen_sequence and antigen_structure
            antigen_sequence, encoded_protein = merge_prompt(model, protein.sequence, encoded_protein, antigen_sequence, antigen_structure)  # 'antigen_sequence' must unupdated
            antigen_structure = encoded_protein.structure.clone()

        else:
            # The first protein in complex
            antigen_sequence = protein.sequence
            antigen_structure = encoded_protein.structure.clone()
        
    # Merge designed prompt with antigen complex
    complex_sequence, encoded_complex = merge_prompt(model, antigen_sequence, encoded_protein, designed_sequence, designed_prompt.structure)

    print(f"Complex prompt:\nRow 1: Sequence prompt; Row 2: Masked sequence token; Row 3: Masked structure token. (Masked: #)")
    print(f"_{complex_sequence}_")
    print("".join(["#" if st == 32 else "_" for st in encoded_complex.sequence]))
    # BUG: the first residue in template, will not less than 4096, so it fail to add the structure of first residue
    print("".join(["#" if st >= 4096 else "_" for st in encoded_complex.structure]))

    return encoded_complex

def sequence_generate(model, prompt:ESMProtein, sequence_sample:int, linker_len):
    num_mask = linker_len*2 # Two terminal for each inserted motif
    generated_sequences = []
    for i in range(sequence_sample):
        sequence_generation = model.generate(
                prompt,
                GenerationConfig(
                    # Generate a structure.
                    track="sequence",
                    # Sample one token per forward pass of the model.
                    num_steps=num_mask,
                    # Sampling temperature trades perplexity with diversity.
                    temperature=0.5,
                )
            )

        print(f"Generated sequence and structure prompt in {i+1} epoch")
        print("".join(["#" if st == 32 else "_" for st in sequence_generation.sequence]))
        print("".join(["#" if st < 4096 else "_" for st in sequence_generation.structure]))
        generated_sequences.append(sequence_generation)
    return generated_sequences

def _is_cuda_oom(exc: BaseException) -> bool:
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    return "out of memory" in str(exc).lower()


def protein_generate(model, structure_sample:int, prompt:torch.Tensor, template_chain:ProteinChain, motifs_inds:np.arange, motifs_inds_in_generation:np.arange, designed_sequence:str, ag, temperature=0.7, batch_size=1):
    all_generated_designs = []
    pass_generated_designs = []
    batch_size = max(1, int(batch_size))
    # We may need to sample more generations from ESM and sort by the generations with the highest predicted TM-score (pTM) by ESM3.
    print(f'Generate {structure_sample} structure (batch_size={batch_size})')
    num_tokens_to_decode = (prompt.structure == 4096).sum().item()
    generation_config = GenerationConfig(
        track="structure",
        num_steps=num_tokens_to_decode,
        temperature=temperature,
    )

    def _prepare_decode_input(structure_generation):
        # When using antigen to generate complex, the designed protein need to extract, unless it will happen 'device-side assert triggered'
        if ag == '':
            return structure_generation
        design_len = len(designed_sequence) + 2   # encoded design need more two letter
        design_sequence = structure_generation.sequence[:design_len].clone()
        design_structure = structure_generation.structure[:design_len].clone()
        design_sequence[-1] = 2
        design_structure[-1] = 4097
        return ESMProteinTensor(sequence=design_sequence, structure=design_structure)

    def _evaluate_decoded_protein(structure_generation, structure_generation_protein):
        print("structure_generation length:", len(structure_generation))
        generation_chain = structure_generation_protein.to_protein_chain()
        ptm = structure_generation_protein.ptm.item()
        print("\nPTM of generated protein: {:.3f}".format(ptm))

        generated_motif_sequence = generation_chain[motifs_inds_in_generation].sequence
        template_motif_sequence = template_chain[motifs_inds].sequence
        print(f"Template motif: {template_motif_sequence}\tGenerated motif: {generated_motif_sequence}")
        if generated_motif_sequence != template_motif_sequence:
            print("ERROR in 'motifs_inds' selection")
            return None

        generation_chain_aligned = generation_chain.align(template_chain, mobile_inds=motifs_inds_in_generation, target_inds=motifs_inds)
        target_crmsd = generation_chain_aligned.rmsd(template_chain, mobile_inds=motifs_inds_in_generation, target_inds=motifs_inds)
        print("Target cRMSD of the motif in the generated structure vs the original structure: {:.3f}".format(target_crmsd))

        c_pass = "Pass" if (float(target_crmsd) < 1.5 and float(ptm) > 0.8) else "Fail"
        print(f"Constrained site RMSD: {target_crmsd:.3f} Ang {c_pass}")

        structure_generation_protein_aligned = ESMProtein.from_protein_chain(generation_chain_aligned)
        return (generation_chain_aligned.sequence, structure_generation_protein_aligned, ptm, target_crmsd, c_pass)

    def _generate_token_batch(num_samples: int):
        if num_tokens_to_decode == 0:
            return [prompt] * num_samples
        if num_samples == 1 and batch_size <= 1:
            return [model.generate(prompt, generation_config)]
        return model.generate_batch(prompt, generation_config, num_samples=num_samples)

    def _decode_token_batch(tensors):
        print(f'Decode {len(tensors)} structure (batch_size={len(tensors)})')
        if len(tensors) == 1:
            return [model.decode(tensors[0])]
        return model.decode_batch(tensors)

    def _decode_with_oom_fallback(tensors):
        decoded = []
        remaining_tensors = list(tensors)
        decode_bs = len(remaining_tensors) if remaining_tensors else 1
        while remaining_tensors:
            decode_bs = min(decode_bs, len(remaining_tensors))
            chunk = remaining_tensors[:decode_bs]
            try:
                decoded.extend(_decode_token_batch(chunk))
                remaining_tensors = remaining_tensors[decode_bs:]
            except Exception as e:
                if decode_bs > 1 and _is_cuda_oom(e):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    decode_bs = max(1, decode_bs // 2)
                    print(f"[Warning] CUDA OOM during structure decode; reducing decode_batch_size to {decode_bs}")
                    continue
                raise
        return decoded

    remaining = structure_sample
    cur_bs = min(batch_size, remaining) if remaining > 0 else 1
    while remaining > 0:
        cur_bs = min(cur_bs, remaining)
        try:
            structure_generations = _generate_token_batch(cur_bs)
        except Exception as e:
            if cur_bs > 1 and _is_cuda_oom(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                cur_bs = max(1, cur_bs // 2)
                print(f"[Warning] CUDA OOM during structure generation; reducing structure_batch_size to {cur_bs}")
                continue
            raise

        prepared = [_prepare_decode_input(item) for item in structure_generations]
        decoded_proteins = _decode_with_oom_fallback(prepared)
        for structure_generation, structure_generation_protein in zip(prepared, decoded_proteins):
            design = _evaluate_decoded_protein(structure_generation, structure_generation_protein)
            if design is None:
                continue
            sequence, aligned_protein, ptm, target_crmsd, c_pass = design
            if c_pass == "Pass":
                pass_generated_designs.append((sequence, aligned_protein, ptm, target_crmsd))
            all_generated_designs.append((sequence, aligned_protein, ptm, target_crmsd))
        remaining -= cur_bs

    return pass_generated_designs, all_generated_designs


# Output: FR name, sequence, pTM, cRMSD. Save structure.
def output_designs(designs, filename, path):
    # designs -- 0: structure; 1: aligned_structure_prediction; 2: ptm; 3:target_crmsd;
    pdb_path = os.path.join(path,"pdb/")
    if not os.path.exists(pdb_path):
        os.makedirs(pdb_path)

    filename = os.path.join(path, filename)
    with open(filename, "w") as f:
        f.write('ID\tPDB_ID\tSequence\tpTM\tcRMSD\tFilename\n')
        for i in range(len(designs)):
            # id, FR_path, seq, ptm, crmsd, filename
            fr_path = os.path.basename(designs[i][0]).removesuffix(".pdb")  # Filename without '.pdb'
            output_formate = "{}\t{}\t{}\t{:.3f}\t{:.3f}\t{}\n".format(i, fr_path, designs[i][1], designs[i][3], designs[i][4], f"{fr_path}_generated_{i}.pdb")
            f.write(output_formate)
            pdb_filename = os.path.join(pdb_path, f"{fr_path}_generated_{i}.pdb")
            designs[i][2].to_pdb(pdb_filename)

def main(graft_info_path, output_path, structure_sample, sample_to_store, sequence_sample, linker_len, linker_design, device, cpu_num, ag, len_limit, model=None, model_lock=None, s_temperature=0.7, output_stream=None, structure_batch_size=1, bf16=False):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    # cpu_num = 16
    set_cpu(cpu_num)

    device = torch.device(device)
    output_stream = output_stream or sys.stdout
    owns_model = model is None
    if model is None:
        model = ESM3.from_pretrained("esm3_sm_open_v1").to(device)
    # Only cast a self-owned model: a shared model may already have fp32
    # structure encoder/decoder registered as submodules, and re-casting
    # would silently convert them to bf16.
    if bf16 and owns_model and device.type == 'cuda':
        model = model.to(torch.bfloat16)
    model.eval()

    def _locked(fn, *args, **kwargs):
        if model_lock is not None:
            with model_lock:
                return fn(*args, **kwargs)
        return fn(*args, **kwargs)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if linker_design != '':
        linker_len = len(linker_design)
    print(f'Linker length will be generated: {linker_len}\nProvided linker: {linker_design}\nSamples to generate structure: {structure_sample}\nStructure batch size: {structure_batch_size}\nModel dtype: {next(model.parameters()).dtype}\nSamples to store: {sample_to_store}\nSamples to generate sequence (function when linker need to be generated): {sequence_sample}', file=output_stream)

    # Read info about regions to be grafted
    graft_region_info = pd.read_csv(graft_info_path, sep = "\t")

    target_info_list = []
    cols = ['cdr3', 'cdr2', 'cdr1']
    for index, row in graft_region_info.iterrows():
        def get_val(col):
            val = row[col]
            if pd.isna(val) or val == '':
                return None
            return val

        if index == 0:
            graft_pep   = row['pdb_path']
            graft_chain = row['chain'] if pd.notna(row['chain']) else None
            # Preserve None so that an absent CDR retains its positional slot.
            graft_regions = [get_val(c) for c in cols]
        else:
            target_pep   = row['pdb_path']
            target_chain = row['chain'] if pd.notna(row['chain']) else None
            target_cdrs  = [get_val(c) for c in cols]
            target_info_list.append((target_pep, target_chain, target_cdrs))

    # Calculate each FR in provided table
    pass_target_generated_designs = []
    all_target_generated_designs = []

    # Read graft pep at first
    try:
        graft_dipep_chain = ProteinChain.from_pdb(graft_pep, graft_chain)
    except Exception as e:
        print(f"Error occurred in ProteinChain.from_pdb of {graft_pep}: {e}")
        graft_dipep_chain = ProteinChain.from_pdb(graft_pep)
        print("Read the first chain in file...")

    graft_regions, grafts_start_end_pos, grafts_sequence, grafts_inds, grafts_atom37_positions = check_structure_info(regions=graft_regions, dipep_chain=graft_dipep_chain, template=graft_regions)


    for target_pep, target_chain, target_cdrs in tqdm(target_info_list, desc="Processing Targets", unit="target"):
    # for target_pep, target_chain, target_cdrs in target_info_list:
        print(f'PDB path: {graft_pep} and {target_pep}')
        # `ProteinChain` class from the `esm` sdk to grab a protein structure from the PDB
        # `ProteinChain` also contains an `atom37_positions` numpy array that contains the atomic coordinates of each of the residues in the protein.
        try:
            target_dipep_chain = ProteinChain.from_pdb(target_pep, target_chain)
        except Exception as e:
            print(f"Error occurred in ProteinChain.from_pdb of {target_pep}: {e}")
            target_dipep_chain = ProteinChain.from_pdb(target_pep)
            print("Read the first chain in file...")
        print(f'Original sequence: {graft_dipep_chain.sequence}\nTarget sequence: {target_dipep_chain.sequence}')

        # Updated: add template region when collect structure info (reflash the target_cdrs based on graft_region length)
        target_cdrs, replaces_start_end_pos, replaces_sequence, replaces_inds, replaces_atom37_positions = check_structure_info(regions=target_cdrs, dipep_chain=target_dipep_chain, template=graft_regions)

        if replaces_start_end_pos is None or len(grafts_start_end_pos) != len(replaces_start_end_pos):
            print(f'[ERROR] {target_pep} miss corresponded region for grafting protein. Skipped...')
            continue

        # Design CDR in all FRs, input motif information
        try:
            designed_prompt, motif_inds_in_generation, sequence_prompt = _locked(
                prompt_design,
                model=model, linker_len=linker_len, target_chain=target_dipep_chain,
                template_chain=graft_dipep_chain, template_pos=grafts_start_end_pos,
                target_pos=replaces_start_end_pos, motifs_sequence=grafts_sequence,
                motifs_atom37_positions=grafts_atom37_positions, graft_regions=graft_regions,
                target_regions=target_cdrs, linker_design=linker_design
            )
        except Exception as e:
            print(f"[ERROR] occurred in prompt_design for Grafting: {graft_pep} and FR: {target_pep}.  \n graft regions: {graft_regions} \n target cdrs: {target_cdrs}: {e}")
            continue
        
        # Filter for sequence_prompt avoid OOM (prompt has additional two signal)
        if len_limit != -1 and len(sequence_prompt) > (len_limit + 2):
            print(f'[Warning] {target_pep} length > len_limit ({len_limit}), skipped...')
            continue

        # Generate complex prompt if provided antigen
        if ag != '':
            print(f'Generate complex prompt based on provided {ag}')
            designed_prompt = _locked(
                complex_generation,
                model=model, designed_prompt=designed_prompt,
                designed_sequence=sequence_prompt, ag_path=ag
            )

        if linker_len != 0 and linker_design == '':
            generated_prompts = _locked(
                sequence_generate,
                model=model, prompt=designed_prompt,
                sequence_sample=sequence_sample, linker_len=linker_len
            )
            for designed_prompt in generated_prompts:
                pass_generated_designs, all_generated_designs = _locked(
                    protein_generate,
                    model=model, structure_sample=structure_sample, prompt=designed_prompt,
                    template_chain=graft_dipep_chain, motifs_inds=grafts_inds,
                    motifs_inds_in_generation=motif_inds_in_generation,
                    designed_sequence=sequence_prompt, ag=ag, temperature=s_temperature,
                    batch_size=structure_batch_size
                )
                for design in pass_generated_designs:
                    pass_target_generated_designs.append((target_pep, design[0], design[1], design[2], design[3]))
                for design in all_generated_designs:
                    all_target_generated_designs.append((target_pep, design[0], design[1], design[2], design[3]))
        else:  # Non linker or linker was specific
            pass_generated_designs, all_generated_designs = _locked(
                protein_generate,
                model=model, structure_sample=structure_sample, prompt=designed_prompt,
                template_chain=graft_dipep_chain, motifs_inds=grafts_inds,
                motifs_inds_in_generation=motif_inds_in_generation,
                designed_sequence=sequence_prompt, ag=ag, temperature=s_temperature,
                batch_size=structure_batch_size
            )
            for design in pass_generated_designs:
                pass_target_generated_designs.append((target_pep, design[0], design[1], design[2], design[3]))
            for design in all_generated_designs:
                all_target_generated_designs.append((target_pep, design[0], design[1], design[2], design[3]))
        

    # Select top N design and store    
    # Clean incomplete generation
    for design in all_target_generated_designs:
        if len(design) <= 4:
            print("Incomplete generation:", design[0])
    all_target_generated_designs = [design for design in all_target_generated_designs if len(design) > 4]
    if len(pass_target_generated_designs) == 0:
        print("Wraming: No generations pass")
    for design in pass_target_generated_designs:
        if len(design) <= 4:
            print("Incomplete passed generation:", design[0])
    pass_target_generated_designs = [design for design in pass_target_generated_designs if len(design) > 4]

    # Sorted by pTM/cRMSD
    sorted_all_generations = sorted(all_target_generated_designs, key=lambda x: x[3], reverse=True)    # Sorted by pTM
    sorted_pass_target_generated_designs = sorted(pass_target_generated_designs, key=lambda x: x[4], reverse=False) # Sorted by cRMSD
    sample_to_store = sample_to_store if len(sorted_pass_target_generated_designs) > sample_to_store else len(sorted_pass_target_generated_designs)
    print(f"Select {sample_to_store} generated protein")
    selected_designs = sorted_pass_target_generated_designs[:sample_to_store]

    # Output
    full_info_path = os.path.join(output_path, "all_info/")
    passed_info_path = os.path.join(output_path, "passed_info/")
    selected_info_path = os.path.join(output_path, "selected_info/")
    if not os.path.exists(full_info_path):
        os.makedirs(full_info_path)
    if not os.path.exists(passed_info_path):
        os.makedirs(passed_info_path)
    if not os.path.exists(selected_info_path):
        os.makedirs(selected_info_path)

    output_designs(selected_designs, "selected_generation.tsv", selected_info_path)
    output_designs(sorted_pass_target_generated_designs, "passed_generation.tsv", passed_info_path)
    output_designs(sorted_all_generations, "all_generation.tsv", full_info_path)

    # Generate FASTA file
    tsv_to_fasta(
        tsv_file=os.path.join(full_info_path, "all_generation.tsv"), 
        fasta_pdb_output=os.path.join(output_path, "all_pdb_id.fasta"), 
        fasta_filename_output=os.path.join(output_path, "all_filename.fasta"), 
        prefix = f"{os.path.basename(graft_pep).removesuffix('.pdb')}_"
        )

if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description="ESM template-based structure prediction")
    parser.add_argument("graft_info_path", type=str, help="graft_info_path")
    parser.add_argument("output_path", type=str, help="output_path")
    parser.add_argument("--gpu", type=str, default='', help="GPU")
    parser.add_argument("--linker_len", type=int, default=0, help="Auto generate linker based on linker_len")
    parser.add_argument("--linker_design", type=str, default='', help="Using linker provided")
    parser.add_argument("--structure_sample", type=int, default=15, help="structure_sample")
    parser.add_argument("--structure_batch_size", type=int, default=1, help="GPU mini-batch size for same-prompt structure samples")
    parser.add_argument("--sequence_sample", type=int, default=5, help="sequence_sample")
    parser.add_argument("--sample_to_store", type=int, default=5, help="sample_to_store")
    parser.add_argument("--cpu_num", type=int, default=8, help="CPU used in program")
    parser.add_argument("--ag", type=str, default='', help="Path to the antigen PDB path")
    parser.add_argument("--len_limit", type=int, default=-1, help="len_limit")
    parser.add_argument("--bf16", action="store_true", help="Cast ESM3 to bfloat16 for faster GPU generation")

    args = parser.parse_args()

    if args.gpu == '':
        print("Program run in CPU...")
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # structure_sample = 15  # Samples for each generation
    # sequence_sample = 5
    # sample_to_store = 5  # Final seleceted designs

    main(graft_info_path=args.graft_info_path, output_path=args.output_path, structure_sample=args.structure_sample, sample_to_store=args.sample_to_store, sequence_sample=args.sequence_sample, linker_len=args.linker_len, linker_design = args.linker_design, device=device, cpu_num=args.cpu_num, ag=args.ag, len_limit=args.len_limit, structure_batch_size=args.structure_batch_size, bf16=args.bf16)

    end_time = time.time()
    runtime = end_time - start_time
    print(f"Total runtime: {runtime} seconds")