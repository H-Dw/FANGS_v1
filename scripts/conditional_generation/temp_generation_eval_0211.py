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
    # 检查cdr是否在full_seq中
    if cdr == None or cdr not in full_seq:
        print(f"WARNNING: CDR sequence '{cdr}' not found!")
        return start, end
    # 如果cdr存在，找到它的起始位置
    start = full_seq.index(cdr)
    end = start + len(cdr)   # +1
    return start, end

def check_structure_info(regions : list, dipep_chain : ProteinChain):
    # Check the motifs
    regions_sequence = []
    regions_atom37_positions = []
    regions_inds = np.arange(0)
    regions_start_end_pos = []
    # For Nb, it need three regions
    for i in range(len(regions)):
        if regions[i] is None:
            continue
        # Obtain start pos and end pos of target region
        region_start, region_end = find_cdr_start(dipep_chain, regions[i])

        if region_start is None:    # Could not find full CDR
            regions_start_end_pos = []  # Initiated
            return regions_start_end_pos, regions_sequence, regions_inds, regions_atom37_positions

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
    return regions_start_end_pos, regions_sequence, regions_inds, regions_atom37_positions

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
    for region in range(len(target_regions)):
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

def protein_generate(model, structure_sample:int, prompt:torch.Tensor, template_chain:ProteinChain, motifs_inds:np.arange, motifs_inds_in_generation:np.arange, designed_sequence:str, ag):
    all_generated_designs = []
    # We may need to sample more generations from ESM and sort by the generations with the highest predicted TM-score (pTM) by ESM3.
    print(f'Generate {structure_sample} structure')
    for i in range(structure_sample):
        num_tokens_to_decode = (prompt.structure == 4096).sum().item()

        structure_generation = model.generate(
            prompt,
            GenerationConfig(
                # Generate a structure.
                track="structure",
                # Sample one token per forward pass of the model.
                num_steps=num_tokens_to_decode,
                # Sampling temperature trades perplexity with diversity.
                temperature=0.7,
            )
        )
        print("structure_generation length:", len(structure_generation))

        # When using antigen to generate complex, the designed protein need to extract, unless it will happen 'device-side assert triggered'
        if ag != '':
            # Extract the designed protein from complex
            design_len = len(designed_sequence) + 2   # encoded design need more two letter
            design_sequence = structure_generation.sequence[:design_len].clone()
            design_structure = structure_generation.structure[:design_len].clone()

            # Replace the <break> as <eos>
            design_sequence[-1] = 2
            design_structure[-1] = 4097

            design_protein = ESMProteinTensor(sequence=design_sequence, structure=design_structure)
            structure_generation_protein:ESMProtein = model.decode(design_protein)
        
        else:
            structure_generation_protein:ESMProtein = model.decode(structure_generation)

        # Decodes structure tokens to backbone coordinates.
        generation_chain = structure_generation_protein.to_protein_chain()
        ptm = structure_generation_protein.ptm.item()
        print("\nPTM of generated protein: {:.3f}".format(ptm))

        # Align the generated structure to the original structure using the motif residues
        # print(f"mobile_inds: {motifs_inds_in_generation}\ntarget_inds: {motifs_inds}")

        # Check motif_inds
        generated_motif_sequence = generation_chain[motifs_inds_in_generation].sequence
        template_motif_sequence = template_chain[motifs_inds].sequence
        print(f"Template motif: {template_motif_sequence}\tGenerated motif: {generated_motif_sequence}")
        if generated_motif_sequence != template_motif_sequence:
            print("ERROR in 'motifs_inds' selection")
            continue

        # generation_chain_aligned = generation_chain.align(template_chain, mobile_inds=motifs_inds_in_generation, target_inds=motifs_inds)
        # target_crmsd = generation_chain_aligned.rmsd(template_chain, mobile_inds=motifs_inds_in_generation, target_inds=motifs_inds)
        # print("Target cRMSD of the motif in the generated structure vs the original structure: {:.3f}".format(target_crmsd))

        # Detect different CDR's cRMSD
        differences = np.diff(motifs_inds_in_generation)
        split_indices = np.where(differences != 1)[0] + 1
        groups = np.split(motifs_inds_in_generation, split_indices)
        target_crmsds = []
        for i, group in enumerate(groups):
            # 'mobile_inds' equal with 'target_inds' when generated the same protein
            generation_chain_aligned = generation_chain.align(template_chain, mobile_inds=group, target_inds=group)
            target_crmsd = generation_chain_aligned.rmsd(template_chain, mobile_inds=group, target_inds=group)
            target_crmsds.append(target_crmsd)
            print(f"Target cRMSD of the motif {i} in the generated structure vs the original structure: {target_crmsd}")

        # Full result
        # Transformat to ESMProtein
        structure_generation_protein_aligned = ESMProtein.from_protein_chain(generation_chain_aligned)
        
        all_generated_designs.append((generation_chain_aligned.sequence, structure_generation_protein_aligned, ptm, target_crmsds))
            
    return all_generated_designs


# Output: FR name, sequence, pTM, cRMSD. Save structure.
def output_designs(designs, filename, path, structure_sample):
    # designs -- 0: structure; 1: aligned_structure_prediction; 2: ptm; 3:target_crmsd;
    pdb_path = os.path.join(path,"pdb/")
    if not os.path.exists(pdb_path):
        os.makedirs(pdb_path)

    filename = os.path.join(path, filename)
    with open(filename, "w") as f:
        f.write('ID\tPDB_ID\tSequence\tpTM\tCDR3_cRMSD\tCDR2_cRMSD\tCDR1_cRMSD\tFilename\n')
        for i in range(len(designs)):
            # id, FR_path, seq, ptm, crmsd, filename
            fr_path = os.path.basename(designs[i][0])[:-4]  # Filename without '.pdb'
            crmsds =  "\t".join(f"{crmsd:.3f}" for crmsd in designs[i][4])
            output_formate = "{}\t{}\t{}\t{:.3f}\t{}\t{}\n".format(i, fr_path, designs[i][1], designs[i][3], crmsds, f"{fr_path}_generated_{i%structure_sample}.pdb")
            f.write(output_formate)
            pdb_filename = os.path.join(pdb_path, f"{fr_path}_generated_{i%structure_sample}.pdb")
            designs[i][2].to_pdb(pdb_filename)

def main(pdb_path, cdr_info_path, output_path, structure_sample, sample_to_store, sequence_sample, linker_len, linker_design, device, cpu_num, ag):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    # cpu_num = 16
    set_cpu(cpu_num)

    device = torch.device(device)
    model = ESM3.from_pretrained("esm3_sm_open_v1").to(device)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if linker_design != '':
        linker_len = len(linker_design)
    print(f'Linker length will be generated: {linker_len}\nProvided linker: {linker_design}\nSamples to generate structure: {structure_sample}\nSamples to store: {sample_to_store}\nSamples to generate sequence (function when linker need to be generated): {sequence_sample}')

    # Read info about regions to be grafted
    cdr_region_info = pd.read_csv(cdr_info_path, sep = "\t")

    target_info_list = []

    for index, row in cdr_region_info.iterrows():
        target_pep = f'{row["PDBChain"]}.pdb'
        target_chain = target_pep[4:]   # PDBID + ChainID
        target_pep = os.path.join(pdb_path, target_pep)
        target_cdrs = [row['CDR3'], row['CDR2'], row['CDR1']]  # 创建一个列表
        target_cdrs = [cdr for cdr in target_cdrs if pd.notna(cdr)]
        target_info_list.append((target_pep, target_chain, target_cdrs))

    # Calculate each FR in provided table
    all_target_generated_designs = []

    for target_pep, target_chain, target_cdrs in tqdm(target_info_list, desc="Processing Targets", unit="target"):
    # for target_pep, target_chain, target_cdrs in target_info_list:
        print(f'PDB path: {target_pep}')
        # `ProteinChain` class from the `esm` sdk to grab a protein structure from the PDB
        # `ProteinChain` also contains an `atom37_positions` numpy array that contains the atomic coordinates of each of the residues in the protein.
        try:
            target_dipep_chain = ProteinChain.from_pdb(target_pep, target_chain)
        except Exception as e:
            print(f"Error occurred in ProteinChain.from_pdb of {target_pep}: {e}")
            target_dipep_chain = ProteinChain.from_pdb(target_pep)
            print("Read the first chain in file...")
        print(f'Target sequence: {target_dipep_chain.sequence}')

        replaces_start_end_pos, replaces_sequence, replaces_inds, replaces_atom37_positions = check_structure_info(regions=target_cdrs, dipep_chain=target_dipep_chain)

        # Set the grafting region as original region
        graft_dipep_chain = target_dipep_chain
        graft_regions = target_cdrs
        grafts_start_end_pos = replaces_start_end_pos
        grafts_sequence = replaces_sequence
        grafts_inds = replaces_inds
        grafts_atom37_positions = replaces_atom37_positions

        if replaces_start_end_pos == []:
            print(f'Protein {target_pep} could not find full CDR, skipping...')
            continue

        # Design CDR in all FRs, input motif information
        designed_prompt, motif_inds_in_generation, sequence_prompt = prompt_design(model=model, linker_len=linker_len, target_chain=target_dipep_chain, template_chain=graft_dipep_chain, template_pos=grafts_start_end_pos, target_pos=replaces_start_end_pos, motifs_sequence=grafts_sequence, motifs_atom37_positions=grafts_atom37_positions, graft_regions=graft_regions, target_regions=target_cdrs, linker_design=linker_design)

        if linker_len != 0 and linker_design == '':
            # Generate sequence (fill mask) after structure information has been provided
            generated_prompts = sequence_generate(model=model, prompt=designed_prompt, sequence_sample=sequence_sample, linker_len=linker_len)
            for designed_prompt in generated_prompts:
                all_generated_designs = protein_generate(model=model, structure_sample=structure_sample, prompt=designed_prompt, template_chain=graft_dipep_chain, motifs_inds=grafts_inds, motifs_inds_in_generation=motif_inds_in_generation, designed_sequence=sequence_prompt, ag=ag)
                for design in all_generated_designs:
                    all_target_generated_designs.append((target_pep, design[0], design[1], design[2], design[3]))
        else:  # Non linker or linker was specific
            # Output format: sequence, aligned_structure_prediction, ptm, crmsds
            all_generated_designs = protein_generate(model=model, structure_sample=structure_sample, prompt=designed_prompt, template_chain=graft_dipep_chain, motifs_inds=grafts_inds, motifs_inds_in_generation=motif_inds_in_generation, designed_sequence=sequence_prompt, ag=ag)
            for design in all_generated_designs:
                all_target_generated_designs.append((target_pep, design[0], design[1], design[2], design[3]))

    # Output
    full_info_path = os.path.join(output_path, "all_info/")
    if not os.path.exists(full_info_path):
        os.makedirs(full_info_path)
    output_designs(all_target_generated_designs, "all_generation.tsv", full_info_path, structure_sample)

if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description="ESM template-based structure prediction")
    parser.add_argument("pdb_path", type=str, help="pdb_path")
    parser.add_argument("cdr_info_path", type=str, help="cdr_info_path")
    parser.add_argument("output_path", type=str, help="output_path")
    parser.add_argument("--gpu", type=str, default='', help="GPU")
    parser.add_argument("--linker_len", type=int, default=0, help="Auto generate linker based on linker_len")
    parser.add_argument("--linker_design", type=str, default='', help="Using linker provided")
    parser.add_argument("--structure_sample", type=int, default=5, help="structure_sample")
    parser.add_argument("--sequence_sample", type=int, default=5, help="sequence_sample")
    parser.add_argument("--sample_to_store", type=int, default=5, help="sample_to_store")
    parser.add_argument("--cpu_num", type=int, default=8, help="CPU used in program")
    parser.add_argument("--ag", type=str, default='', help="Path to the antigen PDB path")

    args = parser.parse_args()

    if args.gpu == '':
        print("Program run in CPU...")
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(pdb_path=args.pdb_path, cdr_info_path=args.cdr_info_path, output_path=args.output_path, structure_sample=args.structure_sample, sample_to_store=args.sample_to_store, sequence_sample=args.sequence_sample, linker_len=args.linker_len, linker_design = args.linker_design, device=device, cpu_num=args.cpu_num, ag=args.ag)

    end_time = time.time()
    runtime = end_time - start_time
    print(f"Total runtime: {runtime} seconds")