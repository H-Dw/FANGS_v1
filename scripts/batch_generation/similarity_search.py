import os, subprocess, sys
import pandas as pd
import readPDBSeq
import extraFR_site

# prefix can set as "foldseek" or "mmseqs"

def main(env, Nb_pdb_path, Nb_CDR_file, db_path, output_path, prefix, cpu_num, gpu_id="", extract=True):
    tsv_filename = os.path.join(output_path, os.path.basename(Nb_pdb_path).replace(".pdb", "") + f"_{prefix}_output.tsv")
    extract_file = os.path.join(output_path, os.path.basename(Nb_pdb_path).replace(".pdb", "") + f"_{prefix}_extract.tsv")

    if prefix == "mmseqs":
        # Build sequences file
        query_fasta_file = os.path.join(output_path, "query.fasta")

        Nb_seq = readPDBSeq.get_sequence_from_pdb(Nb_pdb_path)
        query = f">{os.path.basename(Nb_pdb_path).replace('.pdb', '')}\n{Nb_seq}\n"
        with open(query_fasta_file, 'w') as f:
            f.write(query)

    print(f"Running {prefix}...")
    gpu_setting = f" --gpu {gpu_id}" if gpu_id != "" else ""
    # Select top 1000 protein, ignore e-value
    if prefix == "foldseek":
        # run_command = f"{env}foldseek easy-search {Nb_pdb_path} {db_path} {tsv_filename} tmp --max-seqs 1000 --remove-tmp-files 1 --threads {cpu_num}{gpu_setting} --format-mode 4 --format-output 'query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,qtmscore,bits,qaln,taln,tseq' > /dev/null 2>&1"
        run_command = f"{env}foldseek easy-search {Nb_pdb_path} {db_path} {tsv_filename} tmp --max-seqs 1000 --remove-tmp-files 1 --threads {cpu_num}{gpu_setting} --format-mode 4 --format-output 'query,target,fident,alnlen,nident,mismatch,gapopen,qstart,qend,tstart,tend,evalue,qtmscore,bits,qaln,taln,tseq' > /dev/null 2>&1"
    elif prefix == "mmseqs":
        # run_command = f"{env}mmseqs easy-search {query_fasta_file} {db_path} {tsv_filename} tmp --max-seqs 1000 --remove-tmp-files 1 --threads {cpu_num}{gpu_setting} --format-mode 4 --format-output 'query,target,evalue,qstart,qend,tstart,tend,bits,qaln,taln,tseq' > /dev/null 2>&1"
        run_command = f"{env}mmseqs easy-search {query_fasta_file} {db_path} {tsv_filename} tmp --max-seqs 1000 --remove-tmp-files 1 --threads {cpu_num}{gpu_setting} --format-mode 4 --format-output 'query,target,fident,alnlen,nident,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,qaln,taln,tseq' > /dev/null 2>&1"

    else:
        print(f"Error: Program {prefix} do not support (foldseek, mmseqs)")
        sys.exit(1)

    result = subprocess.run(run_command, shell=True)
    if result.returncode != 0:
        print(f"Error: {prefix} couldn't be finished, return code:", result.returncode)
        sys.exit(1)

    print(f"Extract CDR from {prefix} result...")

    target_df = pd.read_csv(tsv_filename, sep='\t')
    target_df.columns = [col.strip().lower() for col in target_df.columns]

    pdb_column = target_df['target']
    bit_column = target_df['bits']
    qstart_column = target_df['qstart']
    tstart_column = target_df['tstart']
    qaln_column = target_df['qaln']
    taln_column = target_df['taln']
    tseq_column = target_df['tseq']

    similarity_df = pd.DataFrame({
        'pdb_id': pdb_column,
        'bits': bit_column,
        'tSeq': tseq_column
    })
    output_similarity = os.path.join(output_path, f"similarity_{prefix}.tsv")
    similarity_df.to_csv(output_similarity, sep='\t', index=False)

    aln_df = pd.DataFrame({
        'target': pdb_column,
        'qStartPos': qstart_column,
        'dbStartPos': tstart_column,
        'qAln': qaln_column,
        'dbAln': taln_column,
        'tSeq': tseq_column
    })
    aln_df.to_csv(extract_file, sep='\t', index=False)

    if extract:
        # 根据Foldseek的alignment，获取target（FR）对应的CDR seq和end index
        # log_file = os.path.join(output_path, f"{prefix}_extra.log")
        # with open(log_file, "w") as file:
        #     with redirect_stdout(file):
        #         extraFR_site.main(Nb_pdb_path, site_info_path, Nb_CDR_file, extract_file, output_path, prefix=prefix)

        try:
            extraFR_site.main(query_pdb=Nb_pdb_path, site_info_path="", cdr_info_path=Nb_CDR_file, extracted_data_path=extract_file, output_path=output_path, prefix=prefix)
        except Exception as e:
            print(f"[ERROR] Fail to process site extraction: {e}")

        target_output_file = os.path.join(output_path, f"{prefix}_target_pdb_CDR.tsv")
        if not target_output_file:
            return f"[ERROR] extraction fail. {prefix}_target_pdb_CDR.tsv could not generate\n"

        transformat_tsv = os.path.join(output_path, f"{prefix}_target_pdb_cdr_info.tsv")
        # Format: ['Sequence', 'CDR1', 'CDR2', 'CDR3', 'PDBChain']
        out_df = pd.read_csv(target_output_file, sep='\t', dtype=str)
        trans_df = pd.DataFrame({
            'Sequence': out_df['t_seq'],
            'CDR1':     out_df['cdr1_seq'],
            'CDR2':     out_df['cdr2_seq'],
            'CDR3':     out_df['cdr3_seq'],
            'PDBChain': out_df['target']
        })
        trans_df.to_csv(transformat_tsv, sep='\t', index=False)

    return f"Success extracted {len(similarity_df)} files"