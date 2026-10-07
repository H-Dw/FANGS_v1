#!/usr/bin/env python3
"""Extract AF3 samples; CDR C-alpha RMSD uses independent proper Kabsch fits."""
from __future__ import annotations
import argparse
import csv
import json
import shutil
from pathlib import Path
import gemmi
import numpy as np


def rmsd(ref, pred):
    a, b = np.asarray(ref, dtype=float), np.asarray(pred, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or len(a) < 3:
        raise ValueError('Need at least three paired CA coordinates')
    a, b = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(b.T @ a)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    aligned = b @ (u @ correction @ vt)
    return float(np.sqrt(np.mean(np.sum((a - aligned)**2, axis=1))))


def read_chain(path, text=None):
    st = gemmi.make_structure_from_block(gemmi.cif.read_string(text).sole_block()) if text else gemmi.read_structure(str(path))
    chains = []
    for chain in st[0]:
        residues = {}
        for r in chain:
            if r.label_seq is None:
                continue
            i = r.label_seq - 1
            if i in residues:
                raise ValueError(f'Duplicate label_seq_id {i + 1}')
            atoms = [a for a in r if a.name == 'CA']
            atom = max(atoms, key=lambda a: a.occ) if atoms else None
            residues[i] = (gemmi.find_tabulated_residue(r.name).one_letter_code.upper(),
                           np.array([atom.pos.x, atom.pos.y, atom.pos.z]) if atom else None,
                           atom.b_iso if atom else None)
        if residues:
            chains.append(residues)
    if len(chains) != 1:
        raise ValueError(f'Expected one protein chain, got {len(chains)}')
    return chains[0]


def write_tsv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
        w.writeheader()
        w.writerows(rows)


def stats(rows):
    result = []
    for metric in ['CDR1_cRMSD', 'CDR2_cRMSD', 'CDR3_cRMSD', 'global_cRMSD', 'pTM', 'ca_pLDDT']:
        values = np.array([r[metric] for r in rows if isinstance(r.get(metric), (float, int))])
        if len(values):
            result.append(dict(metric=metric, n=len(values), mean=float(values.mean()),
                               std=float(values.std(ddof=1)) if len(values)>1 else 0.,
                               median=float(np.median(values)), min=float(values.min()), max=float(values.max())))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions-dir', type=Path, required=True)
    p.add_argument('--config-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--min-ptm', type=float)
    p.add_argument('--top-k-per-target', type=int)
    p.add_argument('--exclude-protein-ids', default='')
    args = p.parse_args()
    if args.top_k_per_target is not None and args.top_k_per_target < 1:
        p.error('top-k must be positive')
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    mappings = {}
    with (args.config_root/'cdr_mapping_report.tsv').open() as f:
        for r in csv.DictReader(f, delimiter='\t'):
            key = (r['PDBChain'], r['CDR'])
            mappings.setdefault(key, []).append(r)
    configs = {json.loads(f.read_text())['name']: f for f in (args.config_root/'configs').glob('*.json')}
    excluded = set(args.exclude_protein_ids.replace(',', ' ').split())
    candidates, errors, duplicates, seen = {}, [], [], {}
    for job_input in sorted(args.predictions_dir.glob('gpu*/*/*_data.json')):
        x = json.loads(job_input.read_text())
        config_path = configs.get(x['name'])
        if config_path is None:
            errors.append(dict(source=str(job_input), reason='No matching config name')); continue
        target = config_path.name.removesuffix('_config.json')
        if target in excluded:
            continue
        protein = x['sequences'][0]['protein']
        current = json.loads(config_path.read_text())['sequences'][0]['protein']
        if protein['sequence'] != current['sequence']:
            errors.append(dict(source=str(job_input), reason='Query differs from config')); continue
        signature = json.dumps(protein, sort_keys=True)
        for cif in sorted(job_input.parent.glob('seed-*_sample-*/*_model.cif')):
            key = (target, cif.parent.name)
            summary = cif.with_name(cif.name.replace('_model.cif', '_summary_confidences.json'))
            try:
                conf = json.loads(summary.read_text())
                ptm = float(conf['ptm'])
                if not np.isfinite(ptm): raise ValueError('Nonfinite pTM')
                if key in seen:
                    previous_sig, previous_cif = seen[key]
                    if previous_sig != signature or previous_cif.read_bytes() != cif.read_bytes():
                        raise ValueError('Conflicting duplicate target/seed/sample')
                    duplicates.append(dict(source=str(cif), kept=str(previous_cif))); continue
                seen[key] = (signature, cif)
                candidates.setdefault(target, []).append((ptm, cif, conf, protein))
            except Exception as e:
                errors.append(dict(source=str(cif), reason=str(e)))
    rows, best = [], []
    for target, samples in sorted(candidates.items()):
        samples.sort(key=lambda c: (-c[0], str(c[1])))
        selected = [c for c in samples if args.min_ptm is None or c[0] >= args.min_ptm]
        if args.top_k_per_target: selected = selected[:args.top_k_per_target]
        for rank, (ptm, cif, conf, protein) in enumerate(selected, 1):
            try:
                pred = read_chain(cif)
                template = protein['templates'][0]
                # AF3 writes embedded mmCIF into *_data.json; use original config
                # path only after confirming the mapping equals the run's mapping.
                cfg = json.loads((args.config_root/'configs'/f'{target}_config.json').read_text())['sequences'][0]['protein']['templates'][0]
                if any(template[k] != cfg[k] for k in ('queryIndices','templateIndices')):
                    raise ValueError('Run template mapping differs from config')
                template_path = Path(cfg['mmcifPath'])
                if 'mmcif' in template:
                    # Compare parsed coordinates, not serialization formatting.
                    doc = gemmi.cif.read_string(template['mmcif'])
                    ref_st = gemmi.make_structure_from_block(doc.sole_block())
                    disk_st = gemmi.read_structure(str(template_path))
                    if ref_st.make_mmcif_document().as_string() != disk_st.make_mmcif_document().as_string():
                        raise ValueError('Run template differs from current template')
                ref = read_chain(template_path)
                mapping = dict(zip(cfg['queryIndices'], cfg['templateIndices'], strict=True))
                row = dict(target_id=target, sample_id=cif.parent.name, ptm_rank_in_target=rank,
                           pTM=ptm, ranking_score=conf.get('ranking_score'), source_cif=str(cif),
                           template_path=str(template_path), global_cRMSD='', global_coverage=0,
                           ca_pLDDT=float(np.mean([v[2] for v in pred.values() if v[2] is not None])))
                seq = protein['sequence']
                for name in ('CDR1','CDR2','CDR3'):
                    row[name+'_cRMSD'] = ''
                    matches = [m for m in mappings.get((target,name), [])
                               if seq[int(m['query_indices'].split('-')[0]):int(m['query_indices'].split('-')[1])+1] == m['sequence']]
                    if not matches: continue
                    m = matches[-1]
                    q0,q1 = map(int,m['query_indices'].split('-'))
                    t0,t1 = map(int,m['template_indices'].split('-'))
                    qi,ti = list(range(q0,q1+1)),list(range(t0,t1+1))
                    if seq[q0:q1+1] != m['sequence'] or len(qi)!=len(ti): raise ValueError(f'{name}: invalid mapping')
                    if [mapping.get(i) for i in qi] != ti: raise ValueError(f'{name}: config/report mismatch')
                    for q,t,aa in zip(qi,ti,m['sequence'],strict=True):
                        if pred[q][0]!=aa or ref[t][0]!=aa or pred[q][1] is None or ref[t][1] is None:
                            raise ValueError(f'{name}: residue identity or CA missing')
                    row[name+'_cRMSD'] = rmsd([ref[i][1] for i in ti],[pred[i][1] for i in qi])
                    row[name+'_n_CA'] = len(qi)
                if all(i in ref and i in pred and ref[i][0]==pred[i][0]==aa and ref[i][1] is not None and pred[i][1] is not None for i,aa in enumerate(seq)) and len(ref)==len(seq):
                    row['global_cRMSD'] = rmsd([ref[i][1] for i in range(len(seq))],[pred[i][1] for i in range(len(seq))])
                    row['global_coverage'] = 1.0
                dest = out/'structures'/target/f'{cif.parent.name}.cif'
                dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(cif,dest)
                row['extracted_cif'] = str(dest)
                rows.append(row)
                if rank==1:
                    best.append(row)
                    d=out/'best_ptm'; d.mkdir(exist_ok=True)
                    shutil.copy2(cif,d/f'{target}.cif')
                    gemmi.read_structure(str(cif)).write_pdb(str(d/f'{target}.pdb'))
            except Exception as e:
                errors.append(dict(source=str(cif),reason=str(e)))
    write_tsv(out/'af3_results.tsv',rows)
    write_tsv(out/'af3_best_ptm.tsv',best)
    write_tsv(out/'af3_statistics.tsv',stats(rows))
    write_tsv(out/'af3_best_statistics.tsv',stats(best))
    write_tsv(out/'errors.tsv',errors,['source','reason'])
    write_tsv(out/'duplicates.tsv',duplicates,['source','kept'])
    means=[]
    for target in sorted({r['target_id'] for r in rows}):
        group=[r for r in rows if r['target_id']==target]
        means.append(dict(target_id=target,n_samples=len(group),**{s['metric']:s['mean'] for s in stats(group)}))
    write_tsv(out/'af3_target_means.tsv',means)
    report=dict(samples=len(rows),targets=len(means),best_samples=len(best),errors=len(errors),duplicates=len(duplicates),
                candidates=sum(map(len,candidates.values())),min_ptm=args.min_ptm,top_k=args.top_k_per_target,
                rmsd_definition='CA; independent proper Kabsch fit for each CDR; Angstrom',
                global_definition='Full sequence identity only; blank otherwise; no partial-chain global RMSD')
    (out/'run_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not rows: raise SystemExit('No valid samples; see errors.tsv')

if __name__=='__main__': main()
