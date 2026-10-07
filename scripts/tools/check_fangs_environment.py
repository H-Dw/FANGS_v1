#!/usr/bin/env python3
"""Validate the FANGS scientific runtime and optionally load the ESM3 model."""

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--load-model', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['MPLBACKEND'] = 'Agg'
    report = {'python': sys.version.split()[0], 'executable': sys.executable,
              'versions': {}, 'checks': {}, 'model': {'status': 'not_requested'}}
    try:
        assert sys.version_info >= (3, 10)
        modules = ['numpy', 'pandas', 'scipy', 'matplotlib', 'Bio', 'gemmi',
                   'torch', 'torchvision', 'torchaudio', 'torchtext', 'torchdata',
                   'biotite', 'sklearn', 'seaborn', 'transformers',
                   'huggingface_hub', 'anarci']
        for name in modules:
            module = importlib.import_module(name)
            report['versions'][name] = {'version': getattr(module, '__version__', None),
                                        'path': module.__file__}
        from packaging.requirements import Requirement
        for line in (root / 'requirements-fangs.txt').read_text().splitlines():
            line = line.strip()
            if not line or line.startswith(('#', '--')):
                continue
            requirement = Requirement(line)
            actual = importlib.metadata.version(requirement.name)
            assert requirement.specifier.contains(actual, prereleases=True), (line, actual)
        report['checks']['version_constraints'] = True
        result = subprocess.run([sys.executable, '-m', 'pip', 'check'],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        report['checks']['pip_check'] = result.stdout.strip()

        import numpy as np
        import pandas as pd
        from scipy.stats import spearmanr, bootstrap
        from Bio import SeqIO
        from Bio.PDB import PDBParser
        import gemmi
        import matplotlib.pyplot as plt
        import torch
        assert torch.__version__ == '2.2.0+cu118', torch.__version__
        assert torch.version.cuda == '11.8', torch.version.cuda
        import esm.models.esm3
        from esm.models.esm3 import ESM3
        from esm.models.vqvae import StructureTokenEncoder
        from esm.sdk.api import ESMProtein, ESMProteinTensor, GenerationConfig
        from esm.utils.structure.protein_chain import ProteinChain

        selected_source = Path(esm.models.esm3.__file__).resolve()
        assert selected_source == (root / 'esm/models/esm3.py').resolve()
        assert callable(ESM3.structure_encode_full)
        assert callable(ESM3.generate_batch)
        assert callable(StructureTokenEncoder.encode_full)
        report['checks']['project_esm_source'] = str(selected_source)
        report['versions']['esm_distribution'] = importlib.metadata.version('esm')
        values = np.arange(1., 11.)
        assert spearmanr(values, values).statistic > 0.999
        interval = bootstrap((values,), np.mean, n_resamples=100,
                             random_state=42).confidence_interval
        assert interval.low < interval.high
        pd.DataFrame({'value': values}).to_csv(args.output_dir / 'analysis_smoke.tsv',
                                              sep='\t', index=False)
        report['checks']['numpy_pandas_scipy'] = True
        figure, axis = plt.subplots()
        axis.plot(values, values)
        figure.savefig(args.output_dir / 'matplotlib_smoke.svg')
        plt.close(figure)
        report['checks']['matplotlib_export'] = True
        pdb = root / 'example/6lr7B.pdb'
        structure = PDBParser(QUIET=True).get_structure('LaG16', str(pdb))
        assert len(list(structure.get_atoms())) > 0
        cif_root = root / 'data/ESM3-Template_validation/af3_template/af3_observed_chain_canonical_20260929/templates'
        cif = next(iter(sorted(cif_root.glob('*.cif'))))
        assert len(gemmi.read_structure(str(cif))) > 0
        report['checks']['pdb_cif_processing'] = {'pdb': str(pdb), 'cif': str(cif)}

        for executable in ['mafft', 'hmmscan']:
            assert shutil.which(executable), executable
        from anarci import run_anarci
        record = next(SeqIO.parse(str(root / 'example/5m2j_VHH2_sequence.fasta'), 'fasta'))
        numbered = run_anarci([(record.id, str(record.seq))], scheme='k',
                             ncpu=1, allow={'H', 'K', 'L'}, output=False)
        assert numbered[1][0], 'ANARCI did not number VHH2'
        report['checks']['anarci_kabat'] = True
        mafft_input = args.output_dir / 'mafft_input.fasta'
        mafft_input.write_text('>first\n' + str(record.seq) + '\n>second\n' + str(record.seq) + '\n')
        alignment = subprocess.run(['mafft', '--auto', str(mafft_input)],
                                   capture_output=True, text=True, timeout=60)
        assert alignment.returncode == 0 and alignment.stdout.count('>') == 2
        (args.output_dir / 'mafft_aligned.fasta').write_text(alignment.stdout)
        report['checks']['mafft_alignment'] = True
        tensor = torch.from_numpy(values)
        assert np.allclose(tensor.numpy(), values)
        report['checks']['torch_numpy_bridge'] = True
        report['checks']['cuda_available'] = torch.cuda.is_available()
        report['versions']['torch_cuda'] = torch.version.cuda
        if args.device.startswith('cuda'):
            assert torch.cuda.is_available()
            matrix = torch.eye(16, device=args.device)
            assert torch.allclose(matrix @ matrix, matrix)
            report['checks']['cuda_execution'] = True

        if args.load_model:
            report['model']['status'] = 'loading'
            model = ESM3.from_pretrained('esm3_sm_open_v1', device=args.device).eval()
            chain = ProteinChain.from_pdb(str(pdb))
            protein = ESMProtein.from_protein_chain(chain)
            with torch.inference_mode():
                raw, pre_q, quantized, tokens = model.structure_encode_full(protein)
                assert raw.shape[-1] == 1024 and torch.isfinite(raw).all()
                encoded = model.encode(protein)
                assert isinstance(encoded, ESMProteinTensor)
                encoded.sequence[10] = model.tokenizers.sequence.mask_token_id
                generated = model.generate(encoded, GenerationConfig(track='sequence',
                                          num_steps=1, temperature=0.7))
                assert isinstance(generated, ESMProteinTensor)
                assert generated.sequence[10] != model.tokenizers.sequence.mask_token_id
                decoded = model.decode(generated)
                assert decoded.coordinates is not None
            report['model'] = {'status': 'passed', 'name': 'esm3_sm_open_v1',
                               'device': args.device, 'raw_shape': list(raw.shape),
                               'pre_q_shape': list(pre_q.shape),
                               'quantized_shape': list(quantized.shape),
                               'token_shape': list(tokens.shape),
                               'single_step_generation': True, 'structure_decode': True}
        report['status'] = 'passed'
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        (args.output_dir / 'runtime_validation.json').write_text(
            json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
