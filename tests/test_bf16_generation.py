#!/usr/bin/env python3
"""Test whether ESM3 generation/decoding works in bf16, and measure speed/accuracy.

Mimics the grafting workload: sequence prompt + partially templated structure
track, batched iterative structure sampling, then batched decode.

Run from final_version root, conda env ESM3:
    python tests/test_bf16_generation.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, GenerationConfig
from esm.utils.structure.protein_chain import ProteinChain

# Same grafted sequence as the 7b18D -> 7me7A recipient observed in production logs
SEQUENCE = (
    "HVQLVESGGGLVQAGGSLRLSCAASGSIFSSNAMSWYRQAPGKQRELVASIGSSDGRTDYADSVKGRFT"
    "ISRDNAKNTVYPEMSSLKPADTAVYYCALTVGTYYSGNYHYTCSDDMDYWGQGTQVTVSS"
)
# Use a real nanobody structure as template-token donor (CDR3 window)
TEMPLATE_PDB = "data/INDI_database/mmseqs_cluster_structure/id80/cluster_rep_structures/7me7A.pdb"
NUM_SAMPLES = int(os.environ.get("TEST_NUM_SAMPLES", "16"))
SEED = 42


def build_prompt(model):
    prompt = model.encode(ESMProtein(sequence=SEQUENCE))
    prompt.structure = torch.full_like(prompt.sequence, 4096)
    prompt.structure[0] = 4098
    prompt.structure[-1] = 4097

    chain = ProteinChain.from_pdb(TEMPLATE_PDB)
    template_tokens = model.encode(ESMProtein.from_protein_chain(chain))
    # Fill a 22-residue window (CDR3-like) with template structure tokens
    start = 96  # 1-indexed token position used in production logs
    prompt.structure[start + 1 : start + 23] = template_tokens.structure[start + 1 : start + 23]
    return prompt


def run_once(model, prompt, tag):
    num_steps = int((prompt.structure == 4096).sum().item())
    config = GenerationConfig(track="structure", num_steps=num_steps, temperature=0.7)

    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    generations = model.generate_batch(prompt, config, num_samples=NUM_SAMPLES)
    torch.cuda.synchronize()
    gen_time = time.time() - t0

    t0 = time.time()
    decoded = model.decode_batch(generations)
    torch.cuda.synchronize()
    dec_time = time.time() - t0

    tokens = torch.stack([g.structure for g in generations])
    ptms = [float(d.ptm) if d.ptm is not None else float("nan") for d in decoded]
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(
        f"[{tag}] steps={num_steps} gen={gen_time:.1f}s ({gen_time/num_steps:.3f}s/it) "
        f"decode={dec_time:.2f}s peak_mem={peak_gb:.1f}GB "
        f"pTM(mean/min/max)={sum(ptms)/len(ptms):.3f}/{min(ptms):.3f}/{max(ptms):.3f}"
    )
    return tokens, ptms


def main():
    device = torch.device("cuda:0")
    prompt_cache = {}

    # --- fp32 baseline ---
    tokens_fp32 = ptms_fp32 = None
    if not os.environ.get("TEST_SKIP_FP32"):
        model = ESM3.from_pretrained("esm3_sm_open_v1").to(device)
        model.eval()
        prompt_cache["fp32"] = build_prompt(model)
        tokens_fp32, ptms_fp32 = run_once(model, prompt_cache["fp32"], "fp32")
        del model
        torch.cuda.empty_cache()

    # --- bf16 ---
    model = ESM3.from_pretrained("esm3_sm_open_v1").to(device).to(torch.bfloat16)
    model.eval()
    # prompt built with lazily-loaded (fp32) structure encoder, same as production flow
    prompt = build_prompt(model)
    try:
        tokens_bf16, ptms_bf16 = run_once(model, prompt, "bf16")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[bf16] FAILED: {type(e).__name__}: {e}")
        return

    if tokens_fp32 is not None:
        agreement = (tokens_fp32 == tokens_bf16).float().mean().item()
        ptm_diff = [abs(a - b) for a, b in zip(ptms_fp32, ptms_bf16)]
        print(f"token agreement fp32 vs bf16: {agreement * 100:.1f}%")
        print(f"|pTM diff| mean={sum(ptm_diff)/len(ptm_diff):.4f} max={max(ptm_diff):.4f}")


if __name__ == "__main__":
    main()
