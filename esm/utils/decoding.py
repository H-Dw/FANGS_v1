import warnings
from typing import Sequence

import attr
import torch

from esm.models.function_decoder import FunctionTokenDecoder
from esm.models.vqvae import StructureTokenDecoder
from esm.sdk.api import ESMProtein, ESMProteinTensor
from esm.tokenization import TokenizerCollectionProtocol
from esm.tokenization.function_tokenizer import (
    InterProQuantizedTokenizer,
)
from esm.tokenization.residue_tokenizer import (
    ResidueAnnotationsTokenizer,
)
from esm.tokenization.sasa_tokenizer import (
    SASADiscretizingTokenizer,
)
from esm.tokenization.sequence_tokenizer import (
    EsmSequenceTokenizer,
)
from esm.tokenization.ss_tokenizer import (
    SecondaryStructureTokenizer,
)
from esm.tokenization.structure_tokenizer import (
    StructureTokenizer,
)
from esm.tokenization.tokenizer_base import EsmTokenizerBase
from esm.utils.constants import esm3 as C
from esm.utils.function.encode_decode import (
    decode_function_tokens,
    decode_residue_annotation_tokens,
)
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.types import FunctionAnnotation


def _drop_all_pad_tracks(
    input: ESMProteinTensor, tokenizers: TokenizerCollectionProtocol
) -> ESMProteinTensor:
    input = attr.evolve(input)
    for track in attr.fields(ESMProteinTensor):
        tokens: torch.Tensor | None = getattr(input, track.name)
        if track.name == "coordinates" or tokens is None:
            continue
        if tokens.dim() != 1 and not (
            track.name in ("function", "residue_annotations") and tokens.dim() == 2
        ):
            raise ValueError(
                f"decode expects an unbatched ESMProteinTensor, got {track.name} with shape {tuple(tokens.shape)}"
            )
        stripped = tokens[1:-1].flatten()
        track_tokenizer = getattr(tokenizers, track.name)
        if torch.all(stripped == track_tokenizer.pad_token_id):
            setattr(input, track.name, None)
    return input


def _protein_from_decoded_tracks(
    input: ESMProteinTensor,
    tokenizers: TokenizerCollectionProtocol,
    function_token_decoder: FunctionTokenDecoder | None,
    sequence: str | None,
    coordinates: torch.Tensor | None,
    plddt: torch.Tensor | None,
    ptm: torch.Tensor | None,
) -> ESMProtein:
    secondary_structure = None
    sasa = None
    function_annotations = []

    if input.secondary_structure is not None:
        secondary_structure = decode_secondary_structure(
            input.secondary_structure, tokenizers.secondary_structure
        )
    if input.sasa is not None:
        sasa = decode_sasa(input.sasa, tokenizers.sasa)
    if input.function is not None:
        if function_token_decoder is None:
            raise ValueError("function tokens are present but no function decoder was provided")
        function_annotations.extend(
            decode_function_annotations(
                input.function,
                function_token_decoder=function_token_decoder,
                function_tokenizer=tokenizers.function,
            )
        )
    if input.residue_annotations is not None:
        function_annotations.extend(
            decode_residue_annotations(
                input.residue_annotations, tokenizers.residue_annotations
            )
        )

    return ESMProtein(
        sequence=sequence,
        secondary_structure=secondary_structure,
        sasa=sasa,  # type: ignore
        function_annotations=function_annotations if function_annotations else None,
        coordinates=coordinates,
        plddt=plddt,
        ptm=ptm,
    )


def decode_protein_tensor(
    input: ESMProteinTensor,
    tokenizers: TokenizerCollectionProtocol,
    structure_token_decoder: StructureTokenDecoder,
    function_token_decoder: FunctionTokenDecoder | None = None,
) -> ESMProtein:
    input = _drop_all_pad_tracks(input, tokenizers)

    sequence = None
    if input.sequence is not None:
        sequence = decode_sequence(input.sequence, tokenizers.sequence)

    plddt, ptm = None, None
    coordinates = None
    if input.structure is not None:
        coordinates, plddt, ptm = decode_structure(
            structure_tokens=input.structure,
            structure_decoder=structure_token_decoder,
            structure_tokenizer=tokenizers.structure,
            sequence=sequence,
        )
    elif input.coordinates is not None:
        coordinates = input.coordinates[1:-1, ...]

    return _protein_from_decoded_tracks(
        input,
        tokenizers,
        function_token_decoder,
        sequence,
        coordinates,
        plddt,
        ptm,
    )


def decode_protein_tensor_batch(
    inputs: Sequence[ESMProteinTensor],
    tokenizers: TokenizerCollectionProtocol,
    structure_token_decoder: StructureTokenDecoder,
    function_token_decoder: FunctionTokenDecoder | None = None,
) -> list[ESMProtein]:
    """Decode many same-length structure tokens with one GPU structure-decoder call."""
    if not inputs:
        return []
    if len(inputs) == 1:
        return [
            decode_protein_tensor(
                inputs[0],
                tokenizers,
                structure_token_decoder,
                function_token_decoder,
            )
        ]

    prepared = [_drop_all_pad_tracks(item, tokenizers) for item in inputs]
    sequences = [
        decode_sequence(item.sequence, tokenizers.sequence)
        if item.sequence is not None
        else None
        for item in prepared
    ]

    structure_tokens = [item.structure for item in prepared]
    if all(tokens is not None for tokens in structure_tokens):
        stacked = torch.stack(structure_tokens, dim=0)
        decoded_structures = decode_structures_batch(
            structure_tokens=stacked,
            structure_decoder=structure_token_decoder,
            structure_tokenizer=tokenizers.structure,
            sequences=sequences,
        )
    elif any(tokens is not None for tokens in structure_tokens):
        raise ValueError("decode_protein_tensor_batch requires structure tokens on every sample")
    else:
        decoded_structures = []
        for item in prepared:
            if item.coordinates is None:
                decoded_structures.append((None, None, None))
            else:
                decoded_structures.append((item.coordinates[1:-1, ...], None, None))

    return [
        _protein_from_decoded_tracks(
            item,
            tokenizers,
            function_token_decoder,
            sequence,
            coordinates,
            plddt,
            ptm,
        )
        for item, sequence, (coordinates, plddt, ptm) in zip(
            prepared, sequences, decoded_structures
        )
    ]


def _bos_eos_warn(msg: str, tensor: torch.Tensor, tok: EsmTokenizerBase):
    if tensor[0] != tok.bos_token_id:
        warnings.warn(
            f"{msg} does not start with BOS token, token is ignored. BOS={tok.bos_token_id} vs {tensor}"
        )
    if tensor[-1] != tok.eos_token_id:
        warnings.warn(
            f"{msg} does not end with EOS token, token is ignored. EOS='{tok.eos_token_id}': {tensor}"
        )


def decode_sequence(
    sequence_tokens: torch.Tensor,
    sequence_tokenizer: EsmSequenceTokenizer,
    **kwargs,
) -> str:
    _bos_eos_warn("Sequence", sequence_tokens, sequence_tokenizer)
    sequence = sequence_tokenizer.decode(
        sequence_tokens,
        **kwargs,
    )
    sequence = sequence.replace(" ", "")
    sequence = sequence.replace(sequence_tokenizer.mask_token, C.MASK_STR_SHORT)
    sequence = sequence.replace(sequence_tokenizer.cls_token, "")
    sequence = sequence.replace(sequence_tokenizer.eos_token, "")

    return sequence


def _bb_coords_to_atom37(
    bb_coords: torch.Tensor, sequence: str | None
) -> torch.Tensor:
    chain = ProteinChain.from_backbone_atom_coordinates(bb_coords, sequence=sequence)
    chain = chain.infer_oxygen()
    return torch.tensor(chain.atom37_positions)


def _index_ptm(ptm: torch.Tensor | None, index: int) -> torch.Tensor | None:
    if ptm is None:
        return None
    if ptm.ndim == 0:
        return ptm
    return ptm[index]


def decode_structures_batch(
    structure_tokens: torch.Tensor,
    structure_decoder: StructureTokenDecoder,
    structure_tokenizer: StructureTokenizer,
    sequences: Sequence[str | None],
) -> list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]]:
    """GPU-batched structure-token decode. ProteinChain conversion stays per sample."""
    if structure_tokens.dim() == 1:
        structure_tokens = structure_tokens.unsqueeze(0)
    if structure_tokens.dim() != 2:
        raise ValueError(
            f"Expected structure tokens of shape (L,) or (B, L), got {tuple(structure_tokens.shape)}"
        )

    batch_size = structure_tokens.size(0)
    if len(sequences) != batch_size:
        raise ValueError(
            f"Need one sequence per structure sample, got {len(sequences)} sequences for batch {batch_size}"
        )
    _bos_eos_warn("Structure", structure_tokens[0], structure_tokenizer)

    with torch.no_grad():
        decoder_output = structure_decoder.decode(structure_tokens)

    bb_coords = decoder_output["bb_pred"][:, 1:-1, ...].detach().cpu()
    plddt = (
        decoder_output["plddt"][:, 1:-1].detach().cpu()
        if "plddt" in decoder_output
        else None
    )
    ptm = decoder_output["ptm"].detach().cpu() if "ptm" in decoder_output else None

    decoded = []
    for i in range(batch_size):
        decoded.append(
            (
                _bb_coords_to_atom37(bb_coords[i], sequences[i]),
                None if plddt is None else plddt[i],
                _index_ptm(ptm, i),
            )
        )
    return decoded


def decode_structure(
    structure_tokens: torch.Tensor,
    structure_decoder: StructureTokenDecoder,
    structure_tokenizer: StructureTokenizer,
    sequence: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    decoded = decode_structures_batch(
        structure_tokens=structure_tokens,
        structure_decoder=structure_decoder,
        structure_tokenizer=structure_tokenizer,
        sequences=[sequence],
    )
    return decoded[0]


def decode_secondary_structure(
    secondary_structure_tokens: torch.Tensor,
    ss_tokenizer: SecondaryStructureTokenizer,
) -> str:
    _bos_eos_warn("Secondary structure", secondary_structure_tokens, ss_tokenizer)
    secondary_structure_tokens = secondary_structure_tokens[1:-1]
    secondary_structure = ss_tokenizer.decode(
        secondary_structure_tokens,
    )
    return secondary_structure


def decode_sasa(
    sasa_tokens: torch.Tensor,
    sasa_tokenizer: SASADiscretizingTokenizer,
) -> list[float]:
    _bos_eos_warn("SASA", sasa_tokens, sasa_tokenizer)
    sasa_tokens = sasa_tokens[1:-1]

    return sasa_tokenizer.decode_float(sasa_tokens)


def decode_function_annotations(
    function_annotation_tokens: torch.Tensor,
    function_token_decoder: FunctionTokenDecoder,
    function_tokenizer: InterProQuantizedTokenizer,
    **kwargs,
) -> list[FunctionAnnotation]:
    # No need to check for BOS/EOS as function annotations are not affected

    function_annotations = decode_function_tokens(
        function_annotation_tokens,
        function_token_decoder=function_token_decoder,
        function_tokens_tokenizer=function_tokenizer,
        **kwargs,
    )
    return function_annotations


def decode_residue_annotations(
    residue_annotation_tokens: torch.Tensor,
    residue_annotation_decoder: ResidueAnnotationsTokenizer,
) -> list[FunctionAnnotation]:
    # No need to check for BOS/EOS as function annotations are not affected

    residue_annotations = decode_residue_annotation_tokens(
        residue_annotations_token_ids=residue_annotation_tokens,
        residue_annotations_tokenizer=residue_annotation_decoder,
    )
    return residue_annotations
