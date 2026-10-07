from __future__ import annotations

import contextlib
from functools import partial

import attr
import einops
import torch
import torch.nn as nn
from attr import dataclass

from esm.layers.regression_head import RegressionHead
from esm.layers.transformer_stack import TransformerStack
from esm.models.function_decoder import FunctionTokenDecoder
from esm.models.vqvae import (
    StructureTokenDecoder,
    StructureTokenEncoder,
)
from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    ESMProteinTensor,
    ForwardAndSampleOutput,
    ForwardConfig,
    ForwardOutput,
    ForwardTrackData,
    GenerationConfig,
    ProteinType,
    ReturnLogitsConfig,
    SamplingConfig,
    SamplingTrackConfig,
)
from esm.tokenization import get_model_tokenizers
from esm.utils import encoding
from esm.utils.constants import esm3 as C
from esm.utils.constants.models import ESM3_OPEN_SMALL
from esm.utils.decoding import decode_protein_tensor, decode_protein_tensor_batch
from esm.utils.generation import (
    iterative_sampling_raw,
    iterative_sampling_tokens,
    iterative_sampling_tokens_batched,
    repeat_protein_tensor,
    unbatch_protein_tensor,
)
from esm.utils.misc import rbf
from esm.utils.sampling import (
    get_default_sampling_config,
    sample_function_logits,
    sample_logits,
    sample_residue_annotation_logits,
    sample_probs, # Custom
    sample_fused_logits,
)
from esm.utils.structure.affine3d import (
    build_affine3d_from_coordinates,
)


@dataclass
class ESMOutput:
    sequence_logits: torch.Tensor
    structure_logits: torch.Tensor
    secondary_structure_logits: torch.Tensor
    sasa_logits: torch.Tensor
    function_logits: torch.Tensor
    residue_logits: torch.Tensor
    embeddings: torch.Tensor


class EncodeInputs(nn.Module):
    """
    Module for encoding input features in the ESM-3 model.

    Args:
        d_model (int): The dimensionality of the model's hidden states.
    """

    def __init__(self, d_model: int):
        super().__init__()

        # Sequence
        self.sequence_embed = nn.Embedding(64, d_model)
        # Mandatory information
        self.plddt_projection = nn.Linear(16, d_model)
        self.structure_per_res_plddt_projection = nn.Linear(16, d_model)

        # Structure
        self.structure_tokens_embed = nn.Embedding(4096 + 5, d_model)

        # "Structural" features
        self.ss8_embed = nn.Embedding(8 + 3, d_model)
        self.sasa_embed = nn.Embedding(16 + 3, d_model)

        # "Functional" features
        self.function_embed = nn.ModuleList(
            [nn.Embedding(260, d_model // 8, padding_idx=0) for _ in range(8)]
        )

        self.residue_embed = nn.EmbeddingBag(1478, d_model, mode="sum", padding_idx=0)

    def forward(
        self,
        sequence_tokens: torch.Tensor,
        structure_tokens: torch.Tensor,
        average_plddt: torch.Tensor,
        per_res_plddt: torch.Tensor,
        ss8_tokens: torch.Tensor,
        sasa_tokens: torch.Tensor,
        function_tokens: torch.Tensor,
        residue_annotation_tokens: torch.Tensor,
    ) -> torch.Tensor:
        sequence_embed = self.sequence_embed(sequence_tokens)

        rbf_16_fn = partial(rbf, v_min=0.0, v_max=1.0, n_bins=16)
        # the `masked_fill(padding_mask.unsqueeze(2), 0)` for the two below is unnecessary
        # as pad tokens never even interact with the "real" tokens (due to sequence_id)
        plddt_embed = self.plddt_projection(
            rbf_16_fn(average_plddt).to(self.plddt_projection.weight.dtype)
        )
        structure_per_res_plddt = self.structure_per_res_plddt_projection(
            rbf_16_fn(per_res_plddt).to(self.structure_per_res_plddt_projection.weight.dtype)
        )

        # Structure + "structural features" embeds
        structure_embed = self.structure_tokens_embed(structure_tokens)
        ss8_embed = self.ss8_embed(ss8_tokens)
        sasa_embed = self.sasa_embed(sasa_tokens)

        # "Functional" features embeds
        function_embed = torch.cat(
            [
                embed_fn(funcs)
                for embed_fn, funcs in zip(
                    self.function_embed, function_tokens.unbind(-1)
                )
            ],
            -1,
        )

        # Residue embeds
        B, L, N = residue_annotation_tokens.shape
        residue_embed = self.residue_embed(
            einops.rearrange(
                residue_annotation_tokens, "B L N -> (B L) N", B=B, L=L, N=N
            )
        )
        residue_embed = einops.rearrange(residue_embed, "(B L) D -> B L D", B=B, L=L)

        return (
            sequence_embed
            + plddt_embed
            + structure_per_res_plddt
            + structure_embed
            + ss8_embed
            + sasa_embed
            + function_embed
            + residue_embed
        )


class OutputHeads(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.sequence_head = RegressionHead(d_model, 64)
        self.structure_head = RegressionHead(d_model, 4096)
        self.ss8_head = RegressionHead(d_model, 8 + 3)
        self.sasa_head = RegressionHead(d_model, 16 + 3)
        self.function_head = RegressionHead(d_model, 260 * 8)
        self.residue_head = RegressionHead(d_model, 1478)

    def forward(self, x: torch.Tensor, embed: torch.Tensor) -> ESMOutput:
        sequence_logits = self.sequence_head(x)
        structure_logits = self.structure_head(x)
        secondary_structure_logits = self.ss8_head(x)
        sasa_logits = self.sasa_head(x)
        function_logits = self.function_head(x)
        function_logits = einops.rearrange(
            function_logits,
            "... (k v) -> ... k v",
            k=8,
        )

        residue_logits = self.residue_head(x)

        return ESMOutput(
            sequence_logits=sequence_logits,
            structure_logits=structure_logits,
            secondary_structure_logits=secondary_structure_logits,
            sasa_logits=sasa_logits,
            function_logits=function_logits,
            residue_logits=residue_logits,
            embeddings=embed,
        )

class ESM3(nn.Module, ESM3InferenceClient):
    """
    ESM3 model implementation.

    Args:
        d_model (int): The dimensionality of the input and output feature vectors.
        n_heads (int): The number of attention heads in the transformer layers.
        v_heads (int): The number of attention heads in the variational transformer layers.
        n_layers (int): The number of transformer layers.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        v_heads: int,
        n_layers: int,
        structure_encoder_name: str,
        structure_decoder_name: str,
        function_decoder_name: str,
    ):
        super().__init__()
        self.encoder = EncodeInputs(d_model)
        self.transformer = TransformerStack(
            d_model,
            n_heads,
            v_heads,
            n_layers,
            mask_and_zero_frameless=True,
        )
        self.output_heads = OutputHeads(d_model)

        self.structure_encoder_name = structure_encoder_name
        self.structure_decoder_name = structure_decoder_name
        self.function_decoder_name = function_decoder_name

        self.structure_encoder: StructureTokenEncoder | None = None  # type: ignore
        self.structure_decoder: StructureTokenDecoder | None = None  # type: ignore
        self.function_decoder: FunctionTokenDecoder | None = None  # type: ignore

        self.tokenizers = get_model_tokenizers(ESM3_OPEN_SMALL)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = ESM3_OPEN_SMALL,
        device: torch.device | str = "cpu",
    ) -> ESM3:
        from esm.pretrained import load_local_model

        # custom
        # if model_name not in [ESM3_OPEN_SMALL]:
        #     raise ValueError(f"Model name {model_name} is not a valid ESM3 model name.")
        model: ESM3 = load_local_model(model_name, device=device)  # type: ignore
        return model

    def get_structure_token_encoder(self) -> StructureTokenEncoder:
        if self.structure_encoder is None:
            self.structure_encoder = self.load_model(self.structure_encoder_name)  # type: ignore
        return self.structure_encoder  # type: ignore

    def get_structure_token_decoder(self) -> StructureTokenDecoder:
        if self.structure_decoder is None:
            self.structure_decoder = self.load_model(self.structure_decoder_name)  # type: ignore
        return self.structure_decoder  # type: ignore

    def get_function_token_decoder(self) -> FunctionTokenDecoder:
        if self.function_decoder is None:
            self.function_decoder = self.load_model(self.function_decoder_name)  # type: ignore
        return self.function_decoder  # type: ignore

    def load_model(self, model_name: str):
        # Lazy import from pretrained
        from esm.pretrained import load_local_model

        return load_local_model(model_name, device=next(self.parameters()).device)

    def forward(
        self,
        *,
        sequence_tokens: torch.Tensor | None = None,
        structure_tokens: torch.Tensor | None = None,
        ss8_tokens: torch.Tensor | None = None,
        sasa_tokens: torch.Tensor | None = None,
        function_tokens: torch.Tensor | None = None,
        residue_annotation_tokens: torch.Tensor | None = None,
        average_plddt: torch.Tensor | None = None,
        per_res_plddt: torch.Tensor | None = None,
        structure_coords: torch.Tensor | None = None,
        chain_id: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
    ) -> ESMOutput:
        """
        Performs forward pass through the ESM3 model. Check utils to see how to tokenize inputs from raw data.

        Args:
            sequence_tokens (torch.Tensor, optional): The amino acid tokens.
            structure_tokens (torch.Tensor, optional): The structure tokens.
            ss8_tokens (torch.Tensor, optional): The secondary structure tokens.
            sasa_tokens (torch.Tensor, optional): The solvent accessible surface area tokens.
            function_tokens (torch.Tensor, optional): The function tokens.
            residue_annotation_tokens (torch.Tensor, optional): The residue annotation tokens.
            average_plddt (torch.Tensor, optional): The average plddt across the entire sequence.
            per_res_plddt (torch.Tensor, optional): The per residue plddt, if you want to specify exact plddts, use this,
                otherwise, use average_plddt.
            structure_coords (torch.Tensor, optional): The structure coordinates, in the form of (B, L, 3, 3).
            chain_id (torch.Tensor, optional): The chain ID
            sequence_id (torch.Tensor, optional): The sequence ID.

        Returns:
            ESMOutput: The output of the ESM3 model.

        Raises:
            ValueError: If at least one of the inputs is None.

        """
        # Reasonable defaults:
        try:
            L, device = next(
                (x.shape[1], x.device)
                for x in [
                    sequence_tokens,
                    structure_tokens,
                    ss8_tokens,
                    sasa_tokens,
                    structure_coords,
                    function_tokens,
                    residue_annotation_tokens,
                ]
                if x is not None
            )
        except StopIteration:
            raise ValueError("At least one of the inputs must be non-None")

        t = self.tokenizers
        defaults = lambda x, tok: (
            torch.full((1, L), tok, dtype=torch.long, device=device) if x is None else x
        )
        sequence_tokens = defaults(sequence_tokens, t.sequence.mask_token_id)
        ss8_tokens = defaults(ss8_tokens, C.SS8_PAD_TOKEN)
        sasa_tokens = defaults(sasa_tokens, C.SASA_PAD_TOKEN)
        average_plddt = defaults(average_plddt, 1).float()
        per_res_plddt = defaults(per_res_plddt, 0).float()
        chain_id = defaults(chain_id, 0)
        sequence_id = defaults(sequence_id, 0)

        if residue_annotation_tokens is None:
            residue_annotation_tokens = torch.full(
                (1, L, 16), C.RESIDUE_PAD_TOKEN, dtype=torch.long, device=device
            )

        if function_tokens is None:
            function_tokens = torch.full(
                (1, L, 8), C.INTERPRO_PAD_TOKEN, dtype=torch.long, device=device
            )

        if structure_coords is None:
            structure_coords = torch.full(
                (1, L, 3, 3), float("nan"), dtype=torch.float, device=device
            )

        structure_coords = structure_coords[
            ..., :3, :
        ]  # In case we pass in an atom14 or atom37 repr
        affine, affine_mask = build_affine3d_from_coordinates(structure_coords)

        structure_tokens = defaults(structure_tokens, C.STRUCTURE_MASK_TOKEN)
        structure_tokens = (
            structure_tokens.masked_fill(structure_tokens == -1, C.STRUCTURE_MASK_TOKEN)
            .masked_fill(sequence_tokens == C.SEQUENCE_BOS_TOKEN, C.STRUCTURE_BOS_TOKEN)
            .masked_fill(sequence_tokens == C.SEQUENCE_PAD_TOKEN, C.STRUCTURE_PAD_TOKEN)
            .masked_fill(sequence_tokens == C.SEQUENCE_EOS_TOKEN, C.STRUCTURE_EOS_TOKEN)
            .masked_fill(
                sequence_tokens == C.SEQUENCE_CHAINBREAK_TOKEN,
                C.STRUCTURE_CHAINBREAK_TOKEN,
            )
        )

        x = self.encoder(
            sequence_tokens,
            structure_tokens,
            average_plddt,
            per_res_plddt,
            ss8_tokens,
            sasa_tokens,
            function_tokens,
            residue_annotation_tokens,
        )
        x, embedding = self.transformer(x, sequence_id, affine, affine_mask, chain_id)

        return self.output_heads(x, embedding)

    # The following methods are for the ESM3InferenceClient interface
    # real_probs and alpha is custom design for generation using real probs
    def generate(self, input: ProteinType, config: GenerationConfig, real_probs: torch.Tensor = None, alpha: float = 0.6) -> ProteinType:
        if isinstance(input, ESMProtein):
            # return iterative_sampling_raw(self, input, config)
            return iterative_sampling_raw(self, input, config, real_probs, alpha)
        elif isinstance(input, ESMProteinTensor):
            # return iterative_sampling_tokens(self, input, config, self.tokenizers)
            return iterative_sampling_tokens(self, input, config, self.tokenizers, real_probs, alpha)  # real_probs and alpha is custom design for generation using real probs
        else:
            raise ValueError("Input must be an ESMProtein or ESMProteinTensor")

    def generate_batch(
        self,
        input: ESMProteinTensor,
        config: GenerationConfig,
        num_samples: int,
        real_probs: torch.Tensor = None,
        alpha: float = 0.6,
    ) -> list[ESMProteinTensor]:
        if num_samples < 1:
            return []
        if isinstance(input, ESMProtein):
            input = self.encode(input)
        elif not isinstance(input, ESMProteinTensor):
            raise ValueError("Input must be an ESMProtein or ESMProteinTensor")
        if input.sequence is not None and input.sequence.dim() != 1:
            raise ValueError("generate_batch expects an unbatched ESMProteinTensor")
        if input.structure is not None and input.structure.dim() != 1:
            raise ValueError("generate_batch expects an unbatched ESMProteinTensor")

        batched_input = repeat_protein_tensor(input, num_samples)
        batched_output = iterative_sampling_tokens_batched(
            self, batched_input, config, self.tokenizers, real_probs, alpha
        )
        return unbatch_protein_tensor(batched_output)

    def encode(self, input: ESMProtein) -> ESMProteinTensor:
        input = attr.evolve(input)  # Make a copy

        sequence_tokens = None
        structure_tokens = None
        secondary_structure_tokens = None
        sasa_tokens = None
        function_tokens = None
        residue_annotation_tokens = None

        coordinates = None

        if input.sequence is not None:
            sequence_tokens = encoding.tokenize_sequence(
                input.sequence, self.tokenizers.sequence, add_special_tokens=True
            )
        if input.secondary_structure is not None:
            secondary_structure_tokens = encoding.tokenize_secondary_structure(
                input.secondary_structure,
                self.tokenizers.secondary_structure,
                add_special_tokens=True,
            )
        if input.sasa is not None:
            sasa_tokens = encoding.tokenize_sasa(
                input.sasa, self.tokenizers.sasa, add_special_tokens=True
            )

        # Infer input length
        sequence_length = -1
        if sequence_tokens is not None:
            sequence_length = len(sequence_tokens)
        elif secondary_structure_tokens is not None:
            sequence_length = len(secondary_structure_tokens)
        elif sasa_tokens is not None:
            sequence_length = len(sasa_tokens)

        # Try to infer input length from structure data
        if input.coordinates is not None:
            coordinates, _, structure_tokens = encoding.tokenize_structure(
                input.coordinates,
                self.get_structure_token_encoder(),
                structure_tokenizer=self.tokenizers.structure,
                reference_sequence=input.sequence or "",
                add_special_tokens=True,
            )
            if sequence_length == -1:
                sequence_length = len(structure_tokens)

        if sequence_length == -1:
            raise ValueError(
                "Cannot infer input length from input data. Please provide one of: sequence, structure, secondary_structure, sasa.\n"
                "To condition on sequence length only, use ESM3LocalInferenceClient.get_default_sequence(sequence_length) to generate a default sequence input."
            )

        # Function and Residue annotations
        if input.function_annotations is not None:
            if input.sequence is None:
                reference_sequence = encoding.get_default_sequence(sequence_length - 2)
            else:
                reference_sequence = input.sequence
            (
                function_tokens,
                residue_annotation_tokens,
            ) = encoding.tokenize_function_annotations(
                input.function_annotations,
                reference_sequence=reference_sequence,
                function_tokenizer=self.tokenizers.function,
                residue_annotation_tokenizer=self.tokenizers.residue_annotations,
                add_special_tokens=True,
            )

        return ESMProteinTensor(
            sequence=sequence_tokens,
            structure=structure_tokens,
            secondary_structure=secondary_structure_tokens,
            sasa=sasa_tokens,
            function=function_tokens,
            residue_annotations=residue_annotation_tokens,
            coordinates=coordinates,
        ).to(next(self.parameters()).device)

    def decode(
        self,
        input: ESMProteinTensor,
    ) -> ESMProtein:
        return decode_protein_tensor(
            input=input,
            tokenizers=self.tokenizers,
            structure_token_decoder=self.get_structure_token_decoder(),
            function_token_decoder=(
                self.get_function_token_decoder() if input.function is not None else None
            ),
        )

    def decode_batch(self, inputs: list[ESMProteinTensor]) -> list[ESMProtein]:
        if not inputs:
            return []
        if len(inputs) == 1:
            return [self.decode(inputs[0])]
        need_function = any(item.function is not None for item in inputs)
        return decode_protein_tensor_batch(
            inputs=inputs,
            tokenizers=self.tokenizers,
            structure_token_decoder=self.get_structure_token_decoder(),
            function_token_decoder=(
                self.get_function_token_decoder() if need_function else None
            ),
        )

    def _forward(
        self, input: ESMProteinTensor, config: ForwardConfig = ForwardConfig()
    ) -> ForwardOutput:
        # Default plddt conditioning for inference. 1s where coordinates are provided.
        if input.coordinates is None:
            per_res_plddt = None
        else:
            # 1.0 if all coordinates at specific indices have valid non-nan values.
            per_res_plddt = input.coordinates.isfinite().all(dim=-1).any(dim=-1).float()

        # with torch.no_grad() if self.eval else contextlib.nullcontext():
        with contextlib.nullcontext() if self.training else torch.no_grad():
            output = self.forward(
                sequence_tokens=input.sequence,
                structure_tokens=input.structure,
                ss8_tokens=input.secondary_structure,
                sasa_tokens=input.sasa,
                function_tokens=input.function,
                residue_annotation_tokens=input.residue_annotations,
                average_plddt=torch.tensor(1.0, device=input.device),
                per_res_plddt=per_res_plddt,
                structure_coords=input.coordinates,
                chain_id=None,
                sequence_id=None,
            )

            if config.return_logits:
                logits = ForwardTrackData(
                    sequence=output.sequence_logits,
                    structure=output.structure_logits,
                    secondary_structure=output.secondary_structure_logits,
                    sasa=output.sasa_logits,
                    function=output.function_logits,
                )
            else:
                logits = None

            return ForwardOutput(
                logits=logits,
                residue_annotation_logits=output.residue_logits,
                embeddings=output.embeddings if config.return_embeddings else None,
            )

    # real_probs is custom design for generation using real probs
    def forward_and_sample(
        self, input: ESMProteinTensor, sampling_configuration: SamplingConfig, real_probs: torch.Tensor, alpha: float, 
    ) -> ForwardAndSampleOutput:
        protein_tensor = attr.evolve(input)  # Make a copy

        def maybe_clone(x: torch.Tensor | None) -> torch.Tensor | None:
            return x.clone() if x is not None else None

        device = next(self.parameters()).device

        sampling_config = sampling_configuration
        if sampling_config is None:
            sampling_config = get_default_sampling_config(self.tokenizers)

        def _infer_batch_and_length(protein: ESMProteinTensor) -> tuple[bool, int, int]:
            for track in [
                "sequence",
                "structure",
                "secondary_structure",
                "sasa",
            ]:
                tensor = getattr(protein, track, None)
                if tensor is None:
                    continue
                if tensor.dim() == 1:
                    return False, 1, tensor.size(0)
                if tensor.dim() == 2:
                    return True, tensor.size(0), tensor.size(1)
                raise ValueError(
                    f"Unexpected rank {tensor.dim()} for track {track}: {tuple(tensor.shape)}"
                )
            for track in ["function", "residue_annotations"]:
                tensor = getattr(protein, track, None)
                if tensor is None:
                    continue
                if tensor.dim() == 2:
                    return False, 1, tensor.size(0)
                if tensor.dim() == 3:
                    return True, tensor.size(0), tensor.size(1)
                raise ValueError(
                    f"Unexpected rank {tensor.dim()} for track {track}: {tuple(tensor.shape)}"
                )
            if protein.coordinates is not None:
                coordinates = protein.coordinates
                if coordinates.dim() == 3:
                    return False, 1, coordinates.size(0)
                if coordinates.dim() == 4:
                    return True, coordinates.size(0), coordinates.size(1)
            raise ValueError("No input data provided")

        keep_batch, batch_size, encoded_length = _infer_batch_and_length(input)

        # Initialize default values for missing tracks
        default_protein_tensor = ESMProteinTensor.empty(
            encoded_length - 2, tokenizers=self.tokenizers, device=input.device
        )
        for track in attr.fields(ESMProteinTensor):
            if getattr(protein_tensor, track.name, None) is None:
                setattr(
                    protein_tensor,
                    track.name,
                    getattr(default_protein_tensor, track.name, None),
                )

        # Preprocessing
        sequence_length: int = -1
        for track in [
            "sequence",
            "structure",
            "secondary_structure",
            "sasa",
            "function",
            "residue_annotations",
        ]:
            input_tensor: torch.Tensor | None = getattr(protein_tensor, track, None)
            if input_tensor is not None:
                # Add batch dimension if necessary
                if track in ["sequence", "structure", "secondary_structure", "sasa"]:
                    if len(input_tensor.size()) == 1:
                        input_tensor = input_tensor.unsqueeze(0)  # (L,) -> (1, L)
                    if keep_batch and input_tensor.size(0) == 1 and batch_size > 1:
                        input_tensor = input_tensor.expand(batch_size, -1).contiguous()
                elif track in ["function", "residue_annotations"]:
                    if len(input_tensor.size()) == 2:
                        input_tensor = input_tensor.unsqueeze(0)  # (L, O) -> (1, L, O)
                    if keep_batch and input_tensor.size(0) == 1 and batch_size > 1:
                        input_tensor = input_tensor.expand(batch_size, -1, -1).contiguous()

                # Check length consistency
                if sequence_length == -1:
                    sequence_length = input_tensor.size(1)
                else:
                    if input_tensor.size(1) != sequence_length:
                        raise ValueError(
                            f"Length mismatch for track {track}. Expected {sequence_length}, got {input_tensor.size(1)}"
                        )

                # Move input tensor to model device
                input_tensor = input_tensor.to(device)
                setattr(protein_tensor, track, input_tensor)

        if protein_tensor.coordinates is not None:
            coordinates = protein_tensor.coordinates
            if len(coordinates.size()) == 3:
                coordinates = coordinates.unsqueeze(0)
            if keep_batch and coordinates.size(0) == 1 and batch_size > 1:
                coordinates = coordinates.expand(batch_size, -1, -1, -1).contiguous()
            protein_tensor.coordinates = coordinates.to(device)
            sequence_length = coordinates.size(1)

        if sequence_length == -1:
            raise ValueError("No input data provided")

        # Forward pass
        forward_output = self._forward(
            protein_tensor,
            ForwardConfig(
                ReturnLogitsConfig(
                    sequence=True,
                    structure=True,
                    secondary_structure=True,
                    sasa=True,
                    function=True,
                    residue_annotations=True,
                ),
                return_embeddings=True,
            ),
        )

        # Sampling
        tokens_dir = {}
        track_sampling_metadata_dir: dict[str, dict | None] = {}
        for track in ["sequence", "structure", "secondary_structure", "sasa"]:
            config = getattr(sampling_config, track)
            if config is None:
                tokens_dir[track] = maybe_clone(getattr(input, track))
                continue
            logits = getattr(forward_output.logits, track)
            tokens = getattr(protein_tensor, track)
            if not keep_batch:
                logits = logits[0, ...]
                tokens = tokens[0, ...]
            sampling_metadata = self._sample_track(
                logits=logits,
                tokens=tokens,
                sampling_track_config=config,
                mask_idx=getattr(self.tokenizers, track).mask_token_id,
                real_probs=real_probs,   # Custom, using actual probs instead of logits
                alpha=alpha, # Custom, defined the contribution of real probs
            )
            tokens_dir[track] = sampling_metadata.pop("sampled_tokens")
            track_sampling_metadata_dir[track] = sampling_metadata

        # Sample function and residue annotations separately
        config = getattr(sampling_config, "function")
        if config is None:
            tokens_dir["function"] = maybe_clone(getattr(input, "function"))
            tokens_dir["residue_annotations"] = maybe_clone(
                getattr(input, "residue_annotations")
            )
        else:
            function_tokens = getattr(protein_tensor, "function")
            function_logits = getattr(forward_output.logits, "function")
            residue_logits = forward_output.residue_annotation_logits
            if not keep_batch:
                function_tokens = function_tokens[0, ...]
                function_logits = function_logits[0, ...]
                residue_logits = residue_logits[0, ...]
            sampling_metadata = self._sample_function_track(
                tokens=function_tokens,
                logits=function_logits,
                sampling_track_config=config,
            )
            tokens_dir["function"] = sampling_metadata.pop("sampled_tokens")
            track_sampling_metadata_dir["function"] = sampling_metadata

            sampled_tokens, _ = sample_residue_annotation_logits(
                logits=residue_logits  # type: ignore
            )
            tokens_dir["residue_annotations"] = sampled_tokens

        # Format output
        forward_and_sample_output_dir = {}
        forward_and_sample_output_dir["protein_tensor"] = ESMProteinTensor(**tokens_dir)
        for property in [
            "entropy",
            "prob",
            "logprob",
            "top_prob",
            "topk_logprob",
            "topk_tokens",
        ]:
            is_all_none = True
            forward_track_data_dir = {}
            for track in track_sampling_metadata_dir.keys():
                values = track_sampling_metadata_dir[track]
                # 检查当前轨道的元数据 values 是否存在，并且是否包含当前属性 property 的值。
                if values is not None and values.get(property, None) is not None:
                    # 如果存在，则将 property 对应的值添加到 forward_track_data_dir 中。
                    forward_track_data_dir[track] = values.get(property, None)
                    is_all_none = False
            if not is_all_none:
                forward_and_sample_output_dir[property] = ForwardTrackData(
                    **forward_track_data_dir
                )
            else:
                forward_and_sample_output_dir[property] = None

        perres_embed = (
            (forward_output.embeddings if keep_batch else forward_output.embeddings[0])  # type: ignore
            if sampling_configuration.return_per_residue_embeddings
            else None
        )
        mean_embedding = (
            (forward_output.embeddings.mean(1) if keep_batch else forward_output.embeddings[0].mean(0))  # type: ignore
            if sampling_configuration.return_mean_embedding
            else None
        )

        return ForwardAndSampleOutput(
            per_residue_embedding=perres_embed,
            mean_embedding=mean_embedding,
            **forward_and_sample_output_dir,
        )

    def _sample_track(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        sampling_track_config: SamplingTrackConfig,
        mask_idx: int,
        real_probs: torch.Tensor,
        alpha: float = 0.6,
    ) -> dict[str, torch.Tensor]:
        
        # print(f'\ntokens: {tokens}\nshape: {tokens.size()}')
        # print(f'logits: {logits}\nshape: {logits.size()}')
        # Sample in all positions
        temperature = sampling_track_config.temperature
        # sampled_tokens = sample_logits(
        #     logits, temperature=temperature, top_p=sampling_track_config.top_p
        # )

        # Custom design, using actual probs instead of logits
        if real_probs is not None:
            sampled_tokens = sample_fused_logits(
                logits=logits, prior_probs=real_probs, temperature=temperature, top_p=sampling_track_config.top_p, alpha=alpha
            )
        else:
            sampled_tokens = sample_logits(
                logits, temperature=temperature, top_p=sampling_track_config.top_p
            )
        # print(f'\nsampled_tokens after sample_logits: {sampled_tokens}\nshape: {sampled_tokens.size()}')

        # log_probs = logits.log_softmax(-1)

        # Custom: 转换为对数概率分布，此时logits实际上是真实概率分布，用于后续计算熵
        log_probs = torch.log(real_probs) if real_probs is not None else logits.log_softmax(-1)

        # print(f'log_probs (logits.log_softmax(-1)): {log_probs}\nshape: {log_probs.size()}')

        # Do not sample at BOS and EOS tokens
        sampling_mask = torch.ones_like(tokens, dtype=torch.bool)
        sampling_mask[..., 0] = False
        sampling_mask[..., -1] = False

        # Do not sample at special token positions but allow sampling at mask token
        special_minus_mask = list(set(sampling_track_config.invalid_ids) - {mask_idx})
        if len(special_minus_mask) > 0:
            special_tokens = torch.tensor(special_minus_mask, device=tokens.device)
            assert special_tokens.numel() > 0
            sampling_mask = sampling_mask & (
                tokens[..., None] != special_tokens[None, :]
            ).all(-1)

        # Keep only samples from masked positions (if specified)
        if sampling_track_config.only_sample_masked_tokens:
            masked_tokens = tokens == mask_idx
            sampling_mask = sampling_mask & masked_tokens
        sampled_tokens = torch.where(sampling_mask, sampled_tokens, tokens)

        # print(f'sampled_tokens in _sample_track output: {sampled_tokens}')
        return self._compute_track_metadata(
            sampled_tokens,
            log_probs,
            sampling_mask,
            top_k=sampling_track_config.topk_logprobs,
        )

    def _sample_function_track(
        self,
        tokens: torch.Tensor,
        logits: torch.Tensor,
        sampling_track_config: SamplingTrackConfig,
    ) -> dict[str, torch.Tensor]:
        # Do not sample at BOS and EOS tokens
        sampling_mask = torch.ones_like(tokens, dtype=torch.bool)
        sampling_mask[0] = False
        sampling_mask[-1] = False

        sampled_tokens, probs = sample_function_logits(
            logits,
            self.tokenizers.function,
            top_p=sampling_track_config.top_p,
            temperature=sampling_track_config.temperature,
        )

        if sampling_track_config.only_sample_masked_tokens:
            raise ValueError(
                "Sampling only masked tokens is undefined for function tokens."
            )

        sampled_tokens = torch.where(sampling_mask, sampled_tokens, tokens)  # (L, D)

        return self._compute_track_metadata(
            sampled_tokens,
            probs,
            sampling_mask,
            top_k=sampling_track_config.topk_logprobs,
        )

    @staticmethod
    def _compute_track_metadata(
        sampled_tokens: torch.Tensor,
        log_probs: torch.Tensor,
        sampling_mask: torch.Tensor,
        top_k: int,
    ) -> dict:
        probs = torch.exp(log_probs)  # (B, L)
        entropy = torch.distributions.Categorical(probs=probs).entropy()  # (B, L)

        # Only compute probabilities for sampled tokens
        sampled_logprob = torch.zeros_like(
            sampled_tokens, dtype=torch.float32
        )  # (B, L)
        sampled_tokens_valid = sampled_tokens[sampling_mask]
        masked_log_probs = log_probs[sampling_mask]
        if sampled_tokens_valid.numel() == 0:
            sampled_log_probs_valid = sampled_tokens_valid.to(dtype=torch.float32)
        else:
            sampled_log_probs_valid = masked_log_probs[
                torch.arange(sampled_tokens_valid.shape[0], device=sampled_tokens.device),
                sampled_tokens_valid,
            ]
        sampled_logprob[sampling_mask] = sampled_log_probs_valid.to(dtype=torch.float32)

        # Calculate extra metadata
        sampled_prob = torch.exp(sampled_logprob)
        top_prob = torch.max(probs, dim=-1).values
        topk_logprobs, topk_tokens = torch.topk(log_probs, top_k, dim=-1)
        topk_logprobs = None if top_k == 0 else topk_logprobs
        topk_tokens = None if top_k == 0 else topk_tokens

        return {
            "entropy": entropy,
            "sampled_tokens": sampled_tokens,
            "prob": sampled_prob,
            "logprob": sampled_logprob,
            "top_prob": top_prob,
            "topk_logprob": topk_logprobs,
            "topk_tokens": topk_tokens,
        }

    # Customized forward module
    # def finetuning(self, labels: torch.tensor, input_ids: torch.tensor, structures: torch.tensor, coordinates: torch.tensor) -> dict:
    def finetuning(self, labels: torch.tensor, input_ids: torch.tensor, structures: torch.tensor) -> dict:
        device = next(self.parameters()).device
        # Make a copy of the tensor
        labels = labels.clone().view(-1)
        sequence_tokens = input_ids.clone().view(-1)
        structures_tokens = structures.clone().view(-1)
        # coordinates_tokens = coordinates.clone().view(-1)

        # # Remove the padding
        # # Should not let the pads pass 'forward' - 241205
        # pad_token_id = -200
        # labels = self.remove_padding(labels, pad_token_id)
        # sequence_tokens = self.remove_padding(sequence_tokens, pad_token_id)
        # structures_tokens = self.remove_padding(structures_tokens, pad_token_id)
        # coordinates_tokens = self.remove_padding(coordinates_tokens, pad_token_id)

        # # Restore coordinates
        # # The shape of the array is `(n_residues, 37, 3)` where `n_residues` is the number of residues in the protein and 37 is the number of possible distinct atoms that may be present across all amino acids
        # n_residues = len(sequence_tokens)
        # shape = (n_residues, 37, 3)
        # # coordinates_tokens = coordinates_tokens.reshape(shape)
        # coordinates_tokens = coordinates_tokens.contiguous().view(shape)

        # # NOTE should set structure tokens None? -> generate structure tokens based on coordinates
        # if coordinates_tokens is not None:
        #     structures_tokens = None

        # Form ESMProteinTensor
        protein_tensor = ESMProteinTensor(
            sequence=sequence_tokens,
            structure=structures_tokens,
            secondary_structure=None,
            sasa=None,
            function=None,
            residue_annotations=None,
            coordinates=None,
            # coordinates=coordinates_tokens,
        ).to(device)

        # print(f'protein_tensor:\n{protein_tensor}')

        # Initialize default values for missing tracks
        default_protein_tensor = ESMProteinTensor.empty(
            len(protein_tensor) - 2, tokenizers=self.tokenizers, device=device
        )
        for track in attr.fields(ESMProteinTensor):
            if getattr(protein_tensor, track.name, None) is None:
                setattr(
                    protein_tensor,
                    track.name,
                    getattr(default_protein_tensor, track.name, None),
                )

        # Preprocessing
        sequence_length: int = -1
        for track in [
            "sequence",
            "structure",
            "secondary_structure",
            "sasa",
            "function",
            "residue_annotations",
        ]:
            input_tensor: torch.Tensor | None = getattr(protein_tensor, track, None)
            if input_tensor is not None:
                # Add batch dimension if necessary
                if track in ["sequence", "structure", "secondary_structure", "sasa"]:
                    if len(input_tensor.size()) == 1:
                        # 通过 unsqueeze(0) 方法在第 0 维增加一个维度，将其变为 (1, L)。这相当于为该张量增加了一个批处理维度。
                        input_tensor = input_tensor.unsqueeze(0)  # (L,) -> (1, L)
                elif track in ["function", "residue_annotations"]:
                    if len(input_tensor.size()) == 2:
                        input_tensor = input_tensor.unsqueeze(0)  # (L, O) -> (1, L, O)

                # Check length consistency
                if sequence_length == -1:
                    sequence_length = input_tensor.size(1)
                else:
                    if input_tensor.size(1) != sequence_length:
                        raise ValueError(
                            f"Length mismatch for track {track}. Expected {sequence_length}, got {input_tensor.size(1)}"
                        )

                # Move input tensor to model device
                input_tensor = input_tensor.to(device)
                setattr(protein_tensor, track, input_tensor)

        if protein_tensor.coordinates is not None:
            coordinates = protein_tensor.coordinates
            if len(coordinates.size()) == 3:
                coordinates = coordinates.unsqueeze(0)
            protein_tensor.coordinates = coordinates.to(device)
            sequence_length = coordinates.size(1)

        if sequence_length == -1:
            raise ValueError("No input data provided")

        # Forward pass
        forward_output = self._forward(
            protein_tensor,
            ForwardConfig(
                ReturnLogitsConfig(
                    sequence=True,
                    structure=True,
                    secondary_structure=True,
                    sasa=True,
                    function=True,
                    residue_annotations=True,
                ),
                return_embeddings=True,
            ),
        )

        # Obtain logits for further calculation
        output_logits: torch.Tensor = getattr(forward_output.logits, "sequence")
        # embeddings: torch.Tensor = forward_output.embeddings

        # Calculate loss
        # TODO: Combine the extention of batches and selected region in labels
        loss = None
        outputs = None
        if labels is not None:
            # loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            # logits = output_logits.detach()
            # logits = logits.view(-1, logits.size(-1))
            # print(f'logits: {output_logits.detach()}')
            # loss = loss_fct(logits, labels)

            criterion = nn.CrossEntropyLoss(ignore_index=-100)
            target = output_logits.view(-1, output_logits.size(-1))
            loss = criterion(target, labels)
            # print(f'loss: {loss}')
            # requires specifying ``grad_tensors``
            outputs = (loss, output_logits)

        return outputs
    
    @staticmethod
    def remove_padding(tensor: torch.Tensor, pad_value: int) -> torch.Tensor:
        """
        Removes the padding from the tensor by removing all elements equal to pad_value.
        
        Args:
            tensor (torch.Tensor): The tensor from which to remove padding.
            pad_value (int): The value that represents padding to be removed.
        
        Returns:
            torch.Tensor: The tensor with padding removed.
        """
        # Create a mask where elements not equal to pad_value are kept
        mask = tensor != pad_value
        # Apply the mask to the tensor to remove padding
        filtered_tensor = tensor[mask]
        return filtered_tensor.contiguous()
    
    
    def structure_encode_full(self, input: ESMProtein) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from esm.utils.structure.protein_chain import ProteinChain

        input = attr.evolve(input)  # Make a copy
        structure_raw_embeddings = None
        structure_pre_q_embeddings = None
        structure_q_embeddings = None
        structure_tokens = None
        coordinates = None

        # Try to infer input length from structure data
        if input.coordinates is not None:
            coordinates = input.coordinates
            structure_encoder: StructureTokenEncoder = self.get_structure_token_encoder()

            device = next(structure_encoder.parameters()).device
            reference_sequence = input.sequence or ""
            chain = ProteinChain.from_atom37(
                coordinates, sequence=reference_sequence if reference_sequence else None
            )

            # Setup padding
            if reference_sequence and len(reference_sequence) != coordinates.size(0):
                raise ValueError(
                    f"Reference sequence length ({len(reference_sequence)}) does not match the number of residues in the coordinates ({coordinates.size(0)})"
                )
            
            # BOS and EOS is unnecessary for structure encode, since we do not generate design when using 'structure_encode_full'
            coordinates, plddt, residue_index = chain.to_structure_encoder_inputs()
            coordinates = coordinates.to(device)  # (1, L, 37, 3)
            plddt = plddt.to(device)  # (1, L)
            residue_index = residue_index.to(device)  # (1, L)

            # Obtain embeddings and tokens
            structure_raw_embeddings, structure_pre_q_embeddings, structure_q_embeddings, structure_tokens = structure_encoder.encode_full(
                coordinates, residue_index=residue_index
            )

        return (structure_raw_embeddings, structure_pre_q_embeddings, structure_q_embeddings, structure_tokens)

    # Customized forward module
    # def finetuning(self, labels: torch.tensor, input_ids: torch.tensor, structures: torch.tensor, coordinates: torch.tensor) -> dict:
    def md_finetuning(self, labels: torch.tensor, input_ids: torch.tensor, epsilon: float = 0.05) -> dict:
        import torch.nn.functional as F

        device = next(self.parameters()).device
        # Make a copy of the tensor
        labels = labels.clone()
        # sequence_tensor = input_ids.clone().view(-1)
        sequence_tensor = input_ids.clone()

        # 拿到 B, L
        if sequence_tensor.dim() != 2:
            raise ValueError(f"Expect sequence_tensor to be 2D, got {sequence_tensor.shape}")
        B, L = sequence_tensor.shape

        # Form ESMProteinTensor
        protein_tensor = ESMProteinTensor(
            sequence=sequence_tensor,
            structure=None,
            secondary_structure=None,
            sasa=None,
            function=None,
            residue_annotations=None,
            coordinates=None,
        ).to(device)

        # print(f'protein_tensor:\n{protein_tensor}')

        # Initialize default values for missing tracks
        # Original 'len(protein_tensor) - 2' will extract the batch length, instead of encoded sequence length
        default_protein_tensor = ESMProteinTensor.empty(
            L - 2, tokenizers=self.tokenizers, device=device
        )
        for track in attr.fields(ESMProteinTensor):
            if getattr(protein_tensor, track.name, None) is None:
                setattr(
                    protein_tensor,
                    track.name,
                    getattr(default_protein_tensor, track.name, None),
                )

        # Preprocessing
        sequence_length: int = -1
        for track in [
            "sequence",
            "structure",
            "secondary_structure",
            "sasa",
            "function",
            "residue_annotations",
        ]:
            input_tensor: torch.Tensor | None = getattr(protein_tensor, track, None)
            if input_tensor is not None:
                # Add batch dimension if necessary
                if track in ["sequence", "structure", "secondary_structure", "sasa"]:
                    if len(input_tensor.size()) == 1:
                        # 通过 unsqueeze(0) 方法在第 0 维增加一个维度，将其变为 (B, L)。这相当于为该张量增加了一个批处理维度。
                        input_tensor = input_tensor.unsqueeze(0).expand(B, L).contiguous()  # (L,) -> (B, L)
                    elif len(input_tensor.size()) == 2:
                        # 已经是 (B, L)，什么都不做
                        pass
                    else:
                        raise ValueError(f"Unexpected dim for track {track}: {input_tensor.dim()}")
                elif track in ["function", "residue_annotations"]:
                    O = input_tensor.size(-1)
                    if len(input_tensor.size()) == 2:
                        input_tensor = input_tensor.unsqueeze(0).expand(B, L, O).contiguous()  # (L, O) -> (B, L, O)
                    elif len(input_tensor.size()) == 3:
                        # 已经是 (B, L, O)
                        pass
                    else:
                        raise ValueError(f"Unexpected dim for track {track}: {input_tensor.dim()}")

                # Check length consistency
                if sequence_length == -1:
                    sequence_length = input_tensor.size(1)  # Defined based on sequence track
                else:
                    if input_tensor.size(1) != sequence_length:
                        raise ValueError(
                            f"Length mismatch for track {track}. Expected {sequence_length}, got {input_tensor.size(1)}.\n Current input_tensor: {input_tensor.size()}"
                        )

                # Move input tensor to model device
                input_tensor = input_tensor.to(device)
                setattr(protein_tensor, track, input_tensor)

        if protein_tensor.coordinates is not None:
            coordinates = protein_tensor.coordinates
            if len(coordinates.size()) == 3:
                coordinates = coordinates.unsqueeze(0)
            protein_tensor.coordinates = coordinates.to(device)
            sequence_length = coordinates.size(1)

        if sequence_length == -1:
            raise ValueError("No input data provided")

        # Forward pass
        forward_output = self._forward(
            protein_tensor,
            ForwardConfig(
                ReturnLogitsConfig(
                    sequence=True,
                    structure=True,
                    secondary_structure=True,
                    sasa=True,
                    function=True,
                    residue_annotations=True,
                ),
                return_embeddings=True,
            ),
        )

        output_logits: torch.Tensor = getattr(forward_output.logits, "structure")

        # 得到模型的 log‑prob Q
        model_log_probs = F.log_softmax(output_logits, dim=-1)  # [B, L, V]
        B, L, V = output_logits.shape

        if model_log_probs.shape != labels.shape:
            print()
            print("[Warning] Shape mismatch between model_log_probs and label!")
            print(">> input:",  model_log_probs.shape)
            print(">> target:", labels.shape)

        # Calculate loss
        # labels: [B, L, V]，padding 部分的值都是 -100
        valid_mask = labels.ne(-100).any(dim=-1)   # [B, L]

        # labels_clamped = labels.clamp(min=1e-8)    # [B, L, V]
        P_raw = labels.clone()    # [B, L, V]


        # # reduction='batchmean' 会把所有批次和维度上的 loss sum 后除以 batch size
        # kl_elem = F.kl_div(
        #     input=model_log_probs,    # log Q
        #     target=labels_clamped,     # P
        #     reduction='none'     # 或者 'sum' / 'mean'，根据需求选择
        # )

        # label smoothing
        # P_smooth = (1 - ε) P + ε / V
        smooth_val = epsilon / V
        P_smooth = P_raw * (1.0 - epsilon) + smooth_val            # [B, L, V]
        log_P_smooth = torch.log(P_smooth.clamp(min=1e-8))
        kl_elem = F.kl_div(
            input=model_log_probs,
            target=log_P_smooth,
            reduction='none',
            log_target=True
        )  # [B, L, V]

        # 把 pad 处（valid_mask=False）的 kl 值置为 0
        mask = valid_mask.unsqueeze(-1).float()       # [B, L, 1]
        kl_masked = kl_elem * mask  # [B, L, V]

        # sum over vocab，再 sum over batch × length，然后除以总有效位置数
        total_kl = kl_masked.sum(dim=-1).sum()        # 标量
        num_valid = valid_mask.sum()                  # 标量
        loss = total_kl / num_valid.clamp(min=1.0)

        # print(f'loss: {loss}')
        # requires specifying ``grad_tensors``
        outputs = (loss, output_logits)

        return outputs