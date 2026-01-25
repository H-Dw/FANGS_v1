import attr
import torch
import torch.nn.functional as F

from esm.sdk.api import (
    SamplingConfig,
    SamplingTrackConfig,
)
from esm.tokenization import (
    TokenizerCollection,
    get_invalid_tokenizer_ids,
)
from esm.tokenization.function_tokenizer import (
    InterProQuantizedTokenizer,
)
from esm.utils.constants.esm3 import MAX_RESIDUE_ANNOTATIONS


def get_default_sampling_config(tokenizers: TokenizerCollection) -> SamplingConfig:
    tracks = [f.name for f in attr.fields(SamplingConfig)]
    sampling_config = SamplingConfig()
    for current_track in tracks:
        setattr(
            sampling_config,
            current_track,
            SamplingTrackConfig(
                invalid_ids=get_invalid_tokenizer_ids(
                    getattr(tokenizers, current_track)
                ),
                temperature=1.0,
                top_p=1.0,
                # TODO: Add different mask and padding tokens for all tracks
                # Some tracks have the same pad and mask, which causes ambiguity when sampling
                only_sample_masked_tokens=current_track
                not in ["secondary_structure", "sasa", "function"],
            ),
        )
    return sampling_config


def sample_logits(
    logits: torch.Tensor,
    temperature: float | torch.Tensor,
    top_p: float | torch.Tensor = 1.0,
):
    """Default sampling from logits.

    Args:
        logits is shape (..., vocab_size)
        temperature is broadcastable to (...)
    """

    # 1. 如果指定了 top_p < 1.0，先做 nucleus（top-p）裁剪：
    if top_p < 1.0:
        logits = top_p_logits(logits, top_p=top_p)

    # 2. 将 temperature 扩展到与 logits 相同的 batch 形状
    temperature = _tensorize_like(temperature, logits)

    # 3. 如果 temperature 全部为 0，则做贪心（argmax）：
    if torch.all(temperature == 0):
        ids = logits.argmax(-1)
        return ids

    # 4. 不支持部分 temperature==0 的情况
    assert not torch.any(temperature == 0), "Partial temperature 0 not supported."

    # 5. 展平所有的 batch 维度，保留最后一个 vocab_size 维度
    batch_dims = logits.size()[:-1]
    logits = logits.reshape(-1, logits.shape[-1])

    # 6. 通过 softmax(logits/temperature) 得到概率分布
    # Sample from all logits
    probs = F.softmax(logits / temperature[..., None], dim=-1)
    # 7. 从该分布中多项式采样一个 token
    ids = torch.multinomial(probs, 1).squeeze(1)

    # 8. 恢复原始的 batch 形状
    ids = ids.reshape(*batch_dims)
    return ids

# Custom
def sample_probs(
    probs: torch.Tensor,
    top_p: float | torch.Tensor = 1.0,
):
    """
    Args:
        probs: 形状为 (..., vocab_size)，表示每个 token 位置的真实概率分布。
        top_p: Top-p 采样阈值（可选）。
    Returns:
        采样得到的 token 索引，形状为 (...,)
    """
    if top_p < 1.0:
        # 对概率排序并计算累积概率
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 排除累积概率超过 top_p 的词汇
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 0] = False  # 至少保留一个 token

        # 将被移除的词汇概率置零
        probs = probs.scatter(-1, sorted_indices[sorted_indices_to_remove], 0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)  # 重新归一化

    batch_dims = probs.size()[:-1]
    probs_flat = probs.reshape(-1, probs.shape[-1])  # 展平为二维张量
    ids_flat = torch.multinomial(probs_flat, 1).squeeze(1)  # 采样
    ids = ids_flat.reshape(*batch_dims)  # 恢复原始形状
    return ids

def sample_fused_logits(
    logits: torch.Tensor,
    prior_probs: torch.Tensor,
    alpha: float,
    temperature: float | torch.Tensor,
    top_p: float | torch.Tensor = 1.0,
):
    # 截断：将小于 1e-8 的概率都“拉”到 1e-8，避免 log(0)
    # 确保调用者传入的确实是 真 的概率分布张量（在 [0,1] 区间且 sum=1）
    probs_clamped = prior_probs.clamp(min=1e-8)
    prior_log_probs = torch.log(probs_clamped)

    # 融合
    model_log_probs = F.log_softmax(logits, dim=-1) # 归一化模型 logits
    fused_logit = alpha * model_log_probs + (1-alpha) * prior_log_probs

    # 直接用之前的采样函数
    return sample_logits(fused_logit, temperature, top_p)



def sample_function_logits(
    logits: torch.Tensor,
    tokenizer: InterProQuantizedTokenizer,
    top_p: float | torch.Tensor = 1.0,
    temperature: float | torch.Tensor = 1.0,
    p_none_threshold: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    [L, D, V] = logits.shape
    assert D == tokenizer.depth

    if top_p < 1.0:
        logits = top_p_logits(logits, top_p=top_p)

    temperature = torch.ones_like(logits[..., 0]) * temperature

    log_p = F.log_softmax(logits / temperature[..., None], dim=-1)  # (L, D, V)

    # Choose which positions have no predicted function.
    log_p_nones = log_p[..., tokenizer.vocab_to_index["<none>"]]  # (L, D)
    p_none = torch.exp(log_p_nones).mean(dim=-1)  # "Ensemble of <none> predictions"
    where_none = p_none > p_none_threshold  # (L, )

    # Set probability of <none> to 0 for all not-none positions
    none_index = tokenizer.vocab_to_index["<none>"]
    log_p[~where_none, :, none_index] = -torch.inf

    ids = torch.argmax(log_p, dim=-1)  # (L, D)
    ids[where_none, :] = tokenizer.vocab_to_index["<none>"]

    return ids, log_p


def sample_residue_annotation_logits(
    logits: torch.Tensor, annotation_threshold: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    # Take top residue annotations
    top_residue_annotations_idx = logits.argsort(dim=-1, descending=True)[
        ..., :MAX_RESIDUE_ANNOTATIONS
    ]  # (L, MAX_R)
    top_residue_annotations_logprobs = torch.gather(
        F.logsigmoid(logits), -1, top_residue_annotations_idx
    )  # (L, MAX_R)
    top_residue_annotations_probs = top_residue_annotations_logprobs.exp()
    # Keep only positive predictions
    is_negative = top_residue_annotations_probs < annotation_threshold
    top_residue_annotations_idx[is_negative] = 0

    top_residue_annotations_logprobs = top_residue_annotations_logprobs

    return top_residue_annotations_idx, top_residue_annotations_logprobs


def top_p_logits(
    logits: torch.Tensor,
    top_p: float | torch.Tensor,
) -> torch.Tensor:
    top_p = _tensorize_like(top_p, logits)

    batch_dims = logits.size()[:-1]
    logits = logits.reshape(-1, logits.shape[-1])

    # Sort logits in descending order and extract the mask for the top_p
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cumsum_logits = sorted_logits.softmax(-1).cumsum(-1)
    top_p_mask = cumsum_logits <= top_p[:, None]

    # Make sure at least one token is sampled
    top_p_mask[:, 0] = True

    # Mask out the logits that are not in the top_p
    batch_indices_to_mask, _ = torch.where(~top_p_mask)
    vocab_indices_to_mask = sorted_indices[~top_p_mask]
    logits[batch_indices_to_mask, vocab_indices_to_mask] = torch.finfo(logits.dtype).min

    return logits.reshape(*batch_dims, -1)


def _tensorize_like(value: int | float | torch.Tensor, logits: torch.Tensor):
    if isinstance(value, (float, int)):
        value = torch.full_like(logits[..., 0], value, dtype=logits.dtype)
    return value.to(logits.device).expand_as(logits[..., 0]).reshape(-1)
