"""Shared visual-token compression primitives for MSD training and inference."""

from typing import Optional, Tuple

import torch


def validate_visual_compression(num_tokens: int) -> None:
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}")


def build_shifted_visual_padding_mask(
    sequence_length: int,
    visual_start: int,
    original_visual_tokens: int,
    kept_visual_tokens: int,
    device=None,
) -> torch.Tensor:
    """Mask padding slots after the one-token shift used by the draft model."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if visual_start < 0:
        raise ValueError("visual_start must be non-negative")
    if not 0 < kept_visual_tokens <= original_visual_tokens:
        raise ValueError("kept_visual_tokens must be in [1, original_visual_tokens]")
    if visual_start + original_visual_tokens > sequence_length:
        raise ValueError("visual span exceeds sequence length")

    attention_mask = torch.ones((1, sequence_length), dtype=torch.bool, device=device)
    padding_start = visual_start + kept_visual_tokens - 1
    padding_stop = visual_start + original_visual_tokens - 1
    attention_mask[:, padding_start:padding_stop] = False
    return attention_mask


def _nearest_euclidean_centers(features: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Assign each feature to its nearest center without materializing Euclidean distances.

    For a fixed feature, ||x||^2 is identical for every center, so minimizing
    ||x-c||^2 is equivalent to maximizing 2*x*c^T-||c||^2. This keeps the hot
    path as a GEMM and avoids ``cdist``'s square root and large general-purpose
    workspace. The scale and center bias are applied in place to avoid extra
    score matrices.
    """
    center_norms = centers.square().sum(dim=-1)
    scores = features @ centers.transpose(0, 1)
    scores.mul_(2.0).sub_(center_norms)
    return scores.argmax(dim=-1)


def _cluster_means(
    features: torch.Tensor,
    assignments: torch.Tensor,
    counts: torch.Tensor,
    num_clusters: int,
) -> torch.Tensor:
    """Reduce assigned features, using the fastest measured backend per device."""
    float_features = features.float()
    merged = torch.zeros(
        num_clusters,
        float_features.shape[1],
        device=float_features.device,
        dtype=torch.float32,
    )
    if float_features.is_cuda:
        expanded_assignments = assignments.unsqueeze(1).expand_as(float_features)
        merged.scatter_reduce_(
            0,
            expanded_assignments,
            float_features,
            reduce="sum",
            include_self=True,
        )
    else:
        merged.index_add_(0, assignments, float_features)
    return (merged / counts).to(features.dtype)


def _single_pass_kmeans(
    features: torch.Tensor,
    num_output_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match the repository's existing random-center, single-assignment K-Means baseline."""
    num_input_tokens = features.shape[0]
    keep = min(num_input_tokens, num_output_tokens)
    if keep == num_input_tokens:
        assignments = torch.arange(num_input_tokens, device=features.device)
        counts = torch.ones(keep, 1, device=features.device, dtype=torch.float32)
        return features, assignments, counts

    float_features = features.float()
    centers = float_features[torch.randperm(num_input_tokens, device=features.device)[:keep]]
    assignments = _nearest_euclidean_centers(float_features, centers)
    counts = torch.bincount(assignments, minlength=keep).clamp(min=1).to(float_features.dtype).unsqueeze(1)
    return _cluster_means(features, assignments, counts, keep), assignments, counts


def compress_visual_pair(
    visual_inputs: torch.Tensor,
    visual_hidden: torch.Tensor,
    num_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compress aligned input embeddings and target hidden states to the same fixed length."""
    validate_visual_compression(num_tokens)
    if visual_inputs.ndim != 2 or visual_hidden.ndim != 2:
        raise ValueError("visual_inputs and visual_hidden must both have shape [N, H]")
    if visual_inputs.shape != visual_hidden.shape:
        raise ValueError(
            f"visual tensor shapes must match, got {tuple(visual_inputs.shape)} and {tuple(visual_hidden.shape)}"
        )
    if visual_inputs.shape[0] == 0:
        raise ValueError("visual tensors must contain at least one token")

    merged_inputs, assignments, counts = _single_pass_kmeans(visual_inputs, num_tokens)
    keep = merged_inputs.shape[0]
    return merged_inputs, _cluster_means(visual_hidden, assignments, counts, keep)


def compress_visual_sequence(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    hidden_states: torch.Tensor,
    image_token_id: int,
    num_tokens: int,
    loss_mask: Optional[torch.Tensor] = None,
    pad_to_original: bool = False,
    padding_token_id: int = 0,
):
    """Compress one multimodal sequence while preserving text order and tensor alignment.

    When ``pad_to_original`` is enabled, the compressed visual tokens stay at the
    beginning of the original visual span and the remaining slots are filled with
    padding IDs and zero features. This preserves all following text positions;
    callers must mask the inserted padding slots in their attention mask.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"only batch size 1 is supported, got input_ids shape {tuple(input_ids.shape)}")
    if inputs_embeds.shape[:2] != input_ids.shape or hidden_states.shape[:2] != input_ids.shape:
        raise ValueError("input_ids, inputs_embeds, and hidden_states must have aligned batch/sequence dimensions")
    if loss_mask is not None and loss_mask.shape != input_ids.shape:
        raise ValueError("loss_mask must match input_ids shape")

    visual_mask = input_ids == image_token_id
    visual_indices = torch.nonzero(visual_mask.squeeze(0), as_tuple=False).squeeze(1)
    if visual_indices.numel() == 0:
        return input_ids, inputs_embeds, hidden_states, loss_mask, 0

    text_indices = torch.nonzero(~visual_mask.squeeze(0), as_tuple=False).squeeze(1)
    visual_inputs = inputs_embeds[:, visual_indices, :].squeeze(0)
    visual_hidden = hidden_states[:, visual_indices, :].squeeze(0)
    compressed_inputs, compressed_hidden = compress_visual_pair(
        visual_inputs,
        visual_hidden,
        num_tokens=num_tokens,
    )
    kept_visual_tokens = compressed_inputs.shape[0]
    compressed_ids = torch.full(
        (1, kept_visual_tokens), image_token_id, dtype=input_ids.dtype, device=input_ids.device
    )
    if loss_mask is None:
        compressed_loss_mask = None
    else:
        compressed_loss_mask = loss_mask[:, visual_indices[:kept_visual_tokens]]

    sorted_visual_indices = torch.sort(visual_indices).values
    is_contiguous = bool(
        sorted_visual_indices[-1] - sorted_visual_indices[0] + 1 == sorted_visual_indices.numel()
    ) and (
        sorted_visual_indices.numel() == 1 or bool(torch.diff(sorted_visual_indices).min() == 1)
    )
    if is_contiguous:
        first_visual = sorted_visual_indices[0].item()
        last_visual = sorted_visual_indices[-1].item()
        pre_text = text_indices[text_indices < first_visual]
        post_text = text_indices[text_indices > last_visual]
        if pad_to_original:
            pad_count = visual_indices.numel() - kept_visual_tokens
            padding_ids = torch.full(
                (1, pad_count), padding_token_id, dtype=input_ids.dtype, device=input_ids.device
            )
            visual_ids = torch.cat((compressed_ids, padding_ids), dim=1)
            padded_inputs = torch.zeros(
                pad_count,
                compressed_inputs.shape[1],
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )
            padded_hidden = torch.zeros(
                pad_count,
                compressed_hidden.shape[1],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            visual_inputs = torch.cat(
                (compressed_inputs.to(inputs_embeds.dtype), padded_inputs), dim=0
            )
            visual_hidden = torch.cat(
                (compressed_hidden.to(hidden_states.dtype), padded_hidden), dim=0
            )
            if loss_mask is not None:
                padding_loss_mask = torch.zeros(
                    (1, pad_count), dtype=loss_mask.dtype, device=loss_mask.device
                )
                visual_loss_mask = torch.cat((compressed_loss_mask, padding_loss_mask), dim=1)
        else:
            visual_ids = compressed_ids
            visual_inputs = compressed_inputs.to(inputs_embeds.dtype)
            visual_hidden = compressed_hidden.to(hidden_states.dtype)
            visual_loss_mask = compressed_loss_mask

        input_ids = torch.cat((input_ids[:, pre_text], visual_ids, input_ids[:, post_text]), dim=1)
        inputs_embeds = torch.cat(
            (
                inputs_embeds[:, pre_text, :],
                visual_inputs.unsqueeze(0),
                inputs_embeds[:, post_text, :],
            ),
            dim=1,
        )
        hidden_states = torch.cat(
            (
                hidden_states[:, pre_text, :],
                visual_hidden.unsqueeze(0),
                hidden_states[:, post_text, :],
            ),
            dim=1,
        )
        if loss_mask is not None:
            loss_mask = torch.cat(
                (loss_mask[:, pre_text], visual_loss_mask, loss_mask[:, post_text]), dim=1
            )
    else:
        if pad_to_original:
            raise ValueError("pad_to_original requires one contiguous visual-token span")
        sorted_text = torch.sort(text_indices).values
        input_ids = torch.cat((input_ids[:, sorted_text], compressed_ids), dim=1)
        inputs_embeds = torch.cat(
            (inputs_embeds[:, sorted_text, :], compressed_inputs.unsqueeze(0).to(inputs_embeds.dtype)), dim=1
        )
        hidden_states = torch.cat(
            (hidden_states[:, sorted_text, :], compressed_hidden.unsqueeze(0).to(hidden_states.dtype)), dim=1
        )
        if loss_mask is not None:
            loss_mask = torch.cat((loss_mask[:, sorted_text], compressed_loss_mask), dim=1)

    reduced_len = visual_indices.numel() - kept_visual_tokens
    return input_ids, inputs_embeds, hidden_states, loss_mask, reduced_len
