"""The text-encoder projection used by the validated rank experiments."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import torch


def get_target_projection_layer(pipe: Any, layer_id: int = 0) -> Any:
    """Return the selected CLIP self-attention output projection.

    The notebooks expose ``text_encoder.encoder`` directly. Standard
    ``CLIPTextModel`` objects expose the same encoder via ``text_model``.
    """
    if isinstance(layer_id, bool) or not isinstance(layer_id, Integral):
        raise TypeError("layer_id must be an integer")
    text_encoder = pipe.text_encoder
    encoder = getattr(text_encoder, "encoder", None)
    if encoder is None:
        text_model = getattr(text_encoder, "text_model", None)
        encoder = getattr(text_model, "encoder", None)
    if encoder is None:
        raise AttributeError("The text encoder does not expose CLIP encoder layers")
    if not 0 <= layer_id < len(encoder.layers):
        raise ValueError(f"layer_id must be between 0 and {len(encoder.layers) - 1}")
    return encoder.layers[layer_id].self_attn.out_proj


def build_projection(
    U: torch.Tensor, rank: int, lambda_value: float = 4
) -> torch.Tensor:
    """Build ``I - lambda_value * U[:, :rank] @ U[:, :rank].T``.

    This deliberately retains the notebook's strength of four; it is not an
    orthogonal projector at that strength. Construction uses CPU float32, and
    conversion to the model's dtype happens only when applying the matrix.
    """
    if not isinstance(U, torch.Tensor) or U.ndim != 2 or U.shape[0] != 768:
        raise ValueError("U must be a tensor of shape (768, number_of_modes)")
    if not U.is_floating_point() or not torch.isfinite(U).all().item():
        raise ValueError("U must contain finite floating-point values")
    if isinstance(rank, bool) or not isinstance(rank, Integral):
        raise TypeError("rank must be an integer")
    if not 1 <= rank <= U.shape[1]:
        raise ValueError(f"rank must be between 1 and {U.shape[1]}")
    basis = U.detach().cpu().to(torch.float32)[:, :rank]
    subspace = torch.matmul(basis, basis.T)
    return torch.eye(subspace.shape[0], dtype=torch.float32) - lambda_value * subspace


@torch.no_grad()
def project_text_conder_attn(
    pipe: Any, proj_matrix: torch.Tensor, layer_id: int = 0
) -> None:
    """Right-multiply the attention output weight, preserving its bias.

    The historical function-name spelling is retained for notebook continuity.
    Applying this function repeatedly compounds the projection; load or restore
    the original model before evaluating another candidate rank.
    """
    target_layer = get_target_projection_layer(pipe, layer_id)
    org_weight = target_layer.weight.data.clone()
    if proj_matrix.ndim != 2 or proj_matrix.shape != (
        org_weight.shape[1], org_weight.shape[1]
    ):
        raise ValueError("Projection shape must match the attention input dimension")
    target_layer.weight.data = torch.matmul(
        org_weight, proj_matrix.to(org_weight.device).to(org_weight.dtype)
    )
