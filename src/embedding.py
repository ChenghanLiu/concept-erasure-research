"""Extract notebook activations and read the existing cached SVD subspaces."""

from __future__ import annotations

from numbers import Integral
from pathlib import Path
from typing import Any

import torch

from .projection import get_target_projection_layer


@torch.no_grad()
def get_embedding(
    pipe: Any,
    template_path: str | Path,
    concept: str,
    num_prompt: int = 20,
    layer_id: int = 0,
) -> torch.Tensor:
    """Collect inputs to the CLIP self-attention output projection.

    As in the notebook, each of the first 20 nonempty templates is tokenized
    separately, without padding, and its attention mask is passed to CLIP.
    Activations include all actual tokens and are neither centered nor pooled.
    """
    if isinstance(num_prompt, bool) or not isinstance(num_prompt, Integral):
        raise TypeError("num_prompt must be an integer")
    if num_prompt < 1:
        raise ValueError("num_prompt must be positive")
    with Path(template_path).open("r", encoding="utf-8") as handle:
        templates = [line.strip() for line in handle if line.strip()][:num_prompt]
    if not templates:
        raise ValueError(f"No nonempty templates found in {template_path}")

    pipe.set_progress_bar_config(disable=True)
    activations: list[torch.Tensor] = []

    def collect_input(_layer: Any, inputs: tuple, _output: Any) -> None:
        activations.append(inputs[0].cpu().detach())

    target_module = get_target_projection_layer(pipe, layer_id)
    hook_handle = target_module.register_forward_hook(collect_input)
    try:
        for template in templates:
            prompt = template.format(concept)
            text_inputs = pipe.tokenizer(
                prompt,
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            pipe.text_encoder(
                text_inputs.input_ids.to(pipe.device),
                attention_mask=text_inputs.attention_mask.to(pipe.device),
            )
    finally:
        hook_handle.remove()
    if not activations:
        raise RuntimeError("The selected attention layer did not produce activations")
    return torch.cat(activations, dim=1).squeeze(0)


def analyze_spectrum(embedding: torch.Tensor, concept: str) -> dict[str, Any]:
    """Compute the notebook's uncentered SVD and energy threshold ranks."""
    U, S, _V = torch.svd(embedding.reshape(-1, 768).cpu().to(torch.float32).T)
    variance = S ** 2
    ratio = variance / variance.sum()
    cumulative = torch.cumsum(ratio, dim=0)
    threshold_ranks = {}
    for threshold in [0.70, 0.80, 0.90, 0.95]:
        threshold_ranks[threshold] = int(
            torch.searchsorted(cumulative, torch.tensor(threshold)).item()
        ) + 1
    return {
        "concept": concept,
        "U": U,
        "S": S,
        "ratio": ratio,
        "cumulative": cumulative,
        "threshold_ranks": threshold_ranks,
    }


def cache_filename(concept: str) -> str:
    """Use the exact lowercase-and-underscore cache naming from cell 55."""
    return concept.lower().replace(" ", "_") + ".pt"


def load_cached_subspace(
    path: str | Path, concept: str | None = None
) -> dict[str, Any]:
    """Read and validate a notebook cache without modifying or recomputing it.

    Tensor-only loading restricts pickle deserialization. All cached tensor
    values and dtypes are returned unchanged on the CPU.
    """
    cache = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(cache, dict):
        raise ValueError(f"Subspace cache must contain a dictionary: {path}")
    required = {"concept", "U", "S", "ratio", "cumulative", "threshold_ranks"}
    missing = required.difference(cache)
    if missing:
        raise ValueError(f"Subspace cache is missing fields: {sorted(missing)}")
    if not isinstance(cache["concept"], str) or not cache["concept"]:
        raise ValueError("Cached concept must be a nonempty string")
    if concept is not None and cache["concept"] != concept:
        raise ValueError(
            f"Requested concept {concept!r} does not match cache {cache['concept']!r}"
        )

    U = cache["U"]
    if (
        not isinstance(U, torch.Tensor)
        or U.ndim != 2
        or U.shape[0] != 768
        or U.shape[1] < 8
    ):
        raise ValueError("Cached U must have shape (768, at least 8 modes)")
    for name in ("U", "S", "ratio", "cumulative"):
        tensor = cache[name]
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            raise ValueError(f"Cached {name} must be a floating-point tensor")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"Cached {name} contains nonfinite values")
        if name != "U" and (tensor.ndim != 1 or tensor.shape[0] != U.shape[1]):
            raise ValueError(f"Cached {name} must contain one value per column of U")
    if torch.any(cache["S"] < 0).item():
        raise ValueError("Cached singular values must be nonnegative")
    ranks = cache["threshold_ranks"]
    if not isinstance(ranks, dict) or set(ranks) != {0.70, 0.80, 0.90, 0.95}:
        raise ValueError("Cached threshold_ranks must contain 0.70, 0.80, 0.90 and 0.95")
    if any(
        isinstance(rank, bool)
        or not isinstance(rank, Integral)
        or not 1 <= rank <= U.shape[1]
        for rank in ranks.values()
    ):
        raise ValueError("Cached threshold ranks must lie within the available modes")
    return cache
