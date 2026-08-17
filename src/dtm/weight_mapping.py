"""
Weight-mapping helpers: dense FFN → MoE expert tensors.

Supported strategies
--------------------
repeat : copy the dense FFN weight into every expert unchanged.
noise  : copy + add a small Gaussian perturbation per expert.
slice  : row-wise slice the dense weight across experts (dimension 0).
"""

from __future__ import annotations

import math
from typing import Dict, List, Set

try:
    import torch
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "torch is required for weight mapping. "
        "Install it with: pip install 'dtm[inference]'"
    ) from _exc

from .config_utils import MoeSpec, select_moe_layers


# ---------------------------------------------------------------------------
# Layer-name helpers
# ---------------------------------------------------------------------------

# Substrings that identify FFN weights in a Qwen-style state dict.
_FFN_NEEDLES = (
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
    ".feed_forward.",
    ".ffn.",
)


def is_ffn_weight(name: str) -> bool:
    """Return *True* if *name* refers to a dense FFN weight tensor."""
    return any(needle in name for needle in _FFN_NEEDLES)


def layer_index_from_name(name: str) -> int | None:
    """Extract the integer layer index embedded in a parameter name."""
    for part in name.split("."):
        if part.isdigit():
            return int(part)
    return None


# ---------------------------------------------------------------------------
# Expert weight initialisation strategies
# ---------------------------------------------------------------------------


def _clone(t: torch.Tensor) -> torch.Tensor:
    return t.detach().clone()


def repeat_into_experts(
    dense_weight: torch.Tensor,
    num_experts: int,
    strategy: str = "repeat",
    noise_scale: float = 0.01,
    seed: int = 0,
) -> torch.Tensor:
    """Stack *dense_weight* into an expert tensor of shape ``[num_experts, ...]``.

    Parameters
    ----------
    dense_weight:
        The source dense FFN weight.
    num_experts:
        Number of MoE experts to create.
    strategy:
        ``"repeat"``, ``"noise"``, or ``"slice"``.
    noise_scale:
        Gaussian noise standard deviation as a fraction of the weight std
        (only used for ``"noise"``).
    seed:
        RNG seed for reproducible noise (only used for ``"noise"``).
    """
    if strategy == "repeat":
        return torch.stack([_clone(dense_weight) for _ in range(num_experts)], dim=0)

    if strategy == "noise":
        gen = torch.Generator(device=dense_weight.device)
        gen.manual_seed(seed)
        scale = dense_weight.float().std().clamp_min(1e-6) * noise_scale
        experts = []
        for _ in range(num_experts):
            noise = torch.randn(dense_weight.shape, generator=gen, device=dense_weight.device)
            experts.append(
                _clone(dense_weight) + (noise * scale).to(dense_weight.dtype)
            )
        return torch.stack(experts, dim=0)

    if strategy == "slice":
        if dense_weight.dim() != 2:
            # Fall back to repeat for non-matrix tensors.
            return torch.stack([_clone(dense_weight) for _ in range(num_experts)], dim=0)
        rows = dense_weight.shape[0]
        chunk = math.ceil(rows / num_experts)
        experts = []
        for i in range(num_experts):
            start = i * chunk
            end = min(rows, (i + 1) * chunk)
            part = dense_weight[start:end]
            if part.shape[0] == 0:
                part = dense_weight[:1]
            # Pad to uniform chunk size so torch.stack succeeds.
            if part.shape[0] < chunk:
                pad_rows = chunk - part.shape[0]
                part = torch.cat([part, part[:pad_rows]], dim=0)
            experts.append(part.clone())
        return torch.stack(experts, dim=0)

    raise ValueError(
        f"Unknown expert strategy '{strategy}'. "
        "Choose one of: repeat, noise, slice."
    )


# ---------------------------------------------------------------------------
# Full state-dict mapping
# ---------------------------------------------------------------------------


def map_state_dict_to_moe(
    dense_state: Dict[str, torch.Tensor],
    spec: MoeSpec,
    expert_strategy: str = "repeat",
    router_init: str = "zeros",
) -> Dict[str, torch.Tensor]:
    """Convert a dense state dict to an MoE-shaped state dict.

    Parameters
    ----------
    dense_state:
        State dict from the dense model.
    spec:
        Inferred MoE layout parameters.
    expert_strategy:
        How to populate expert tensors (``"repeat"``, ``"noise"``, ``"slice"``).
    router_init:
        ``"zeros"`` or ``"random"`` initialisation for router gate weights.
    """
    if router_init not in ("zeros", "random"):
        raise ValueError(
            f"Unknown router_init '{router_init}'. Choose 'zeros' or 'random'."
        )

    out: Dict[str, torch.Tensor] = {}
    moe_layer_ids: Set[int] = set(
        select_moe_layers(spec.num_hidden_layers, spec.moe_layer_freq)
    )

    for name, tensor in dense_state.items():
        if not is_ffn_weight(name):
            out[name] = _clone(tensor)
            continue

        layer_idx = layer_index_from_name(name)
        if layer_idx is None or layer_idx not in moe_layer_ids:
            out[name] = _clone(tensor)
            continue

        if name.endswith(".weight"):
            out[name] = repeat_into_experts(
                dense_weight=tensor,
                num_experts=spec.num_experts,
                strategy=expert_strategy,
            )
        else:
            out[name] = _clone(tensor)

    # Ensure router weights exist for every MoE layer.
    for layer_idx in sorted(moe_layer_ids):
        router_w = f"model.layers.{layer_idx}.mlp.gate.weight"
        router_b = f"model.layers.{layer_idx}.mlp.gate.bias"

        if router_w not in out:
            out[router_w] = torch.zeros(
                (spec.num_experts, spec.hidden_size), dtype=torch.float32
            )
        if router_b not in out:
            out[router_b] = torch.zeros(spec.num_experts, dtype=torch.float32)

        if router_init == "random":
            out[router_w] = torch.randn_like(out[router_w]) * 0.01
            out[router_b] = torch.zeros_like(out[router_b])

    return out
