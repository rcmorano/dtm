"""
Helpers for reading and inferring Qwen MoE config fields.

Qwen config field names vary slightly across releases; the functions here
handle those variations gracefully.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


# ---------------------------------------------------------------------------
# MoE specification dataclass
# ---------------------------------------------------------------------------


@dataclass
class MoeSpec:
    """Inferred MoE layout parameters."""

    num_experts: int
    num_experts_per_tok: int
    moe_layer_freq: int
    shared_expert_intermediate_size: Optional[int]
    expert_intermediate_size: Optional[int]
    router_top_k: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

# Ordered lists of alternative config field names (most-to-least-common).
_NUM_EXPERTS_KEYS = ("num_experts", "moe_num_experts", "num_local_experts")
_TOP_K_KEYS = ("num_experts_per_tok", "moe_top_k", "num_experts_per_token")
_FREQ_KEYS = ("moe_layer_freq", "decoder_sparse_step")


def _first_int(d: Dict[str, Any], *keys: str, default: int) -> int:
    for k in keys:
        v = d.get(k)
        if v is not None:
            return int(v)
    return default


def infer_moe_spec(
    moe_config: Dict[str, Any],
    dense_config: Dict[str, Any],
) -> MoeSpec:
    """Build a :class:`MoeSpec` from the MoE template config.

    Fields that are absent from the MoE config fall back to the dense config,
    then to a safe default.
    """
    num_experts = _first_int(moe_config, *_NUM_EXPERTS_KEYS, default=8)
    num_experts_per_tok = _first_int(moe_config, *_TOP_K_KEYS, default=2)
    moe_layer_freq = _first_int(moe_config, *_FREQ_KEYS, default=1)

    hidden_size = _first_int(
        moe_config, "hidden_size", default=int(dense_config["hidden_size"])
    )
    intermediate_size = _first_int(
        moe_config,
        "intermediate_size",
        default=int(dense_config["intermediate_size"]),
    )
    num_hidden_layers = _first_int(
        moe_config,
        "num_hidden_layers",
        default=int(dense_config["num_hidden_layers"]),
    )

    shared_expert_intermediate_size = moe_config.get("shared_expert_intermediate_size")
    expert_intermediate_size = moe_config.get("expert_intermediate_size")

    return MoeSpec(
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_layer_freq=moe_layer_freq,
        shared_expert_intermediate_size=int(shared_expert_intermediate_size)
        if shared_expert_intermediate_size is not None
        else None,
        expert_intermediate_size=int(expert_intermediate_size)
        if expert_intermediate_size is not None
        else None,
        router_top_k=num_experts_per_tok,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
    )


def select_moe_layers(num_layers: int, freq: int) -> List[int]:
    """Return layer indices that become MoE layers given *freq*.

    A layer at index *i* is an MoE layer when ``(i + 1) % freq == 0``.
    When *freq* is 1 every layer is an MoE layer.
    """
    if freq <= 0:
        raise ValueError(f"moe_layer_freq must be >= 1, got {freq}")
    return [i for i in range(num_layers) if (i + 1) % freq == 0]


def build_output_config(
    dense_config: Dict[str, Any],
    moe_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge dense and MoE configs into the output config dict.

    Architecture / model-family metadata is taken from the MoE template;
    the dense model's layer parameters are used as the base.
    """
    cfg = dict(dense_config)

    # Top-level metadata: prefer the MoE template.
    for key in (
        "architectures",
        "model_type",
        "rope_theta",
        "hidden_act",
        "initializer_range",
        "max_position_embeddings",
        "tie_word_embeddings",
        "vocab_size",
    ):
        if key in moe_config:
            cfg[key] = moe_config[key]

    # MoE-specific fields: copy all that exist in the template.
    moe_specific_keys = (
        "num_experts",
        "moe_num_experts",
        "num_local_experts",
        "num_experts_per_tok",
        "moe_top_k",
        "num_experts_per_token",
        "moe_layer_freq",
        "decoder_sparse_step",
        "shared_expert_intermediate_size",
        "expert_intermediate_size",
    )
    for key in moe_specific_keys:
        if key in moe_config:
            cfg[key] = moe_config[key]

    return cfg
