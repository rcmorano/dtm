"""Tests for config_utils: MoeSpec inference and related helpers."""

import pytest

from dtm.config_utils import (
    MoeSpec,
    build_output_config,
    infer_moe_spec,
    select_moe_layers,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DENSE_CFG = {
    "model_type": "qwen2",
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_hidden_layers": 32,
    "vocab_size": 152064,
}

MOE_CFG_FULL = {
    "model_type": "qwen2_moe",
    "architectures": ["Qwen2MoeForCausalLM"],
    "hidden_size": 2048,
    "intermediate_size": 2048,
    "num_hidden_layers": 24,
    "num_experts": 64,
    "num_experts_per_tok": 4,
    "moe_layer_freq": 2,
    "expert_intermediate_size": 1408,
    "shared_expert_intermediate_size": 2816,
    "vocab_size": 152064,
}


# ---------------------------------------------------------------------------
# infer_moe_spec
# ---------------------------------------------------------------------------


def test_infer_moe_spec_full_moe_config():
    spec = infer_moe_spec(MOE_CFG_FULL, DENSE_CFG)
    assert isinstance(spec, MoeSpec)
    assert spec.num_experts == 64
    assert spec.num_experts_per_tok == 4
    assert spec.moe_layer_freq == 2
    assert spec.expert_intermediate_size == 1408
    assert spec.shared_expert_intermediate_size == 2816
    assert spec.hidden_size == 2048
    assert spec.num_hidden_layers == 24


def test_infer_moe_spec_fallback_to_dense():
    """Fields missing from moe_config fall back to dense_config values."""
    minimal_moe = {
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_layer_freq": 1,
    }
    spec = infer_moe_spec(minimal_moe, DENSE_CFG)
    assert spec.hidden_size == DENSE_CFG["hidden_size"]
    assert spec.intermediate_size == DENSE_CFG["intermediate_size"]
    assert spec.num_hidden_layers == DENSE_CFG["num_hidden_layers"]


def test_infer_moe_spec_alternative_field_names():
    """Older Qwen configs use moe_num_experts / moe_top_k / decoder_sparse_step."""
    alt_moe = {
        "moe_num_experts": 16,
        "moe_top_k": 3,
        "decoder_sparse_step": 4,
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_hidden_layers": 8,
    }
    spec = infer_moe_spec(alt_moe, DENSE_CFG)
    assert spec.num_experts == 16
    assert spec.num_experts_per_tok == 3
    assert spec.moe_layer_freq == 4


def test_infer_moe_spec_defaults_when_nothing_provided():
    """When neither config has the field, safe defaults are returned."""
    spec = infer_moe_spec({}, {"hidden_size": 512, "intermediate_size": 2048, "num_hidden_layers": 4})
    assert spec.num_experts == 8
    assert spec.num_experts_per_tok == 2
    assert spec.moe_layer_freq == 1


# ---------------------------------------------------------------------------
# select_moe_layers
# ---------------------------------------------------------------------------


def test_select_moe_layers_freq1():
    layers = select_moe_layers(4, freq=1)
    assert layers == [0, 1, 2, 3]


def test_select_moe_layers_freq2():
    layers = select_moe_layers(6, freq=2)
    assert layers == [1, 3, 5]


def test_select_moe_layers_freq_larger_than_num_layers():
    layers = select_moe_layers(3, freq=10)
    assert layers == []


def test_select_moe_layers_invalid_freq():
    with pytest.raises(ValueError, match="moe_layer_freq"):
        select_moe_layers(4, freq=0)


# ---------------------------------------------------------------------------
# build_output_config
# ---------------------------------------------------------------------------


def test_build_output_config_moe_type_wins():
    cfg = build_output_config(DENSE_CFG, MOE_CFG_FULL)
    assert cfg["model_type"] == "qwen2_moe"
    assert cfg["architectures"] == ["Qwen2MoeForCausalLM"]


def test_build_output_config_moe_fields_present():
    cfg = build_output_config(DENSE_CFG, MOE_CFG_FULL)
    assert cfg["num_experts"] == 64
    assert cfg["expert_intermediate_size"] == 1408
    assert cfg["moe_layer_freq"] == 2


def test_build_output_config_dense_base_preserved():
    """Dense-only fields (e.g. num_hidden_layers) are kept from the dense config."""
    cfg = build_output_config(DENSE_CFG, {"model_type": "qwen2_moe", "num_experts": 8})
    assert cfg["num_hidden_layers"] == DENSE_CFG["num_hidden_layers"]
