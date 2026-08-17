"""Tests for weight_mapping helpers."""

import pytest
import torch

from dtm.config_utils import MoeSpec
from dtm.weight_mapping import (
    is_ffn_weight,
    layer_index_from_name,
    map_state_dict_to_moe,
    repeat_into_experts,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SPEC = MoeSpec(
    num_experts=4,
    num_experts_per_tok=2,
    moe_layer_freq=2,
    shared_expert_intermediate_size=None,
    expert_intermediate_size=None,
    router_top_k=2,
    hidden_size=8,
    intermediate_size=16,
    num_hidden_layers=4,
)

# With freq=2 and num_hidden_layers=4, MoE layers are at index 1 and 3.
MOE_LAYERS = {1, 3}


def _make_dense_state(spec: MoeSpec) -> dict:
    """Minimal fake dense state dict."""
    state = {}
    for i in range(spec.num_hidden_layers):
        # Attention (should pass through unchanged)
        state[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.ones(
            spec.hidden_size, spec.hidden_size
        )
        # FFN
        state[f"model.layers.{i}.mlp.gate_proj.weight"] = torch.ones(
            spec.intermediate_size, spec.hidden_size
        )
        state[f"model.layers.{i}.mlp.up_proj.weight"] = torch.ones(
            spec.intermediate_size, spec.hidden_size
        )
        state[f"model.layers.{i}.mlp.down_proj.weight"] = torch.ones(
            spec.hidden_size, spec.intermediate_size
        )
    return state


# ---------------------------------------------------------------------------
# is_ffn_weight
# ---------------------------------------------------------------------------


def test_is_ffn_weight_gate_proj():
    assert is_ffn_weight("model.layers.0.mlp.gate_proj.weight")


def test_is_ffn_weight_up_proj():
    assert is_ffn_weight("model.layers.0.mlp.up_proj.weight")


def test_is_ffn_weight_down_proj():
    assert is_ffn_weight("model.layers.0.mlp.down_proj.weight")


def test_is_ffn_weight_attn_not_ffn():
    assert not is_ffn_weight("model.layers.0.self_attn.q_proj.weight")


# ---------------------------------------------------------------------------
# layer_index_from_name
# ---------------------------------------------------------------------------


def test_layer_index_from_name_normal():
    assert layer_index_from_name("model.layers.7.mlp.gate_proj.weight") == 7


def test_layer_index_from_name_no_index():
    assert layer_index_from_name("model.embed_tokens.weight") is None


# ---------------------------------------------------------------------------
# repeat_into_experts: shape and strategy
# ---------------------------------------------------------------------------


def _w(rows=4, cols=4) -> torch.Tensor:
    return torch.randn(rows, cols)


def test_repeat_shape():
    w = _w()
    out = repeat_into_experts(w, num_experts=4, strategy="repeat")
    assert out.shape == (4, *w.shape)


def test_repeat_all_same():
    w = _w()
    out = repeat_into_experts(w, num_experts=3, strategy="repeat")
    assert torch.allclose(out[0], out[1]) and torch.allclose(out[1], out[2])


def test_noise_shape():
    w = _w()
    out = repeat_into_experts(w, num_experts=3, strategy="noise")
    assert out.shape == (3, *w.shape)


def test_noise_experts_differ():
    torch.manual_seed(0)
    w = torch.randn(4, 4)  # non-constant weight so std > 0
    out = repeat_into_experts(w, num_experts=4, strategy="noise", seed=42)
    # Not all expert tensors should be identical (noise added).
    assert not torch.allclose(out[0], out[1])


def test_slice_shape():
    w = _w(8, 4)
    out = repeat_into_experts(w, num_experts=4, strategy="slice")
    assert out.shape[0] == 4


def test_slice_uneven_rows():
    """Rows not divisible by num_experts must produce uniform expert shapes."""
    w = _w(5, 4)  # 5 rows, 3 experts → sizes 2, 2, 1 before padding
    out = repeat_into_experts(w, num_experts=3, strategy="slice")
    assert out.shape[0] == 3
    # All experts must have the same shape so torch.stack succeeded.
    assert out[0].shape == out[1].shape == out[2].shape


def test_invalid_strategy():
    with pytest.raises(ValueError, match="Unknown expert strategy"):
        repeat_into_experts(_w(), num_experts=2, strategy="invalid")


# ---------------------------------------------------------------------------
# map_state_dict_to_moe
# ---------------------------------------------------------------------------


def test_map_attn_weights_unchanged():
    state = _make_dense_state(SPEC)
    out = map_state_dict_to_moe(state, SPEC, expert_strategy="repeat")
    for i in range(SPEC.num_hidden_layers):
        k = f"model.layers.{i}.self_attn.q_proj.weight"
        assert torch.allclose(out[k], state[k])


def test_map_ffn_on_moe_layers_stacked():
    state = _make_dense_state(SPEC)
    out = map_state_dict_to_moe(state, SPEC, expert_strategy="repeat")
    for i in MOE_LAYERS:
        k = f"model.layers.{i}.mlp.gate_proj.weight"
        assert out[k].shape[0] == SPEC.num_experts, f"layer {i}: expected expert dim"


def test_map_ffn_on_dense_layers_unchanged():
    state = _make_dense_state(SPEC)
    out = map_state_dict_to_moe(state, SPEC, expert_strategy="repeat")
    dense_only = {0, 2}
    for i in dense_only:
        k = f"model.layers.{i}.mlp.gate_proj.weight"
        assert torch.allclose(out[k], state[k])


def test_map_router_weights_created():
    state = _make_dense_state(SPEC)
    out = map_state_dict_to_moe(state, SPEC, expert_strategy="repeat")
    for i in sorted(MOE_LAYERS):
        assert f"model.layers.{i}.mlp.gate.weight" in out
        assert f"model.layers.{i}.mlp.gate.bias" in out


def test_map_router_zeros_default():
    state = _make_dense_state(SPEC)
    out = map_state_dict_to_moe(state, SPEC, router_init="zeros")
    for i in sorted(MOE_LAYERS):
        w = out[f"model.layers.{i}.mlp.gate.weight"]
        assert torch.all(w == 0)


def test_map_invalid_router_init():
    state = _make_dense_state(SPEC)
    with pytest.raises(ValueError, match="router_init"):
        map_state_dict_to_moe(state, SPEC, router_init="bad")
