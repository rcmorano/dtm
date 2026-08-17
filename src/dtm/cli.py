"""
CLI entry point: ``dtm-convert`` / ``python -m dtm``.

Usage
-----
dtm-convert \\
    --dense-model  path/to/qwen-27b-dense \\
    --moe-model    path/to/qwen-moe-template \\
    --output-dir   path/to/output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "torch and transformers are required for conversion. "
        "Install them with: pip install 'dtm[inference]'"
    ) from _exc

from .config_utils import build_output_config, infer_moe_spec, save_json
from .weight_mapping import map_state_dict_to_moe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dtm-convert",
        description=(
            "Convert a dense Qwen checkpoint into an MoE-style initialisation "
            "using an existing Qwen MoE checkpoint as the architectural template."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dense-model",
        required=True,
        metavar="PATH_OR_REPO",
        help="Local path or Hugging Face repo ID of the dense Qwen model.",
    )
    parser.add_argument(
        "--moe-model",
        required=True,
        metavar="PATH_OR_REPO",
        help="Local path or Hugging Face repo ID of the MoE Qwen template model.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        metavar="DIR",
        help="Directory to write the converted checkpoint.",
    )
    parser.add_argument(
        "--expert-strategy",
        choices=["repeat", "noise", "slice"],
        default="repeat",
        help="Strategy for initialising expert weights from the dense FFN weights.",
    )
    parser.add_argument(
        "--router-init",
        choices=["zeros", "random"],
        default="zeros",
        help="Initialisation strategy for router gate weights.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Pass trust_remote_code=True when loading configs / tokenizers.",
    )
    return parser


def convert(
    dense_model: str,
    moe_model: str,
    output_dir: str | Path,
    expert_strategy: str = "repeat",
    router_init: str = "zeros",
    trust_remote_code: bool = False,
) -> None:
    """Programmatic entry point (also used by the CLI)."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load configs
    # ------------------------------------------------------------------ #
    print(f"[dtm] Loading dense config from {dense_model!r} …")
    dense_cfg = AutoConfig.from_pretrained(
        dense_model, trust_remote_code=trust_remote_code
    ).to_dict()

    print(f"[dtm] Loading MoE template config from {moe_model!r} …")
    moe_cfg = AutoConfig.from_pretrained(
        moe_model, trust_remote_code=trust_remote_code
    ).to_dict()

    # ------------------------------------------------------------------ #
    # Infer MoE layout
    # ------------------------------------------------------------------ #
    spec = infer_moe_spec(moe_cfg, dense_cfg)
    print(f"[dtm] Inferred MoE spec:")
    print(f"      num_experts           = {spec.num_experts}")
    print(f"      num_experts_per_tok   = {spec.num_experts_per_tok}")
    print(f"      moe_layer_freq        = {spec.moe_layer_freq}")
    print(f"      expert_intermediate   = {spec.expert_intermediate_size}")
    print(f"      shared_intermediate   = {spec.shared_expert_intermediate_size}")
    print(f"      hidden_size           = {spec.hidden_size}")
    print(f"      num_hidden_layers     = {spec.num_hidden_layers}")

    # ------------------------------------------------------------------ #
    # Save tokenizer (from dense model as semantic source)
    # ------------------------------------------------------------------ #
    print(f"[dtm] Saving tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(
        dense_model, trust_remote_code=trust_remote_code
    )
    tokenizer.save_pretrained(out_dir)

    # ------------------------------------------------------------------ #
    # Save merged config
    # ------------------------------------------------------------------ #
    out_cfg = build_output_config(dense_cfg, moe_cfg)
    save_json(out_dir / "config.json", out_cfg)
    print(f"[dtm] Wrote config.json")

    # ------------------------------------------------------------------ #
    # Load dense weights
    # ------------------------------------------------------------------ #
    print(f"[dtm] Loading dense model weights (this may take a while) …")
    dense_model_obj = AutoModelForCausalLM.from_pretrained(
        dense_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
    )
    dense_state = dense_model_obj.state_dict()

    # ------------------------------------------------------------------ #
    # Map weights
    # ------------------------------------------------------------------ #
    print(
        f"[dtm] Mapping dense weights → MoE experts "
        f"(strategy={expert_strategy!r}, router_init={router_init!r}) …"
    )
    converted_state = map_state_dict_to_moe(
        dense_state=dense_state,
        spec=spec,
        expert_strategy=expert_strategy,
        router_init=router_init,
    )

    # ------------------------------------------------------------------ #
    # Save checkpoint
    # ------------------------------------------------------------------ #
    out_weights = out_dir / "pytorch_model.bin"
    print(f"[dtm] Saving checkpoint to {out_weights} …")
    torch.save(converted_state, out_weights)

    print(f"\n[dtm] ✓ Conversion complete → {out_dir}")
    print(
        "[dtm] NOTE: Fine-tune the resulting model before production use."
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        convert(
            dense_model=args.dense_model,
            moe_model=args.moe_model,
            output_dir=args.output_dir,
            expert_strategy=args.expert_strategy,
            router_init=args.router_init,
            trust_remote_code=args.trust_remote_code,
        )
    except FileNotFoundError as exc:
        print(f"[dtm] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"[dtm] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
