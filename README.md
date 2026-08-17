# dtm – Dense-to-MoE

Convert a dense Qwen checkpoint into a Qwen MoE-style checkpoint using an
existing Qwen MoE model as the architectural template.

> **Note:** This is an *initialisation / upcycling* tool.  The resulting model
> should be fine-tuned before production use.

---

## Installation

```bash
# Core package only (no heavy ML deps – suitable for CI / config tooling):
pip install -e .

# Full runtime (needed to actually run conversions):
pip install -e ".[inference]"

# Development (tests + torch):
pip install -e ".[dev]"
```

## Usage

```bash
dtm-convert \
    --dense-model  path/to/qwen-27b-dense \
    --moe-model    path/to/qwen-moe-template \
    --output-dir   path/to/output \
    --expert-strategy repeat \
    --router-init zeros
```

Or via the module:

```bash
python -m dtm \
    --dense-model  path/to/qwen-27b-dense \
    --moe-model    path/to/qwen-moe-template \
    --output-dir   path/to/output
```

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--dense-model` | *required* | Local path or HF repo of the dense Qwen model |
| `--moe-model` | *required* | Local path or HF repo of the MoE template model |
| `--output-dir` | *required* | Directory to write the converted checkpoint |
| `--expert-strategy` | `repeat` | `repeat` / `noise` / `slice` |
| `--router-init` | `zeros` | `zeros` / `random` |
| `--trust-remote-code` | off | Pass `trust_remote_code=True` to Transformers loaders |

### Expert strategies

| Strategy | Description |
|----------|-------------|
| `repeat` | Copy dense FFN weight into every expert unchanged |
| `noise`  | Copy + add a small Gaussian perturbation per expert |
| `slice`  | Row-wise slice of dense weight distributed across experts |

## Package layout

```
src/
  dtm/
    __init__.py           package version
    __main__.py           python -m dtm entry point
    cli.py                argument parsing and top-level orchestration
    config_utils.py       MoE spec inference and config merging
    weight_mapping.py     dense → expert tensor mapping strategies
tests/
  test_config_utils.py
  test_weight_mapping.py
```

## Running tests

```bash
python -m pytest tests/ -v
```
