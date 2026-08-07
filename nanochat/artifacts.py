"""
Model tags and tag-scoped artifact directories.

A model tag names the directory a run's checkpoints (and eval outputs) live under. Tags are
derived from the model architecture plus the random seed, so that a sweep over depths,
Laplacian head counts, ablations, and seeds can never have two cells overwrite each other.

Kept separate from common.py so that checkpoint_manager can import these without pulling in
anything that imports gpt.py.
"""

import os

from nanochat.common import get_base_dir

CHECKPOINT_DIRS = {
    "base": "base_checkpoints",
    "sft": "chatsft_checkpoints",
    "rl": "chatrl_checkpoints",
}


def parse_laplacian_heads_spec(value):
    """argparse type for --laplacian-heads: '2' -> 2, '0,2,2,0' -> [0, 2, 2, 0]."""
    if isinstance(value, (int, list)):
        return value
    value = str(value).strip()
    if "," not in value:
        return int(value)
    parts = [part.strip() for part in value.split(",")]
    if any(part == "" for part in parts):
        raise ValueError(f"Invalid laplacian head spec: {value!r}")
    return [int(part) for part in parts]


def _format_laplacian_heads(laplacian_heads):
    """(2, 2, 2) -> '2' (uniform), (0, 2, 2) -> '0x2x2' (per-layer)."""
    counts = list(laplacian_heads)
    if not counts:
        return "0"
    if len(set(counts)) == 1:
        return str(counts[0])
    return "x".join(str(count) for count in counts)


def make_model_tag(config, seed):
    """Derive a model tag from the architecture and seed, e.g. d20-e1280-h10-kv10-wsssl-lap0-s42.

    The seed suffix is always present so that multi-seed runs of the same architecture land in
    separate directories.
    """
    return (
        f"d{config.n_layer}"
        f"-e{config.n_embd}"
        f"-h{config.n_head}"
        f"-kv{config.n_kv_head}"
        f"-w{config.window_pattern.lower()}"
        f"-lap{_format_laplacian_heads(config.laplacian_heads)}"
        f"{'' if config.use_ve else '-nove'}"
        f"{'' if config.use_resid_lambdas else '-stdresid'}"
        f"{'' if config.use_x0 else '-nox0'}"
        f"{'' if config.use_smear else '-nosmear'}"
        f"{'' if config.use_backout else '-nobackout'}"
        f"-s{seed}"
    )


def resolve_model_tag(config, seed, explicit_tag=None):
    """An explicit --model-tag always wins; otherwise derive one from the architecture."""
    if explicit_tag:
        return explicit_tag
    return make_model_tag(config, seed)


def get_checkpoints_dir(source, base_dir=None):
    base_dir = get_base_dir() if base_dir is None else base_dir
    return os.path.join(base_dir, CHECKPOINT_DIRS[source])


def get_checkpoint_dir(source, model_tag, base_dir=None):
    return os.path.join(get_checkpoints_dir(source, base_dir=base_dir), model_tag)


def get_base_eval_dir(model_tag=None, base_dir=None):
    """Where base_eval writes its CORE csv. Tag-scoped so two models that stop at the same step
    do not overwrite each other (they used to share base_eval/base_model_<step>.csv)."""
    base_dir = get_base_dir() if base_dir is None else base_dir
    if model_tag is None:
        return os.path.join(base_dir, "base_eval")
    return os.path.join(base_dir, "base_eval", model_tag)
