"""
Helpers for model tags and tag-scoped artifact directories.
"""

import os

from nanochat.common import get_base_dir


CHECKPOINT_DIRS = {
    "base": "base_checkpoints",
    "sft": "chatsft_checkpoints",
    "rl": "chatrl_checkpoints",
}


def parse_laplacian_heads_spec(value):
    """Parse CLI input like '2' or '0,2,2,0' into int or list[int]."""
    if isinstance(value, int):
        return value
    if isinstance(value, list):
        return value
    value = str(value).strip()
    if "," not in value:
        return int(value)
    parts = [part.strip() for part in value.split(",")]
    if any(part == "" for part in parts):
        raise ValueError(f"Invalid laplacian head spec: {value!r}")
    return [int(part) for part in parts]


def _format_laplacian_heads(laplacian_heads):
    counts = list(laplacian_heads)
    if not counts:
        return "0"
    if len(set(counts)) == 1:
        return str(counts[0])
    return "x".join(str(count) for count in counts)


def make_model_tag(config):
    lap_tag = _format_laplacian_heads(config.laplacian_heads)
    window_tag = config.window_pattern.lower()
    return (
        f"d{config.n_layer}"
        f"-e{config.n_embd}"
        f"-h{config.n_head}"
        f"-kv{config.n_kv_head}"
        f"-w{window_tag}"
        f"-lap{lap_tag}"
    )


def resolve_model_tag(config, explicit_tag=None):
    """Manual tag wins; otherwise derive one from model architecture."""
    if explicit_tag:
        return explicit_tag
    return make_model_tag(config)


def get_checkpoints_dir(source, base_dir=None):
    base_dir = get_base_dir() if base_dir is None else base_dir
    return os.path.join(base_dir, CHECKPOINT_DIRS[source])


def get_checkpoint_dir(source, model_tag, base_dir=None):
    return os.path.join(get_checkpoints_dir(source, base_dir=base_dir), model_tag)


def get_report_dir(model_tag=None, base_dir=None):
    base_dir = get_base_dir() if base_dir is None else base_dir
    if model_tag is None:
        return os.path.join(base_dir, "report")
    return os.path.join(base_dir, "reports", model_tag)


def get_base_eval_dir(model_tag=None, base_dir=None):
    base_dir = get_base_dir() if base_dir is None else base_dir
    if model_tag is None:
        return os.path.join(base_dir, "base_eval")
    return os.path.join(base_dir, "base_eval", model_tag)
