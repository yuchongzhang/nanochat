"""
Utilities for saving and loading model/optim/state checkpoints.
"""
import os
import re
import json
import logging
import torch

from nanochat.artifacts import get_checkpoints_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import setup_default_logging

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)

# Config keys that did not exist in older checkpoints, with the value that reproduces the
# behavior those checkpoints were trained with.
_CONFIG_KEY_DEFAULTS = {
    "window_pattern": "L",      # old models were trained with full context (no sliding window)
    "laplacian_heads": 0,       # old models had no Laplacian heads
    "use_ve": True,             # every ablation switch was implicitly on
    "use_resid_lambdas": True,
    "use_x0": True,
    "use_smear": True,
    "use_backout": True,
}

def _patch_missing_config_keys(model_config_kwargs):
    """Add default values for new config keys missing in old checkpoints."""
    for key, default in _CONFIG_KEY_DEFAULTS.items():
        if key not in model_config_kwargs:
            model_config_kwargs[key] = default
            log0(f"Patching missing {key} in model config to {default!r}")

def _make_like_param(model_data, shape, fill_value):
    """Build a default tensor for a missing scalar parameter.

    Device is taken from the checkpoint's other tensors (load_state_dict runs with assign=True,
    so a CPU tensor here would end up inside a CUDA model). dtype is pinned to fp32 because
    that is what these parameters are on the model - only wte and value_embeds get cast to
    COMPUTE_DTYPE, so copying dtype from a sample tensor would silently downcast them to bf16.
    """
    sample = next((t for t in model_data.values() if torch.is_tensor(t) and t.is_floating_point()), None)
    device = None if sample is None else sample.device
    return torch.full(shape, fill_value, dtype=torch.float32, device=device)

def _patch_missing_keys(model_data, model_config):
    """Reconcile a checkpoint's parameters with the components the config actually enables.

    Disabled components are None on the model and therefore absent from its state_dict, while
    build_model loads with strict=True. So this has to work in both directions: synthesize
    defaults for enabled components an older checkpoint lacks, and drop parameters belonging
    to components this config disables.
    """
    n_layer = model_config.n_layer
    # (state_dict key, enabled, shape, default fill value)
    expected = [
        ("resid_lambdas", model_config.use_resid_lambdas, (n_layer,), 1.0),  # 1.0 = identity scaling
        ("x0_lambdas", model_config.use_x0, (n_layer,), 0.0),                # 0.0 = disabled
        ("smear_lambda", model_config.use_smear, (1,), 0.0),
        ("smear_gate.weight", model_config.use_smear, (1, 24), 0.0),
        ("backout_lambda", model_config.use_backout, (1,), 0.2),
    ]
    for key, enabled, shape, fill_value in expected:
        if not enabled:
            if key in model_data:
                model_data.pop(key)
                log0(f"Dropping {key} from model data because it is disabled in the model config")
        elif key not in model_data:
            model_data[key] = _make_like_param(model_data, shape, fill_value)
            log0(f"Patching missing {key} in model data to {fill_value}")

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save the model state parameters
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(model_data, model_path)
        logger.info(f"Saved model parameters to: {model_path}")
        # Save the metadata dict as json
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        logger.info(f"Saved metadata to: {meta_path}")
    # Note that optimizer state is sharded across ranks, so each rank must save its own.
    if optimizer_data is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        torch.save(optimizer_data, optimizer_path)
        logger.info(f"Saved optimizer state to: {optimizer_path}")

def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    # Load the model state
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)
    # Load the metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step, device, phase):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    if device.type in {"cpu", "mps"}:
        # Convert bfloat16 tensors to float for CPU inference
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = dict(meta_data["model_config"]) # copy so patching doesn't mutate the caller's meta
    _patch_missing_config_keys(model_config_kwargs)
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    # Hand back the normalized config, and a model tag for callers that need to name outputs
    meta_data["model_config"] = model_config.to_dict()
    meta_data.setdefault("model_tag", os.path.basename(checkpoint_dir))
    _patch_missing_keys(model_data, model_config)
    with torch.device("meta"):
        model = GPT(model_config)
    # Load the model state
    model.to_empty(device=device)
    model.init_weights() # note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init
    model.load_state_dict(model_data, strict=True, assign=True)
    # Put the model in the right training phase / mode
    if phase == "eval":
        model.eval()
    else:
        model.train()
    # Load the Tokenizer
    tokenizer = get_tokenizer()
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"], f"Tokenizer vocab size {tokenizer.get_vocab_size()} does not match model config vocab size {model_config_kwargs['vocab_size']}"
    return model, tokenizer, meta_data


def find_largest_model(checkpoints_dir):
    # attempt to guess the model tag: take the biggest model available
    model_tags = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    # Many tags now share a depth prefix (d20-e1280-...-lap0-s1, d20-e1280-...-lap2-s1, ...),
    # so break ties on mtime and take the most recently trained. Note this makes auto-selection
    # mtime-dependent: pass an explicit model_tag whenever the choice matters.
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            model_mtime = os.path.getmtime(os.path.join(checkpoints_dir, model_tag))
            candidates.append((model_depth, model_mtime, model_tag))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][2]
    # 2) if that failed, take the most recently updated model:
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if re.search(r'model_(\d+)\.pt$', f)]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = max(int(f.split("_")[-1].split(".")[0]) for f in checkpoint_files)
    return last_step

# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure

def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None):
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    meta_data["model_tag"] = model_tag # the tag we actually resolved to, for naming derived artifacts
    return model, tokenizer, meta_data

def load_model(source, *args, **kwargs):
    checkpoints_dir = get_checkpoints_dir(source)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)

def load_optimizer_state(source, device, rank, model_tag=None, step=None):
    """Load just the optimizer shard for a given rank, without re-loading the model."""
    checkpoints_dir = get_checkpoints_dir(source)
    if model_tag is None:
        model_tag = find_largest_model(checkpoints_dir)
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        step = find_last_step(checkpoint_dir)
    optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
    if not os.path.exists(optimizer_path):
        log0(f"Optimizer checkpoint not found: {optimizer_path}")
        return None
    log0(f"Loading optimizer state from {optimizer_path}")
    optimizer_data = torch.load(optimizer_path, map_location=device)
    return optimizer_data
