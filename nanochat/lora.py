"""
LoRA (Low-Rank Adaptation) utilities for nanochat.

Wraps bias-free Linear layers (in particular nanochat.gpt.Linear) with low-rank
adapters: y = base(x) + (dropout(x) @ A^T) @ B^T * (alpha / rank). The base weight
is kept on the wrapper as a frozen submodule; only A and B are trained. B is
zero-initialised so the wrapped layer is mathematically identical to the base
layer at step 0.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """LoRA-adapted bias-free Linear layer."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        assert base.bias is None, "LoRALinear only supports bias-free Linear layers"
        self.base = base
        self.base.weight.requires_grad = False
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = (self.alpha / self.rank) if self.rank > 0 else 0.0
        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()

    def forward(self, x):
        out = F.linear(x, self.base.weight.to(x.dtype))
        if self.rank > 0:
            lora_x = self.dropout(x)
            tmp = F.linear(lora_x, self.lora_A.to(x.dtype))
            delta = F.linear(tmp, self.lora_B.to(x.dtype))
            out = out + delta * self.scaling
        return out


def _resolve_attr(root: nn.Module, qualified_name: str):
    parts = qualified_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def apply_lora(model: nn.Module, rank: int, alpha: float, dropout: float, target_modules):
    """
    Replace each target submodule inside every transformer block with a LoRALinear.
    target_modules is a list of dotted names relative to a Block, e.g.
    ["attn.c_q", "attn.c_k", "attn.c_v", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"].
    Returns the list of newly-created LoRA parameters (lora_A and lora_B).
    """
    target_modules = list(target_modules)
    lora_params = []
    for block in model.transformer.h:
        for qualified_name in target_modules:
            parent, attr = _resolve_attr(block, qualified_name)
            child = getattr(parent, attr)
            assert isinstance(child, nn.Linear), (
                f"target {qualified_name} resolved to {type(child).__name__}, expected nn.Linear"
            )
            wrapped = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
            setattr(parent, attr, wrapped)
            lora_params.append(wrapped.lora_A)
            lora_params.append(wrapped.lora_B)
    return lora_params


def freeze_base(model: nn.Module):
    """Freeze every parameter except those whose name contains '.lora_A' or '.lora_B'."""
    for name, p in model.named_parameters():
        p.requires_grad = (".lora_A" in name) or (".lora_B" in name)


def _strip_compile_prefix(key: str) -> str:
    return key.removeprefix("_orig_mod.")


def lora_state_dict(model: nn.Module) -> dict:
    """Return only LoRA parameters (A and B), with torch.compile prefix stripped."""
    sd = {}
    for k, v in model.state_dict().items():
        if ".lora_A" in k or ".lora_B" in k:
            sd[_strip_compile_prefix(k)] = v.detach().cpu().clone()
    return sd


def load_lora_state_dict(model: nn.Module, sd: dict):
    """Load a LoRA-only state dict into a model that has had apply_lora() called.

    Validates that every key in `sd` corresponds to an existing LoRA parameter and
    that no expected LoRA parameter is missing. Non-LoRA parameters are untouched.
    """
    model_keys = set(model.state_dict().keys())
    expected_lora_keys = {k for k in model_keys if ".lora_A" in k or ".lora_B" in k}
    expected_keys_stripped = {_strip_compile_prefix(k) for k in expected_lora_keys}
    given_keys = set(sd.keys())
    missing = expected_keys_stripped - given_keys
    unexpected = given_keys - expected_keys_stripped
    if missing:
        raise RuntimeError(f"Missing {len(missing)} LoRA keys in state dict, e.g. {sorted(missing)[:3]}")
    if unexpected:
        raise RuntimeError(f"Unexpected {len(unexpected)} keys in LoRA state dict, e.g. {sorted(unexpected)[:3]}")
    has_orig_mod = any(k.startswith("_orig_mod.") for k in model_keys)
    sd_to_load = {f"_orig_mod.{k}": v for k, v in sd.items()} if has_orig_mod else dict(sd)
    incompatible = model.load_state_dict(sd_to_load, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected keys after load: {incompatible.unexpected_keys[:3]}")


def count_trainable(model: nn.Module):
    """Return (trainable_params, total_params, percentage)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = (100.0 * trainable / total) if total > 0 else 0.0
    return trainable, total, pct
