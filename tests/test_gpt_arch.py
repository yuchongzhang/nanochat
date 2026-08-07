"""
Tests for the configurable parts of the GPT architecture:
- GPTConfig validation / normalization / JSON round-trip
- Laplacian attention heads
- the use_* ablation switches
- checkpoint round-trip across all of the above
- seeded weight initialization

Run as: python -m pytest tests/test_gpt_arch.py -v
"""

import json

import pytest
import torch
import torch.nn.functional as F

from nanochat.artifacts import make_model_tag, parse_laplacian_heads_spec, resolve_model_tag
from nanochat.gpt import GPT, GPTConfig, CausalSelfAttention, apply_rotary_emb, norm

DEVICE = "cpu"

ALL_FLAGS = ["use_ve", "use_resid_lambdas", "use_x0", "use_smear", "use_backout"]


def make_config(**overrides):
    kwargs = dict(
        sequence_len=64, vocab_size=128, n_layer=4,
        n_head=2, n_kv_head=2, n_embd=64, window_pattern="L",
    )
    kwargs.update(overrides)
    return GPTConfig(**kwargs)


def build_model(config):
    """Same meta-device dance the training scripts use."""
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=DEVICE)
    model.init_weights()
    return model


# =============================================================================
# GPTConfig
# =============================================================================
class TestGPTConfig:

    def test_laplacian_int_broadcasts_to_every_layer(self):
        config = make_config(n_layer=4, laplacian_heads=2)
        assert config.laplacian_heads == (2, 2, 2, 2)

    def test_laplacian_default_is_zero_everywhere(self):
        assert make_config(n_layer=4).laplacian_heads == (0, 0, 0, 0)

    def test_laplacian_list_is_kept_per_layer(self):
        config = make_config(n_layer=4, laplacian_heads=[0, 1, 2, 0])
        assert config.laplacian_heads == (0, 1, 2, 0)

    def test_laplacian_list_length_must_match_n_layer(self):
        with pytest.raises(ValueError, match="length n_layer"):
            make_config(n_layer=4, laplacian_heads=[0, 1])

    @pytest.mark.parametrize("bad", [3, -1, [0, 0, 0, 5]])
    def test_laplacian_count_must_be_within_head_count(self, bad):
        with pytest.raises(ValueError, match="must be in"):
            make_config(n_layer=4, n_head=2, laplacian_heads=bad)

    @pytest.mark.parametrize("bad", ["two", 1.5, True, [0, 0, 0, True]])
    def test_laplacian_rejects_non_int_specs(self, bad):
        with pytest.raises(TypeError):
            make_config(n_layer=4, laplacian_heads=bad)

    def test_window_pattern_is_uppercased(self):
        assert make_config(window_pattern="ssl").window_pattern == "SSL"

    def test_invalid_dims_are_rejected(self):
        with pytest.raises(ValueError, match="divisible"):
            make_config(n_embd=65, n_head=2)
        with pytest.raises(ValueError, match="n_kv_head"):
            make_config(n_head=2, n_kv_head=4)

    def test_to_dict_round_trips_through_json(self):
        config = make_config(laplacian_heads=[0, 1, 2, 0], use_ve=False, use_smear=False)
        restored = GPTConfig(**json.loads(json.dumps(config.to_dict())))
        assert restored == config
        # asdict() would emit a tuple here, which json turns into a list; to_dict is explicit
        assert isinstance(config.to_dict()["laplacian_heads"], list)


# =============================================================================
# Laplacian heads
# =============================================================================
class TestLaplacianHeads:

    def _reference_attention(self, attn, x, cos, sin, B, T, H, D):
        """Recompute q/k/v and vanilla attention the long way, for comparison."""
        q = attn.c_q(x).view(B, T, H, D)
        k = attn.c_k(x).view(B, T, H, D)
        v = attn.c_v(x).view(B, T, H, D)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q) * 1.2, norm(k) * 1.2
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        ).transpose(1, 2)
        return v, y

    @pytest.mark.parametrize("n_laplacian", [0, 1, 2])
    def test_laplacian_heads_return_v_minus_attention(self, n_laplacian):
        """Trailing n_laplacian heads must output (v - Av); leading heads stay untouched."""
        torch.manual_seed(0)
        B, T, H, D, C = 2, 8, 2, 16, 32
        config = make_config(sequence_len=T, n_layer=1, n_head=H, n_kv_head=H, n_embd=C,
                             laplacian_heads=n_laplacian)
        attn = CausalSelfAttention(config, 0)
        for p in attn.parameters():
            torch.nn.init.uniform_(p, -0.3, 0.3)

        x = torch.randn(B, T, C)
        inv_freq = 1.0 / (100000 ** (torch.arange(0, D, 2).float() / D))
        freqs = torch.outer(torch.arange(T).float(), inv_freq)
        cos, sin = freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :]

        got = attn(x, None, (cos, sin), (T, 0), None)

        v, y = self._reference_attention(attn, x, cos, sin, B, T, H, D)
        n_vanilla = H - n_laplacian
        expected = y if n_laplacian == 0 else torch.cat(
            [y[:, :, :n_vanilla, :], v[:, :, n_vanilla:, :] - y[:, :, n_vanilla:, :]], dim=2
        )
        expected = attn.c_proj(expected.contiguous().view(B, T, -1))
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)

    def test_laplacian_heads_add_no_parameters(self):
        """The mechanism reuses the existing projections, so param counts must not move."""
        baseline = build_model(make_config(laplacian_heads=0)).num_scaling_params()
        for spec in [1, 2, [0, 1, 2, 0]]:
            counts = build_model(make_config(laplacian_heads=spec)).num_scaling_params()
            assert counts == baseline, f"laplacian_heads={spec} changed the parameter count"

    def test_laplacian_heads_change_the_output(self):
        """Sanity check that the flag is actually wired into the forward pass.

        Note init_weights() zero-inits attn.c_proj, so at step 0 the attention branch
        contributes nothing and Laplacian heads are provably a no-op. We give c_proj real
        weights first, mimicking a partially trained model.
        """
        def build_trained_ish(spec):
            torch.manual_seed(0)
            model = build_model(make_config(laplacian_heads=spec))
            gen = torch.Generator().manual_seed(7)
            with torch.no_grad():
                for block in model.transformer.h:
                    block.attn.c_proj.weight.copy_(
                        torch.randn(block.attn.c_proj.weight.shape, generator=gen) * 0.05
                    )
            return model

        idx = torch.randint(0, 128, (2, 16))
        assert not torch.allclose(build_trained_ish(0)(idx), build_trained_ish(2)(idx))

    def test_laplacian_is_a_noop_at_initialization(self):
        """Documents the consequence of zero-init c_proj: Laplacian heads only start acting
        once the output projection has moved away from zero."""
        torch.manual_seed(0)
        plain = build_model(make_config(laplacian_heads=0))
        torch.manual_seed(0)
        lap = build_model(make_config(laplacian_heads=2))
        idx = torch.randint(0, 128, (2, 16))
        torch.testing.assert_close(plain(idx), lap(idx))

    def test_all_heads_laplacian_is_allowed(self):
        """n_laplacian == n_head leaves an empty vanilla slice, which must not break."""
        model = build_model(make_config(n_head=2, n_kv_head=2, laplacian_heads=2))
        assert torch.isfinite(model(torch.randint(0, 128, (2, 16)), torch.randint(0, 128, (2, 16))))

    def test_laplacian_works_under_gqa(self):
        """With n_kv_head < n_head, v must be expanded to line up with the query heads."""
        model = build_model(make_config(n_head=4, n_kv_head=2, n_embd=64, laplacian_heads=2))
        loss = model(torch.randint(0, 128, (2, 16)), torch.randint(0, 128, (2, 16)))
        assert torch.isfinite(loss)


# =============================================================================
# Ablation flags
# =============================================================================
class TestAblationFlags:

    @pytest.mark.parametrize("flag", ALL_FLAGS)
    def test_single_ablation_builds_and_trains(self, flag):
        model = build_model(make_config(**{flag: False}))
        idx = torch.randint(0, 128, (2, 16))
        loss = model(idx, idx)
        assert torch.isfinite(loss)
        loss.backward()
        # num_scaling_params and setup_optimizer both assert internally that every parameter
        # is accounted for exactly once; if a disabled component leaked they would fail here
        model.num_scaling_params()
        model.setup_optimizer()

    def test_all_ablations_off_together(self):
        model = build_model(make_config(**{flag: False for flag in ALL_FLAGS}))
        idx = torch.randint(0, 128, (2, 16))
        assert torch.isfinite(model(idx, idx))
        assert model.num_scaling_params()["scalars"] == 0
        model.setup_optimizer()

    @pytest.mark.parametrize("flag", ALL_FLAGS)
    def test_disabled_component_has_no_parameters(self, flag):
        """Disabled components register as None, so they leave parameters() and state_dict()."""
        enabled = build_model(make_config()).num_scaling_params()["total"]
        disabled = build_model(make_config(**{flag: False})).num_scaling_params()["total"]
        assert disabled < enabled, f"{flag}=False did not remove any parameters"

    def test_ablations_combine_with_laplacian_heads(self):
        config = make_config(laplacian_heads=[0, 1, 2, 0], use_ve=False, use_backout=False)
        model = build_model(config)
        idx = torch.randint(0, 128, (2, 16))
        assert torch.isfinite(model(idx, idx))

    def test_no_ve_removes_value_embeddings_and_gates(self):
        model = build_model(make_config(use_ve=False))
        assert len(model.value_embeds) == 0
        assert all(block.attn.ve_gate is None for block in model.transformer.h)

    def test_no_smear_removes_gate_and_lambda(self):
        model = build_model(make_config(use_smear=False))
        assert model.smear_gate is None
        assert model.smear_lambda is None
        assert "smear_gate.weight" not in model.state_dict()

    def test_default_config_keeps_every_component(self):
        model = build_model(make_config())
        assert model.smear_gate is not None
        assert model.resid_lambdas is not None and model.x0_lambdas is not None
        assert model.backout_lambda is not None
        assert len(model.value_embeds) > 0


# =============================================================================
# Optimization
# =============================================================================
class TestOptimizerSteps:
    """setup_optimizer() alone does not prove MuonAdamW can actually step the reduced param
    groups an ablated model produces, so take a few real steps. Marked slow: the fused
    optimizer kernels are torch.compile'd and cold compilation on CPU is not cheap."""

    @pytest.mark.slow
    @pytest.mark.parametrize("label,overrides", [
        ("baseline", {}),
        ("laplacian", {"laplacian_heads": 2}),
        ("all ablations off", {flag: False for flag in ALL_FLAGS}),
        ("all off + laplacian", dict({flag: False for flag in ALL_FLAGS}, laplacian_heads=2)),
    ])
    def test_training_steps_reduce_the_loss(self, label, overrides):
        torch.manual_seed(0)
        model = build_model(make_config(**overrides))
        optimizer = model.setup_optimizer()
        idx = torch.randint(0, 128, (2, 16))

        losses = []
        for _ in range(3):
            loss = model(idx, idx)
            loss.backward()
            optimizer.step()
            model.zero_grad(set_to_none=True)
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"{label}: loss did not decrease ({losses})"
        for name, p in model.named_parameters():
            assert torch.isfinite(p).all(), f"{label}: {name} went non-finite"


# =============================================================================
# Checkpoint round-trip
# =============================================================================
class TestCheckpointRoundTrip:

    @pytest.mark.parametrize("overrides", [
        {},
        {"laplacian_heads": [0, 1, 2, 0]},
        {flag: False for flag in ALL_FLAGS},
        {"use_ve": False, "use_smear": False, "laplacian_heads": 2},
    ])
    def test_state_dict_round_trips(self, overrides, tmp_path):
        """Save a config to json + params to a file, rebuild, and load with strict=True."""
        from nanochat.checkpoint_manager import _patch_missing_config_keys, _patch_missing_keys

        config = make_config(**overrides)
        model = build_model(config)
        state = {k: v.clone() for k, v in model.state_dict().items()}
        config_kwargs = json.loads(json.dumps(config.to_dict()))

        _patch_missing_config_keys(config_kwargs)
        rebuilt_config = GPTConfig(**config_kwargs)
        assert rebuilt_config == config
        _patch_missing_keys(state, rebuilt_config)

        rebuilt = build_model(rebuilt_config)
        rebuilt.load_state_dict(state, strict=True, assign=True)
        idx = torch.randint(0, 128, (2, 16))
        torch.testing.assert_close(model(idx), rebuilt(idx))

    def test_legacy_config_without_new_keys_is_patched(self):
        """A checkpoint predating these features must still load, unchanged in behavior."""
        from nanochat.checkpoint_manager import _patch_missing_config_keys

        legacy = {"sequence_len": 64, "vocab_size": 128, "n_layer": 4,
                  "n_head": 2, "n_kv_head": 2, "n_embd": 64}
        _patch_missing_config_keys(legacy)
        config = GPTConfig(**legacy)
        assert config.laplacian_heads == (0, 0, 0, 0)
        assert all(getattr(config, flag) for flag in ALL_FLAGS)
        assert config.window_pattern == "L" # old models had no sliding window

    def test_legacy_params_are_synthesized(self):
        """An old state dict missing the scalar params gets sensible defaults filled in."""
        from nanochat.checkpoint_manager import _patch_missing_keys

        config = make_config()
        state = {"transformer.wte.weight": torch.zeros(128, 64)}
        _patch_missing_keys(state, config)
        assert torch.equal(state["resid_lambdas"], torch.ones(config.n_layer))
        assert torch.equal(state["x0_lambdas"], torch.zeros(config.n_layer))
        assert torch.equal(state["backout_lambda"], torch.full((1,), 0.2))
        assert state["smear_gate.weight"].shape == (1, 24)

    def test_synthesized_params_match_the_model_dtype_and_device(self):
        """load_state_dict runs with assign=True, so a synthesized tensor with the wrong dtype
        or device silently becomes the model's parameter. These scalars are fp32 on the model
        even when wte has been cast to bf16."""
        from nanochat.checkpoint_manager import _patch_missing_keys

        state = {"transformer.wte.weight": torch.zeros(128, 64, dtype=torch.bfloat16)}
        _patch_missing_keys(state, make_config())
        for key in ["resid_lambdas", "x0_lambdas", "smear_lambda", "backout_lambda", "smear_gate.weight"]:
            assert state[key].dtype == torch.float32, f"{key} was synthesized as {state[key].dtype}"
            assert state[key].device == torch.device(DEVICE)

    def test_params_of_disabled_components_are_dropped(self):
        """Loading an all-on checkpoint under an ablated config must not trip strict=True."""
        from nanochat.checkpoint_manager import _patch_missing_keys

        state = dict(build_model(make_config()).state_dict())
        _patch_missing_keys(state, make_config(**{flag: False for flag in ALL_FLAGS}))
        for key in ["resid_lambdas", "x0_lambdas", "smear_lambda", "smear_gate.weight", "backout_lambda"]:
            assert key not in state, f"{key} should have been dropped"


# =============================================================================
# Seeding
# =============================================================================
class TestSeeding:

    def test_same_seed_gives_identical_weights(self):
        config = make_config()
        torch.manual_seed(1234)
        a = build_model(config)
        torch.manual_seed(1234)
        b = build_model(config)
        for (name, pa), pb in zip(a.named_parameters(), b.parameters()):
            torch.testing.assert_close(pa, pb, msg=f"{name} differs under the same seed")

    def test_different_seeds_give_different_weights(self):
        config = make_config()
        torch.manual_seed(1)
        a = build_model(config)
        torch.manual_seed(2)
        b = build_model(config)
        assert not torch.allclose(a.transformer.wte.weight, b.transformer.wte.weight)

    def test_compute_init_seeds_the_global_rng(self):
        from nanochat.common import compute_init

        compute_init("cpu", seed=7)
        first = torch.randn(4)
        compute_init("cpu", seed=7)
        torch.testing.assert_close(first, torch.randn(4))


# =============================================================================
# Seeded data order
# =============================================================================
class TestShardSeeding:
    """The pretraining loader is otherwise fully deterministic, so shard_seed is the only thing
    that makes two seeds see different data."""

    N_SHARDS = 20

    @pytest.fixture
    def shard_order(self, monkeypatch):
        import nanochat.dataloader as dl

        shards = [f"shard_{i:03d}.parquet" for i in range(self.N_SHARDS)]
        monkeypatch.setattr(dl, "list_parquet_files", lambda warn_on_legacy=False: list(shards))

        class FakeColumn:
            def to_pylist(self):
                return ["doc"]

        class FakeRowGroup:
            def column(self, name):
                return FakeColumn()

        def order(split, seed, n_yields):
            opened = []

            class FakeParquetFile:
                num_row_groups = 1
                def __init__(self, path):
                    opened.append(path)
                def read_row_group(self, i):
                    return FakeRowGroup()

            monkeypatch.setattr(dl.pq, "ParquetFile", FakeParquetFile)
            batches = dl._document_batches(split, None, 8, shard_seed=seed)
            for _ in range(n_yields):
                next(batches)
            return opened

        order.shards = shards
        return order

    def test_no_seed_keeps_sorted_order(self, shard_order):
        # base_eval and any other caller that omits shard_seed must see today's exact behavior
        assert shard_order("train", None, self.N_SHARDS - 1) == shard_order.shards[:-1]

    def test_same_seed_is_reproducible(self, shard_order):
        n = self.N_SHARDS - 1
        assert shard_order("train", 1, n) == shard_order("train", 1, n)

    def test_different_seeds_give_different_order(self, shard_order):
        n = self.N_SHARDS - 1
        assert shard_order("train", 1, n) != shard_order("train", 2, n)

    def test_permutation_preserves_the_shard_set(self, shard_order):
        n = self.N_SHARDS - 1
        assert sorted(shard_order("train", 1, n)) == sorted(shard_order.shards[:-1])

    def test_val_split_is_never_permuted(self, shard_order):
        # val must stay fixed so that bpb is comparable across seeds
        assert shard_order("val", 1, 1) == shard_order("val", None, 1) == [shard_order.shards[-1]]


# =============================================================================
# Model tags
# =============================================================================
class TestModelTags:

    def test_tag_encodes_architecture_and_seed(self):
        tag = make_model_tag(make_config(n_layer=4, n_embd=64, n_head=2, n_kv_head=2), seed=42)
        assert tag == "d4-e64-h2-kv2-wl-lap0-s42"

    def test_uniform_and_per_layer_laplacian_specs_format_differently(self):
        uniform = make_model_tag(make_config(laplacian_heads=2), seed=1)
        per_layer = make_model_tag(make_config(laplacian_heads=[0, 1, 2, 0]), seed=1)
        assert "-lap2-" in uniform
        assert "-lap0x1x2x0-" in per_layer

    def test_ablations_appear_as_suffixes(self):
        tag = make_model_tag(make_config(use_ve=False, use_x0=False), seed=3)
        assert "-nove" in tag and "-nox0" in tag
        assert "-nosmear" not in tag

    def test_seed_distinguishes_otherwise_identical_runs(self):
        config = make_config()
        assert make_model_tag(config, seed=1) != make_model_tag(config, seed=2)

    def test_explicit_tag_wins(self):
        assert resolve_model_tag(make_config(), seed=1, explicit_tag="mytag") == "mytag"
        assert resolve_model_tag(make_config(), seed=1) == make_model_tag(make_config(), seed=1)

    @pytest.mark.parametrize("value,expected", [
        ("2", 2), ("0", 0), ("0,2,2,0", [0, 2, 2, 0]), (" 3 ", 3), (2, 2), ([1, 2], [1, 2]),
    ])
    def test_parse_laplacian_heads_spec(self, value, expected):
        assert parse_laplacian_heads_spec(value) == expected

    @pytest.mark.parametrize("bad", ["0,,2", "1,2,", "abc"])
    def test_parse_laplacian_heads_spec_rejects_garbage(self, bad):
        with pytest.raises(ValueError):
            parse_laplacian_heads_spec(bad)
