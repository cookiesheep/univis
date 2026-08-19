"""Tests for Pilot intervention (layer-skip + early-exit)."""

import pytest
import torch
import torch.nn as nn

from univis import pilot as pilot_mod
from univis.pilot import Pilot, PilotPolicy


class _FakeBlock(nn.Module):
    """Fake transformer block: doubles input (so skip=identity is detectable)."""

    def forward(self, hidden_states, *args, **kwargs):
        return (hidden_states * 2,)


class _FakeModel(nn.Module):
    """model.layers.{0..N} structure mimicking HF naming. Eval by default."""

    def __init__(self, n: int = 6) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FakeBlock() for _ in range(n)])
        self.eval()  # avoid Pilot training-mode warning

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            if isinstance(x, tuple):
                x = x[0]
        return x


# ----- PilotPolicy -----


def test_policy_marks_redundant_layers():
    summary = [
        {'idx': 0, 'avg_cosim': 0.5},
        {'idx': 1, 'avg_cosim': 0.96},
        {'idx': 2, 'avg_cosim': 0.98},
        {'idx': 3, 'avg_cosim': 0.7},
    ]
    policy = PilotPolicy.from_layer_summary(summary, cos_threshold=0.95, max_skip_ratio=1.0)
    assert policy.skip_layers == {1, 2}


def test_policy_cap_respected():
    summary = [{'idx': i, 'avg_cosim': 0.99} for i in range(6)]
    policy = PilotPolicy.from_layer_summary(
        summary, cos_threshold=0.95, max_skip_ratio=0.3)
    assert len(policy.skip_layers) == 1  # int(6 * 0.3) = 1


def test_policy_empty_summary():
    policy = PilotPolicy.from_layer_summary([])
    assert policy.skip_layers == set()


# ----- Pilot layer-skip -----


def test_apply_skips_layers_and_restore_reverts():
    model = _FakeModel(6)
    pilot = Pilot(model, PilotPolicy(skip_layers={1, 3}))
    x = torch.tensor([1.0])

    assert model(x).item() == 64.0

    n = pilot.apply()
    assert n == 2
    assert pilot.skipped_layer_count == 2
    assert model(x).item() == 16.0  # layers 0,2,4,5 double; 1,3 identity

    pilot.restore()
    assert model(x).item() == 64.0
    assert pilot.skipped_layer_count == 0


def test_apply_is_idempotent():
    model = _FakeModel(4)
    pilot = Pilot(model, PilotPolicy(skip_layers={0}))
    n1 = pilot.apply()
    n2 = pilot.apply()
    assert n1 == n2 == 1


def test_apply_with_empty_skip_set_is_noop():
    model = _FakeModel(4)
    pilot = Pilot(model, PilotPolicy(skip_layers=set()))
    assert pilot.apply() == 0
    assert pilot.skipped_layer_count == 0


def test_context_manager_restores_on_exception():
    """CRITICAL fix: context manager restores forwards even if body raises."""
    model = _FakeModel(4)
    pilot = Pilot(model, PilotPolicy(skip_layers={0, 1}))
    assert model(torch.tensor([1.0])).item() == 16.0

    with pytest.raises(RuntimeError):
        with pilot:
            assert pilot.skipped_layer_count == 2
            raise RuntimeError('simulated generation failure')

    # restored despite exception
    assert pilot.skipped_layer_count == 0
    assert model(torch.tensor([1.0])).item() == 16.0


def test_context_manager_normal_exit_restores():
    model = _FakeModel(4)
    pilot = Pilot(model, PilotPolicy(skip_layers={0}))
    with pilot:
        assert pilot.skipped_layer_count == 1
    assert pilot.skipped_layer_count == 0


def test_wraps_tuple_false_returns_tensor(monkeypatch):
    """transformers>=5 path: pass-through returns tensor, not tuple."""
    monkeypatch.setattr(pilot_mod, '_WRAPS_TUPLE', False)
    model = _FakeModel(3)
    pilot = Pilot(model, PilotPolicy(skip_layers={1}))
    pilot.apply()
    out = model.layers[1](torch.tensor([1.0, 2.0]))
    assert isinstance(out, torch.Tensor)
    assert torch.equal(out, torch.tensor([1.0, 2.0]))
    pilot.restore()


def test_training_mode_warns():
    model = _FakeModel(3).train()  # force training mode
    pilot = Pilot(model, PilotPolicy(skip_layers={0}))
    with pytest.warns(UserWarning, match='training mode'):
        pilot.apply()
    pilot.restore()


# ----- Pilot early-exit -----


def test_early_exit_forces_eos_on_low_entropy():
    model = _FakeModel(2)
    pilot = Pilot(model, PilotPolicy(entropy_threshold=0.5))
    proc = pilot.logits_processor(eos_token_id=0)

    out = proc(None, torch.zeros(1, 10))  # uniform -> high entropy -> no EOS
    assert not torch.isinf(out).any()
    assert pilot.early_exit_count == 0

    peaked = torch.full((1, 10), -100.0)  # one-hot -> entropy ~0 -> force EOS
    peaked[0, 3] = 100.0
    out = proc(None, peaked)
    assert out[0, 0].item() == 0.0
    assert pilot.early_exit_count == 1


def test_early_exit_noop_without_eos_id():
    model = _FakeModel(2)
    pilot = Pilot(model, PilotPolicy(entropy_threshold=0.5))
    proc = pilot.logits_processor(eos_token_id=None)
    scores = torch.zeros(1, 10)
    assert torch.equal(proc(None, scores), scores)


def test_early_exit_invalid_eos_id_is_safe():
    """Out-of-range eos_token_id must not raise (bounds check)."""
    model = _FakeModel(2)
    pilot = Pilot(model, PilotPolicy(entropy_threshold=0.5))
    peaked = torch.full((1, 4), -100.0)
    peaked[0, 1] = 100.0
    for bad_eos in (-1, 99):
        proc = pilot.logits_processor(eos_token_id=bad_eos)
        out = proc(None, peaked)  # must not raise IndexError
        assert out.shape == peaked.shape
