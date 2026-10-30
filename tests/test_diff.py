"""Tests for layer-wise diff (tiny random Llama, CPU only)."""

import pytest
import torch

transformers = pytest.importorskip('transformers')

from univis import diff

TOKENS = [[1, 5, 9, 2, 7, 3, 11, 4], [1, 8, 6, 12, 10]]


def _model(seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    cfg = transformers.LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=6, num_attention_heads=4,
                                   num_key_value_heads=2)
    return transformers.LlamaForCausalLM(cfg).eval()


def _faulty(model: torch.nn.Module, block: int, scale: float = 0.1) -> torch.nn.Module:
    """Perturb one block's output by ~scale of its magnitude, mimicking a wrong kernel."""
    gen = torch.Generator().manual_seed(1)

    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h + scale * h.abs().mean() * torch.randn(h.shape, generator=gen).to(h.dtype)
        return (h, *output[1:]) if isinstance(output, tuple) else h

    model.model.layers[block].register_forward_hook(hook)
    return model


def test_projection_is_deterministic():
    a, b = diff.projection(32, 16, seed=3), diff.projection(32, 16, seed=3)
    assert torch.equal(a, b)
    assert torch.allclose(a.abs(), torch.full_like(a, 16 ** -0.5))
    assert not torch.equal(a, diff.projection(32, 16, seed=4))


def test_record_roundtrip(tmp_path):
    fp = diff.record(_model(), token_ids=TOKENS, dim=8, label='cpu')
    assert fp['num_layers'] == 6 and len(fp['layers']) == 7
    loaded = diff.load(diff.save(fp, tmp_path / 'fp.json'))
    assert loaded == fp


def test_identical_runs_pass_with_baseline():
    ref = diff.record(_model(), token_ids=TOKENS)
    res = diff.compare(ref, diff.record(_model(), token_ids=TOKENS), baseline=ref)
    assert res['verdict'] == 'PASS'
    assert max(res['rel_error']) == 0.0
    assert res['top1_agreement'] == 1.0


@pytest.mark.parametrize('block', [1, 4])
def test_fault_onset_is_localized(block):
    ref = diff.record(_model(), token_ids=TOKENS)
    bad = diff.record(_faulty(_model(), block), token_ids=TOKENS)
    jump = diff.compare(ref, bad)
    assert jump['verdict'] == 'DIVERGE' and jump['onset'] == block + 1  # index 0 is the embedding
    good = diff.record(_model().to(torch.bfloat16), token_ids=TOKENS)
    res = diff.compare(ref, diff.record(_faulty(_model(), block).to(torch.bfloat16), token_ids=TOKENS),
                       baseline=good)
    assert res['verdict'] == 'DIVERGE' and res['onset'] == block + 1


def test_precision_noise_alone_is_not_flagged():
    ref = diff.record(_model(), token_ids=TOKENS)
    cand = diff.record(_model().to(torch.bfloat16), token_ids=TOKENS)
    assert diff.compare(ref, cand, baseline=cand)['verdict'] == 'PASS'
    assert diff.compare(ref, cand)['verdict'] in ('UNKNOWN', 'DIVERGE')


def test_mismatched_tokens_rejected():
    a = diff.record(_model(), token_ids=TOKENS)
    b = diff.record(_model(), token_ids=TOKENS[:1])
    with pytest.raises(ValueError, match='token_ids'):
        diff.compare(a, b)


def test_render_html(tmp_path):
    ref = diff.record(_model(), token_ids=TOKENS)
    res = diff.compare(ref, diff.record(_faulty(_model(), 2), token_ids=TOKENS))
    html = diff.render_html(res, tmp_path / 'r.html').read_text(encoding='utf-8')
    assert 'DIVERGE' in html and 'echarts' in html


def _faulty_sublayer(model: torch.nn.Module, block: int, name: str, scale: float = 0.5) -> torch.nn.Module:
    """Perturb one sublayer (self_attn or mlp) output of a block."""
    gen = torch.Generator().manual_seed(2)

    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h + scale * h.abs().mean() * torch.randn(h.shape, generator=gen).to(h.dtype)
        return (h, *output[1:]) if isinstance(output, tuple) else h

    getattr(model.model.layers[block], name).register_forward_hook(hook)
    return model


# random-init MLP outputs are tiny next to the residual stream, hence the larger scale
@pytest.mark.parametrize('name,expected,scale', [('mlp', 'mlp', 50.0), ('self_attn', 'attn', 0.5)])
def test_onset_sublayer_attribution(name, expected, scale):
    ref = diff.record(_model(), token_ids=TOKENS)
    good = diff.record(_model().to(torch.bfloat16), token_ids=TOKENS)
    bad = diff.record(_faulty_sublayer(_model(), 3, name, scale).to(torch.bfloat16), token_ids=TOKENS)
    res = diff.compare(ref, bad, baseline=good)
    assert res['verdict'] == 'DIVERGE' and res['onset'] == 4
    assert res['onset_sublayer'] == expected
    assert ('MLP' if expected == 'mlp' else 'attention') in res['summary']


def test_sublayers_optional():
    fp = diff.record(_model(), token_ids=TOKENS, sublayers=False)
    assert 'sublayers' not in fp
    res = diff.compare(fp, diff.record(_faulty(_model(), 2), token_ids=TOKENS, sublayers=False))
    assert res['onset_sublayer'] is None
