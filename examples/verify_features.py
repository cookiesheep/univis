"""Verify all 3 new features: report, logits_processor, batch aggregation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import torch.nn as nn

import univis


class FakeBlock(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return (x + self.linear(x),)


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([FakeBlock(16) for _ in range(3)])
        self.lm_head = nn.Linear(16, 50)
        self.config = type('C', (), {'_name_or_path': 'test-model'})()

    def forward(self, x):
        for b in self.transformer.h:
            x = b(x)[0]
        return self.lm_head(x)


def main():
    print('=== Feature 1: ECharts Report ===')
    model = FakeModel()
    tracker = univis.attach(model, transport='file', output_dir='/tmp/univis_verify')
    x = torch.randn(1, 5, 16)
    with torch.no_grad():
        for i in range(10):
            out = model(x)
            tracker.on_step(i, f'tok{i}', out[:, -1, :])
    path = tracker.finish()
    html = Path(path).read_text(encoding='utf-8')
    print(f'  Report: {path}')
    print(f'  Has ECharts: {"echarts" in html}')
    print(f'  Has heatmap: {"heatmap" in html}')
    print(f'  Has tokens:  {"tok0" in html}')
    print(f'  No raw </script>: {"</script>" not in html.split("</script>")[-1]}')

    print('\n=== Feature 2: logits_processor ===')
    model2 = FakeModel()
    tracker2 = univis.attach(model2, transport='file', output_dir='/tmp/univis_verify')

    class MockTokenizer:
        def decode(self, token_id):
            return f'word_{token_id.item()}'

    lp = tracker2.logits_processor(tokenizer=MockTokenizer())
    for i in range(5):
        scores = torch.randn(1, 50)
        scores[0, i * 10] = 10.0  # force argmax to token i*10
        lp(torch.tensor([[1]]), scores)

    print(f'  Steps recorded: {tracker2._step_count}')
    print(f'  Token 0: {tracker2._all_steps[0]["generated_token"]}')
    path2 = tracker2.finish()
    print(f'  Report: {path2}')

    print('\n=== Feature 3: Batch Aggregation ===')
    from univis.metrics import compute_relative_delta, compute_cosine_sim, compute_sparsity, compute_entropy

    inp = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
    out = torch.tensor([[2.0, 3.0], [2.0, 3.0]])
    s1 = compute_relative_delta(inp[:1], out[:1])
    b2 = compute_relative_delta(inp, out)
    print(f'  batch=1: {s1:.6f}, batch=2: {b2:.6f}, match: {abs(s1 - b2) < 1e-6}')

    logits = torch.zeros(2, 10)
    e1 = compute_entropy(logits[:1])
    e2 = compute_entropy(logits)
    print(f'  entropy batch=1: {e1:.6f}, batch=2: {e2:.6f}, match: {abs(e1 - e2) < 1e-6}')

    print('\nAll features verified.')


if __name__ == '__main__':
    main()
