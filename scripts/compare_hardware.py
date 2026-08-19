"""Compare two UniVis JSONL sessions layer-by-layer (cross-hardware analysis).

Usage: python compare_hw.py <label_a> <jsonl_a> <label_b> <jsonl_b> [--chart out.png]
Prints per-layer averaged metrics, Pearson correlation of layer profiles, and
mean absolute deviation between the two runs.
"""

import json
import sys


def layer_profile(path):
    """Mean per-layer metrics over all steps in a session JSONL."""
    prof = {}
    nsteps = 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            if rec.get('type') != 'step':
                continue
            nsteps += 1
            for lay in rec['layers']:
                d = prof.setdefault(lay['idx'], {'rd': [], 'cs': [], 'sp': []})
                d['rd'].append(lay['relative_delta'])
                d['cs'].append(lay['cosine_sim'])
                d['sp'].append(lay['sparsity'])
    out = {i: {k: sum(v) / len(v) for k, v in d.items()} for i, d in prof.items()}
    return out, nsteps


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (vx * vy) if vx and vy else float('nan')


def main() -> None:
    la, pa, lb, pb = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    chart = sys.argv[6] if len(sys.argv) > 6 and sys.argv[5] == '--chart' else None
    A, na = layer_profile(pa)
    B, nb = layer_profile(pb)
    common = sorted(set(A) & set(B))
    depth = max(common) if common else 1
    cs_a = [A[i]['cs'] for i in common]
    cs_b = [B[i]['cs'] for i in common]
    rd_a = [A[i]['rd'] for i in common]
    rd_b = [B[i]['rd'] for i in common]
    r_cs = pearson(cs_a, cs_b)
    r_rd = pearson(rd_a, rd_b)
    mae_cs = sum(abs(x - y) for x, y in zip(cs_a, cs_b)) / len(common)
    mae_rd = sum(abs(x - y) for x, y in zip(rd_a, rd_b)) / len(common)

    print(f'{la}: {na} steps, {len(A)} layers')
    print(f'{lb}: {nb} steps, {len(B)} layers')
    print(f'common layers: {len(common)}')
    print(f'Pearson r (cosine_sim profile): {r_cs:.4f}')
    print(f'Pearson r (relative_delta profile): {r_rd:.4f}')
    print(f'MAE cosine_sim: {mae_cs:.4f} | MAE relative_delta: {mae_rd:.4f}')
    print()
    print('| layer | depth | cos(A) | cos(B) | d_cos | rd(A) | rd(B) |')
    print('|---|---|---|---|---|---|---|')
    for i in common:
        print(f'| {i} | {i / depth:.2f} | {A[i]["cs"]:.4f} | {B[i]["cs"]:.4f} | '
              f'{A[i]["cs"] - B[i]["cs"]:+.4f} | {A[i]["rd"]:.3f} | {B[i]["rd"]:.3f} |')

    if chart:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        x = [i / depth for i in common]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(x, cs_a, 'o-', label=la, color='#d62728')
        axes[0].plot(x, cs_b, 's--', label=lb, color='#1f77b4')
        axes[0].set_xlabel('normalized layer depth')
        axes[0].set_ylabel('cosine similarity (layer avg)')
        axes[0].set_title(f'cosine profile  r={r_cs:.3f}')
        axes[0].legend()
        axes[1].plot(x, rd_a, 'o-', label=la, color='#d62728')
        axes[1].plot(x, rd_b, 's--', label=lb, color='#1f77b4')
        axes[1].set_xlabel('normalized layer depth')
        axes[1].set_ylabel('relative delta (layer avg)')
        axes[1].set_title(f'relative-delta profile  r={r_rd:.3f}')
        axes[1].legend()
        fig.suptitle(f'{la} vs {lb} — same prompt, greedy, bf16', fontsize=11)
        fig.tight_layout()
        fig.savefig(chart, dpi=150)
        print(f'\nchart saved: {chart}')


if __name__ == '__main__':
    main()
