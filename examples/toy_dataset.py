"""How to plug a new dataset into HERON: write each window in the standard format of heron/data.py.

This toy dataset places N particles on a ring graph. Each window follows one of two regimes (clockwise or
counter-clockwise rotation) and may switch once; one particle can appear mid-window. Node features are
[x, y, particle type] with only (x, y) dynamic; the edge feature is the per-step distance between neighbours.

    python examples/toy_dataset.py --out data/toy
    python scripts/run_cv.py --config examples/toy.yaml
"""
import argparse
import math
from pathlib import Path
import torch


def create_window(seed: int, num_steps: int = 8, num_nodes: int = 6) -> dict:
    """ Create one window of particles rotating on a ring, with at most one regime switch. """
    generator = torch.Generator().manual_seed(seed)
    # Regime sequence
    switch_step = int(torch.randint(2, num_steps, (1,), generator=generator)) if seed % 2 else num_steps
    first_regime = int(torch.randint(0, 2, (1,), generator=generator))
    regime = torch.tensor([first_regime if t < switch_step else 1 - first_regime for t in range(num_steps)])
    # Initial particle positions
    radius = 1 + torch.rand(num_nodes, generator=generator)
    angle = 2 * math.pi * torch.rand(num_nodes, generator=generator)
    particle_type = torch.randint(0, 2, (num_nodes,), generator=generator).float()
    # Rotation in the regime direction
    x = torch.zeros(num_steps, num_nodes, 3)
    for t in range(num_steps):
        x[t, :, 0], x[t, :, 1], x[t, :, 2] = radius * angle.cos(), radius * angle.sin(), particle_type
        angle = angle + (0.3 if regime[t] == 0 else -0.3) + 0.02 * torch.randn(num_nodes, generator=generator)
    # Node presence
    node_mask = torch.ones(num_steps, num_nodes, dtype=torch.bool)
    if seed % 3 == 0:
        node_mask[:2, -1] = False                  # the last particle appears at step 2
        x[:2, -1] = 0
    # Ring edges with distance features
    edge_index, edge_attr = [], []
    for t in range(num_steps):
        node_idx = torch.nonzero(node_mask[t]).squeeze(1)
        ring = torch.stack([node_idx, node_idx.roll(-1)])
        ring = torch.cat([ring, ring.flip(0)], 1)  # undirected
        edge_index.append(ring)
        edge_attr.append((x[t, ring[0], :2] - x[t, ring[1], :2]).norm(dim=-1, keepdim=True))
    return {"x": x, "node_mask": node_mask, "edge_index": edge_index, "edge_attr": edge_attr, "y": regime,
            "seq_id": seed // 2}


def generate_dataset(out_dir, num_windows: int = 48, num_steps: int = 8) -> Path:
    """ Generate the toy windows and save each one as a separate file in out_dir. """
    dataset_dir = Path(out_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_windows):
        torch.save(create_window(i, num_steps), dataset_dir / f"{i}.pt")
    return dataset_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/toy")
    parser.add_argument("--windows", type=int, default=48)
    args = parser.parse_args()
    print(generate_dataset(args.out, args.windows))
