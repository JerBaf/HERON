""" Graph-window datasets, batching, normalization statistics and splits.

    A dataset is a directory of ``.pt`` files, one window of T = C + H steps per file, in the standard format:

        x          float [T, N, F]   absolute node features (dynamic and static); slot n is the same entity at all steps
        node_mask  bool  [T, N]      entity n exists at step t
        edge_index long  [2, E] or a list of T tensors [2, E_t] (slot indices; edges only between existing entities)
        edge_attr  float [E, D_e] or a list of T tensors [E_t, D_e]   (optional)
        y          long  [T]         regime label per step, -1 if unknown (used for evaluation only)
        seq_id     int               (optional) source sequence; windows of one sequence never cross splits
        split      str               (optional) "test" pins the window to the test set (e.g. an official partition)

    The paper's synthetic files (a list of T ``torch_geometric.data.Data`` with ``x``, ``edge_index``, ``node_ids``,
    ``y``) are converted on load. Files are unpickled with ``torch.load(weights_only=False)``: load trusted data only.

"""
import json
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Optional
# Custom Library
from .config import Config


# Window Loading

def _pyg_to_window(graph_list: list) -> dict:
    """ Convert a list of PyG graphs (paper's synthetic format) to a window dict with one slot per entity. """
    # Entity slots
    node_ids = torch.cat([graph.node_ids.long() for graph in graph_list]).unique()
    x = torch.zeros(len(graph_list), len(node_ids), graph_list[0].x.shape[1])
    node_mask = torch.zeros(len(graph_list), len(node_ids), dtype=torch.bool)
    # Scatter each step into the slots
    edges = []
    for t, graph in enumerate(graph_list):
        slot_idx = torch.searchsorted(node_ids, graph.node_ids.long())
        x[t, slot_idx], node_mask[t, slot_idx] = graph.x.float(), True
        edges.append(slot_idx[graph.edge_index])
    # Regime label per step
    regime_labels = torch.stack([graph.y.reshape(-1)[0] for graph in graph_list]).long()
    return {"x": x, "node_mask": node_mask, "edge_index": edges, "y": regime_labels}


def load_window(window_path) -> dict:
    """ Load one window file in the standard format, converting PyG lists and repeating static edges. """
    window = torch.load(window_path, weights_only=False)
    if isinstance(window, (list, tuple)):
        window = _pyg_to_window(window)
    # Repeat static edges at every step
    num_steps = window["x"].shape[0]
    for key in ("edge_index", "edge_attr"):
        if torch.is_tensor(window.get(key)):
            window[key] = [window[key]] * num_steps
    # Optional entries
    window.setdefault("edge_attr", None)
    window.setdefault("y", torch.full((num_steps,), -1, dtype=torch.long))
    # Zero the features of absent entities, which are ignored
    window["node_mask"] = window["node_mask"].bool()
    window["x"] = window["x"].float().masked_fill(~window["node_mask"].unsqueeze(-1), 0.0)
    return window


# Dataset

class GraphWindowDataset(Dataset):
    """ Dataset of graph windows held in memory, loaded from a directory of .pt files. """
    def __init__(self, root: str, max_files: int = 0):
        """ Load the window files of root, shorter names first, keeping only the first max_files if nonzero. """
        # Sorted window files
        files = sorted(Path(root).glob("*.pt"), key=lambda file_path: (len(file_path.stem), file_path.stem))
        if max_files:
            files = files[:max_files]
        if not files:
            raise FileNotFoundError(f"no .pt windows in {root}")
        self.windows = [load_window(file_path) for file_path in files]
        # Store key dimensions
        first_window = self.windows[0]
        self.num_steps, self.feat_dim = first_window["x"].shape[0], first_window["x"].shape[-1]
        self.edge_dim = 0 if first_window["edge_attr"] is None else first_window["edge_attr"][0].shape[-1]

    def __len__(self) -> int:
        """ Retrieve the number of windows. """
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        """ Retrieve the window at position idx. """
        return self.windows[idx]

    def get_labels(self) -> np.ndarray:
        """ Retrieve the majority regime label per window (-1 if unlabeled), used only to stratify splits. """
        label_list = []
        for window in self.windows:
            known_labels = window["y"][window["y"] >= 0]
            label_list.append(int(torch.mode(known_labels).values) if len(known_labels) else -1)
        return np.array(label_list)

    def get_group_ids(self) -> np.ndarray:
        """ Retrieve the source sequence id per window, or its position if it has none. """
        return np.array([int(window.get("seq_id", i)) for i, window in enumerate(self.windows)])

    def get_fixed_test_mask(self) -> np.ndarray:
        """ Retrieve the boolean mask of the windows pinned to the test set. """
        return np.array([window.get("split") == "test" for window in self.windows])


# Batching

def collate(window_list: list) -> dict:
    """ Pad windows to the largest node count and merge their per-step edges into one graph (offset b * N). """
    # Batch dimensions
    B, T = len(window_list), window_list[0]["x"].shape[0]
    N, F = max(window["x"].shape[1] for window in window_list), window_list[0]["x"].shape[-1]
    # Padded containers
    x, mask = torch.zeros(B, T, N, F), torch.zeros(B, T, N, dtype=torch.bool)
    edge_index = [[] for _ in range(T)]
    edge_attr = None if window_list[0]["edge_attr"] is None else [[] for _ in range(T)]
    # Fill each window
    for b, window in enumerate(window_list):
        num_nodes = window["x"].shape[1]
        x[b, :, :num_nodes], mask[b, :, :num_nodes] = window["x"], window["node_mask"]
        for t in range(T):
            edge_index[t].append(window["edge_index"][t] + b * N)
            if edge_attr is not None:
                edge_attr[t].append(window["edge_attr"][t].float())
    # Batch dict
    return {
        "x": x,
        "mask": mask,
        "y": torch.stack([window["y"] for window in window_list]),
        "edge_index": [torch.cat(step_edges, 1) for step_edges in edge_index],
        "edge_attr": None if edge_attr is None else [torch.cat(step_attrs) for step_attrs in edge_attr],
    }


def to_device(batch: dict, device: torch.device) -> dict:
    """ Move every tensor of the batch, including those in per-step lists, to the device. """
    move = lambda value: value.to(device) if torch.is_tensor(value) else value
    return {k: [move(element) for element in v] if isinstance(v, list) else move(v) for k, v in batch.items()}


def create_dataloader(dataset: GraphWindowDataset, indices: list, batch_size: int, shuffle: bool,
                      num_workers: int = 0) -> DataLoader:
    """ Create a dataloader over the given windows of the dataset, batched with collate. """
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate, num_workers=num_workers)


# Normalization Statistics

def _compute_mean_std(rows: list, num_channels: int) -> tuple:
    """ Compute the per-channel mean and std of the rows (0 and 1 if empty, std 1 for constant channels). """
    if not rows or sum(len(row_block) for row_block in rows) == 0:
        return torch.zeros(num_channels), torch.ones(num_channels)
    rows = torch.cat(rows)
    std = rows.std(0)
    return rows.mean(0), torch.where(std < 1e-6, torch.ones_like(std), std)


def compute_stats(dataset: GraphWindowDataset, indices: list, dynamic_dims: list) -> dict:
    """ Compute per-channel normalization statistics from the training windows only (valid entries only). """
    # Valid feature, velocity and edge rows
    feature_rows, velocity_rows, edge_rows = [], [], []
    for i in indices:
        window = dataset[i]
        mask = window["node_mask"]
        feature_rows.append(window["x"][mask])
        dynamic_features = window["x"][..., dynamic_dims]
        velocity_rows.append((dynamic_features[1:] - dynamic_features[:-1])[mask[1:] & mask[:-1]])
        if window["edge_attr"] is not None:
            edge_rows.extend(window["edge_attr"])
    # Mean and std per channel
    stats_dict = {}
    stats_dict["x_mean"], stats_dict["x_std"] = _compute_mean_std(feature_rows, dataset.feat_dim)
    stats_dict["v_mean"], stats_dict["v_std"] = _compute_mean_std(velocity_rows, len(dynamic_dims))
    stats_dict["e_mean"], stats_dict["e_std"] = _compute_mean_std(edge_rows, dataset.edge_dim)
    return stats_dict


# Splits

def create_splits(labels: np.ndarray, group_ids: np.ndarray, num_folds: int = 4, test_fraction: float = 0.2,
                  val_fraction: float = 0.1, seed: int = 42, fixed_test_mask: Optional[np.ndarray] = None) -> dict:
    """ Create group-level splits stratified by label: a test set, `num_folds` CV folds and a final train/val split. """
    rng = np.random.default_rng(seed)
    # Majority label per group: labels only balance the strata, they never reach training
    unique, inverse = np.unique(group_ids, return_inverse=True)
    group_labels = np.array([np.bincount(labels[inverse == i] + 1).argmax() - 1 for i in range(len(unique))])

    def sample_stratified_groups(group_idx: np.ndarray, fraction: float) -> np.ndarray:
        """ Sample the given fraction of the groups within each label stratum. """
        chosen = [rng.permutation(group_idx[group_labels[group_idx] == label])[: round(
                      fraction * (group_labels[group_idx] == label).sum())]
                  for label in np.unique(group_labels[group_idx])]
        return np.sort(np.concatenate(chosen)).astype(int)

    def get_window_indices(group_idx: np.ndarray) -> list:
        """ Retrieve the indices of the windows that belong to the given groups. """
        return np.flatnonzero(np.isin(inverse, group_idx)).tolist()

    # Test set, drawn or made of the windows flagged in fixed_test_mask
    if fixed_test_mask is not None and fixed_test_mask.any():
        test = np.unique(inverse[fixed_test_mask])
    else:
        test = sample_stratified_groups(np.arange(len(unique)), test_fraction)
    # CV folds on the remaining pool, balanced per label
    pool = np.setdiff1d(np.arange(len(unique)), test)
    fold_of, offset = np.zeros(len(unique), dtype=int), 0
    for label in np.unique(group_labels[pool]):
        members = rng.permutation(pool[group_labels[pool] == label])
        fold_of[members] = (np.arange(len(members)) + offset) % num_folds
        offset += len(members)
    # Train/val split of the pool for the final run
    val = sample_stratified_groups(pool, val_fraction)
    return {
        "test": get_window_indices(test),
        "folds": [{"train": get_window_indices(pool[fold_of[pool] != k]),
                   "val": get_window_indices(pool[fold_of[pool] == k])} for k in range(num_folds)],
        "final": {"train": get_window_indices(np.setdiff1d(pool, val)), "val": get_window_indices(val)},
    }


def prepare_data(config: Config, out_dir=None) -> tuple:
    """ Load the dataset, check its window length and create its splits, saved as JSON in out_dir if given. """
    # Load and check windows
    dataset = GraphWindowDataset(config.data.root, config.data.max_files)
    if dataset.num_steps != config.data.context + config.data.horizon:
        raise ValueError(f"windows have {dataset.num_steps} steps, expected C + H = "
                         f"{config.data.context + config.data.horizon}")
    # Create splits
    data_config = config.data
    splits = create_splits(dataset.get_labels(), dataset.get_group_ids(), data_config.folds, data_config.test_fraction,
                           data_config.val_fraction, data_config.split_seed, dataset.get_fixed_test_mask())
    # Save splits
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "splits.json").write_text(json.dumps(splits))
    return dataset, splits
