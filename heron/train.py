""" Training and evaluation of HERON (paper Sec. 4, App. 10.2).

    Training is label-free: early stopping and checkpoint selection use the validation loss; regime labels are only
    read by `evaluate` to report recovery metrics.

"""
from collections import defaultdict
import json
import numpy as np
from pathlib import Path
import random
import torch
# Custom Library
from .config import Config, config_to_dict, dict_to_config
from .data import GraphWindowDataset, compute_stats, create_dataloader, to_device
from .losses import HERONLoss, gaussian_nll
from .metrics import compute_classification_metrics, compute_displacement_errors, compute_regime_mapping
from .model import HERON


# Setup

def set_seed(seed: int):
    """ Set the seed of the python, numpy and torch random generators. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(device_name: str) -> torch.device:
    """ Retrieve the torch device, where `auto` selects cuda when available. """
    return torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else device_name)


def create_model(config: Config, dataset: GraphWindowDataset) -> HERON:
    """ Create an instance of the model defined in the config, sized for the dataset. """
    dynamic_dims = config.data.dynamic_dims or list(range(dataset.feat_dim))  # all features when empty
    return HERON(config.model, dataset.feat_dim, dynamic_dims, config.data.context, dataset.edge_dim)


def get_temperature(config: Config, epoch: int) -> float:
    """ Retrieve the Gumbel-softmax temperature, annealed linearly from tau_max to tau_min (App. 10.1.1). """
    span = max(1, round(config.train.tau_anneal * config.train.epochs))  # first `tau_anneal` fraction of training
    return config.train.tau_max + min(1.0, epoch / span) * (config.train.tau_min - config.train.tau_max)


# Training

def train_one_epoch(model: HERON, criterion: HERONLoss, dataloader, device: torch.device, tau: float, optimizer=None,
                    grad_clip: float = 1.0, max_batches: int = 0) -> dict:
    """ Perform one pass over the dataloader, with gradient updates when an optimizer is given. """
    # Set training mode
    training = optimizer is not None
    model.train(training)
    criterion.train(training)
    log_sums, num_batches = defaultdict(float), 0
    with torch.set_grad_enabled(training):
        for batch_id, batch in enumerate(dataloader):
            if max_batches and batch_id >= max_batches:
                break
            # Forward pass
            loss, logs = criterion(model(to_device(batch, device), tau))
            # Update using the gradients
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            # Accumulate loss logs
            for key, value in logs.items():
                log_sums[key] += value
            num_batches += 1
    return {key: value / max(num_batches, 1) for key, value in log_sums.items()}


def save_model(checkpoint_path: Path, model: HERON, criterion: HERONLoss, config: Config):
    """ Save the model and loss states with the metadata and config needed to rebuild the model. """
    torch.save({"model": model.state_dict(), "loss": criterion.state_dict(), "meta": model.meta_dict,
                "config": config_to_dict(config)}, checkpoint_path)


def load_model(checkpoint_path: Path, device: str = "cpu") -> tuple:
    """ Rebuild a trained HERON with its normalization statistics, without the training data. """
    # Create model from checkpoint metadata
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config, meta_dict = dict_to_config(checkpoint["config"]), checkpoint["meta"]
    model = HERON(config.model, meta_dict["feat_dim"], meta_dict["dynamic_dims"], meta_dict["context"],
                  meta_dict["edge_dim"])
    # Load weights
    legacy_prefix = "entity_gru."  # checkpoints written before the backbone interface
    model.load_state_dict({"backbone.gru." + k[len(legacy_prefix):] if k.startswith(legacy_prefix) else k: v
                           for k, v in checkpoint["model"].items()})
    return model.to(device), config


def train_model(config: Config, dataset: GraphWindowDataset, train_idx: list, val_idx: list, out_dir: Path) -> HERON:
    """ Perform training on `train_idx`, keeping the epoch with the lowest validation loss. """
    # Initialization
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.train.seed)
    device = get_device(config.train.device)
    # Create model and loss
    model = create_model(config, dataset)
    model.set_normalizer(compute_stats(dataset, train_idx, model.meta_dict["dynamic_dims"]))
    model.to(device)
    criterion = HERONLoss(config.loss, config.model.num_regimes).to(device)
    # Optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.train.lr_step, config.train.lr_gamma)
    # Dataloaders
    data_config = config.data
    train_dataloader = create_dataloader(dataset, train_idx, data_config.batch_size, True, data_config.num_workers)
    val_dataloader = create_dataloader(dataset, val_idx, data_config.batch_size, False, data_config.num_workers)
    # Training state
    rng_devices = [device.index or 0] if device.type == "cuda" else []
    best_val_loss, wait, history = float("inf"), 0, []
    # Model training
    for epoch in range(config.train.epochs):
        tau = get_temperature(config, epoch)
        train_logs = train_one_epoch(model, criterion, train_dataloader, device, tau, optimizer,
                                     config.train.grad_clip, config.train.max_batches)
        # Validation pass (tau_min, fixed seed)
        with torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(config.train.seed)
            val_logs = train_one_epoch(model, criterion, val_dataloader, device, config.train.tau_min,
                                       max_batches=config.train.max_batches)
        scheduler.step()
        # Register logs
        history.append({"epoch": epoch, "tau": tau, "train": train_logs, "val": val_logs})
        print(json.dumps({"epoch": epoch, "tau": round(tau, 3), "train": round(train_logs["total"], 4),
                          "val": round(val_logs["total"], 4), "usage_min": round(val_logs["usage_min"], 3)}),
              flush=True)
        # Checkpoint and early stopping
        if val_logs["total"] < best_val_loss:
            best_val_loss, wait = val_logs["total"], 0
            save_model(out_dir / "best.pt", model, criterion, config)
        else:
            wait += 1
            if wait >= config.train.patience:
                break
    # Save history, reload best model
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device, weights_only=False)["model"])
    return model


# Evaluation

@torch.no_grad()
def evaluate(model: HERON, dataset: GraphWindowDataset, indices: list, config: Config) -> tuple:
    """ Perform evaluation of the autonomous rollout and of the per-step posterior regime probabilities. """
    # Initialization
    model.eval()
    set_seed(config.train.seed)
    device = next(model.parameters()).device
    C, tau = config.data.context, config.train.tau_min
    dynamic_dims = model.meta_dict["dynamic_dims"]
    pos_idx = [dynamic_dims.index(dim) for dim in (config.data.pos_dims or dynamic_dims)]
    ade, fde, min_fde, horizon_labels, probs, labels = [], [], [], [], [], []
    nll_sum, nll_count = 0.0, 0.0
    for batch in create_dataloader(dataset, indices, config.data.batch_size, False, config.data.num_workers):
        batch = to_device(batch, device)
        true = batch["x"][:, C:][..., model.dyn_index][..., pos_idx]  # [B, H, N, P]
        mask = batch["mask"][:, C:] & batch["mask"][:, C - 1:C]  # [B, H, N]
        # ADE and FDE of the mean rollout
        mean_forecast = model.forecast(batch, "mean", tau)
        batch_ade, batch_fde, valid = compute_displacement_errors(mean_forecast.positions[..., pos_idx], true, mask)
        # FDE of sampled rollouts (minFDE)
        sampled_fde = torch.stack(
            [compute_displacement_errors(model.forecast(batch, "sample", tau).positions[..., pos_idx], true, mask)[1]
             for _ in range(config.eval.num_samples)])
        # Accumulate valid windows
        ade.append(batch_ade[valid])
        fde.append(batch_fde[valid])
        min_fde.append(sampled_fde[:, valid].min(0).values)
        horizon_labels.append(batch["y"][:, C][valid])
        # Horizon NLL of the mean rollout
        _, target, target_mask, _ = model.get_inputs_and_targets(batch)
        nll = gaussian_nll(mean_forecast.velocity_mean, mean_forecast.velocity_logvar, target[:, C - 1:-1])
        nll_sum += (nll * target_mask[:, C - 1:-1]).sum().item()
        nll_count += target_mask[:, C - 1:-1].sum().item()
        # Posterior regimes of labeled steps
        context_labels = batch["y"][:, :C]
        keep = (context_labels >= 0) & batch["mask"][:, :C].any(-1)
        probs.append(mean_forecast.context_regimes[keep].cpu().numpy())
        labels.append(context_labels[keep].cpu().numpy())
    # Forecasting metrics
    to_numpy = lambda tensor_list: torch.cat(tensor_list).cpu().numpy()
    ade, horizon_labels = to_numpy(ade), to_numpy(horizon_labels)
    metric_dict = {"ade": float(np.mean(ade)), "fde": float(np.nanmean(to_numpy(fde))),
                   "min_fde": float(np.nanmean(to_numpy(min_fde))), "horizon_nll": nll_sum / max(nll_count, 1.0)}
    # ADE per true horizon regime (reporting only)
    for k in np.unique(horizon_labels[horizon_labels >= 0]):
        metric_dict[f"ade_regime_{k}"] = float(ade[horizon_labels == k].mean())
    return metric_dict, (np.concatenate(probs), np.concatenate(labels))  # probs [n, K], labels [n]


def compute_regime_metrics(fit_regimes: tuple, eval_regimes: tuple, num_regimes: int) -> tuple:
    """ Compute the regime scores of `eval_regimes` with the mapping frozen on `fit_regimes` (validation). """
    if len(fit_regimes[1]) == 0 or len(eval_regimes[1]) == 0:
        return {}, None
    mapping = compute_regime_mapping(fit_regimes[0].argmax(1), fit_regimes[1], num_regimes)
    return compute_classification_metrics(eval_regimes[0], eval_regimes[1], mapping), mapping
