"""CPU smoke test and invariant checks on a tiny toy dataset (about a minute).

    PYTHONPATH=. python scripts/smoke_test.py [--synthetic /path/to/paper/synthetic/files]
"""
import argparse
import numpy as np
from pathlib import Path
import sys
import tempfile
import torch
# Custom Library
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))  # makes toy_dataset importable
from heron.config import Config, apply_overrides, config_to_dict, dict_to_config, load_config  # noqa: E402
from heron.data import GraphWindowDataset, collate, create_splits, prepare_data  # noqa: E402
from heron.losses import HERONLoss, balance_kl, gaussian_kl  # noqa: E402
from heron.metrics import compute_classification_metrics, compute_regime_mapping  # noqa: E402
from heron.model import HERON  # noqa: E402
from heron.search import run_trial, sample_trial, summarize_trials  # noqa: E402
from heron.train import compute_regime_metrics, create_model, evaluate, load_model, train_model  # noqa: E402
from toy_dataset import generate_dataset  # noqa: E402


# Constants
TINY_OVERRIDES = ["data.context=4", "data.horizon=4", "data.dynamic_dims=[0, 1]", "data.batch_size=8",
                  "data.num_workers=0", "model.num_regimes=2", "model.entity_embedding_dim=8",
                  "model.encoder_hidden_dim=8", "model.latent_dim=4", "model.entity_hidden_dim=8",
                  "model.system_hidden_dim=8", "model.regime_hidden_dim=8", "model.expert_hidden_dim=8",
                  "model.decoder_hidden_dim=8", "train.epochs=2", "train.max_batches=2", "eval.num_samples=3",
                  "train.device=cpu"]


# Helper Functions

def validate_condition(name: str, condition: bool):
    """ Print the result of a named check and exit with an error if it failed. """
    print(f"[{'ok' if condition else 'FAIL'}] {name}")
    if not condition:
        raise SystemExit(1)


def is_future_ignored(model: HERON, batch: dict, C: int) -> bool:
    """ Check whether the forecast is unchanged when the frames after the context are perturbed. """
    # Perturb future frames
    future_batch = {**batch, "x": batch["x"].clone(), "mask": batch["mask"].clone()}
    future_batch["x"][:, C:] = torch.randn_like(future_batch["x"][:, C:])
    future_batch["mask"][:, C:] = ~future_batch["mask"][:, C:]
    # Drop future edges
    future_batch["edge_index"] = batch["edge_index"][:C] + [edges[:, :0] for edges in batch["edge_index"][C:]]
    if batch["edge_attr"] is not None:
        future_batch["edge_attr"] = batch["edge_attr"][:C] + [edge_attr[:0] for edge_attr in batch["edge_attr"][C:]]
    # Compare mean forecasts
    return torch.allclose(model.forecast(batch, "mean").positions, model.forecast(future_batch, "mean").positions)


# Smoke Tests

def validate_invariants(config: Config, dataset: GraphWindowDataset):
    """ Validate model, loss, metric, config, search and split invariants on the toy dataset. """
    # Forward and backward pass
    model = create_model(config, dataset).eval()
    batch = collate([dataset[i] for i in range(4)])
    C = config.data.context
    outputs = model(batch, tau=1.0)
    loss, logs = HERONLoss(config.loss, config.model.num_regimes)(outputs)
    loss.backward()
    validate_condition("forward/backward finite",
                       torch.isfinite(loss).item() and all(np.isfinite(v) for v in logs.values()))
    # Causal forecast
    validate_condition("forecast ignores future frames", is_future_ignored(model, batch, C))
    # Regime clamping
    forecast_0, forecast_1 = (model.forecast(batch, "mean", clamp_regime=0),
                              model.forecast(batch, "mean", clamp_regime=1))
    validate_condition("regime clamping changes the rollout",
                       not torch.allclose(forecast_0.positions, forecast_1.positions))
    validate_condition("clamped regimes are respected",
                       bool((forecast_0.horizon_regimes[:, 1:].argmax(-1) == 0).all()))
    # Soft routing
    config.model.routing = "soft"
    validate_condition("soft routing runs",
                       torch.isfinite(create_model(config, dataset)(batch).posterior_mean).all().item())
    config.model.routing = "hard"
    # Gaussian KL against torch.distributions
    posterior_mu, posterior_logvar, prior_mu, prior_logvar = torch.randn(3, 4, 5).unbind(1)
    reference_kl = torch.distributions.kl_divergence(
        torch.distributions.Normal(posterior_mu, (0.5 * posterior_logvar).exp()),
        torch.distributions.Normal(prior_mu, (0.5 * prior_logvar).exp())).sum(-1)
    validate_condition("Gaussian KL matches torch.distributions", torch.allclose(
        gaussian_kl(posterior_mu, posterior_logvar, prior_mu, prior_logvar), reference_kl, atol=1e-5))
    # Balance loss at uniform usage
    uniform = torch.full((6, 3, 2), 0.5)
    validate_condition("balance loss is zero at uniform usage", balance_kl(
        uniform, torch.ones(6, 3, dtype=torch.bool), torch.full((2,), 0.5)).abs().item() < 1e-6)
    # Regime alignment transfer
    labels = np.repeat([0, 1, 2], 20)
    probs = np.eye(3)[(labels + 1) % 3] * 0.8 + 0.1     # regime k predicts label k - 1
    mapping = compute_regime_mapping(probs.argmax(1), labels, 3)
    validate_condition("alignment fit on validation transfers to test",
                       compute_classification_metrics(probs[::-1], labels[::-1], mapping)["macro_f1"] == 1.0)
    # One-step horizon
    one_step_config = dict_to_config(config_to_dict(config))
    one_step_config.data.context, one_step_config.data.horizon = 7, 1
    validate_condition("forecast with H = 1",
                       create_model(one_step_config, dataset).forecast(batch).positions.shape[1] == 1)
    # Grid search enumeration
    space = {"a": [1, 2], "b": [3, 4, 5]}
    grid = [sample_trial(space, i, "grid") for i in range(7)]
    validate_condition("grid search enumerates every combination",
                       grid[-1] is None and len({tuple(trial.values()) for trial in grid[:6]}) == 6)
    # Fixed test partition
    pinned = np.zeros(len(dataset), dtype=bool)
    pinned[:6] = True
    splits = create_splits(dataset.get_labels(), dataset.get_group_ids(), fixed_test_mask=pinned)
    validate_condition("a fixed test partition is honoured", set(range(6)) <= set(splits["test"])
                       and not set(range(6)) & set(splits["folds"][0]["train"] + splits["folds"][0]["val"]))


def validate_pipeline(config: Config, dataset: GraphWindowDataset, splits: dict, out_dir: Path):
    """ Validate training, evaluation, checkpoint reload, the learned balance prior and one search trial. """
    # Train and evaluate one fold
    fold = splits["folds"][0]
    model = train_model(config, dataset, fold["train"], fold["val"], out_dir / "fit")
    metrics, regimes = evaluate(model, dataset, fold["val"], config)
    metrics.update(compute_regime_metrics(regimes, regimes, config.model.num_regimes)[0])
    validate_condition(f"fit + evaluate {metrics}", all(np.isfinite(v) for v in metrics.values()))
    # Checkpoint reload
    reloaded, _ = load_model(out_dir / "fit" / "best.pt")
    validate_condition("checkpoint reload reproduces the forecast", torch.allclose(
        reloaded.forecast(collate([dataset[0]]), "mean").positions,
        model.forecast(collate([dataset[0]]), "mean").positions))
    # Learned balance prior
    config.loss.balance_target = "learned"
    train_model(config, dataset, fold["train"], fold["val"], out_dir / "learned")
    config.loss.balance_target = "uniform"
    usage_prior = torch.load(out_dir / "learned" / "best.pt", weights_only=False)["loss"]["usage_prior"]
    validate_condition(f"learned balance prior moves from uniform {usage_prior.tolist()}",
                       not torch.allclose(usage_prior, torch.full_like(usage_prior, 0.5)))
    # Search trial and summary
    trial = sample_trial({"train.lr": {"loguniform": [1e-4, 1e-3]}, "loss.entropy_weight": [0.1, 0.2]}, 0, seed=1)
    run_trial(config, dataset, {"folds": splits["folds"][:2]}, trial, out_dir / "search" / "trial_000")
    validate_condition("search trial + summary", summarize_trials(out_dir / "search")[0]["trial"] == "trial_000")


def validate_plugin_interface(config: Config, dataset: GraphWindowDataset, splits: dict, out_dir: Path):
    """ Validate that pre-interface checkpoints load and that the example backbone runs through the config. """
    # Toy batch
    batch = collate([dataset[i] for i in range(4)])
    # Checkpoint with pre-interface keys
    checkpoint = torch.load(out_dir / "fit" / "best.pt", weights_only=False)
    checkpoint["model"] = {k.replace("backbone.gru.", "entity_gru."): v for k, v in checkpoint["model"].items()}
    torch.save(checkpoint, out_dir / "legacy.pt")
    validate_condition("checkpoints with pre-interface keys load", torch.equal(
        load_model(out_dir / "legacy.pt")[0].forecast(batch).positions,
        load_model(out_dir / "fit" / "best.pt")[0].forecast(batch).positions))
    # Plug-in config overrides
    override_dict = {"model.backbone.hops": 2,
                     "model.backbone.class": "examples.graph_gru:GraphGRU"}  # class is applied first
    config = apply_overrides(dict_to_config(config_to_dict(config)), override_dict)
    backbone_dict = apply_overrides(dict_to_config(config_to_dict(config)),
                                    {"model.backbone.class": "gru"}).model.backbone
    validate_condition("overriding a plug-in class replaces its kwargs",
                       backbone_dict == {"class": "gru"} and config.model.backbone["hops"] == 2)
    # Forward and backward pass
    model = create_model(config, dataset).eval()
    loss, _ = HERONLoss(config.loss, config.model.num_regimes)(model(batch))
    loss.backward()
    validate_condition("plug-in backbone: forward/backward finite", torch.isfinite(loss).item())
    validate_condition("plug-in backbone: forecast ignores future frames",
                       is_future_ignored(model, batch, config.data.context))
    # Train and reload
    fold = splits["folds"][0]
    model = train_model(config, dataset, fold["train"], fold["val"], out_dir / "plugin")
    reloaded, _ = load_model(out_dir / "plugin" / "best.pt")
    validate_condition("plug-in backbone: trains, and reloads from its config",
                       type(reloaded.backbone).__name__ == "GraphGRU"
                       and torch.allclose(reloaded.forecast(batch).positions, model.cpu().forecast(batch).positions))


# Main

def main():
    """ Run every check on a temporary toy dataset, then optionally on the paper synthetic files. """
    # Parse arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", help="directory of the paper's synthetic files (optional)")
    args = parser.parse_args()
    # Global seed
    torch.manual_seed(0)
    # Toy dataset checks
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = generate_dataset(Path(tmp_dir) / "toy", num_windows=48, num_steps=8)
        config = load_config(None, TINY_OVERRIDES + [f"data.root={root}", f"out_dir={tmp_dir}/run"])
        dataset, splits = prepare_data(config, Path(tmp_dir) / "run")
        validate_condition("splits are disjoint and grouped",
                           not set(splits["test"]) & set(splits["folds"][0]["train"])
                           and len({int(dataset[i]["seq_id"]) for i in splits["test"]}
                                   & {int(dataset[i]["seq_id"]) for i in splits["folds"][0]["train"]}) == 0)
        validate_invariants(config, dataset)
        validate_pipeline(config, dataset, splits, Path(tmp_dir) / "run")
        validate_plugin_interface(config, dataset, splits, Path(tmp_dir) / "run")
    # Paper synthetic files
    if args.synthetic:
        config = load_config(None, TINY_OVERRIDES + ["data.context=5", "data.horizon=5", "data.dynamic_dims=[]",
                                                     "model.num_regimes=3", f"data.root={args.synthetic}",
                                                     "data.max_files=4"])
        synthetic = GraphWindowDataset(config.data.root, config.data.max_files)
        outputs = create_model(config, synthetic)(collate([synthetic[i] for i in range(4)]))
        validate_condition(f"paper synthetic files load (feat_dim={synthetic.feat_dim})",
                           torch.isfinite(outputs.posterior_mean).all().item())
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
