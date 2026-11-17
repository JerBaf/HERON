"""Random or grid hyper-parameter search with k-fold cross-validation on the training pool.

Trials are ranked by the mean validation horizon NLL over folds (label-free). Trials are indexed, so several
jobs can run disjoint ranges into the same output directory; finished trials are skipped.

    python -m heron.search --config configs/network.yaml --space configs/search.yaml --out runs/search --trials 0-7
    python -m heron.search --out runs/search --summarize
"""
import argparse
import json
import math
import numpy as np
from pathlib import Path
import random
import yaml
# Custom Library
from .config import Config, apply_overrides, config_to_dict, dict_to_config, load_config
from .data import GraphWindowDataset, prepare_data
from .train import evaluate, train_model


# Search Space

def sample_trial(space: dict, trial_id: int, mode: str = "random", seed: int = 0) -> dict:
    """ Sample the overrides of trial `trial_id`, enumerating the product of the value lists in grid mode. """
    keys = sorted(space)
    # Grid mode (lists only)
    if mode == "grid":
        if not all(isinstance(space[k], list) for k in keys):
            raise ValueError("grid mode needs a list of values for every key")
        if trial_id >= get_grid_size(space):
            return None  # past the last combination
        trial = {}
        for key in reversed(keys):  # mixed-radix decoding of the trial index
            trial_id, j = divmod(trial_id, len(space[key]))
            trial[key] = space[key][j]
        return trial
    # Random mode
    rng, trial = random.Random(seed + trial_id), {}
    for key in keys:
        spec = space[key]
        if isinstance(spec, list):  # set of choices
            trial[key] = rng.choice(spec)
        elif "loguniform" in spec:  # range {loguniform: [a, b]}
            low, high = map(float, spec["loguniform"])
            trial[key] = math.exp(rng.uniform(math.log(low), math.log(high)))
        else:  # range {uniform: [a, b]}
            trial[key] = rng.uniform(*map(float, spec["uniform"]))
    return trial


def get_grid_size(space: dict) -> int:
    """ Retrieve the number of value combinations of the grid search space. """
    return math.prod(len(values) for values in space.values())


# Trials

def run_trial(config: Config, dataset: GraphWindowDataset, splits: dict, override_dict: dict, out_dir: Path) -> dict:
    """ Perform the k-fold cross-validation of one trial and save its result in `out_dir`. """
    config = apply_overrides(dict_to_config(config_to_dict(config)), override_dict)
    # Cross-validation
    fold_metrics = []
    for fold_id, fold in enumerate(splits["folds"]):
        model = train_model(config, dataset, fold["train"], fold["val"], Path(out_dir) / f"fold_{fold_id}")
        fold_metrics.append(evaluate(model, dataset, fold["val"], config)[0])
    # Save result
    result = {"overrides": override_dict, "config": config_to_dict(config), "folds": fold_metrics,
              "horizon_nll": float(np.mean([metrics["horizon_nll"] for metrics in fold_metrics])),
              "ade": float(np.mean([metrics["ade"] for metrics in fold_metrics]))}
    (Path(out_dir) / "result.json").write_text(json.dumps(result, indent=1))
    return result


def summarize_trials(out_dir: Path) -> list:
    """ Rank the finished trials by validation horizon NLL, then save the ranking and the best config. """
    # Rank trials
    results = [json.loads(result_path.read_text()) | {"trial": result_path.parent.name}
               for result_path in sorted(Path(out_dir).glob("trial_*/result.json"))]
    results.sort(key=lambda result: result["horizon_nll"])
    ranking = [{key: result[key] for key in ("trial", "horizon_nll", "ade", "overrides")} for result in results]
    # Save ranking and best config
    (Path(out_dir) / "ranking.json").write_text(json.dumps(ranking, indent=1))
    with open(Path(out_dir) / "best.yaml", "w") as config_file:
        yaml.safe_dump(results[0]["config"], config_file, sort_keys=False)
    return ranking


# Command Line

def main():
    """ Run the hyper-parameter search over a range of trials, or summarize the finished trials. """
    # Parse arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("--space", help="search space YAML: {mode, num_trials, seed, space}")
    parser.add_argument("--out", required=True)
    parser.add_argument("--trials", help="range a-b (default: all)")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("overrides", nargs="*", help="section.key=value applied to the base config")
    args = parser.parse_args()
    # Summary only
    if args.summarize:
        for row in summarize_trials(args.out)[:5]:
            print(json.dumps(row))
        return
    # Load space, config and data
    with open(args.space) as space_file:
        search_dict = yaml.safe_load(space_file)
    config = load_config(args.config, args.overrides)
    dataset, splits = prepare_data(config, args.out)
    # Trial range
    mode = search_dict.get("mode", "random")
    num_trials = get_grid_size(search_dict["space"]) if mode == "grid" else search_dict["num_trials"]
    first, last = map(int, args.trials.split("-")) if args.trials else (0, num_trials - 1)
    # Run trials
    for trial_id in range(first, last + 1):
        trial_dir = Path(args.out) / f"trial_{trial_id:03d}"
        override_dict = sample_trial(search_dict["space"], trial_id, mode, search_dict.get("seed", 0))
        if override_dict is None or (trial_dir / "result.json").exists():
            continue
        result = run_trial(config, dataset, splits, override_dict, trial_dir)
        print(json.dumps({"trial": trial_id, "horizon_nll": result["horizon_nll"], "ade": result["ade"]}), flush=True)


if __name__ == "__main__":
    main()
