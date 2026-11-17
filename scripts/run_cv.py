"""k-fold cross-validation on the training pool: train on k-1 folds, early-stop and evaluate on the held-out fold.

    python scripts/run_cv.py --config configs/network.yaml data.root=/path/to/windows

Regime metrics of a fold are aligned on that same fold (its only held-out data); see run_test.py for the
validation-to-test protocol.
"""
import argparse
import json
import numpy as np
from pathlib import Path
# Custom Library
from heron.config import load_config, save_config
from heron.data import prepare_data
from heron.train import compute_regime_metrics, evaluate, train_model


def main():
    """ Perform k-fold cross-validation and save the metrics of every fold and their summary. """
    # Parse arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("overrides", nargs="*", help="section.key=value")
    args = parser.parse_args()
    # Load config and data
    config = load_config(args.config, args.overrides)
    out_dir = Path(config.out_dir) / "cv"
    dataset, splits = prepare_data(config, out_dir)
    save_config(config, out_dir / "config.yaml")
    # Train and evaluate every fold
    fold_metric_list = []
    for fold_id, fold in enumerate(splits["folds"]):
        model = train_model(config, dataset, fold["train"], fold["val"], out_dir / f"fold_{fold_id}")
        metrics, regimes = evaluate(model, dataset, fold["val"], config)
        # Regime alignment on this fold
        metrics.update(compute_regime_metrics(regimes, regimes, config.model.num_regimes)[0])
        # Register metrics
        (out_dir / f"fold_{fold_id}" / "metrics.json").write_text(json.dumps(metrics, indent=1))
        print(json.dumps({"fold": fold_id} | metrics), flush=True)
        fold_metric_list.append(metrics)
    # Summarize across folds
    summary = {}
    for key in dict.fromkeys(k for fold_metrics in fold_metric_list for k in fold_metrics):
        values = np.array([fold_metrics.get(key, np.nan) for fold_metrics in fold_metric_list])
        summary[key] = {"mean": float(np.nanmean(values)),
                        "std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
