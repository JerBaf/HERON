"""Final run: train on the training pool (minus an internal validation split), evaluate on the held-out test set.

    python scripts/run_test.py --config configs/network.yaml data.root=/path/to/windows

The regime alignment (Hungarian) is fit on the internal validation split and applied unchanged to the test set.
"""
import argparse
import json
from pathlib import Path
# Custom Library
from heron.config import load_config, save_config
from heron.data import prepare_data
from heron.train import compute_regime_metrics, evaluate, train_model


def main():
    """ Perform the final training and evaluate the model once on the held-out test set. """
    # Parse arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("overrides", nargs="*", help="section.key=value")
    args = parser.parse_args()
    # Load config and data
    config = load_config(args.config, args.overrides)
    out_dir = Path(config.out_dir) / "test"
    dataset, splits = prepare_data(config, out_dir)
    save_config(config, out_dir / "config.yaml")
    # Final training
    model = train_model(config, dataset, splits["final"]["train"], splits["final"]["val"], out_dir)
    # Evaluate on validation and test
    _, val_regimes = evaluate(model, dataset, splits["final"]["val"], config)
    metrics, test_regimes = evaluate(model, dataset, splits["test"], config)
    # Regime alignment fit on validation
    scores, mapping = compute_regime_metrics(val_regimes, test_regimes, config.model.num_regimes)
    metrics.update(scores)
    metrics["regime_mapping"] = None if mapping is None else mapping.tolist()
    # Register metrics
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
