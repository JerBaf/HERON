""" Typed configuration: defaults are the paper's synthetic setting (Table 20) and YAML files override them. """
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Optional, Sequence
import yaml


# Config Sections

@dataclass
class DataConfig:
    """ Dataset location, window lengths, loading and split settings. """
    root: str = ""                 # Directory of window files (see heron/data.py)
    context: int = 5               # C observed steps
    horizon: int = 5               # H forecast steps
    dynamic_dims: list = field(default_factory=list)  # Node features that evolve and are predicted ([] = all)
    pos_dims: list = field(default_factory=list)      # Subset of dynamic_dims scored by ADE/FDE ([] = all dynamic)
    batch_size: int = 256
    num_workers: int = 4
    folds: int = 4                 # CV folds on the training pool
    test_fraction: float = 0.2
    val_fraction: float = 0.1      # Internal validation split of the final (test) run
    split_seed: int = 42
    max_files: int = 0             # Load only the first files (debugging); 0 = all


@dataclass
class ModelConfig:
    """ Model sizes, regime routing and plug-in modules. """
    num_regimes: int = 3           # K
    entity_embedding_dim: int = 32
    encoder_hidden_dim: int = 48
    encoder_layers: int = 2
    entity_hidden_dim: int = 32
    system_hidden_dim: int = 32
    latent_dim: int = 8
    regime_hidden_dim: int = 48
    expert_hidden_dim: int = 16
    decoder_hidden_dim: int = 64
    # Regime routing (App. 10.5.4), hard: one-hot straight-through gate | soft: posterior-weighted experts
    routing: str = "hard"
    # Plug-ins (heron/backbone.py): {"class": short name or "package.module:Class", **constructor kwargs}
    backbone: dict = field(default_factory=lambda: {"class": "gru"})   # Entity-level transition
    encoder: dict = field(default_factory=lambda: {"class": "gine"})   # Relational encoder of both paths


@dataclass
class LossConfig:
    """ Loss weights and regime balance target. """
    prediction_weight: float = 5.0
    prior_prediction_weight: float = 0.5   # gamma
    dynamics_kl_weight: float = 0.8
    representation_kl_weight: float = 0.2
    structural_weight: float = 2.0
    entropy_weight: float = 0.2
    orthogonality_weight: float = 0.05
    balance_target: str = "uniform"        # uniform | learned (slow EMA of regime usage, App. 10.5.3)
    prior_momentum: float = 0.99


@dataclass
class TrainConfig:
    """ Optimization, temperature annealing and early stopping settings. """
    epochs: int = 30
    lr: float = 8.42e-4
    weight_decay: float = 1e-4
    lr_step: int = 10
    lr_gamma: float = 0.5
    grad_clip: float = 1.0
    tau_max: float = 1.2
    tau_min: float = 0.5
    tau_anneal: float = 0.6        # Fraction of the epochs over which tau decays linearly
    patience: int = 8              # Early stopping on the validation loss
    seed: int = 42
    device: str = "auto"
    max_batches: int = 0           # Per epoch, 0 = all (debugging)


@dataclass
class EvalConfig:
    """ Evaluation settings. """
    num_samples: int = 20          # M stochastic rollouts for minFDE


# Main Config

@dataclass
class Config:
    """ Full configuration, one section per YAML block. """
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    out_dir: str = "runs/default"


# Updates and Overrides

def _cast_value(current, value):
    """ Cast a scalar to the type of its default, since PyYAML reads "1e-3" as a string. """
    if isinstance(current, float) and isinstance(value, (int, str)):
        return float(value)
    if isinstance(current, int) and isinstance(value, str):
        return int(value)
    return value


def _update_config(config, values_dict: dict, prefix: str = ""):
    """ Update a config dataclass in place from a nested dict, recursing into its sections. """
    field_names = {config_field.name for config_field in fields(config)}
    for key, value in values_dict.items():
        # Check keys
        if key not in field_names:
            raise KeyError(f"unknown config key: {prefix}{key}")
        current = getattr(config, key)
        if is_dataclass(current):
            _update_config(current, value, f"{prefix}{key}.")
        elif isinstance(current, dict) and isinstance(value, dict):
            # Plug-in spec: a new class replaces it, otherwise kwargs are merged
            same_class = value.get("class", current["class"]) == current["class"]
            setattr(config, key, {**current, **value} if same_class else dict(value))
        else:
            setattr(config, key, _cast_value(current, value))


def apply_overrides(config: Config, override_dict: dict) -> Config:
    """ Apply dotted overrides such as {"train.lr": 1e-3} to the config in place and return it. """
    # Plug-in classes first, so that the kwargs given with them are kept
    for key, value in sorted(override_dict.items(), key=lambda item: not item[0].endswith(".class")):
        # Nest the dotted key
        *parents, leaf = key.split(".")
        nested_dict = {leaf: value}
        for parent in reversed(parents):
            nested_dict = {parent: nested_dict}
        _update_config(config, nested_dict)
    return config


def load_config(config_path: Optional[str] = None, override_list: Sequence[str] = ()) -> Config:
    """ Load the config: defaults <- YAML file <- `section.key=value` command-line overrides. """
    # Defaults and YAML file
    config = Config()
    if config_path:
        with open(config_path) as config_file:
            _update_config(config, yaml.safe_load(config_file) or {})
    # Command-line overrides
    return apply_overrides(config, {k: yaml.safe_load(v)
                                    for k, v in (override.split("=", 1) for override in override_list)})


# Conversion and Saving

def dict_to_config(config_dict: dict) -> Config:
    """ Convert a nested dict, such as a saved config, to a Config on top of the defaults. """
    config = Config()
    _update_config(config, config_dict)
    return config


def config_to_dict(config: Config) -> dict:
    """ Convert the config to a nested dict. """
    return asdict(config)


def save_config(config: Config, config_path):
    """ Save the config as a YAML file, in field order. """
    with open(config_path, "w") as config_file:
        yaml.safe_dump(config_to_dict(config), config_file, sort_keys=False)
