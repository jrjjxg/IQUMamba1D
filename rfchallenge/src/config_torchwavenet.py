from dataclasses import MISSING, asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional

try:
    from omegaconf import DictConfig, OmegaConf
except ImportError:  # Checkpoint unpickling needs the dataclasses, not parsing.
    DictConfig = Any
    OmegaConf = None

if OmegaConf is not None:
    OmegaConf.register_new_resolver(
        "datetime", lambda s: f'{s}_{datetime.now().strftime("%H_%M_%S")}')


@dataclass
class ModelConfig:
    input_channels: int = 2
    residual_layers: int = 30
    residual_channels: int = 64
    dilation_cycle_length: int = 10


@dataclass
class DataConfig:
    root_dir: str = MISSING
    batch_size: int = 16
    num_workers: int = 4
    train_fraction: float = 0.8


@dataclass
class DistributedConfig:
    distributed: bool = False
    world_size: int = 2


@dataclass
class TrainerConfig:
    learning_rate: float = 2e-4
    max_steps: int = 1000
    max_grad_norm: Optional[float] = None
    fp16: bool = False

    log_every: int = 50
    save_every: int = 2000
    validate_every: int = 100


@dataclass
class Config:
    model_dir: str = MISSING

    # The upstream Python 3.7 code used dataclass instances as defaults.
    # Python 3.11+ rejects those mutable defaults during checkpoint unpickling,
    # so preserve the same values through factories instead.
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=lambda: DataConfig(root_dir=""))
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)


def parse_configs(cfg: DictConfig, cli_cfg: Optional[DictConfig] = None) -> DictConfig:
    if OmegaConf is None:
        raise ModuleNotFoundError(
            "omegaconf is required to parse upstream training YAML, but not "
            "to unpickle released WaveNet checkpoint dataclasses"
        )
    base_cfg = OmegaConf.structured(Config)
    merged_cfg = OmegaConf.merge(base_cfg, cfg)
    if cli_cfg is not None:
        merged_cfg = OmegaConf.merge(merged_cfg, cli_cfg)
    return merged_cfg


if __name__ == "__main__":
    base_config = OmegaConf.structured(Config)
    config = OmegaConf.load("configs/short_ofdm.yaml")
    config = OmegaConf.merge(base_config, OmegaConf.from_cli(), config)
    config = Config(**config)

    print(asdict(config))
