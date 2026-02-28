import yaml
from dataclasses import dataclass


@dataclass
class Config:
    dataset_path: str
    dataset_limit: int | None
    batch_size: int
    epochs: int
    num_workers: int
    model: str
    checkpoint_interval: int
    failure_prob: float


def load_config(path: str) -> Config:
    with open(path) as f:
        data = yaml.safe_load(f)

    return Config(**data)