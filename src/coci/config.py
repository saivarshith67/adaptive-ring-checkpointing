import yaml
from dataclasses import dataclass


# FaceForensics++ Configuration Defaults
FACEFORENSICS_DATASET_PATH = "./data/faceforensics"
MODEL_NAME = "efficientnet_b0"
NUM_CLASSES = 2
DATASET_TYPE = "faceforensics"
FACEFORENSICS_COMPRESSION = "c23"


@dataclass
class Config:
    dataset_path: str
    dataset_limit: int | None
    batch_size: int
    epochs: int
    num_workers: int
    model: str
    checkpoint_interval: int
    failure_rate_per_second: float
    strategy: str
    fixed_interval: float
    checkpoint_cost_estimate: float


def load_config(path: str) -> Config:
    with open(path) as f:
        data = yaml.safe_load(f)

    if "failure_rate_per_sec" in data and "failure_rate_per_second" not in data:
        data["failure_rate_per_second"] = data.pop("failure_rate_per_sec")
    data.setdefault("strategy", "epoch")
    data.setdefault("fixed_interval", float(data.get("checkpoint_interval", 30)))
    data.setdefault("checkpoint_cost_estimate", 1.0)

    return Config(**data)
