import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


@dataclass
class Config:
    regions: dict
    project: str | None = None
    year: int = 2020
    mangrove_label: str = "extent"
    scales: list = field(default_factory=lambda: [0, 1, 3, 7])
    coast_buffer_m: int = 5000
    n_per_class: int = 300
    n_unlabeled: int = 3000
    n_edge: int = 1500
    edge_radius_px: int = 3
    tile_deg: float = 0.25
    workers: int = 6
    block_deg: float = 0.05
    anchor: str | None = None
    seed: int = 42
    data_dir: str = "data"
    results_dir: str = "results"

    @property
    def raw_dir(self) -> Path:
        return Path(self.data_dir) / "raw"

    @property
    def processed_dir(self) -> Path:
        return Path(self.data_dir) / "processed"


def load(path=None, **overrides) -> Config:
    with open(path or DEFAULT) as f:
        d = yaml.safe_load(f)
    d.update({k: v for k, v in overrides.items() if v is not None})
    d["project"] = os.environ.get("EE_PROJECT") or d.get("project")
    if overrides.get("project"):
        d["project"] = overrides["project"]
    return Config(**d)
