import dataclasses
import random

from numpy.random import RandomState, MT19937, SeedSequence

import torch


@dataclasses.dataclass
class _GlobalConfig:
    # Global system-level parameters.
    DEVICE: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    SEED: int = 451

    # Dataset parameters.
    DATASETS_BASE_DIR: str = "~/datasets"
    IMG_SIZE: tuple[int, int] = (28, 28)  # MNIST image size.
    NUM_WORKERS: int = 4

GlobalConfig = _GlobalConfig()


def set_seeds(
    seed: int=GlobalConfig.SEED,
) -> torch.Generator:
    # Sets seeds, essentially an extension of torch.manual_seed.
    random.seed(seed)
    RandomState(MT19937(SeedSequence(seed)))
    return torch.manual_seed(seed)


@dataclasses.dataclass
class ConfigBaseClass:
    batch_size: int

    epochs: int
    evals_per_epoch: int  # Including the one at the end of epoch.
