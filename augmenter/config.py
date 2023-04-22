from typing import Tuple
import torch
import dataclasses

MNIST_SIZE = (28, 28)

@dataclasses.dataclass
class HyperParameters:
    IMG_SIZE:Tuple[int, int] = MNIST_SIZE
    DEVICE:str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    SEED:int = 451
    BATCH_SIZE:int = 4
    NUM_WORKERS:int = 4
    DATASETS_BASE_DIR:str = "~/datasets"
    TB_BASE_DIR:str = "."
    TB_RUNS_DIR:str = "nano-dreamer-runs"

    IMG_MODE:str = 'RGB'

    # ColourPicker params.
    # TODO: MOAR colours.
    hue_ranges = {
        'red': (-20, 20),
        'yellow': (40, 80),
        'green': (100, 140),
        'cyan': (160, 200),
        'blue': (220, 260),
        'magenta': (280, 320),
    }

    # TruncNorm mu and sigma are all for [a, b] = [0, 1],
    # except in the case of greyscale distribution.
    # TruncNorm distribution, (min, max, mu, sigma).
    hue_distrib = (0.5, 0.6)
    saturation_distrib = (0, 1, 0.8, 0.25)
    lightness_distrib = (0, 1, 0.5, 0.25)

    # TruncNorm distribution, (min, max, mu, sigma).
    greyscale_distrib = {
        'black': (0, 0.25, 0, 0.125),
        'grey': (0.25, 0.75, 0.5, 0.125),
        'white': (0.75, 1, 1, 0.125)
    }

    background_hues = ['red', 'yellow', 'blue']

    max_num_transforms = 2

    circle_center_direction_angle_distrib = (0, 360)
    # TruncNorm distribution, (min, max, mu, sigma).
    circle_center_dist_distrib = (IMG_SIZE[0] // 6, IMG_SIZE[0] // 2, 0.5, 0.25)
    circle_diameter_distrib = (IMG_SIZE[0] // 6, IMG_SIZE[0] // 3, 0.5, 0.25)
    circle_hues = ['green', 'cyan']

    line_perpendicular_angle_distrib = (0, 360)
    # TruncNorm distribution, (min, max, mu, sigma).
    line_center_dist_distrib = (0, IMG_SIZE[0] // 2, 0.5, 0.25)
    line_hues = ['cyan', 'magenta']
