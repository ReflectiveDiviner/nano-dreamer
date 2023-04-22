import torch
import dataclasses

UniformDistParams = tuple[float, float]

# TruncNorm distribution parameters are given in sample space in the form
# (left_bound, right_bound, rel_mu, rel_sigma)
SampleSpaceTruncNormDistParams = tuple[float, float, float, float]

_MNIST_SIZE = (28, 28)

@dataclasses.dataclass
class HyperParameters:
    IMG_SIZE: tuple[int, int] = _MNIST_SIZE
    DEVICE: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    SEED: int = 451
    BATCH_SIZE: int = 8
    NUM_WORKERS: int = 4
    DATASETS_BASE_DIR: str = "~/datasets"
    TB_BASE_DIR: str = "."
    TB_RUNS_DIR: str = "nano-dreamer-runs"

    IMG_MODE: str | None = 'RGB'

    # ColourPicker params.
    # (left_bound, right_bound) for the hue TruncNorm in sample space.
    # TODO: MOAR colours.
    hue_ranges: dict[str, tuple[float, float]] = dataclasses.field(
        default_factory=lambda: {
            'red': (-20, 20),
            'yellow': (40, 80),
            'green': (100, 140),
            'cyan': (160, 200),
            'blue': (220, 260),
            'magenta': (280, 320),
        }
    )

    # (rel_mu, rel_sigma) for the hue TruncNorm in sample space.
    hue_distrib: tuple[float, float] = (0.5, 0.2)
    saturation_distrib: SampleSpaceTruncNormDistParams = (0, 1, 0.8, 0.25)
    lightness_distrib: SampleSpaceTruncNormDistParams = (0, 1, 0.5, 0.25)

    greyscale_distrib: dict[str, SampleSpaceTruncNormDistParams] = \
        dataclasses.field(
            default_factory=lambda: {
                'black': (0, 0.25, 0, 0.5),
                'grey': (0.25, 0.75, 0.5, 0.25),
                'white': (0.75, 1, 1, 0.5)
            }
        )

    background_hues: set[str] = dataclasses.field(
        default_factory=lambda: {'red', 'yellow', 'blue'}
    )

    max_num_transforms = 2

    circle_center_direction_angle_distrib: \
        UniformDistParams | SampleSpaceTruncNormDistParams \
        = (0, 360)
    # TruncNorm distribution, (min, max, mu, sigma).
    circle_center_dist_distrib: SampleSpaceTruncNormDistParams \
        = (IMG_SIZE[0] // 8, 3 * IMG_SIZE[0] // 8, 0.5, 0.25)
    circle_diameter_distrib: SampleSpaceTruncNormDistParams \
        = (IMG_SIZE[0] // 6, IMG_SIZE[0] // 4, 0.5, 0.4)
    circle_hues: set[str] = dataclasses.field(
        default_factory=lambda: {'green', 'cyan'}
    )

    line_perpendicular_angle_distrib: \
        UniformDistParams | SampleSpaceTruncNormDistParams \
        = (0, 360)
    # TruncNorm distribution, (min, max, mu, sigma).
    line_center_dist_distrib: SampleSpaceTruncNormDistParams \
        = (0, IMG_SIZE[0] // 2, 0.5, 0.25)
    line_hues: set[str] = dataclasses.field(
        default_factory=lambda: {'cyan', 'magenta'}
    )
