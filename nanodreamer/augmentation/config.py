import dataclasses

UniformDistParams = tuple[float, float]

# TruncNorm distribution parameters are given in sample space in the form
# (left_bound, right_bound, rel_mu, rel_sigma)
SampleSpaceTruncNormDistParams = tuple[float, float, float, float]
DistributionParameters = UniformDistParams | SampleSpaceTruncNormDistParams


@dataclasses.dataclass
class AugmenterParameters:
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

    background_hues: list[str] = dataclasses.field(
        default_factory=lambda: ['red', 'yellow', 'blue']
    )

    max_num_transforms: int = 2

    # TODO: Make transform configuration hierarchical.
    # All angles must be within (0, 360).
    circle_center_direction_angle_distrib: list[DistributionParameters] \
        = dataclasses.field(default_factory=lambda: [(0, 360), (0, 50, 0.5, 0.25)])
    # min and max are in fractions of half of image spatial sizes.
    circle_center_dist_distrib: SampleSpaceTruncNormDistParams \
        = (1 / 4, 3 / 4, 0.5, 0.25)
    # min and max are in fractions of the smallest of image spatial sizes.
    circle_diameter_distrib: SampleSpaceTruncNormDistParams \
        = (1 / 6, 1 / 4, 0.5, 0.4)
    circle_hues: list[str] = dataclasses.field(
        default_factory=lambda: ['green', 'cyan']
    )
    circle_max_num: int = 3

    # All angles must be within (0, 360).
    line_perpendicular_angle_distrib: list[DistributionParameters] \
        = dataclasses.field(default_factory=lambda: [(0, 360), (0, 50, 0.5, 0.25)])
    # min and max are in fractions of half of image spatial sizes.
    line_center_dist_distrib: SampleSpaceTruncNormDistParams \
        = (0, 0.9, 0.5, 0.25)
    line_hues: list[str] = dataclasses.field(
        default_factory=lambda: ['cyan', 'magenta']
    )
    line_max_num: int = 2
