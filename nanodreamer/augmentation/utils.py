from typing import Callable
import math

from scipy.stats import truncnorm

from .config import (
    UniformDistParams,
    SampleSpaceTruncNormDistParams,
    DistributionParameters
)

UniformRVSFn = Callable[[*UniformDistParams], float]
SampleSpaceTruncNormRVSFn = Callable[[*SampleSpaceTruncNormDistParams], float]
DistributionRVSFn = UniformRVSFn | SampleSpaceTruncNormRVSFn


def truncnorm_in_sample_space(
    left_bound: float,
    right_bound: float,
    rel_mu: float,
    rel_sigma: float
) -> float:
    # rel_mu is original normal distribution mu given in terms of
    # relative position from left_bound, loc is then calculated via:
    # left_bound + rel_mu * (right_bound - left_bound).
    #
    # rel_sigma is original normal distribution sigma give in terms of
    # relative size vs the size of the truncated interval,
    # scale is then calculated via:
    # rel_sigma * (right_bound - left_bound).
    if right_bound <= left_bound:
        raise ValueError(
            f"right_bound <= left_bound ({right_bound} <= {left_bound})"
        )

    interval_size = right_bound - left_bound
    loc = left_bound + rel_mu * interval_size
    scale = rel_sigma * interval_size
    a, b = (
        (bound - loc) / scale
        for bound in (left_bound, right_bound)
    )

    return truncnorm.rvs(a, b, loc=loc, scale=scale)


def generate_distance_and_angle(
    area_size: tuple[int, int],
    distance_dist_params: DistributionParameters,
    distance_dist_rvs_fn: DistributionRVSFn,
    angle_dist_params: DistributionParameters,
    angle_dist_rvs_fn: DistributionRVSFn
) -> tuple[float, float]:
    # Get distance from area center (0, 0) to point.
    # dist = sqrt(dist_x ** 2 + dist_y ** 2)
    # dist_<x|y> = 0.5 * area_size[x|y] * <random fraction>
    distance = math.sqrt(sum(
        (
            0.5 *
            area_size[i] *
            distance_dist_rvs_fn(*distance_dist_params)
        ) ** 2
        for i in range(2)
    ))

    # Get angle between (1,0) and direction to point from (0, 0).
    angle = angle_dist_rvs_fn(*angle_dist_params)
    angle *= math.pi / 180

    return distance, angle
