from typing import Callable
from itertools import chain
import math
import random

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


def recompute_distrib_params(
        dist_p: SampleSpaceTruncNormDistParams,
        new_interval: tuple[float, float]
    ) -> SampleSpaceTruncNormDistParams:
        # Recompute relative sample space distribution parameters
        # for the smaller new interval that is still using same distribution.
        old_left_bound, old_right_bound, old_rel_mu, old_rel_sigma = dist_p
        new_left_bound, new_right_bound = new_interval

        old_interval_size = old_right_bound - old_left_bound
        loc = old_left_bound + old_rel_mu * old_interval_size
        scale = old_rel_sigma * old_interval_size

        new_interval_size = new_right_bound - new_left_bound
        new_rel_mu = (loc - new_left_bound) / new_interval_size
        new_rel_sigma = scale / new_interval_size

        assert new_interval_size <= old_interval_size

        return new_left_bound, new_right_bound, new_rel_mu, new_rel_sigma


def pick_distribution_function(
    distribution_params: DistributionParameters
) -> DistributionRVSFn:
    if len(distribution_params) < 3:
        return random.uniform
    else:
        return truncnorm_in_sample_space


def generate_distance_and_angle(
    area_size: tuple[int, int],
    distance_dist_params: DistributionParameters,
    angle_dist_params: list[DistributionParameters],
) -> tuple[float, float]:  # Distance in fraction of pixels, angle in radians.
    # Get distance from area center (0, 0) to point.
    # dist = sqrt(dist_x ** 2 + dist_y ** 2)
    # dist_<x|y> = 0.5 * area_size[x|y] * <random fraction>
    distance_dist_rvs_fn = pick_distribution_function(distance_dist_params)
    distance = math.sqrt(sum(
        (
            area_size[i] // 2 *
            distance_dist_rvs_fn(*distance_dist_params)
        ) ** 2
        for i in range(2)
    ))

    if distance > (math.sqrt(sum(side ** 2 for side in area_size)) / 2):
        raise ValueError(
            "distance > diagonal, object is not within area, "
            "check distribution parameters."
        )

    valid_angle_iterval_points = []
    if distance < area_size[0] // 2:
        valid_angle_iterval_points.append(0)
    for i in range(2):
        side = area_size[i] // 2
        if distance < side:  # No incidence of a r=distance circle with area.
            continue
        alpha = abs(math.acos(side / distance) * 180 / math.pi - 90 * i)
        # By mirror and central symmetry, all angles to incidences are:
        valid_angle_iterval_points.extend([
            alpha, 180 - alpha, 180 + alpha, 360 - alpha
        ])
    if distance < area_size[0] // 2:
        valid_angle_iterval_points.append(360)
    valid_angle_iterval_points = sorted(valid_angle_iterval_points)

    # Line sweep intersection search.
    new_interval_point_lists: list[list[float]] = [
        [] for _ in range(len(angle_dist_params))
    ]

    def is_inetrval_start(idx):
        return (idx % 2) == 0
    valid_angle_iterval_points = [
        (interval_point, is_inetrval_start(idx), -1)
        for idx, interval_point in enumerate(valid_angle_iterval_points)
    ]

    current_open_intervals: set[int] = set()
    is_in_valid_interval = False
    for interval_point in sorted(
        chain(
            (
                # (interval_point, is_iterval_end, angle_dist_params_idx)
                (adp[i], is_inetrval_start(i), idx)
                for idx, adp in enumerate(angle_dist_params)
                for i in range(2)
            ),
            valid_angle_iterval_points
        )
    ):
        if interval_point[2] == -1:
            is_in_valid_interval = interval_point[1]
            for open_interval in current_open_intervals:
                new_interval_point_lists[open_interval].append(
                    interval_point[0]
                )
            continue

        if interval_point[1]:
            current_open_intervals.add(interval_point[2])
        else:
            current_open_intervals.remove(interval_point[2])

        if not is_in_valid_interval:
            continue
        new_interval_point_lists[interval_point[2]].append(interval_point[0])

    if all(len(item) == 0 for item in new_interval_point_lists):
        raise ValueError(
            "No valid angle intervals, check distribution parameters."
        )

    # Convert a lists of points into lists of intervals.
    assert all(len(item) % 2 == 0 for item in new_interval_point_lists)
    new_interval_lists = [
        [
            tuple(interval_point_list[i:i + 2])
            for i in range(0, len(interval_point_list), 2)
        ]
        for interval_point_list in new_interval_point_lists
    ]

    new_angle_dist_params = []
    for adp_idx, new_intervals in enumerate(new_interval_lists):
        for interval in new_intervals:
            if len(angle_dist_params[adp_idx]) < 3:  # is UniformDistParams
                new_angle_dist_params.append(interval)
            else:  # is SampleSpaceTruncNormDistParams
                new_angle_dist_params.append(
                    recompute_distrib_params(
                        angle_dist_params[adp_idx],  # type: ignore
                        interval
                    )
                )

    # Get angle between (1,0) and direction to point from (0, 0).
    angle = random.choice([
        pick_distribution_function(angle_dist_p)(*angle_dist_p)
        for angle_dist_p in new_angle_dist_params
    ])
    angle *= math.pi / 180

    return distance, angle
