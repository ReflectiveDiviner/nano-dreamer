from scipy.stats import truncnorm


def truncnorm_in_sample_space(
    left_bound: float,
    right_bound: float,
    rel_mu: float,
    rel_sigma: float
) -> int:
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
