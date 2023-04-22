from scipy.stats import truncnorm

def truncnorm_intAB(
    int_min: int,
    int_max: int,
    mu: float,
    sigma: float
) -> int:
    # (
    # min,
    # max,
    # mu (for (a,b)=(0,1)),
    # sigma (for (a,b)=(0,1))
    # )
    return (
        int_min +
        int(
            (int_max - int_min) *
            truncnorm.rvs(0, 1, mu, sigma)
    ))
