import numpy as np
import argparse
import csv


def get_inv_window(distribution):
    if distribution not in ["delta", "gamma", "skewnorm"]:
        raise AssertionError()
    inv_window_left = 0
    inv_window_right = 0
    if distribution == "gamma":
        inv_window_left = -70
    if distribution == "skewnorm":
        inv_window_left = -70
        inv_window_right = 70
    return inv_window_left, inv_window_right


def get_fwd_window(distribution):
    if distribution not in ["delta", "gamma", "skewnorm"]:
        raise AssertionError()
    fwd_window_left = 0
    fwd_window_right = 0
    if distribution == "gamma":
        fwd_window_left = 70
    if distribution == "skewnorm":
        fwd_window_left = -70
        fwd_window_right = 70
    return fwd_window_left, fwd_window_right


def compute_fwd_weights(fwd_window_left, fwd_window_right, distribution):
    if distribution == "delta":
        return np.array([1.0])
    if distribution == "gamma":
        return _discrete_gamma(np.arange(fwd_window_left, fwd_window_right + 1))
    if distribution == "skewnorm":
        return _discrete_skewnorm(np.arange(fwd_window_left, fwd_window_right + 1))


def compute_inv_weights(inv_window_left, inv_window_right, distribution):
    if distribution == "delta":
        return np.array([1.0])
    if distribution == "gamma":
        return np.flip(
            _discrete_gamma(np.arange(-inv_window_right, -inv_window_left + 1))
        )
    if distribution == "skewnorm":
        return np.flip(
            _discrete_skewnorm(np.arange(inv_window_left, inv_window_right + 1))
        )


def _discrete_skewnorm(days, shape=0.828, loc=2.045, scale=5.199):

    from scipy.stats import skewnorm

    cdf = skewnorm(shape, loc, scale).cdf(days)
    pmf = np.diff(cdf, prepend=0.0)
    assert np.isclose(pmf.sum(), 1.0, 1e-3)
    assert cdf.size == pmf.size
    pmf /= pmf.sum()
    return pmf


def _discrete_gamma(days, mean=5.6, std=4.2):

    from scipy.stats import gamma

    scale = std * std / mean
    shape = mean / scale
    loc = 0.0

    cdf = gamma(shape, loc, scale).cdf(days)
    pmf = np.diff(cdf, prepend=0.0)
    assert np.isclose(pmf.sum(), 1.0, 1e-4)
    assert cdf.size == pmf.size
    pmf /= pmf.sum()
    return pmf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fwd-distribution",
        type=str,
        choices=["gamma", "skewnorm", "delta"],
        help="type of distribution to compute the fwd weights",
        default="delta",
        metavar="fwd_distribution",
    )
    parser.add_argument(
        "--inv-distribution",
        type=str,
        choices=["gamma", "skewnorm", "delta"],
        help="type of distribution to compute the fwd weights",
        default="gamma",
        metavar="inv_distribution",
    )

    args = parser.parse_args()

    inv_window_left, inv_window_right = get_inv_window(args.inv_distribution)
    fwd_window_left, fwd_window_right = get_fwd_window(args.fwd_distribution)

    fwd_weights = compute_fwd_weights(
        fwd_window_left, fwd_window_right, args.fwd_distribution
    )
    inv_weights = compute_inv_weights(
        inv_window_left, inv_window_right, args.inv_distribution
    )

    f_inv_weights = open("inv_weights.csv", "w")
    writer = csv.writer(f_inv_weights)
    writer.writerow(np.arange(inv_window_left, inv_window_right + 1))
    writer.writerow(inv_weights)

    f_fwd_weights = open("fwd_weights.csv", "w")
    writer = csv.writer(f_fwd_weights)
    writer.writerow(np.arange(fwd_window_left, fwd_window_right + 1))
    writer.writerow(fwd_weights)


if __name__ == "__main__":
    main()
