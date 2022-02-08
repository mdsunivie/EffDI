import numpy as np
import os
import pandas as pd


def apply_weights(data, weights, window_left, window_right):
    """multiply data with weights"""

    n_days = data.size
    conv = np.convolve(data, np.flip(weights), "full")
    assert window_right + n_days == conv.size + window_left
    return conv[window_right : window_right + n_days]


def correct_space_in_input(dict, input):
    output = []
    for k in range(len(input)):
        if input[k] in dict.keys():
            output.append(input[k])
        elif input[k] + " " + input[min(k + 1, len(input) - 1)] in dict.keys():
            output.append(input[k] + " " + input[k + 1])
    return output


def get_data_dict(filename):
    ts_dict = {}
    data = (
        pd.read_csv(filename)
        .drop(["Province/State", "Lat", "Long"], axis=1)
        .groupby("Country/Region")
        .sum()
    )
    ts_dict["dates"] = pd.to_datetime(data.columns).to_numpy("datetime64[D]")
    for index in data.index:
        ts_dict[index] = data.loc[index].values.astype(np.float32)
    return ts_dict


def get_pvals(
    incid_daily,
    secondary_infections,
    kappas,
    tau1,
    tau2,
    n_samples,
    distribution="gamma",
    mode="t",
):
    r_eff_case, r_eff_fits = get_r_eff_case(
        incid_daily, secondary_infections, tau1, tau2, mode=mode
    )
    p_vals_full = np.zeros([len(kappas), len(incid_daily)])
    p_vals_full[:] = np.nan
    for k, kappa in enumerate(kappas):
        p_vals = np.zeros_like(incid_daily)
        p_vals[:tau1] = np.nan
        if tau2 != 0:
            p_vals[-tau2:] = np.nan

        for idx in range(tau1, len(p_vals) - tau2):
            l = len(r_eff_fits[idx])
            idxs_nan = np.isnan(r_eff_fits[idx])
            if np.sum(idxs_nan) > l // 2:
                p_vals[idx] = np.nan
            else:
                sample_idxs = np.array(range(idx - tau1, idx + tau2 + 1))
                sample_idxs = sample_idxs[~idxs_nan]

                r_eff_fit = r_eff_fits[idx][~idxs_nan]

                incid_samples = sample_from_model(
                    incid_daily,
                    sample_idxs,
                    r_eff_fit,
                    distribution,
                    n=n_samples,
                    k=kappa,
                )

                means = r_eff_fit * incid_daily[idx - tau1 : idx + tau2 + 1][~idxs_nan]
                sec_inf = secondary_infections[idx - tau1 : idx + tau2 + 1][~idxs_nan]

                test_statistic_incid = np.sum(np.square(means - sec_inf))
                test_statistic_samples = np.sum(
                    np.square(means - incid_samples), axis=1
                )

                p_vals[idx] = (
                    np.sum(test_statistic_samples >= test_statistic_incid) / n_samples
                )

        p_vals_full[k, :] = p_vals
    return p_vals_full, r_eff_case, r_eff_fits


def get_linear_system(incid_daily, secondary_infections, mode="t"):
    if mode == "c":
        n_cols = 1
    elif mode == "t":
        n_cols = 2
    elif mode == "st":
        n_cols = 8
    else:
        raise ValueError("unknown mode")

    n_days = len(incid_daily)

    A = np.reshape(np.repeat(incid_daily, n_cols), [n_days, n_cols])
    if mode == "t" or mode == "st":
        A[:, 0] = np.array(range(n_days)) * A[:, 0]
    if mode == "st":
        for k2 in range(1, 8):
            season_helper = np.zeros(n_days)
            season_helper[list(range(k2 - 1, n_days, 7))] = 1
            A[:, k2] = season_helper * A[:, k2]

    b = np.reshape(secondary_infections, [-1, 1])
    return A, b


def get_r_eff_case(incid_daily, secondary_infections, tau1, tau2, mode="t"):
    n_days = tau1 + tau2 + 1

    ts_r_eff = np.zeros_like(incid_daily)
    ts_r_eff[0:tau1] = np.nan
    if tau2 != 0:
        ts_r_eff[-tau2:] = np.nan

    r_eff_fits = np.empty([len(incid_daily), n_days])
    r_eff_fits[:] = np.nan

    for k in range(tau1, len(ts_r_eff) - tau2):
        # define linear system
        A, b = get_linear_system(
            incid_daily[k - tau1 : k + tau2 + 1],
            secondary_infections[k - tau1 : k + tau2 + 1],
            mode,
        )

        # only fith where daily incid is nonzero
        incid_gt_zero = incid_daily[k - tau1 : k + tau2 + 1] > 0
        weekday_gt_zero = [np.sum(incid_gt_zero[k::7]) for k in range(7)]
        b2 = b[incid_gt_zero]
        A2 = A[incid_gt_zero, :]

        # solve linear system
        coeffs = np.matmul(np.linalg.pinv(A2), b2)

        # compute r_eff case
        x = incid_daily[k - tau1 : k + tau2 + 1]
        x2 = x[incid_gt_zero]

        r_eff_fits[k, ~incid_gt_zero] = np.nan
        r_eff_fits[k, incid_gt_zero] = np.divide(np.squeeze(np.matmul(A2, coeffs)), x2)

        if mode == "c":
            ts_r_eff[k] = coeffs[0]
        if mode == "t":
            ts_r_eff[k] = coeffs[0] * tau1 + coeffs[1]
        if mode == "st":
            if np.sum(incid_gt_zero) > 0:
                ts_r_eff[k] = coeffs[0] * tau1 + np.sum(
                    np.reshape(weekday_gt_zero, [7, 1]) * coeffs[1:]
                ) / np.sum(incid_gt_zero)
            else:
                ts_r_eff[k] = np.nan
    return ts_r_eff, r_eff_fits


def sample_from_model(incid_daily, sample_idxs, r_eff, distribution, n=1, k=0.5):
    samples = np.empty([n, len(sample_idxs)])
    samples[:] = np.nan
    for j, idx in enumerate(sample_idxs):
        if distribution == "gamma":
            samples[:, j] = np.random.gamma(
                k,
                scale=np.clip(
                    (np.clip(r_eff[j], None, 100) * incid_daily[idx]) / k, 0, None
                ),
                size=n,
            )
        elif distribution == "NB":
            p = 1 / (
                1
                + np.clip(
                    (np.clip(r_eff[j], None, 100) * incid_daily[idx]) / k, 0, None
                )
            )
            samples[:, j] = np.random.negative_binomial(k, p, size=n)
        else:
            raise ValueError
    return samples


def compute_level_set_lines(pvals, kappas, p0s):
    kappa_level = np.zeros((len(p0s), len(pvals[0])))
    for k in range(len(p0s)):
        for j in range(pvals.shape[1]):
            for i in range(pvals.shape[0]):
                if pvals[i, j] > p0s[k]:
                    kappa_level[k, j] = kappas[i]
                    break
    return kappa_level


def save_data(dir, dict):
    if not os.path.exists(dir):
        os.makedirs(dir)
    np.savez_compressed(dir + "/pvals.npz", **dict)
    with open(dir + "/info.txt", "w") as f:
        print("The file pvals.npz contains the following variables:\n", file=f)
        for key in dict:
            print(key, file=f)
