import argparse
import csv

import pandas as pd

from compute.computation import *

"""
def apply_weights(data, weights, window_left, window_right):


    n_days = data.size
    conv = np.convolve(data, np.flip(weights), 'full')
    assert window_right + n_days == conv.size + window_left
    return conv[window_right:window_right + n_days]

def correct_space_in_input(dict, input):
    output = []
    for k in range(len(input)):
        if input[k] in dict.keys():
            output.append(input[k])
        elif input[k] + " " + input[min(k+1, len(input) -1)] in dict.keys():
            output.append(input[k] + " " + input[k+1])
    return output

def get_data_dict(filename):
    ts_dict = {}
    data = pd.read_csv(filename).drop(['Province/State', 'Lat', 'Long'], axis=1).groupby("Country/Region").sum()
    ts_dict["dates"] = pd.to_datetime(data.columns).to_numpy("datetime64[D]")
    for index in data.index:
        ts_dict[index] = data.loc[index].values.astype(np.float32)
    return ts_dict
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        nargs=1,
        type=str,
        help=".csv file with daily incidence time series",
        default="time_series_covid19_confirmed_global.csv",
    )
    parser.add_argument(
        "--inv_weights",
        type=str,
        help="cvs file with inverse weights",
        default="inv_weights.csv",
    )
    parser.add_argument(
        "--fwd_weights",
        type=str,
        help="cvs file with forward weights",
        default="fwd_weights.csv",
    )
    parser.add_argument(
        "--countries", nargs="*", type=str, help="countries", default=["Austria"]
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="only trend part of seasonal trend model",
        choices=["c", "t", "st"],
        default="st",
    )
    parser.add_argument(
        "--tau", nargs=2, type=int, help="tau1 and tau2", default=[6, 7]
    )
    parser.add_argument(
        "--k_range",
        nargs=2,
        type=int,
        help="range of parameter k (logarithmic scale, base 10)",
        default=[np.log10(0.1), 4],
    )
    parser.add_argument(
        "--k_samp", type=int, help="samples of k in logarithmic scale", default=300
    )
    parser.add_argument("--n", type=int, help="number of sample for model", default=500)
    parser.add_argument(
        "--distribution",
        type=str,
        choices=["gamma", "NB"],
        help="distribution used to model secondary infections",
        default="gamma",
    )

    args = parser.parse_args()

    data_dict = get_data_dict(os.path.expanduser(args.data_file))
    # correct the country keys for countries that one space in them
    countries = correct_space_in_input(data_dict, args.countries)

    inv_window_left = 0
    inv_window_right = 0
    fwd_window_left = 0
    fwd_window_right = 0
    inv_weights_content = []
    fwd_weights_content = []
    # load inv and fwd weights
    with open(args.inv_weights) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            inv_weights_content.append(np.array(row, dtype=float))

    inv_window_left = int(np.amin(inv_weights_content[0]))
    inv_window_right = int(np.amax(inv_weights_content[0]))
    inv_weights = inv_weights_content[1]

    with open(args.fwd_weights) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            fwd_weights_content.append(np.array(row, dtype=float))

    fwd_window_left = int(np.amin(fwd_weights_content[0]))
    fwd_window_right = int(np.amax(fwd_weights_content[0]))
    fwd_weights = fwd_weights_content[1]

    for country in countries:
        ts_reported_cases = np.convolve(data_dict[country], [1, -1], mode="same")
        ts_dates = data_dict["dates"]

        # ts_infection_potential : currently infected, ...
        # ts_infection_activity : new cases, secondary cases, ...
        # ts_infection_potential, ts_infection_activity = preprocess(ts_dates, ts_reported_cases, mode=args.prepro)
        ts_infection_activity = apply_weights(
            ts_reported_cases, fwd_weights, fwd_window_left, fwd_window_right
        )

        ts_infection_potential = apply_weights(
            ts_reported_cases, inv_weights, inv_window_left, inv_window_right
        )
        # Attn: the next line fixed an issue for CH
        # Should think of  general fix
        # could be a spatial case that arises for delta distr
        # could catch this case by values for difference of window_right and window_left
        # ts_infection_potential[ts_reported_cases == 0] = 0
        if inv_window_left == inv_window_right:
            ts_infection_potential[ts_reported_cases == 0] = 0

        kappas = np.flip(np.logspace(args.k_range[0], args.k_range[1], num=args.k_samp))

        pvals, r_eff_case, r_eff_fits = get_pvals(
            ts_infection_potential,
            ts_infection_activity,
            kappas,
            args.tau[0],
            args.tau[1],
            args.n,
            args.distribution,
            mode=args.mode,
        )

        p0s = [0.8, 0.85, 0.9, 0.95]
        kappa_levels = compute_level_set_lines(pvals, kappas, p0s)

        save_dict = {
            "pvals": pvals,
            "dates": ts_dates,
            "reported_cases": ts_reported_cases,
            "infectious_load": ts_infection_potential,
            "infectious_activity": ts_infection_activity,
            "kappas": kappas,
            "r_eff_case": r_eff_case,
            "r_eff_fits": r_eff_fits,
            "kappa_level_set_lines": kappa_levels,
            "p0s": p0s,
        }

        country_str = country.replace(" ", "").replace(",", "").lower()
        save_data("results/" + country_str + "_" + args.mode, save_dict)


if __name__ == "__main__":
    main()
