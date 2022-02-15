
import csv

import pandas as pd

from EffDI.computation import *



def compute(data_file="time_series_covid19_confirmed_global.csv",
            inv_weights="inv_weights.csv",
            fwd_weights="fwd_weights.csv",
            countries=["Austria"],
            mode="st",
            tau=[6, 7],
            k_range=[np.log10(0.1), 4],
            k_samp=300,
            n=500,
            distribution="gamma"):


    data_dict = get_data_dict(os.path.expanduser(data_file))
    # correct the country keys for countries that one space in them
    countries = correct_space_in_input(data_dict, countries)

    inv_weights_content = []
    fwd_weights_content = []
    # load inv and fwd weights
    with open(inv_weights) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            inv_weights_content.append(np.array(row, dtype=float))

    inv_window_left = int(np.amin(inv_weights_content[0]))
    inv_window_right = int(np.amax(inv_weights_content[0]))
    inv_weights = inv_weights_content[1]

    with open(fwd_weights) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            fwd_weights_content.append(np.array(row, dtype=float))

    fwd_window_left = int(np.amin(fwd_weights_content[0]))
    fwd_window_right = int(np.amax(fwd_weights_content[0]))
    fwd_weights = fwd_weights_content[1]

    for country in countries:
        ts_reported_cases = np.convolve(data_dict[country], [1, -1], mode="same")
        ts_dates = data_dict["dates"]

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

        kappas = np.flip(np.logspace(k_range[0], k_range[1], num=k_samp))

        pvals, r_eff_case, r_eff_fits = get_pvals(
            ts_infection_potential,
            ts_infection_activity,
            kappas,
            tau[0],
            tau[1],
            n,
            distribution,
            mode=mode,
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
        save_data("results/" + country_str + "_" + mode, save_dict)


