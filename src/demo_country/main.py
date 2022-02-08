import argparse

import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
from matplotlib.colors import *

from demo_country.visualization import *
from datetime import datetime

from matplotlib.lines import Line2D

import matplotlib.colors as mcolors

cmap = mcolors.LinearSegmentedColormap.from_list(
    "map", ["white", "#14f000", "#14f000", "#0d9900"]
)


ZORDER_DEFAULT = -10000
ZORDER_OVERLAY = -1000
ZORDER_LEGEND = -100
ALPHA_OVERLAY = 0.85


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--country",
        nargs="*",
        type=str,
        help="Country, for which stochasticity is visualized",
        default=["Austria"],
    )
    parser.add_argument(
        "--dates",
        nargs=2,
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        help="date range of time series",
        default=None,
    )
    parser.add_argument(
        "--low_incid_dates",
        nargs=2,
        type=lambda s: np.datetime64(datetime.strptime(s, "%Y-%m-%d")),
        help="date range of time series",
        default=[np.datetime64("2020-05-27"), np.datetime64("2020-06-17")],
    )
    parser.add_argument(
        "--high_incid_dates",
        nargs=2,
        type=lambda s: np.datetime64(datetime.strptime(s, "%Y-%m-%d")),
        help="date range of time series",
        default=[np.datetime64("2021-01-26"), np.datetime64("2021-02-16")],
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="only trend part of seasonal trend model",
        choices=["c", "t", "st"],
        default="st",
    )

    args = parser.parse_args()

    country = args.country[0]
    country = country.replace(" ", "").replace(",", "").lower()

    # check dates
    if args.dates and args.dates[0] > args.dates[1]:
        print(
            "dates "
            + args.dates[0].strftime("%Y/%m/%d")
            + " and "
            + args.dates[1].strftime("%Y/%m/%d")
            + " are incompatible"
        )
        quit()

    # load data
    data = load_data("./results/" + country + "_" + args.mode)

    # fontsizes
    fontsize_label = 12
    fontsize_legend = 12
    fontsize_xticks = 12
    fontsize_yticks = 10

    linewidth = 2

    # basic figure setup
    fig = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(
        left=0.065, bottom=0.03, right=0.99, top=0.985, wspace=None, hspace=0.05
    )

    axes = fig.subplots(
        4,
        1,
        sharex=True,
        sharey=False,
        gridspec_kw={"height_ratios": [1.0, 2.5, 1.0, 1.0]},
    )

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # compute start and end date and adjust xlim
    first_non_zero_date = data["dates"][np.where(data["reported_cases"] > 0)[0][0]]
    last_non_zero_date = data["dates"][np.where(data["reported_cases"] > 0)[0][-1]]

    for ax in axes:
        ax.set_xlim(first_non_zero_date, last_non_zero_date - 14)
        if args.dates:
            dates = [
                min(args.dates[0], args.low_incid_dates[0] - 45),
                max(args.dates[1], args.high_incid_dates[1] + 45),
            ]
            ax.set_xlim(
                max(dates[0], first_non_zero_date),
                min(dates[1], last_non_zero_date - 14),
            )

    # compute y_lim for axes[0]
    idx_first_non_zero_date = np.where(data["dates"] == first_non_zero_date)
    idx_last_non_zero_date = np.where(data["dates"] == last_non_zero_date)
    y_lim0 = int(
        max(
            data["reported_cases"][
                idx_first_non_zero_date[0][0] : idx_last_non_zero_date[0][0] - 13
            ]
        )
    )
    axes[0].set_ylim(0, y_lim0)

    # include high and low intervals
    place_intervals(
        axes,
        data["dates"],
        data["reported_cases"],
        ZORDER_OVERLAY,
        low_incid_dates=args.low_incid_dates,
        high_incid_dates=args.high_incid_dates,
    )

    # EffDI and corresponding Reff is only computable up to 2 weeks before last
    # date in the reported cases ts
    omit_data_parameter = 20

    # get dates where reported cases is non-zero
    idxs_nonzero = np.nonzero(data["kappa_level_set_lines"][2])
    kappa_dates = (data["dates"][idxs_nonzero]).min(), (
        data["dates"][idxs_nonzero]
    ).max()

    plot_pvalues_reff(
        axes[1:5],
        data["dates"],
        data["pvals"],
        data["kappas"],
        data["kappa_level_set_lines"][2],
        data["r_eff_case"],
        data["p0s"][2],
        kappa_dates,
        omit_data_parameter,
        ZORDER_DEFAULT,
        linewidth,
        fontsize_label=fontsize_label,
        fontsize_yticks=fontsize_yticks,
    )

    plot_incidence(
        axes[0],
        data["dates"],
        data["reported_cases"],
        data["infectious_load"],
        data["infectious_activity"],
        ZORDER_DEFAULT,
        linewidth,
    )

    # SOME ADJUSTMENTS
    adjustments(axes)

    axes[0].ticklabel_format(
        useOffset=False, style="sci", axis="y", useMathText=True, scilimits=(0, 0)
    )

    # added stuff
    add_text(fig, 0.003, 0.96, "A", fontsize=14, fontweight="bold")
    add_text(fig, 0.003, 0.78, "B", fontsize=14, fontweight="bold")
    add_text(fig, 0.003, 0.36, "C", fontsize=14, fontweight="bold")
    add_text(fig, 0.003, 0.18, "D", fontsize=14, fontweight="bold")

    # show and save the figure
    fig.show()
    if not os.path.exists("./plots"):
        os.mkdir("plots")

    fig.savefig("plots/demo_country_" + country + ".pdf")


if __name__ == "__main__":
    main()
