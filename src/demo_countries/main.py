import argparse
import os


from demo_countries.visualization import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--countries",
        nargs="*",
        type=str,
        help="Country, for which stochasticity is visualized",
        default=None,
    )
    parser.add_argument(
        "--dates",
        nargs=2,
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        help="date range of time series",
        default=None,
    )
    parser.add_argument(
        "--mode",
        type=str,
        nargs="*",
        help="only trend part of seasonal trend model",
        choices=["c", "t", "st"],
        default=["st"],
    )
    parser.add_argument(
        "--name", type=str, help="name of calculation", default="demo_other_countries"
    )
    parser.add_argument(
        "--markdates",
        type=str,
        nargs="*",
        help="list of special dates followed by index of directory",
        default=None,
    )

    args = parser.parse_args()

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

    fontsize_label = 12
    fontsize_legend = 12
    fontsize_xticks = 12
    fontsize_yticks = 10
    linewidth = [2, 1.5, 1.0]

    n_countries = len(args.countries)

    fig = plt.figure(figsize=(10, n_countries * 1.6 + 0.5))
    fig.subplots_adjust(
        left=0.05, bottom=0.03, right=0.997, top=0.985, wspace=None, hspace=0.05
    )

    axes = fig.subplots(n_countries, 1, sharex=True, sharey=False)

    for idx_mode, mode in enumerate(args.mode):
        for idx, country in enumerate(args.countries):
            # load data
            country_dir_label = country.replace(" ", "").replace(",", "").lower()
            data = load_data("./results/" + country_dir_label + "_" + mode)

            plot_country_incid_kappa_line(
                axes[idx],
                data["dates"],
                data["reported_cases"],
                data["kappa_level_set_lines"][2],
                data["p0s"][2],
                args.countries[idx],
                args.dates,
                idx=idx,
                linewidth=linewidth[idx_mode],
                fontsize_label=fontsize_label,
                fontsize_yticks=fontsize_yticks,
                fontsize_legend=fontsize_legend,
                idx_mode=idx_mode,
            )

    if args.markdates is not None:
        plot_marker(axes, args.markdates, fontsize_legend=fontsize_legend)

    fig.show()
    if not os.path.exists("./plots"):
        os.mkdir("plots")

    fig.savefig("plots/" + args.name + ".pdf")


if __name__ == "__main__":
    main()
