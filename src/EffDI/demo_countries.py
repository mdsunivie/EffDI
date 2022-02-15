import os


from EffDI.visualization import *


def demo_countries(countries=None,
                   dates=None,
                   modes=["st"],
                   name="demo_other_countries",
                   markdates=None):


    # check dates
    if dates and dates[0] > dates[1]:
        print(
            "dates "
            + dates[0].strftime("%Y/%m/%d")
            + " and "
            + dates[1].strftime("%Y/%m/%d")
            + " are incompatible"
        )
        quit()

    fontsize_label = 12
    fontsize_legend = 12
    fontsize_yticks = 10
    linewidth = [2, 1.5, 1.0]

    n_countries = len(countries)

    fig = plt.figure(figsize=(10, n_countries * 1.6 + 0.5))
    fig.subplots_adjust(
        left=0.05, bottom=0.03, right=0.997, top=0.985, wspace=None, hspace=0.05
    )

    axes = fig.subplots(n_countries, 1, sharex=True, sharey=False)

    for idx_mode, mode in enumerate(modes):
        for idx, country in enumerate(countries):
            # load data
            country_dir_label = country.replace(" ", "").replace(",", "").lower()
            data = load_data("./results/" + country_dir_label + "_" + mode)

            plot_country_incid_kappa_line(
                axes[idx],
                data["dates"],
                data["reported_cases"],
                data["kappa_level_set_lines"][2],
                data["p0s"][2],
                countries[idx],
                dates,
                idx=idx,
                linewidth=linewidth[idx_mode],
                fontsize_label=fontsize_label,
                fontsize_yticks=fontsize_yticks,
                fontsize_legend=fontsize_legend,
                idx_mode=idx_mode,
            )

    if markdates is not None:
        plot_marker(axes, markdates, fontsize_legend=fontsize_legend)

    fig.show()
    if not os.path.exists("./plots"):
        os.mkdir("plots")

    fig.savefig("plots/" + name + ".pdf")


