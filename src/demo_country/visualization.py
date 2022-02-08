from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
import matplotlib.dates as mdates
import pandas as pd

import matplotlib.colors as mcolors

cmap = mcolors.LinearSegmentedColormap.from_list(
    "map", ["white", "#14f000", "#14f000", "#0d9900"]
)


# https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
def smooth(x, window_len=15, window="hanning"):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        raise ValueError(
            "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        )

    s = np.r_[x[window_len - 1 : 0 : -1], x, x[-2 : -window_len - 1 : -1]]
    # print(len(s))
    if window == "flat":  # moving average
        w = np.ones(window_len, "d")
    else:
        w = eval("np." + window + "(window_len)")

    y = np.convolve(w / w.sum(), s, mode="valid")

    offset = (window_len - 1) / 2
    assert offset == int(offset)
    offset = int(offset)

    return y[offset:-offset]


def load_data(dir):
    return dict(np.load(dir + "/pvals.npz"))


def compute_average_range(ts_data, ts_dates, date_interval):
    idx_low_start = np.where(ts_dates == date_interval[0])
    idx_low_end = np.where(ts_dates == date_interval[1])
    average_low = ts_data[idx_low_start[0][0] : idx_low_end[0][0] + 1].mean()
    return round(average_low)


def plot_incidence(
    ax,
    ts_dates,
    ts_reported_cases,
    infectious_load,
    infectious_activity,
    zorder,
    linewidth=1,
):

    ts_dates = pd.DatetimeIndex(ts_dates)
    ax.fill_between(
        ts_dates,
        ts_reported_cases,
        0.0,
        linewidth=0,
        color="k",
        alpha=0.2,
        label="reported cases $I_{{t}}$",
    )

    ax.plot(
        ts_dates,
        infectious_load,
        linewidth=linewidth,
        color="blue",
        label="load $I^{{\\ast}}_{{t}}$",
        zorder=zorder,
    )
    ax.plot(
        ts_dates,
        infectious_activity,
        linewidth=linewidth,
        color="red",
        label="activity $I^{{\dagger}}_{{t}}$",
        zorder=zorder,
    )

    handles, labels = ax.get_legend_handles_labels()
    order_lines = [2, 0, 1]
    legend1 = ax.legend(
        [handles[idx] for idx in order_lines],
        [labels[idx] for idx in order_lines],
        # loc=2, ncol=1,
        frameon=True,
        loc=2,
        ncol=3,
        bbox_to_anchor=(0.3, 1.1),
        # bbox_to_anchor=(0.685, 0.8),
        framealpha=0.9,
        facecolor="white",
        edgecolor="white",
    )
    ax.set_ylabel("population")
    p = ax.get_yaxis().get_offset_text().get_position()
    ax.get_yaxis().get_offset_text().set_position((p[0] + 0.01, 0.0))
    ax.get_yaxis().get_offset_text().set_va("top")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.add_artist(legend1)


def plot_pvalues_reff(
    axes,
    ts_dates,
    pvals,
    kappas,
    kappa_level_set_line,
    r_eff_case,
    p0,
    kappas_dates,
    omit_data_parameter,
    zorder,
    linewidth=1,
    fontsize_label=16,
    fontsize_yticks=16,
):

    idx_kappas_start = np.where(ts_dates == np.datetime64(kappas_dates[0]))[0][0]
    idx_kappas_end = np.where(ts_dates == np.datetime64(kappas_dates[1]))[0][0] + 1

    kappas_dates = ts_dates[idx_kappas_start:idx_kappas_end]
    # exclude zero values at boundary
    pvals = pvals[:, idx_kappas_start:idx_kappas_end]
    kappa_level_set_line = kappa_level_set_line[idx_kappas_start:idx_kappas_end]
    r_eff_case = r_eff_case[idx_kappas_start:idx_kappas_end]

    # set parameters
    linestyle = "-"
    color_kappa = "C2"

    # plot pvals
    x_start = mdates.date2num(kappas_dates[0])
    x_end = mdates.date2num(kappas_dates[-omit_data_parameter])
    y_start = len(kappas)
    y_end = 0

    im = axes[0].imshow(
        np.array(pvals[:, :-omit_data_parameter]),
        extent=(x_start, x_end, y_start, y_end),
        aspect="auto",
        cmap=cmap,
        zorder=zorder,
    )
    cb = plt.colorbar(
        im, ax=axes[0], fraction=0.05, pad=0.01, aspect=30, format=ScalarFormatter()
    )
    cb.set_label(label="$p$ ... plausibility of $\kappa_t$")
    cb.ax.tick_params(labelsize=fontsize_yticks)

    # correct the y-axis, since data of pvals comes in log-scales described by kappas
    yticks_base10 = [-1, 0, 1, 2, 3, 4]
    yticks = [np.argmin(np.abs(kappas - 10**exponent)) for exponent in yticks_base10]
    axes[0].set_yticks(yticks)
    axes[0].set_yticklabels(
        ["$10^{" + str(exponent) + "}$" for exponent in yticks_base10],
        fontsize=fontsize_yticks,
    )
    axes[0].set_ylabel("$\kappa_t$", fontsize=fontsize_label)

    # plot level set line into phase plot
    alpha_data = 0.2
    alpha_smooth = 1.0
    level_set_line_transformed = len(kappas) - len(kappas) * (
        np.log(kappa_level_set_line) - (np.log(np.min(kappas)))
    ) / (np.log(np.max(kappas)) - np.log(np.min(kappas)))
    axes[0].plot(
        kappas_dates[:-omit_data_parameter],
        smooth(level_set_line_transformed[:-omit_data_parameter]),
        linestyle="-",
        linewidth=linewidth,
        color="k",
        alpha=alpha_smooth,
        label="plausibility level set line $\kappa^{level}_t(p)$ for p=%.1f" % p0,
    )
    axes[0].plot(
        kappas_dates[:-omit_data_parameter],
        level_set_line_transformed[:-omit_data_parameter],
        linestyle="-",
        linewidth=linewidth,
        color=color_kappa,
        alpha=alpha_data,
    )
    axes[0].legend(
        loc=2,
        ncol=1,
        frameon=True,
        framealpha=0.9,
        bbox_to_anchor=(0.3, 0.2),
        facecolor="white",
        edgecolor="white",
    )
    axes[0].set_ylim(300, 0)

    axes[1].plot(
        kappas_dates[:-omit_data_parameter],
        1 / np.sqrt(kappa_level_set_line[:-omit_data_parameter]),
        linestyle=linestyle,
        color=color_kappa,
        alpha=alpha_data,
    )
    axes[1].plot(
        kappas_dates[:-omit_data_parameter],
        smooth(1 / np.sqrt(kappa_level_set_line[:-omit_data_parameter])),
        linestyle=linestyle,
        linewidth=linewidth,
        color=color_kappa,
        alpha=alpha_smooth,
        label="EffDI $\kappa_t(p)^{-1/2}$ for p=%.1f" % p0,
    )

    stoch_max = max(1 / np.sqrt(kappa_level_set_line[:-omit_data_parameter]))
    axes[1].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[1].set_ylim(0.0, stoch_max)
    axes[1].tick_params(axis="y", labelsize=fontsize_yticks)
    axes[1].set_ylabel("stochasticity")
    axes[1].tick_params(axis="y", colors="green")
    axes[1].legend(
        loc=2,
        ncol=1,
        frameon=True,
        framealpha=0.9,
        bbox_to_anchor=(0.3, 0.95),
        facecolor="white",
        edgecolor="white",
    )

    # plot r_eff
    axes[2].plot(
        kappas_dates[:-omit_data_parameter],
        r_eff_case[:-omit_data_parameter],
        linewidth=linewidth,
        color="k",
        label="estimated case reproduction number",
    )
    axes[2].plot(
        ts_dates,
        np.ones(len(ts_dates)),
        linewidth=linewidth,
        color="k",
        linestyle="dashed",
    )
    axes[2].set_ylim([0, 3])
    axes[2].tick_params(axis="y", labelsize=fontsize_yticks)
    axes[2].tick_params(axis="x", labelsize=fontsize_yticks)
    axes[2].set_ylabel("reprod. number")
    plt.tight_layout()


def place_intervals(
    axes, ts_dates, ts_values, zorder, low_incid_dates=None, high_incid_dates=None
):

    for ax in axes:
        if low_incid_dates is not None:
            ax.fill_between(
                low_incid_dates, 1e10, -1e10, color="yellow", alpha=0.1, zorder=zorder
            )
            ax.plot(
                low_incid_dates[:1] * 2,
                [-1e10, 1e10],
                color="k",
                linestyle="--",
                zorder=zorder,
            )
            ax.plot(
                low_incid_dates[-1:] * 2,
                [-1e10, 1e10],
                color="k",
                linestyle="--",
                zorder=zorder,
            )
        if high_incid_dates is not None:
            ax.fill_between(
                high_incid_dates, 1e10, -1e10, color="orange", alpha=0.1, zorder=zorder
            )
            ax.plot(
                high_incid_dates[:1] * 2,
                [-1e10, 1e10],
                color="k",
                linestyle="--",
                zorder=zorder,
            )
            ax.plot(
                high_incid_dates[-1:] * 2,
                [-1e10, 1e10],
                color="k",
                linestyle="--",
                zorder=zorder,
            )

    y_lim0 = axes[0].get_ylim()[1]
    y_position = (7.0 / 12.0) * y_lim0
    if low_incid_dates is not None:
        average_low = compute_average_range(ts_values, ts_dates, low_incid_dates)
        axes[0].text(
            low_incid_dates[0] + 0.5 * (low_incid_dates[1] - low_incid_dates[0]),
            y_position,
            "low incidence period\n%.0f daily cases in average" % average_low,
            horizontalalignment="center",
            bbox=dict(facecolor="white", alpha=0.9),
            zorder=zorder,
        )
    if high_incid_dates is not None:
        average_high = compute_average_range(ts_values, ts_dates, high_incid_dates)
        axes[0].text(
            high_incid_dates[0] + 0.5 * (high_incid_dates[1] - high_incid_dates[0]),
            y_position,
            "high incidence period\n%.0f daily cases in average" % average_high,
            horizontalalignment="center",
            bbox=dict(facecolor="white", alpha=0.9),
            zorder=zorder,
        )


def add_text(fig, x, y, text, fontweight=550, fontsize=14):

    ax = fig.add_axes(
        (x, y, 10, 10),
        alpha=1.0,
        frame_on=False,
    )
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for side in ["top", "bottom", "left", "right"]:
        ax.spines[side].set_visible(False)
    ax.text(
        0, 0, text, fontfamily="sans-serif", fontweight=fontweight, fontsize=fontsize
    )


def adjustments(axes):
    # adjust right border because of colorbar
    p0 = axes[0].get_position()  # bbox# get_position().get_points()
    p1 = axes[1].get_position()  # bbox# get_position().get_points()
    p2 = axes[2].get_position()
    p3 = axes[3].get_position()
    p0.x1 = p1.x1
    p2.x1 = p1.x1
    p3.x1 = p1.x1
    axes[0].set_position(p0)
    axes[2].set_position(p2)
    axes[3].set_position(p3)
