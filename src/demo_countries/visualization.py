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


def plot_country_incid_kappa_line(
    ax,
    ts_dates,
    ts_reported_cases,
    kappa_level,
    p0,
    label,
    kappas_dates,
    idx=0,
    linewidth=1,
    fontsize_label=16,
    fontsize_legend=16,
    fontsize_yticks=16,
    idx_mode=0,
):

    # clear zeros from kappa_line
    idxs_non_zero = kappa_level > 0.01

    if label == "Korea, South":
        label = "South Korea"
    if idx == 0:
        ax_label = "reported cases in " + label
    else:
        ax_label = label

    idxs_active = idxs_non_zero
    if kappas_dates:
        if np.datetime64(kappas_dates[0]) < ts_dates.min():
            kappas_dates[0] = ts_dates.min()
        if np.datetime64(kappas_dates[1]) > ts_dates.max():
            kappas_dates[1] = ts_dates.max()
        idx_kappas_start = np.where(ts_dates == np.datetime64(kappas_dates[0]))[0][0]
        idx_kappas_end = np.where(ts_dates == np.datetime64(kappas_dates[1]))[0][0] + 1
        idxs_active[: idx_kappas_start + 1] = False
        idxs_active[idx_kappas_end:] = False

    ax_ys = ts_reported_cases[idxs_active]
    if idx_mode == 0:
        ax.fill_between(
            ts_dates[idxs_active],
            ax_ys,
            0.0,
            linewidth=0,
            color="k",
            alpha=0.2,
            label=ax_label,
        )

    kappas_dates = ts_dates[idxs_active]
    kappa_level = kappa_level[idxs_active]

    ax2 = ax.twinx()
    ax2_label = "EffDI for p=%.1f" % p0
    ax2_ys = 1.0 / np.sqrt(smooth(kappa_level))
    ax2.plot(kappas_dates, ax2_ys, color="green", label=ax2_label, linewidth=linewidth)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    if idx == 0:
        ax.legend(
            lines + lines2,
            labels + labels2,
            ncol=1,
            frameon=True,
            fontsize=fontsize_legend,
        )
    else:
        ax.legend(loc=0, ncol=1, frameon=True, fontsize=fontsize_legend)

    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    # modify yticks
    ax.set_ylabel("number of cases", fontsize=fontsize_label)

    axmax = np.max(ax_ys)
    ax_ylim = [0.0, np.ceil(0.0005 * axmax) / 0.0005]
    ax_yticks = [ax_ylim[0], np.mean(ax_ylim), ax_ylim[1]]
    ax.set_yticks(ax_yticks)
    ax.set_yticklabels(
        [
            f"{int(ax_yticks[0])}",
            f"{int(ax_yticks[1]/1000)}k",
            f"{int(ax_yticks[2]/1000)}k",
        ],
        fontsize=fontsize_yticks,
    )
    ax.tick_params(axis="y", labelsize=fontsize_yticks)

    ax2max = np.max(ax2_ys)
    ax2_ylim = [0.0, np.ceil(5 * ax2max) / 5]
    ax2.set_yticks([ax2_ylim[0], np.mean(ax2_ylim), ax2_ylim[1]])
    ax2.tick_params(axis="y", labelsize=fontsize_yticks)

    ax.tick_params(axis="x", labelsize=fontsize_yticks)
    plt.tight_layout()


def plot_marker(axes, markdates, fontsize_legend=16):
    marker = "a"
    for k in range(0, len(markdates), 2):
        idx = int(markdates[k])
        date = datetime.strptime(markdates[k + 1], "%Y-%m-%d")
        axes[idx].axvline(date, color="red", alpha=1.0, linestyle="-", linewidth=2)
        ylim = axes[idx].get_ylim()
        axes[idx].text(
            date, ylim[1], marker, color="red", rotation=0, fontsize=fontsize_legend
        )
        marker = chr(ord(marker) + 1)
