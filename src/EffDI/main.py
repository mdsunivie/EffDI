import argparse

from EffDI.demo_country import *
from EffDI.demo_countries import *
from EffDI.compute import *
from EffDI.pre_compute_weights import *

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    parser_pcw = subparsers.add_parser("pre_compute_weights")

    #arguments for pre_compute_weights
    parser_pcw.add_argument(
        "--fwd-distribution",
        type=str,
        choices=["gamma", "skewnorm", "delta"],
        help="type of distribution to compute the fwd weights",
        default="delta",
        metavar="fwd_distribution",
    )
    parser_pcw.add_argument(
        "--inv-distribution",
        type=str,
        choices=["gamma", "skewnorm", "delta"],
        help="type of distribution to compute the fwd weights",
        default="gamma",
        metavar="inv_distribution",
    )
    #arguments for compute
    parser_c = subparsers.add_parser("compute")
    parser_c.add_argument(
        "--countries", nargs="*", type=str, help="countries", default=["Austria"]
    )
    parser_c.add_argument(
        "--data_file",
        nargs=1,
        type=str,
        help=".csv file with daily incidence time series",
        default="time_series_covid19_confirmed_global.csv",
    )
    parser_c.add_argument(
        "--inv_weights",
        type=str,
        help="cvs file with inverse weights",
        default="inv_weights.csv",
    )
    parser_c.add_argument(
        "--fwd_weights",
        type=str,
        help="cvs file with forward weights",
        default="fwd_weights.csv",
    )
    parser_c.add_argument(
        "--mode",
        type=str,
        help="only trend part of seasonal trend model",
        choices=["c", "t", "st"],
        default="st",
    )
    parser_c.add_argument(
        "--tau", nargs=2, type=int, help="tau1 and tau2", default=[6, 7]
    )
    parser_c.add_argument(
        "--k_range",
        nargs=2,
        type=int,
        help="range of parameter k (logarithmic scale, base 10)",
        default=[np.log10(0.1), 4],
    )
    parser_c.add_argument(
        "--k_samp", type=int, help="samples of k in logarithmic scale", default=300
    )
    parser_c.add_argument("--n", type=int, help="number of sample for model", default=500)
    parser_c.add_argument(
        "--distribution",
        type=str,
        choices=["gamma", "NB"],
        help="distribution used to model secondary infections",
        default="gamma",
    )
    #arguments for demo_country
    parser_dc = subparsers.add_parser("demo_country")
    parser_dc.add_argument(
        "--country",
        type=str,
        help="Country, for which stochasticity is visualized",
        default=["Austria"],
    )
    parser_dc.add_argument(
        "--dates",
        nargs=2,
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        help="date range of time series",
        default=None,
    )
    parser_dc.add_argument(
        "--low_incid_dates",
        nargs=2,
        type=lambda s: np.datetime64(datetime.strptime(s, "%Y-%m-%d")),
        help="date range of time series",
        default=None,
    )
    parser_dc.add_argument(
        "--high_incid_dates",
        nargs=2,
        type=lambda s: np.datetime64(datetime.strptime(s, "%Y-%m-%d")),
        help="date range of time series",
        default=None,
    )
    parser_dc.add_argument(
        "--mode",
        type=str,
        help="only trend part of seasonal trend model",
        choices=["c", "t", "st"],
        default="st",
    )
    #arguments for demo_countries
    parser_dcs = subparsers.add_parser("demo_countries")
    parser_dcs.add_argument(
        "--countries", nargs="*", type=str, help="countries", default=["Austria"]
    )
    parser_dcs.add_argument(
        "--dates",
        nargs=2,
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        help="date range of time series",
        default=None,
    )
    parser_dcs.add_argument(
        "--modes",
        nargs="*",
        type=str,
        help="only trend part of seasonal trend model",
        choices=["c", "t", "st"],
        default=["st"],
    )
    parser_dcs.add_argument(
        "--name", type=str, help="name of calculation", default="demo_other_countries"
    )
    parser_dcs.add_argument(
        "--markdates",
        type=str,
        nargs="*",
        help="list of special dates followed by index of directory",
        default=None,
    )

    args = parser.parse_args()

    if args.command == "pre_compute_weights":
        pre_compute_weights(fwd_distribution=args.fwd_distribution,
                            inv_distribution=args.inv_distribution)

    if args.command == "compute":
        compute(data_file=args.data_file,
                inv_weights=args.inv_weights,
                fwd_weights=args.fwd_weights,
                countries=args.countries,
                mode=args.mode,
                tau=args.tau,
                k_range=args.k_range,
                k_samp=args.k_samp,
                n=args.n,
                distribution=args.distribution)

    if args.command == "demo_country":
        demo_country(country_arg=args.country,
                     dates=args.dates,
                     low_incid_dates=args.low_incid_dates,
                     high_incid_dates=args.high_incid_dates,
                     mode=args.mode)

    if args.command == "demo_countries":
        demo_countries(countries=args.countries,
                       dates=args.dates,
                       modes=args.modes,
                       name=args.name,
                       markdates=args.markdates)



