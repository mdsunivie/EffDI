
# Description

Python package to compute the effective dispersion index (EffDI). 


# Publication


G. Schneckenreither, L. Herrmann, R. Reisenhofer, N. Popper, P. Grohs <br /> 
Assessing the heterogeneity in the transmission of infectious diseases from time series of epidemiological data <br />
Medrxiv 2022 <br />
[DOI](https://doi.org/10.1101/2022.02.21.22271241)




Please consult Section 4.2 of this paper for technical explanations of the EffDI.




When you find this code useful for your own research, please cite the mentioned paper above. 


# Installation

Clone the repository and use

```bash
pip install -e EffDI/
```

to install the package.

# Usage of EffDI

In the first step results for the evaluation of the EffDI are computed and stored in the results/ folder. 
In the second step, these results are visualized and stored in the plots/ folder.

In the example of COVID-19 reported cases based on the file
[time_series_covid19_confirmed_global.csv](https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv), 
using default parameters, EffDI for any country in this data file may be computed. 
Please download and store this at the location, from where EffDI called. Otherwise parse the location to the respective function calls.
The following commands serve as an example of how to use EffDI.


Pre-compute the inverse and forward weights by 

```bash
effdi pre_compute_weights
```

Compute results for EffDI for a selection of countries by

```bash
effdi compute --countries "Austria" "Italy" "Korea, South"
```

Create a detailed plot for any of the countries, where you computed results for, by
```bash 
effdi demo_country --country "Austria"
```

Create a plot comparing a selection of countries, where you computed results for, by

```bash
effdi demo_countries --countries "Austria" "Italy" "Korea, South"
```
