This directory contains necessary files to reproduce the results in the article A climatology of weather-driven anomalies in European photovoltaic and wind power production (Ho and Fiedler, 2024), including:

(1) Output of the Renewable Energy Model (REM) as described in (Ho and Fiedler, 2024), last modification on 30.10.2023 from Linh Ho, named year_PV_wind_generation_v2.nc, with 23 years from 1995 to 2017.
REM includes one simulation of photovoltaic (PV) power production and one simulation of wind power production across European domain, with a horizontal resolution of 48 km, hourly output for the period 1995--2017.

The output has a European domain with the same size as in the reanalysis dataset COSMO-REA6. This is a rotated grid with the coordinates of the rotated North Pole −162.0, 39.25, and of the lower left corner −23.375, −28.375. See Bollmeyer et al. (2014, http://doi.org/10.1002/qj.2486).
Data downloaded from https://opendata.dwd.de/climate_environment/REA/COSMO_REA6/ 

(2) Weather pattern classification daily for Europe from 1995 to April 2020, named EGWL_LegacyGWL.txt, from James (2007, http://doi.org/10.1007/s00704-006-0239-3)

(3) The installation data of PV and wind power in Europe for one scenario in 2050 from the CLIMIX model, processed to have the same horizontal resolution as in REM, named installed_capacity_PV_wind_power_from_CLIMIX_final.nc.
Original data were provided at 0.11 degree resolution, acquired from personal communication with the author from Jerez et al. (2015, http://doi.org/10.1016/j.rser.2014.09.041)

(4) Python scripts of REM, including:
- model_PV_wind_complete_v2.py: the main script to produce REM output
- model_PV_wind_potential_v2.py: produce potential (capacity factor) of PV and wind power for model evaluations, e.g., against CDS and Renewables Ninja data, as descript in Ho and Fiedler (2024)
- model_PV_wind_complete_v1_ONLYyear2000.py: a separate Python script to produce REM output only for the year 2000. Note that the data for 2000 from COSMO-REA6 were read in a different approach (using cfgrib) probably due to the time stamp changes at the beginning of the milenium, also explains the larger size of the final output
- utils_LH_archive_Oct2022.py: contains necessary Python functions to run the other scripts

(5) Jupyter notebook files to reproduce the figures in Ho and Fiedler (2024), named Paper1_Fig*_**.ipynb

(6) Time series of European-aggregated PV and wind power production hourly during the period 1995--2017, processed data from the dataset (1) to facilitate the reproduction of the figures, including two installations scale-2019 and scenario-2050:
- Timeseries_all_hourly_1995_2017_GW_scale2019.csv
- Timeseries_all_hourly_1995_2017_GW_scen2050.csv



