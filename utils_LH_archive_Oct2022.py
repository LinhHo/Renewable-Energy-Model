#!/usr/bin/env python
# -*- coding: utf-8 -*-

# useful general functions
###################################

import pandas as pd
import numpy as np
import datetime as dt
import xarray as xr

Dir = {'out_dir'   : '/net/respos/linhho/energy_model/',
       'path_const' : '/home/linhho/Documents/Code_PhD/CONST/',
       'dir_plot_test': '/net/respos/linhho/PLOTtest/',
       'dir_plot_save': '/home/linhho/Documents/Code_PhD/code_cosmo/PLOT_tosave/',
       'path_era5' : '/net/respos/linhho/ERA5_reanalysis_models_update_2020/ERA5_reanalysis_models/',
       'path_cosmo_48km': '/data/etcc/cosmo-rea6/hourly/48km/',
       'processed' : '/net/respos/linhho/processed/'
      }

gribfile = Dir['path_const'] + 'COSMO_REA6_CONST'
dir_cosmorea6_processed = '/net/respos/linhho/cosmo-rea6_processed/'
print(Dir['out_dir'])

###########################
### READ REA6 CONSTANTS ###
###########################

fname = Dir['path_const'] + '/COSMO_REA6_CONST_withOUTsponge.nc'
ncfile = xr.open_dataset(fname)
print(fname)

lat = ncfile.variables['RLAT'][:]
lon = ncfile.variables['RLON'][:]

nx = 848					# resolution
ny = 824					# resolution

new_lat = lat[0:ny:8, 0:nx:8]
new_lon = lon[0:ny:8, 0:nx:8]

# Pixels of interest
ilat = np.arange(0,ny);  ioi_lat = np.arange(0,ny,8)
ilon = np.arange(0,nx);  ioi_lon = np.arange(0,nx,8)

# Projection to plot COSMO rotated grid data with ccrs
import cartopy.crs as ccrs
rotated_projection_cosmo = ccrs.RotatedPole(pole_longitude=-162.0,
                  pole_latitude=39.25,
                  globe=ccrs.Globe(semimajor_axis=6370000,
                                   semiminor_axis=6370000))
cosmo_map_extent = [-23, 11, -16, 21] # original
cosmo_map_extent_DENA = [-23, 11, -16, 16] # to avoid empty value of radiation in the winter # [-23, 11, -16, 21] # original



###############################
#                                 
## read GWL file from Paul James 
#
###############################
"""
GWL (legacy 29 types) group by cyclone, anticyclone, westerly.. (James 2007)
HM, BM, TM: no wind direction!

Returns
-----------
daily
df1: dataframe of GWL with date and season
dict_GWL_id_winter, dict_GWL_id_summer: dictionary of indices of date where each GWL happens (for summer and winter)

Customised by each script
-----------
period = [start_year, end_year]

"""    

GWL_table = [['Nz',  'cyclonic', 'northerly',      'Cyclonic Northerly'],
            ['NWz',  'cyclonic', 'north-westerly', 'Cyclonic North-Westerly'],
            ['NEz',  'cyclonic', 'north-easterly', 'Cyclonic North-Easterly'],
            ['Sz',   'cyclonic', 'southerly',      'Cyclonic Southerly'],
            ['SEz',  'cyclonic', 'south-easterly', 'Cyclonic South-Easterly'],
            ['SWz',  'cyclonic', 'south-westerly', 'Cyclonic South-Westerly'],
            ['Wz',   'cyclonic', 'westerly',       'Cyclonic Westerly'],
            ['Ww',   'cyclonic', 'westerly',       'Maritime Westerly (Block Eastern Europe)'],
            ['Ws',   'cyclonic', 'westerly',       'South-Shifted Westerly'],
            ['TrM',  'cyclonic', 'northerly',      'Trough over Central Europe'],
            ['TrW',  'cyclonic', 'southerly',      'Trough over Western Europe'],
            ['TB',   'cyclonic', 'southerly',      'Low over the British Isles'],
            ['HNz',  'cyclonic', 'northerly',      'Icelandic High, Trough Central Europe'],
            ['HNFz', 'cyclonic', 'easterly',       'High Scandi-Iceland, Trough Central Europe'],
            ['HFz',  'cyclonic', 'easterly',       'Scandi- High, Trough Central Europe'],
            ['TM',   'cyclonic', ' ',              'Low (Cut-Off) over Central Europe'],
            ['HFa',  'anticyclonic', 'easterly',   'Scandi- High, Ridge Central Europe'],
            ['HNFa', 'anticyclonic', 'easterly',   'High Scandi-Iceland, Ridge Central Europe'],
            ['HNa',  'anticyclonic', 'northerly',  'Icelandic High, Ridge Central Europe'],
            ['Na',   'anticyclonic', 'northerly',  'Anticycloinic Northerly'],
            ['NWa',  'anticyclonic', 'north-westerly', 'Anticyclonic North-Westerly'],
            ['NEa',  'anticyclonic', 'north-easterly', 'Anticyclonic North-Easterly'],
            ['Wa',   'anticyclonic', 'westerly',   'Anticyclonic Westerly'],
            ['Sa',   'anticyclonic', 'southerly',  'Antiyclonic Southerly'],
            ['SWa',  'anticyclonic', 'south-westerly', 'Anticyclonic South-Westerly'],
            ['SEa',  'anticyclonic', 'south-easterly', 'Anticyclonic South-Easterly'],
            ['HM',   'anticyclonic', ' ',          'High over Central Europe'],
            ['HB',   'anticyclonic', 'northerly',  'High over the British Isles'],
            ['BM',   'anticyclonic', ' ',          'Zonal Ridge across Central Europe']]
GWL_table = pd.DataFrame(GWL_table, columns = ['lgcGWL', 'circulation', 'direction', 'fullname'])
GWL_table.set_index("lgcGWL", inplace=True)

df_GWL = pd.read_csv('/home/linhho/Documents/Code_PhD/DATA/EGWL_LegacyGWL.txt', delim_whitespace=True, header=None, error_bad_lines = False,
                 names=["Year", "Month", "Day", "id_EGWL", "EGWL", "id_lgcGWL", "lgcGWL"])
df_GWL['Date'] = pd.to_datetime(df_GWL[['Year', 'Month', 'Day']])
list_lgcGWL = df_GWL.lgcGWL.unique()
# print(list_lgcGWL)


# # Groupping GWLs based on scatter plot in Figure 3 Paper 1 (Climatology)
# GWL_groups = {'group1_highwindlowPV'   : ['SWz', 'Ww', 'Wa', 'Wz', 'NWz', 'SWa', 'TB'],
#              'group2_average' : ['Ws', 'Nz', 'NWa', 'TrW', 'TM', 'Sz', 'Sa', 'SEz', 'HFz', 'HNFz', 'HFa', 'SEa'],
#              'group3_lowwindhighPV': ['NEz', 'Na', 'NEa', 'TrM', 'BM', 'HB', 'HM', 'HNFa', 'HNz', 'HNa']
#              }
# # new version 2 for 1995-2017
# GWL_groups = {'High wind low PV'   : ['Ww', 'SWz', 'Wa', 'Wz', 'SWa', 'NWz', 'TB'],
#              'Average condition'   : ['Ws', 'Nz', 'NWa', 'Sz', 'TrW', 'Sa', 'HFz', 'SEz', 'HNFz'],
#              'Low wind high PV'    : ['BM', 'HFa', 'TM', 'TrM', 'SEa', 'HM', 'NEz', 'NEa', 'HB', 'Na', 'HNz', 'HNa', 'HNFa']
#              }
# new version 3 for 1995-2017 - with dark doldrum
GWL_groups = {'High wind'    : ['Ww', 'SWz', 'Wa', 'Wz', 'SWa', 'NWz', 'TB'],
              'Moderate'     : ['Nz', 'NWa', 'Sz', 'TrW', 'Sa', 'HFz'],
              'High PV'      : ['BM', 'HFa', 'TM', 'TrM', 'SEa', 'HM', 'NEz', 'NEa', 'HB', 'Na', 'HNz', 'HNa', 'HNFa'],
              'Dark doldrum' : ['Ws', 'SEz', 'HNFz'],
             }

# # select only one period corresponding to meteorology data
# if not period:  # empty list []
#     df1 = df.copy()
# else:
#     df1 = df_GWL.loc[(df_GWL['Year']>=period[0]) & (df_GWL['Year']<=period[1])].copy()  # deep copy to create a real new copy not depend on the old df
#     df1.set_index('Date')
#     df =  df1.set_index('Date').to_period().resample('H').ffill()  # convert from daily to hourly dataframe
#     df.reset_index(drop=False, inplace=True)

# new **SEASONAL** column with 'summer' assigned for 16 Apr to 15 Oct, and 'winter' for the rest
df_GWL_season = df_GWL.copy()
df_GWL_season['Season'] = np.nan
for year in df_GWL_season.Year.unique():
    df_GWL_season.loc[((df_GWL_season['Date']>dt.datetime(year,4,15)) & (df_GWL_season['Date']<dt.datetime(year,10,16))),'Season'] = "summer"
idx = df_GWL_season.index[df_GWL_season['Season'].isna()]  # where not summer
df_GWL_season.Season.iloc[idx] = "winter"


# count frequency per month
def count_frequency_GWL(df_input):
    df_GWL = df_input[['Month', 'lgcGWL']].copy()
    counts = df_GWL.groupby(['lgcGWL', 'Month'])
    # print(counts.size())
    df_frequency = counts.size()/len(df_input)*100  # actually a pd Series
    max_frequency = max(df_frequency)
    # print(max_frequency)
    dict_GWL_frequency = dict.fromkeys(list_lgcGWL)
    for GWL in list_lgcGWL:
        tmp = df_frequency.loc[GWL]
        gwl_count = pd.DataFrame({'Month':tmp.index, 'lgcGWL':tmp.values}) 
        month_missing = set(range(1,13)) - set(gwl_count.Month.unique())
        df_missing = pd.DataFrame({'Month': list(month_missing), 'lgcGWL': [0]*len(month_missing)})

        # plot seasonal frequency of each GWL
        # make sure to have complete 12 months statistics
        toplot = pd.concat([gwl_count, df_missing]).sort_values('Month').reset_index(drop=True)
        toplot['Month'] = ['J','F','M','A','M','J','J','A','S','O','N','D']
        dict_GWL_frequency[GWL] = toplot
    return dict_GWL_frequency

# Dictionary of indices of dates when each GWL type occurs for summer/winter
# ------------------------------------------------------------
df2 = df_GWL_season.loc[df_GWL_season['Season']=="winter"].copy()
df3 = df_GWL_season.loc[df_GWL_season['Season']=="summer"].copy()
print(df2.head(5))
print(df2.tail(5))
# get index of winter/summer GWL
dict_GWL_id = dict.fromkeys(list_lgcGWL)
dict_GWL_id_winter = dict.fromkeys(list_lgcGWL)
dict_GWL_id_summer = dict.fromkeys(list_lgcGWL)
for idx, GWL in enumerate(list_lgcGWL):
    dict_GWL_id[GWL] = df_GWL_season.index[df_GWL_season['lgcGWL']==GWL]
    dict_GWL_id_winter[GWL] = df2.index[df2['lgcGWL']==GWL]
    dict_GWL_id_summer[GWL] = df3.index[df3['lgcGWL']==GWL]
# print(df_GWL_season.loc[dict_GWL_id_winter['HM']])
# print(df_GWL_season.loc[dict_GWL_id_summer['HM']])
    
#     # Function: *mean* of
#     # ------------------------------------------------------------
#     def get_GWL_var(var):
#         dict_GWL_var_annual = dict.fromkeys(list_lgcGWL)
#         dict_GWL_var_winter = dict.fromkeys(list_lgcGWL)
#         dict_GWL_var_summer = dict.fromkeys(list_lgcGWL)

#         for idx, GWL in enumerate(list_lgcGWL):
#             dict_GWL_var_annual[GWL] = np.nanmean(var[dict_GWL_id[GWL],:,:], axis=0)
#             dict_GWL_var_winter[GWL] = np.nanmean(var[dict_GWL_id_winter[GWL],:,:], axis=0)
#             dict_GWL_var_summer[GWL] = np.nanmean(var[dict_GWL_id_summer[GWL],:,:], axis=0)

#         varmax = max(np.nanmax(dict_GWL_var_winter[max(dict_GWL_var_winter)]), np.nanmax(dict_GWL_var_summer[max(dict_GWL_var_summer)]))
#         varmin = min(np.nanmin(dict_GWL_var_winter[min(dict_GWL_var_winter)]), np.nanmin(dict_GWL_var_summer[min(dict_GWL_var_summer)]))
#         var_range = max(abs(varmax), abs(varmin))
#         print(varmax, varmin)

#     return df_GWL_season, df_frequency, dict_GWL_frequency, dict_GWL_id_winter, dict_GWL_id_summer #, get_GWL_var()

## OLD
# def anomalies_perGWL(var, nan_to_zero=False):
#     # nan_omit, e.g. [1,2,9,NaN]: when True, calculate withOUT nan value, reduce length of data, -> np.nanmean = 4
#     # when False, convert into 0 and calculate keeping the same length of data --> [1,2,9,0] -> np.mean = 3
# #     if nan_to_zero==True: var = np.nan_to_num(var)  # may crash, shouldn't use it

#     # np.nansum treats **NaN as zero** to keep mean over the same length    
#     var_mean_annual = np.nansum(var, axis=0)/var.shape[0]
#     dict_dif_var_annual = dict.fromkeys(list_lgcGWL)
#     for GWL in list_lgcGWL:
#         dict_dif_var_annual[GWL] = np.nansum(var[dict_GWL_id[GWL],:,:], axis=0)/len(dict_GWL_id[GWL]) - var_mean_annual

#     varmax = np.max(dict_dif_var_annual[max(dict_dif_var_annual)])
#     varmin = np.min(dict_dif_var_annual[min(dict_dif_var_annual)])
#     var_range = max(abs(varmax), abs(varmin))
#     print(varmax, varmin, var_range)
#     return dict_dif_var_annual, varmax, varmin, var_range

## New (2021-11-29), save computing var_mean_annual, and check if the shape is correct (103,106) gridded
def anomalies_perGWL(var, var_mean_annual=np.zeros(1), nan_to_zero=False):
    # nan_omit, e.g. [1,2,9,NaN]: when True, calculate withOUT nan value, reduce length of data, -> np.nanmean = 4
    # when False, convert into 0 and calculate keeping the same length of data --> [1,2,9,0] -> np.mean = 3
#     if nan_to_zero==True: var = np.nan_to_num(var)  # may crash, shouldn't use it

    # np.nansum treats **NaN as zero** to keep mean over the same length   
    # if var_mean (103,106 not provided), calculate it
    if var_mean_annual.shape!=(103,106):
        var_mean_annual = np.nansum(var, axis=0)/var.shape[0]
    dict_dif_var_annual = dict.fromkeys(list_lgcGWL)
    for GWL in list_lgcGWL:
        dict_dif_var_annual[GWL] = np.nansum(var[dict_GWL_id[GWL],:,:], axis=0)/len(dict_GWL_id[GWL]) - var_mean_annual

    varmax = np.nanmax(dict_dif_var_annual[max(dict_dif_var_annual)])
    varmin = np.nanmin(dict_dif_var_annual[min(dict_dif_var_annual)])
    var_range = max(abs(varmax), abs(varmin))
    print(varmax, varmin, var_range)
    return dict_dif_var_annual, varmax, varmin, var_range

def anomalies_perGWL_season(var):
    
    id_winter = df_GWL_season.index[df_GWL_season['Season']=="winter"]
    id_summer = df_GWL_season.index[df_GWL_season['Season']=="summer"]

    # mean over whole period, hence no NaN value
    var_mean_annual = np.nanmean(var, axis=0)
    var_mean_winter = np.nanmean(var[id_winter,:,:], axis=0)
    var_mean_summer = np.nanmean(var[id_summer,:,:], axis=0)

    dict_dif_var_annual = dict.fromkeys(list_lgcGWL)
    dict_dif_var_winter = dict.fromkeys(list_lgcGWL)
    dict_dif_var_summer = dict.fromkeys(list_lgcGWL)
    for GWL in list_lgcGWL:
        dict_dif_var_annual[GWL] = np.nanmean(var[dict_GWL_id[GWL],:,:], axis=0) - var_mean_annual
        dict_dif_var_winter[GWL] = np.nanmean(var[dict_GWL_id_winter[GWL],:,:], axis=0) - var_mean_winter
        dict_dif_var_summer[GWL] = np.nanmean(var[dict_GWL_id_summer[GWL],:,:], axis=0) - var_mean_summer

    varmax = max(np.nanmax(dict_dif_var_annual[max(dict_dif_var_annual)]), np.nanmax(dict_dif_var_winter[max(dict_dif_var_winter)]),
              np.nanmax(dict_dif_var_summer[max(dict_dif_var_summer)]))
    varmin = min(np.nanmin(dict_dif_var_annual[min(dict_dif_var_annual)]), np.nanmin(dict_dif_var_winter[min(dict_dif_var_winter)]),
              np.nanmin(dict_dif_var_summer[min(dict_dif_var_summer)]))
    var_range = max(abs(varmax), abs(varmin))
    print(varmax, varmin, var_range)
    return dict_dif_var_annual, dict_dif_var_winter, dict_dif_var_summer, varmax, varmin, var_range

# Divergence COLOUR PALETTE
##############################################
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# for radiation
top = cm.get_cmap('Greys_r', 8)
bottom = cm.get_cmap('Oranges', 8)
newcolors = np.vstack((top(np.linspace(0, 1, 8)),
                       bottom(np.linspace(0, 1, 8))))
newcmap_OrGy = ListedColormap(newcolors, name='OrangeGrey')

# for wind speed
top = cm.get_cmap('Purples_r', 8)
bottom = cm.get_cmap('Greens', 8)
newcolors = np.vstack((top(np.linspace(0, 1, 8)),
                       bottom(np.linspace(0, 1, 8))))
newcmap_PuGn = ListedColormap(newcolors, name='PurpleGreen')

def timerange(start_time, end_time, increment="hourly"):
    """
    Args
    -------
    start_time
        format: "%Y-%m-%d %H:%M"

    Returns
    -------
    list of datetime object within the given range, *EX*clude end_time
    """
    from datetime import datetime, timedelta

    start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M")
    end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M")
    out = []
    if increment=="hourly":
        delta = timedelta(hours=1)
    else:
        pass
    while start_time < end_time:
        # yield start_time
        out.append(start_time)
        start_time += delta
    print("Generate", increment, "time range from", start_time, "to", end_time)
    return out



############################################
#
#   Bivariate choroplete map
#
############################################
import matplotlib.colors as colors
bivariate_cmap = colors.ListedColormap(["#fef1e4", "#fab186", "#f3742d", 
                  "#97d0e7", "#b0988c", "#ab5f37", 
                  "#18aee5", "#407b7f", "#5c473d"])

# Functions from https://chart-studio.plotly.com/~empet/15191/texas-bivariate-choropleth-assoc/#/
def set_interval_value(x, a, b):
    # function that associate to a float x, a value encoding its position with respect to the interval [a, b]
    #  the associated values are 0, 1, 2 assigned as follows:
#     if np.isnan(x):
#         return None
    if x <= a: 
        return 0
    elif a < x <= b: 
        return 1
    else: 
        return 2
    
def data2color(x_array, y_array, a, b, c, d, biv_colors):
    # modify for x, y as 2D array rather than list
    # This function works only with a list of 9 bivariate colors, because of the definition of set_interval_value()
    # x, y: lists or 1d arrays, containing values of the two variables
    #  each x[k], y[k] is mapped to an int  value xv, respectively yv, representing its category,
    # from which we get their corresponding color  in the list of bivariate colors
    # a,b,c,d: var1_thresh1, var1_thresh2, var2_thresh1, var2_thresh2
    if  x_array.shape != y_array.shape:
        raise ValueError('the 2D array of x and y-coordinates must have the same shape')
    n_colors = len(biv_colors)
    if n_colors != 9:
        raise ValueError('the list of bivariate colors must have the length eaqual to 9')
    n = 3    
    x_flat = np.ndarray.flatten(x_array)
    y_flat = np.ndarray.flatten(y_array)
    xcol = [set_interval_value(v, a, b) for v in x_flat]
    ycol = [set_interval_value(v, c, d) for v in y_flat]
    idxcol = [int(xc + n*yc) for xc, yc in zip(xcol,ycol)]# index of the corresponding color in the list of bivariate colors
    colors = np.array(biv_colors)[idxcol]
    colors_array = np.reshape(colors, (-1,x_array.shape[1]))  # return 2D array
    return colors_array

def colorsquare(text_x, text_y, colorscale, n=3, xaxis ='x2', yaxis='y2'): 
    # text_x : list of n strings, representing intervals of values for the first variable or its n percentiles
    # text_y : list of n strings, representing intervals of values for the second variable or its n percentiles
    # colorscale: Plotly bivariate colorscale
    # returns the colorsquare as alegend for the bivariate choropleth, heatmap and more
    
    z = [[j+n*i for j in range(n)] for i in range(n)]
    n = len(text_x)
    if len(text_x) != n   or len(text_y) != n  or len(colorscale) != 2*n**2:
        raise ValueError('Your lists of strings  must have the length {n} and the colorscale, {n**2}')
    
    text = [[text_x[j]+'<br>'+text_y[i] for j in range(len(text_x))] for i in range(len(text_y))]
    return go.Heatmap(x=list(range(n)),
                      y=list(range(n)),
                      z=z,
                      xaxis=xaxis,
                      yaxis=yaxis, 
                      text=text, 
                      hoverinfo='text',  
                      colorscale=colorscale,
                      showscale=False)   

def colors_to_colorscale(biv_colors):
    # biv_colors: list of n**2 color codes in hexa or RGB255
    # returns a discrete colorscale  defined by biv_colors
    n = len(biv_colors)
    biv_colorscale = []
    for k, col in enumerate(biv_colors):
        biv_colorscale.extend([[round(k/n, 2) , col], [round((k+1)/n, 2), col]])
    return biv_colorscale


## scatter plot with colored density - for validation in Thesis (2022-08-22) =============================================

from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.stats import gaussian_kde

def densitycolor_scatter(x, y, ax=None, refline=True, title=None, max_density=None, **kwargs):
    """
    Scatter plot colored the standard way, slow for large number of data points
    https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib
    """

    # Calculate the point density
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    ax=ax
    ax.scatter(x, y, c=z, **kwargs)
    ax.set_aspect('equal')  # keep the plot square
    ax.set_title(title)
    print(ax)
    if refline:  ax.axline([0, 0], slope=1, c='red')  # add y=x line
    
    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    cbar = plt.colorbar(cm.ScalarMappable(norm = norm), ax=ax, shrink=.5)
    cbar.ax.set_ylabel('Density')

    return ax
