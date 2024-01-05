#!/usr/bin/env python
# -*- coding: utf-8 -*-

####################################################
#        Renewable energy model - POTENTIAL (no installed capacity)
#
# Input:
#       - COSMO-REA6 gridded data
#       - 2m-temperature
#       - wind speed u & v
#       - Short wave direct and diffuse radiation post-processed (ca 48 km horizontal) daily from C.Frank
#       - optimal angle from C.Frank
#       - ** NO ** installed capacity for PV and wind power plant 2013 and planned 2050 from CLIMIX model (Jerez et al 2015)
#       - 
# Output:
#       - netCDF filename, path [...]
#       - gridded data (ca 48 km horizontal) 
#       - dimension: lon (106), lat (103), time (daily)
#       - PV and wind power capacity factor (POTENTIAL)
#       - (not in ver 2) 2-m temperature, Mean sea level pressure, Wind speed horizontal 100m 
#
# by Linh Ho (linh.ho@uni-koeln.de) 2021-06-28, 2021-07-30, 2021-08-16
# used in [...]
# Adapt from Christopher Frank's PV simulation and wind power estimate based on power law
# 
####################################################

import numpy as np
import netCDF4 as nc
import xarray as xr
import pandas as pd
import datetime as dt
import glob
import pickle
from calendar import monthrange
import os
import math

from netCDF4 import Dataset
from numpy import isnan, float16, integer  ## LH
from mpl_toolkits.basemap import maskoceans
from skimage.measure import block_reduce
from calendar import monthrange

# only for year 2000 - Old script from CFrank
from cf_read_grib import get_data
from cf_read_grib import get_read_index
from cf_read_grib import read_grib_var_3

from working_with_datetime import array_of_dates
from working_with_datetime import seconds_to_datetime
from working_with_datetime import datetime_to_seconds
from working_with_datetime import match_dates

# Helper functions
def rad2grad(rad):
    return 180/np.pi*rad

def grad2rad(grad):
    return np.pi/180*grad

Dir = {'out_dir'   : '/net/respos/linhho/energy_model/',
       'path_const' : '/home/linhho/Documents/Code_PhD/CONST/',
       'path_cosmo_48km': '/data/etcc/cosmo-rea6/hourly/48km/'
      }
gribfile = Dir['path_const'] + 'COSMO_REA6_CONST'
print(gribfile)
print(Dir['out_dir'])
Param = {'rho'                  : 1.2295,
         'dTdh'                 : 0.0065,    # temperature gradient
         'alpha_on'             : 0.2,
         'alpha_off'            : 0.14,
         'effcoeff'             : 0.35,
         'cutin_speed'          : 3.5,
         'cutout_speed'         : 25,
         'rated_speed'          : 13,
         'hub_height_avg'       : 100}
Const = {'molar_mass'           : 0.02896,
         'gravitation'          : 9.807,
         'gas'                  : 8.314,
         'lapse_rate'           : 0.0065,
         'karman'               : 0.35,
         'heat_capacity'        : 1004.5}

save_COSMO_48km =  False

###########################
### READ REA6 CONSTANTS ###
###########################

fname = Dir['path_const'] + 'COSMO_REA6_CONST_withOUTsponge.nc'
ncfile = Dataset(fname,'r')
print(fname)

lat = ncfile.variables['RLAT'][:]
lon = ncfile.variables['RLON'][:]

# grib_ids = np.arange(44,45)
# lat = get_data(ny,nx,grib_ids,gribfile)
# grib_ids = np.arange(45,46)
# lon = get_data(ny,nx,grib_ids,gribfile)

nx = 848					# resolution
ny = 824					# resolution

ilat = np.arange(0,ny)
ilon = np.arange(0,nx)

# Pixels of interest
ioi_lat = np.arange(0,ny,8)
ioi_lon = np.arange(0,nx,8)

# Define date of interest
years = range(2017,2018)
months = ['01','02', '03','04','05','06','07','08','09','10','11','12']

time_range = range(00,24)
minu= np.array([0])           # Values used from observations

##########################
# read the height of level 36 and 37 (4th and 3rd from the ground) hlev36, hlev37
# by subtracting constant field by height of surface hsurf, reduce to 8 pxiel grid (48 km)
##########################
hlev = ncfile.variables['HHL'][:].data
hsurf = ncfile.variables['HSURF'][:].data
hlev36 = hlev[36][::8,::8] - hsurf[::8,::8]
hlev37 = hlev[37][::8,::8] - hsurf[::8,::8]

hub_height = Param['hub_height_avg']

# Start month for first year:
for year in years:
    datum_year =  array_of_dates([year], time_range, minu)

    yyyy = str(year)
    print('year ' + yyyy)
    

    ################################
    ### Read REA6 temperature 2m ###
    ################################
    fname = '/data/etcc/cosmo-rea6/hourly/T_2M.2D/' + yyyy + '/T_2M.2D.*.grb'
    try:
        ds = xr.open_mfdataset(fname, engine="cfgrib", parallel=False)
        pred_new = ds['t2m'].data
        T2m_all = np.array(pred_new[:][:,:,ioi_lon][:,ioi_lat])
    except:
        ds1 = xr.open_mfdataset(fname, engine="cfgrib", backend_kwargs={'filter_by_keys':{'typeOfLevel': 'surface', u'iDirectionIncrementInDegrees': 0.055}})
        ds2 = xr.open_mfdataset(fname, engine="cfgrib", backend_kwargs={'filter_by_keys':{'typeOfLevel': 'surface', u'iDirectionIncrementInDegrees': 0.05500118063754427}})
        ds_concat = xr.concat([ds1, ds2], dim="time")
        pred_new = ds_concat['t2m'].sortby('time')
        T2m_all = np.array(pred_new[:][:,:,ioi_lon][:,ioi_lat])

    print('=============== 2-m Temperature ==================================')
    print(T2m_all.shape)
#     print(T2m_all)

    ############################
    ### Read REA6 wind speed ###
    ############################
    fname = '/data/etcc/cosmo-rea6/hourly/U_10M.2D/' + yyyy + '/U_10M.2D.*.grb'
    ds = xr.open_mfdataset(fname, engine="cfgrib", parallel=False)
    pred_new = ds['u10'].data
    U_10m = np.array(pred_new[:][:,:,ioi_lon][:,ioi_lat])
    
    fname = '/data/etcc/cosmo-rea6/hourly/V_10M.2D/' + yyyy + '/V_10M.2D.*.grb'
    ds = xr.open_mfdataset(fname, engine="cfgrib", parallel=False)
    pred_new = ds['v10'].data
    V_10m = np.array(pred_new[:][:,:,ioi_lon][:,ioi_lat])

    wsp = (U_10m**2 + V_10m**2)**0.5
    print('============== Wind speed at 10 m =============================')
    print(wsp.shape)
#     print(wsp)
    
    time_step = wsp.shape[0]
    ####################################################################################################
    #                                                                                                  #
    #        PV capacity factor simulation and energy generation                                       #
    #                                                                                                  #
    ####################################################################################################


    #################################
    ### Read REA6 dirpp and difpp ###
    #################################
    # direct radiation
    fname = '/data/etcc/cosmo-rea6/hourly/SWDIRS_RAD.2D/' + yyyy + '/SWDIRS_RAD.2D.*.grb'
    ds = xr.open_mfdataset(fname, engine="cfgrib", parallel=False)
    pred_new = ds['SWDIRS_RAD'].data
    SWDIR = np.array(pred_new[:][:,:,ioi_lon][:,ioi_lat])
    
    # diffuse radiation
    fname = '/data/etcc/cosmo-rea6/hourly/SWDIFDS_RAD.2D/' + yyyy + '/SWDIFDS_RAD.2D.*.grb'
    ds = xr.open_mfdataset(fname, engine="cfgrib", parallel=False)
    pred_new = ds['SWDIFDS_RAD'].data
    SWDIFD = np.array(pred_new[:][:,:,ioi_lon][:,ioi_lat])

    Q_dir_all = SWDIR
    Q_dif_all = SWDIFD
    print('=============== Short-wave radiation direct ==================================')
    print(Q_dir_all.shape)
#     print(Q_dir_all)

    ########################
    ### Read REA6 albedo ###
    ########################
    
    ## Cfgrib cannot read albedo for 2017, I converted it to netCDF file (2023-02-14)
    if year == 2017:
        list_files = []
    for month in months: 
        tmp = '/data/etcc/cosmo-rea6/hourly/ALB_2D.2017' + '/ALB_RAD.2D.' + yyyy + month + '.nc' 
        list_files.append(tmp)
    ds = xr.open_mfdataset(list_files)
    ds = ds.resample(time='1H').asfreq()  # add NA to missing time steps
    pred_new = ds['var84'].data
    albedo = np.array(pred_new[:][:,:,ioi_lon][:,ioi_lat])
    
    else:
        for month in months:
            fname = '/data/etcc/cosmo-rea6/ALB_RAD.2D/' + yyyy + '/' + month + '/ALB_RAD.2D.' + yyyy + month + '.grb'
            if year == 2018:  # I downloaded data for 2018 from Jan Keller later and put in a different directory
                fname = '/data/etcc/cosmo-rea6/hourly/ALBEDO2018/' + month + '/ALB_RAD.2D.' + yyyy + month + '.grb'

            # avoid filter_by_keys DatasetBuildError sometimes happens for whatever reason
            try:
                ds = xr.open_dataset(fname, engine="cfgrib")
                ds = ds.resample(time='1H').asfreq()  # add NA to missing time steps
                pred_new = ds['al'].data
                values = np.array(pred_new[:][:,:,ioi_lon][:,ioi_lat])
            # Only apply from year 2015 onward, no need to fill NA value with resample
            except:
                ds1 = xr.open_dataset(fname, engine="cfgrib", 
                                 backend_kwargs={'filter_by_keys':{'typeOfLevel': 'surface', u'latitudeOfLastGridPointInDegrees': 21.863}})
                ds2 = xr.open_dataset(fname, engine="cfgrib", 
                                     backend_kwargs={'filter_by_keys':{'typeOfLevel': 'surface', u'latitudeOfLastGridPointInDegrees': 21.862}})
                ds_concat = xr.concat([ds1, ds2], dim="time")
                pred_new = ds_concat['al'].sortby('time')
                values = np.array(pred_new[:][:,:,ioi_lon][:,ioi_lat])

    #         del ds, pred_new

            if month == months[0]:
                albedo = values
            else:
                albedo = np.concatenate((albedo,values),axis=0)
        if year < 2015:  # years before 2015 miss the first step
            albedo = np.concatenate((albedo[None,0,:,:], albedo), axis=0)  # repeat first step to add missing data yyyy-01-01 00:00:00
    #     if year == 2007:
    #         albedo = np.insert(albedo, 5755, np.full([6, 103, 106], np.nan), axis=0)

    print('=============== Albedo ==================================')
    print(albedo.shape)

    #################################
    ### Read sun elevation_angles ###
    #################################
    sun_elevation = np.nan * np.empty( (len(datum_year), len(ioi_lat), len(ioi_lon))) #, dtype=float16 )
    sun_azimuth = np.nan * np.empty( (len(datum_year), len(ioi_lat), len(ioi_lon))) #, dtype=float16 )

    for i_month,month in enumerate(np.arange(1,13)):
        mm = '%02d' %month

        days = monthrange(int(yyyy),month)[1]
        for day in np.arange(0,days):
#             print 'Day %s'  %day
            day_str = '%02d'     % (day + 1)

            in_dir = '/data/etcc/cosmo-rea6/15min/sun_position/'+yyyy+'/'+mm+'/'
            in_file = in_dir +yyyy+mm+day_str+'_sun_position_with0.1precision.nc'
            f=nc.Dataset(in_file, 'r')
            #nc_attrs, nc_dims, nc_vars = ncdump(f)
            dt_day = seconds_to_datetime(f.variables['TIME'][:])
            ioi_scratch, ioi_year = match_dates(dt_day, datum_year)
            sun_elevation[ioi_year] = f.variables['elevation_angle'][:][:,:,ioi_lon][:,ioi_lat][ioi_scratch]
            sun_azimuth[ioi_year] = f.variables['azimuth_angle'][:][:,:,ioi_lon][:,ioi_lat][ioi_scratch]


    ################################
    ### Read optimal tilt anlges ###
    ################################
    tmp_path = '/data/herz/cf_data/power_estimate_PV/'
    input_vars = pickle.load( open( tmp_path+"opt_tilt_est_2014_europe_48x48_tmp.npy", "rb" ) )
    opt_tilt_angle = input_vars['opt_tilt'] * 0.7  # 0.7 is the adjustment to real installations (see Saint-Drenan2018)

    #####################################
    ### Calculate the power estimates ###
    #####################################
    # Get direct radiation on tilted angle
    dir_rea6_on_tilt = np.nan * np.empty( (len(datum_year), len(ioi_lat), len(ioi_lon)) )
    theta_rad = np.nan * np.empty( (len(datum_year), len(ioi_lat), len(ioi_lon)) )
    for i_lat in np.arange(len(ioi_lat)):
        for i_lon in np.arange(len(ioi_lon)):
            a = - np.cos(grad2rad(sun_elevation[:,i_lat,i_lon])) * np.sin(grad2rad(opt_tilt_angle[i_lat,i_lon])) * np.cos(grad2rad(sun_azimuth[:,i_lat,i_lon]))
            b = np.sin(grad2rad(sun_elevation[:,i_lat,i_lon])) * np.cos(grad2rad(opt_tilt_angle[i_lat,i_lon]))
            theta_rad[:,i_lat,i_lon] = np.arccos(a+b)
            #del a,b
            dir_rea6_on_tilt[:,i_lat,i_lon] = Q_dir_all[:,i_lat,i_lon] * np.cos(theta_rad[:,i_lat,i_lon])/np.sin(grad2rad(sun_elevation[:,i_lat,i_lon]))
    dir_rea6_on_tilt[dir_rea6_on_tilt<0] = 0

    # Get diffuse radiation on tilted angle
    dif_rea6_on_tilt = np.nan * np.empty( (len(datum_year), len(ioi_lat), len(ioi_lon)) )
    soil_reflection = np.nan * np.empty( (len(datum_year), len(ioi_lat), len(ioi_lon)) )
    for i_lat in np.arange(len(ioi_lat)):
        for i_lon in np.arange(len(ioi_lon)):
            # Klucher model
            #### UNPERTURBED
            F = 1 - (Q_dif_all[:,i_lat,i_lon]/(Q_dir_all[:,i_lat,i_lon]+Q_dif_all[:,i_lat,i_lon]))**2
            F[(Q_dir_all[:,i_lat,i_lon]+Q_dif_all[:,i_lat,i_lon]) == 0] = 0
            dif_rea6_on_tilt[:,i_lat,i_lon] = ( Q_dif_all[:,i_lat,i_lon] * 0.5 * (1+np.cos(grad2rad(opt_tilt_angle[i_lat,i_lon]))) *
            ( 1 + F * (np.sin(grad2rad(opt_tilt_angle[i_lat,i_lon])/2))**3) *
            ( 1 + F * (np.cos(theta_rad[:,i_lat,i_lon]))**2 * (np.cos(sun_elevation[:,i_lat,i_lon]))**3 ) )

            # Bodenreflexion
            soil_reflection[:,i_lat,i_lon] = (Q_dir_all[:,i_lat,i_lon]+Q_dif_all[:,i_lat,i_lon]) * albedo[:,i_lat,i_lon] * 0.01 * 0.5 * (1-np.cos(grad2rad(opt_tilt_angle[i_lat,i_lon])))


    ########################################
    ### Calculate the PV-power estimates ###
    ########################################
    ### Angle of incidence losses in direct radiation ### Martin and Ruiz 2001
    AL = 1-(1-np.exp(-np.cos(theta_rad)/0.16)/(1-np.exp(-1/0.16)))
    dir_rea6_on_tilt_with_AL = dir_rea6_on_tilt * (1-AL)

    #dir_reduction_due_to_AL = 1-nansum(dir_rea6_on_tilt_with_AL)/nansum(dir_rea6_on_tilt)
    #--> Im Jahr 2007 ergibt sich eine Gesamtreduktion (CORDEX-dom) der Direktstrahlung von 1.77%

    # We need module temperature (S. 201)
    #eff_stc = 0.14   # P. 198 Poly-Si
    T_module = (T2m_all-273.15) + (dir_rea6_on_tilt_with_AL + dif_rea6_on_tilt + soil_reflection) / (26.9 + 6.2 * (2./10.)**0.2*wsp)  # Feiman2008 model with coefficients from Koehl2011 for c-SI module

    #T_module_without_wind = (T2m_all-273.15) + (dir_rea6_on_tilt_with_AL + dif_rea6_on_tilt + soil_reflection) / (26.9)
    #1 - nanmean(T_module) / nanmean(T_module_without_wind)
    #-->Im Jahr 2007 ergibt sich eine Gesamtreduktion (CORDEX-dom) der Modultemperatur von 13.75 % von 18.59° auf 16.04°.

    # Model from Huld2011:
    Gs = (dir_rea6_on_tilt_with_AL + dif_rea6_on_tilt + soil_reflection) / 1000
    Ts = T_module - 25
    k1 = -0.017237
    k2 = -0.040465
    k3 = -0.004702
    k4 = 0.000149
    k5 = 0.000170
    k6 = 0.000005

    Pstrich = Gs * ( 1 + k1*np.log(Gs) + k2 * (np.log(Gs))**2 + k3*Ts + k4*Ts*np.log(Gs) + k5*Ts*(np.log(Gs))**2 + k6*Ts**2 )
    Pstrich[Gs==0] = 0
    
#     PV = Pstrich*np.repeat(ic_PV, time_step, axis=0)  # shape of ic_PV now is [1,103,106]
#     Pstrich[Pstrich==0] = np.nan
#     PV[PV==0] = np.nan

    ####################################################################################################
    #                                                                                                  #
    #            Wind power model                                                                      #
    #                                                                                                  #
    ####################################################################################################

    ########################################
    # Get wind speed at level 36 and 37
    # To interpolate wind speed at hub height
    ########################################

    # read nc file wind speed at lev 36-37
    for month in months:    
        filename = '/data/etcc/cosmo-rea6/hourly/Wind_lev36-37/' + yyyy + '/' + yyyy + month + '_wind_speed_level_36-37.nc'
        ds = xr.open_dataset(filename)
        w36 = ds['wsp36'].data
        w37 = ds['wsp36'].data
        if month == months[0]:
            wsp36 = w36
            wsp37 = w37
        else:
            wsp36 = np.vstack((wsp36, w36))
            wsp37 = np.vstack((wsp37, w37))
    time_step = wsp36.shape[0]
    wsp_avg_36_37 = (wsp36 + wsp37)/2
        
    print("Shape of 3D wind speed at levels 36 and 37: ", wsp36.shape, wsp37.shape)
    
    ########################################
    # Calculate air density
    ########################################
    # air density can be set constant or
    # calculated from meteorological data via the barometric formula
    # later: read atm vertical profile, e.g. inversion
    # If keep constant air density

    a = ((Const['molar_mass']*Const['gravitation'])/(Const['gas']*Param['dTdh'])) - 1

    ########################################
    # Vertical interpolation of wind speeds at a reference height (e.g. 2m/10m) using
    ########################################
    # Create installed capacity (ic_wp) and hub_height with corresponding dimension
    # i.e. repeat along time dimension

    # Interpolate wind speed at hub height: v_hub = (hub_height - h_lev37)/(h_lev36-h_lev37) * (v_36-v_37) + v_37
    # Then calculated capacity factor of wind output (lower than rated wind speed): (Matthew 2006, Tobin 2016)
    # (vhub**3 - cutin_speed**3) / (rated_speed**3 - cutin_speed**3)
    
#     IC_timestep = np.repeat(ic_wp[None,k,:,:], time_step, axis=0)

    rhoLEV = Param['rho'] * (1 - (Param['dTdh']/T2m_all)*hub_height )**a
    vhub = (hub_height-hlev37)/(hlev36-hlev37) * (wsp36-wsp37) + wsp37

    # Power law: define where cutin <= v < rated   
    ############
    id_cubic = (vhub >= Param['cutin_speed']) & (vhub < Param['rated_speed'])
    # define where rated <= v <= cutout
    id_rated = (vhub >= Param['rated_speed']) & (vhub <= Param['cutout_speed'])

    # on or offshore turbine will have different capacity and rotor diameter
    # now use installed capacity from CLIMIX
    cf_eout = np.full_like(wsp, 0)  # capacity factor

    ########################################
    # Calculate energy output
    ########################################
    cf_eout[id_cubic] = (vhub[id_cubic]**3 - Param['cutin_speed']**3) / (Param['rated_speed']**3 - Param['cutin_speed']**3)
    cf_eout[id_rated] = 1

#     cf_eout[cf_eout==0] = np.nan

    print('=================== Wind energy ===========================')
    print("Wind output and capacity factor for 2013: ", cf_eout.shape)
#     print(cf_eout)
    

    ########################################
    ### SAVING IN NETCDF4                ###
    ########################################

    # # Save energy POTENTIAL
    # ##################################
    filename = Dir['out_dir'] + 'potential/' +  yyyy + '_PV_wind_potential_v2.nc'
    time = pd.date_range(dt.datetime(year, 1, 1), dt.datetime(year+1, 1, 1), freq="H", closed="left") #, periods=365 * 24)

    ds = xr.Dataset({
                     "Wp": (('time', 'lon','lat'), cf_eout),
                     "PV": (('time', 'lon','lat'), Pstrich),
                     "time": time})
    ds['Wp'].attrs = {'units': 'No unit',
                      'name': 'Capacity factor of wind power in Europe 2013 CLIMIX'}
    ds['PV'].attrs = {'units': 'No unit',
                      'name': 'Capacity factor of PV power in Europe 2050 CLIMIX'}
    
    ds.attrs['Conventions'] = ''  # 'CF-1.7'
    ds.attrs['Title'] = 'Energy POTENTIAL, i.e. capacity factor without installed capcity CLIMIX, version 2 (2021-09-07)'
    ds.attrs['Author'] = 'Linh Ho, Institute of Geophysics and Meteorology, University of Cologne, Germany'
#     ds.attrs['source'] = 'WRF-1.5'
    ds.attrs['History'] = str(dt.datetime.utcnow()) + ' Python2'
    ds.attrs['References'] = ''
    ds.attrs['Notes'] = 'Wind speed at hubheight FIXED at 100 m interpolated from wind at level 36 and 37 (rouhgly 80-150 m)'

    ds.to_netcdf(filename, 'w')
    print('\n =================== Finish saving energy POTENTIAL for year', yyyy, '=========================== \n')

    ## Save meteorological data COSMO-REA6 hourly at 48km for later analysis
    ##########################################################################
    ########################################
    #
    #  Get MSL pressure at 48 km for later plots
    #
    ########################################   
    if save_COSMO_48km == True:
        fname = '/data/etcc/cosmo-rea6/hourly/PMSL.2D/' + yyyy + '/*.grb'

        # avoid filter_by_keys DatasetBuildError sometimes happens for whatever reason
        try:
            ds = xr.open_mfdataset(fname, engine="cfgrib", parallel=False)
            pred_new = ds['msl'].data
            values = np.array(pred_new[:][:,:,ioi_lon][:,ioi_lat])
        except:
            ds = xr.open_dataset(fname, engine="cfgrib", 
                             backend_kwargs={'filter_by_keys':{u'iDirectionIncrementInDegrees': 0.055}})
            pred_new = ds['msl'].data
            values1 = np.array(pred_new[:][:,:,ioi_lon][:,ioi_lat])
            ds = xr.open_dataset(fname, engine="cfgrib", 
                                 backend_kwargs={'filter_by_keys':{u'iDirectionIncrementInDegrees': 0.05500118063754427}})
            pred_new = ds['msl'].data
            values2 = np.array(pred_new[:][:,:,ioi_lon][:,ioi_lat])
            values = np.concatenate((values1, values2), axis=0)
        pmsl = values
        print('=================== Mean see level pressure ===========================')
        print("Mean sea level pressure: ", pmsl.shape)
#         print(pmsl)

        filename = Dir['path_cosmo_48km'] + yyyy + '_COSMO-REA6_hourly_48km.nc'
        time = pd.date_range(dt.datetime(year, 1, 1), dt.datetime(year+1, 1, 1), freq="H", closed="left") #, periods=365 * 24)

        ds = xr.Dataset({"wsp100m_36_37": (('time', 'lon','lat'), vhub),
                         "total_radiation": (('time', 'lon','lat'), SWDIR+SWDIFD),
                         "wsp": (('time', 'lon','lat'), wsp),
                         "t2m": (('time', 'lon','lat'), T2m_all),
                         "pmsl": (('time', 'lon','lat'), pmsl),
                         "time": time})
        ds['wsp100m_36_37'].attrs = {'units': 'ms-1',
                          'name': 'Wind speed at 100m interpolated from level 36 and 37 from COSMO-REA6'}
        ds['total_radiation'].attrs = {'units': 'Wm-2',
                          'name': 'total short-wave surface radiation from COSMO-REA6'}
        ds['wsp'].attrs = {'units': 'ms-1',
                          'name': 'Wind speed calculated from U_10m and V_10m from COSMO-REA6'}
        ds['t2m'].attrs = {'units': 'K',
                          'name': '2m-temperature from COSMO-REA6'}
        ds['pmsl'].attrs = {'units': 'Pa',
                          'name': 'Mean sea level pressure hourly from COSMO-REA6 48 km, selected to every 8 grid point'}
        ds.attrs['Notes'] = 'COSMO-REA6 data reduced to 48 km horizontal resolution for energy model'
        ds.to_netcdf(filename, 'w')
        print('\n =================== Finish saving COSMO 48 km for year', yyyy, '=========================== \n')
