#!/usr/bin/env python
# -*- coding: utf-8 -*-

####################################################
#        Renewable energy model ONLY >>>> YEAR 2000 <<<<<<<<
#
# Input:
#       - COSMO-REA6 gridded data
#       - 2m-temperature
#       - wind speed u & v
#       - Short wave direct and diffuse radiation post-processed (ca 48 km horizontal) daily from C.Frank
#       - optimal angle from C.Frank
#       - installed capacity for PV and wind power plant 2013 and planned 2050 from CLIMIX model (Jerez et al 2015)
#       - 
# Output:
#       - netCDF filename, path [...]
#       - gridded data (ca 48 km horizontal) 
#       - dimension: lon (106), lat (103), time (daily)
#       - PV energy generation (MW) with installed capacity by 2012
#       - (not yet) PV energy generation (MW) plus installed capacity planned 2050
#       - Wind energy generation (MW) with installed capacity by 2013
#       - (not yet) Wind energy generation (MW) plus installed capacity planned 2050
#       - Wind speed horizontal corresponding (squareroot(u**2 + v**2))
#
# by Linh Ho (linh.ho@uni-koeln.de) 2020-10-28
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

from working_with_datetime import array_of_dates
from cf_read_grib import get_data
from cf_read_grib import get_read_index
from cf_read_grib import read_grib_var_3
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

### Some options =================================
outfile_label = 'v2'
scaleit = True  # scale the installed capacity to a certain value, e.g. to fit 2019 installation

save_energy_generation = True
save_energy_potential = False
save_COSMO_48km = False

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
years = range(2000,2001)
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

##########################
# read installed capacity for wind power plant
##########################
filename = '/home/linhho/Documents/Code_PhD/DATA/CLIMIX/installed_capacity_PV_wind_power_from_CLIMIX_final.nc'
ds = xr.open_dataset(filename)
ic_wp = ds['ic_wp'].data
hub_height = ds['hub_height'].data
ic_wp2050 = ds['ic_wp2050'].data
hub_height2050 = ds['hub_height2050'].data
ic_PV = ds['ic_PV2050'].data
print("Shape of the installed capacity and hub height of wind and PV power plant data are: ", ic_wp.shape, " and ", hub_height.shape, ic_PV.shape)
    
if scaleit == True:
#     Dir['out_dir'] = '/net/respos/linhho/energy_model/scale2019/'
#     outfile_label = 'scale2019'
#     ic_wp     = ic_wp*167/440
#     ic_wp2050 = ic_wp2050*167/440
#     ic_PV     = ic_PV*120/870

    ## to test if only the ratio of PV increase matters
    ## here the same wind installation as in scale-2019, but PV installation with the ratio PV/wind of scenario-2050
#     outfile_label = 'scale_ratio2050'
    ic_wp     = ic_wp*167/440
    ic_wp2050 = ic_wp2050*167/440
#     ic_PV     = ic_PV*167/440

    # with 870 GW of PV, 167 GW of wind, increase significantly the ratio of PV/wind ~5.2
    outfile_label = 'newratio5'
    ic_PV     = ic_PV   

    Dir['out_dir'] = '/net/respos/linhho/energy_model/' + outfile_label + '/'

    
# Start month for first year:
for year in years:
    datum_year =  array_of_dates([year], time_range, minu)

    yyyy = str(year)
    print('year ' + yyyy)

    ################################
    ### Read REA6 temperature 2m ###
    ################################
    # Use Christopher Frank's old code to read T2m since xarray gives KeyError: u'paramId' <<<<<<<<<<<<<<
    # ONLY for >>>>>>>>>>>>>>    year 2000    <<<<<<<<<<<<<<<<

    for i_file in np.arange(0,12):
        #print(input_file)
        month = i_file+1
        time_range_min = [00, 15, 30, 45]
        i_in_pred,grib_ids_vec,need_complete_file,dt_expected_mm = get_read_index(str(year),int(month),time_range,time_range_min,
                  etcc_path='/data/etcc/cosmo-rea6/15min/',pred_name='2MT.2D')
        pred_new = read_grib_var_3(str(year),int(month),time_range,time_range_min,
                  '/data/etcc/cosmo-rea6/15min/','2MT.2D',ilat,ilon,ny,nx,
                  i_in_pred,grib_ids_vec,need_complete_file,
                  dt_expected_mm, sites=False)

        #Reduce data to 48 * 48 km
        ioi_lat = np.arange(0,ny,8)
        ioi_lon = np.arange(0,nx,8)
        ioi_date = np.arange(0,len(dt_expected_mm),4)
        values = np.array(pred_new[:,:,ioi_lon][:,ioi_lat][ioi_date])

        del pred_new

        if i_file == 0:
            T2m_all = values
        else:
            T2m_all = np.concatenate((T2m_all,values),axis=0)

    print('=============== 2-m Temperature ==================================')
    print(T2m_all.shape)
    print(T2m_all)

    ############################
    ### Read REA6 wind speed ###
    ############################
    for i_file in np.arange(0,12):
        #print(input_file)
        month = i_file+1
        time_range_min = [00, 15, 30, 45]
        i_in_pred,grib_ids_vec,need_complete_file,dt_expected_mm = get_read_index(str(year),int(month),time_range,time_range_min,
                  etcc_path='/data/etcc/cosmo-rea6/15min/',pred_name='10U.2D')
        pred_new = read_grib_var_3(str(year),int(month),time_range,time_range_min,
                  '/data/etcc/cosmo-rea6/15min/','10U.2D',ilat,ilon,ny,nx,
                  i_in_pred,grib_ids_vec,need_complete_file,
                  dt_expected_mm, sites=False)

        #Reduce data to 48 * 48 km
        ioi_lat = np.arange(0,ny,8)
        ioi_lon = np.arange(0,nx,8)
        ioi_date = np.arange(0,len(dt_expected_mm),4)
        values = np.array(pred_new[:,:,ioi_lon][:,ioi_lat][ioi_date])

        del pred_new

        if i_file == 0:
            U_10m = values
        else:
            U_10m = np.concatenate((U_10m,values),axis=0)

    for i_file in np.arange(0,12):
        #print(input_file)
        month = i_file+1
        time_range_min = [00, 15, 30, 45]
        i_in_pred,grib_ids_vec,need_complete_file,dt_expected_mm = get_read_index(str(year),int(month),time_range,time_range_min,
                  etcc_path='/data/etcc/cosmo-rea6/15min/',pred_name='10V.2D')
        pred_new = read_grib_var_3(str(year),int(month),time_range,time_range_min,
                  '/data/etcc/cosmo-rea6/15min/','10V.2D',ilat,ilon,ny,nx,
                  i_in_pred,grib_ids_vec,need_complete_file,
                  dt_expected_mm, sites=False)

        #Reduce data to 48 * 48 km
        ioi_lat = np.arange(0,ny,8)
        ioi_lon = np.arange(0,nx,8)
        ioi_date = np.arange(0,len(dt_expected_mm),4)
        values = np.array(pred_new[:,:,ioi_lon][:,ioi_lat][ioi_date])

        del pred_new

        if i_file == 0:
            V_10m = values
        else:
            V_10m = np.concatenate((V_10m,values),axis=0)

    wsp = (U_10m**2 + V_10m**2)**0.5
    print('============== Wind speed at 10 m =============================')
    print(wsp.shape)
    print(wsp)
    
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
    print(Q_dir_all)

    ########################
    ### Read REA6 albedo ###
    ########################
    for month in months:
        fname = '/data/etcc/cosmo-rea6/ALB_RAD.2D/' + yyyy + '/' + month + '/ALB_RAD.2D.' + yyyy + month + '.grb'
            
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
    if year != 2015:
        albedo = np.concatenate((albedo[None,0,:,:], albedo), axis=0)  # repeat first step to add missing data yyyy-01-01 00:00:00
#     if year == 2007:
#         albedo = np.insert(albedo, 5755, np.full([6, 103, 106], np.nan), axis=0)

    print('=============== Albedo ==================================')
    print(albedo.shape)
    print(albedo)

    #################################
    ### Read sun elevation_angles ###
    #################################
    sun_elevation = np.nan * np.empty( (len(datum_year), len(ioi_lat), len(ioi_lon)), dtype=float16 )
    sun_azimuth = np.nan * np.empty( (len(datum_year), len(ioi_lat), len(ioi_lon)), dtype=float16 )

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
    
    PV = Pstrich*np.repeat(ic_PV, time_step, axis=0)  # shape of ic_PV now is [1,103,106]
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
    #
    # Wind energy -- POTENTIAL -- (withOUT installed capacity)
    #
    ########################################
    hub_height_fixed = Param['hub_height_avg']
    rhoLEV = Param['rho'] * (1 - (Param['dTdh']/T2m_all)*hub_height_fixed )**a
    vhub = np.dtype('f4')
    vhub = (hub_height_fixed - hlev37) / (hlev36 - hlev37) * (wsp36 - wsp37) + wsp37

    # Power law: define where cutin <= v < rated   
    ############
    id_cubic = (vhub >= Param['cutin_speed']) & (vhub < Param['rated_speed'])
    # define where rated <= v <= cutout
    id_rated = (vhub >= Param['rated_speed']) & (vhub <= Param['cutout_speed'])

    # on or offshore turbine will have different capacity and rotor diameter
    # now use installed capacity from CLIMIX
    cf_eout_potential = np.full_like(wsp, 0, dtype='f8')  # capacity factor

    ########################################
    # Calculate energy output
    ########################################
    cf_eout_potential[id_cubic] = (vhub[id_cubic]**3 - Param['cutin_speed']**3) / (Param['rated_speed']**3 - Param['cutin_speed']**3)
    cf_eout_potential[id_rated] = 1

#     cf_eout_potential = np.dtype('f4')  # to avoid error TypeError: illegal primitive data type, must be one of ['i8', 'f4', 'f8', 'S1', 'i2', 'i4', 'u8', 'u4', 'u1', 'u2', 'i1'], got float16 ## f8 = float64
#     cf_eout_potential[cf_eout_potential==0] = np.nan

    print('=================== Wind energy ===========================')
    print("Wind output and capacity factor for 2013: ", cf_eout_potential.shape)
    print(cf_eout_potential)


    ########################################
    #
    # Wind energy -- generation -- (with installed capacity from CLIMIX)
    #
    ########################################
    # Vertical interpolation of wind speeds at a reference height
    # Create installed capacity (ic_wp) and hub_height with corresponding dimension
    # i.e. repeat along time dimension

    # Interpolate wind speed at hub height: v_hub = (hub_height - h_lev37)/(h_lev36-h_lev37) * (v_36-v_37) + v_37
    # Then calculated capacity factor of wind output (lower than rated wind speed): (Matthew 2006, Tobin 2016)
    # (vhub**3 - cutin_speed**3) / (rated_speed**3 - cutin_speed**3)
    
    # loop over each layer of wind power plant from CLIMIX === 2013 ===============================
    ic_wp_all_layer = np.nansum(ic_wp, axis=0)

    for k in range(0,ic_wp.shape[0]):
        IC_timestep = np.repeat(ic_wp[None,k,:,:], time_step, axis=0)

        rhoLEV = Param['rho'] * (1 - (Param['dTdh']/T2m_all)*hub_height[k,:,:] )**a
        vhub = (hub_height[k,:,:]-hlev37)/(hlev36-hlev37) * (wsp36-wsp37) + wsp37

        # Power law: define where cutin <= v < rated   
        ############
        id_cubic = (vhub >= Param['cutin_speed']) & (vhub < Param['rated_speed'])
        # define where rated <= v <= cutout
        id_rated = (vhub >= Param['rated_speed']) & (vhub <= Param['cutout_speed'])

        # on or offshore turbine will have different capacity and rotor diameter
        # now use installed capacity from CLIMIX
        cf_eout = np.full_like(wsp, 0, dtype=np.float64)  # capacity factor
        layer_eout = np.full_like(wsp, 0, dtype=np.float64)  # wind energy output

        ########################################
        # Calculate energy output
        ########################################
        cf_eout[id_cubic] = (vhub[id_cubic]**3 - Param['cutin_speed']**3) / (Param['rated_speed']**3 - Param['cutin_speed']**3)
        cf_eout[id_rated] = 1
        layer_eout[id_cubic] = cf_eout[id_cubic]*IC_timestep[id_cubic]  # same unit MW as CLIMIX
        layer_eout[id_rated] = IC_timestep[id_rated]

        # add up each layer to total wind power generation in each grid cell
        if k == 0:
            tmp_eout = layer_eout
        else:
            tmp_eout = np.add(tmp_eout, layer_eout)

    wind_output = tmp_eout 
    # Capacity factor: CF each grid = sum(CF each layer * capacity each layer) / sum capacity grid
    CF_output = wind_output/ic_wp_all_layer
    
#     wind_output[wind_output==0] = np.nan
#     CF_output[CF_output==0] = np.nan

    print('=================== Wind energy output 2013 ===========================')
    print("Wind output and capacity factor for 2013: ", wind_output.shape, CF_output.shape)
    print(wind_output)
    print(CF_output)

    # loop over each layer of wind power plant from CLIMIX === ONLY planned 2050 ============
    # Similar for 2050 wind turbines, then combine to get BOTH in 2050
    ic_wp_all_layer2050 = np.nansum(ic_wp2050, axis=0)

    for k in range(0,ic_wp2050.shape[0]):
        IC_timestep2050 = np.repeat(ic_wp2050[None,k,:,:], time_step, axis=0)

        rhoLEV = Param['rho'] * (1 - (Param['dTdh']/T2m_all)*hub_height2050[k,:,:] )**a
        vhub2050 = (hub_height2050[k,:,:]-hlev37)/(hlev36-hlev37) * (wsp36-wsp37) + wsp37

        # Power law: define where cutin <= v < rated   
        ############
        id_cubic = (vhub2050 >= Param['cutin_speed']) & (vhub2050 < Param['rated_speed'])
        # define where rated <= v <= cutout
        id_rated = (vhub2050 >= Param['rated_speed']) & (vhub2050 <= Param['cutout_speed'])

        # on or offshore turbine will have different capacity and rotor diameter
        # now use installed capacity from CLIMIX
        cf_eout2050 = np.full_like(wsp, 0, dtype=np.float64)  # capacity factor
        layer_eout2050 = np.full_like(wsp, 0, dtype=np.float64)  # wind energy output

        ########################################
        # Calculate energy output
        ########################################
        cf_eout2050[id_cubic] = (vhub2050[id_cubic]**3 - Param['cutin_speed']**3) / (Param['rated_speed']**3 - Param['cutin_speed']**3)
        cf_eout2050[id_rated] = 1
        layer_eout2050[id_cubic] = cf_eout2050[id_cubic]*IC_timestep2050[id_cubic]  # same unit MW as CLIMIX
        layer_eout2050[id_rated] = IC_timestep2050[id_rated]

        # add up each layer to total wind power generation in each grid cell
        if k == 0:
            tmp_eout2050 = layer_eout2050
        else:
            tmp_eout2050 = np.add(tmp_eout2050, layer_eout2050)

    wind_output_all = wind_output + tmp_eout2050
    # Capacity factor: CF each grid = sum(CF each layer * capacity each layer) / sum capacity grid
    CF_output_all = wind_output_all / (ic_wp_all_layer + ic_wp_all_layer2050)

#     wind_output_all[wind_output_all==0] = np.nan
#     CF_output_all[CF_output_all==0] = np.nan

    print('=================== Wind energy output ALL 2050 ===========================')
    print("Wind output and capacity factor for BOTH 2013 + 2050: ", wind_output_all.shape, CF_output_all.shape)
    print(wind_output_all)
    print(CF_output_all)

    ########################################
    ### SAVING IN NETCDF4                ###
    ########################################

    # # Save energy generation 
    # ##################################
    if save_energy_generation == True:
        filename = Dir['out_dir'] + yyyy + '_PV_wind_generation_' + outfile_label + '.nc'
        time = pd.date_range(dt.datetime(year, 1, 1), dt.datetime(year+1, 1, 1), freq="H", closed="left") #, periods=365 * 24)

        ds = xr.Dataset({"Wp": (('time', 'lon','lat'), wind_output),
                         "Wp_CF": (('time', 'lon','lat'), CF_output),
                         "Wp2050": (('time', 'lon','lat'), wind_output_all),
                         "Wp2050_CF": (('time', 'lon','lat'), CF_output_all),
                         "PV2050": (('time', 'lon','lat'), PV),
                         "PV2050_CF": (('time', 'lon','lat'), Pstrich),
                         "time": time})
        ds['Wp'].attrs = {'units': 'MW',
                          'name': 'Power production of wind power in Europe 2013 CLIMIX'}
        ds['Wp_CF'].attrs = {'units': 'MW',
                          'name': 'Capacity factor of wind power in Europe 2013 CLIMIX'}
        ds['Wp2050'].attrs = {'units': 'MW',
                          'name': 'Power production of wind power in Europe planned -TOTAL- 2013 and 2050 CLIMIX'}
        ds['Wp2050_CF'].attrs = {'units': 'MW',
                          'name': 'Capacity factor of wind power in Europe planned -TOTAL- 2013 and 2050 CLIMIX'}
        ds['PV2050'].attrs = {'units': 'MW',
                          'name': 'Power production of PV power in Europe 2050 CLIMIX'}
        ds['PV2050_CF'].attrs = {'units': 'MW',
                          'name': 'Capacity factor of PV power in Europe 2050 CLIMIX'}
        ds.attrs['Conventions'] = ''  # 'CF-1.7'
        ds.attrs['Title'] = 'Energy model output version 2 (2021-09-07)'
        ds.attrs['Author'] = 'Linh Ho, Institute of Geophysics and Meteorology, University of Cologne, Germany'
    #     ds.attrs['source'] = 'WRF-1.5'
        ds.attrs['History'] = str(dt.datetime.utcnow()) + ' Python2'
        ds.attrs['References'] = ''   # add my paper later :)
        ds.attrs['Notes'] = 'Wind speed at hubheight interpolated from wind at level 36 and 37 (roughly 80-150 m)'
        ds.to_netcdf(filename, 'w')

        print('\n =================== Finish saving ENERGY OUTPUT for year', yyyy, '=========================== \n')

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
        print(pmsl)

        filename = Dir['path_cosmo_48km'] + yyyy + '_COSMO-REA6_hourly_48km.nc'
        time = pd.date_range(dt.datetime(year, 1, 1), dt.datetime(year+1, 1, 1), freq="H", closed="left") #, periods=365 * 24)

        ds = xr.Dataset({"wsp100m_36_37": (('time', 'lon','lat'), vhub),
    #                      "wsp_avg_36_37": (('time', 'lon','lat'), wsp_avg_36_37),
                         "total_radiation": (('time', 'lon','lat'), SWDIR+SWDIFD),
                         "wsp": (('time', 'lon','lat'), wsp.astype('float32')),  # to avoid error float16
                         "t2m": (('time', 'lon','lat'), T2m_all.astype('float32')),
    #                      "SWDIR": (('time', 'lon','lat'), SWDIR),
    #                      "SWDIFD": (('time', 'lon','lat'), SWDIFD),
                         "pmsl": (('time', 'lon','lat'), pmsl),
                         "time": time})
        ds['wsp100m_36_37'].attrs = {'units': 'ms-1',
                          'name': 'Wind speed at 100m interpolated from level 36 and 37 from COSMO-REA6'}
    #     ds['wsp_avg_36_37'].attrs = {'units': 'ms-1',
    #                       'name': 'Wind speed average from level 36 and 37 from COSMO-REA6'}
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

         # # Save energy POTENTIAL
        # ##################################
    if save_energy_potential == True:
        filename = Dir['out_dir'] + yyyy + '_PV_wind_potential_v2.nc'
        time = pd.date_range(dt.datetime(year, 1, 1), dt.datetime(year+1, 1, 1), freq="H", closed="left") #, periods=365 * 24)

        ds = xr.Dataset({"Wp": (('time', 'lon','lat'), cf_eout_potential),
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


