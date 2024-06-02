#!/glade/u/apps/ch/opt/python/3.7.9/gnu/9.1.0/pkg-library/20201220/bin/python3

'''
Description: python code to compute ACSPO_L3U daily (day/night) SST statistics from 10-minute time resolution to OI grid.
Author: Ligang Chen
Date created: 12/14/2021
Date last modified: 12/14/2021 
'''

import numpy as np
import xarray as xr

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors

# import geocat.datafiles as gdf
from geocat.viz import cmaps as gvcmaps
from geocat.viz import util as gvutil

import os
import calendar
import datetime
import glob
import time

import fort


DIR_AC = '/glade/scratch/lgchen/data/ACSPO_L3U/1984'
FN_AC_TEST = DIR_AC+"/test/19840117000000-STAR-L3U_GHRSST-SSTsubskin-AVHRRG_N07-ACSPO_V2.81-v02.0-fv01.0.nc"
ds_ac_test = xr.open_dataset(filename_or_obj=FN_AC_TEST)
ds_ac_test.coords['lon'] = xr.where(ds_ac_test.coords['lon']<0, 360+ds_ac_test.coords['lon'], ds_ac_test.coords['lon'])
ds_ac_test.sortby(ds_ac_test.coords['lon'])
lat_ac = ds_ac_test.coords['lat']  #  9000: 89.99, 89.97, ..., -89.97, -89.99;
lon_ac = ds_ac_test.coords['lon']  # 18000: -179.99, -179.97, ..., 179.97, 179.99 (initially);
N_LAT_AC = 9000
N_LON_AC = 18000

DIR_OI = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master'
FN_OI_MASK = DIR_OI + '/bin2netCDF/quarter-mask-extend_Ligang.nc'
ds_oi_mask = xr.open_dataset(filename_or_obj=FN_OI_MASK)
ds_oi_mask['landmask'] = ds_oi_mask.landmask.astype(np.int32)
oi_mask_fort = ds_oi_mask.landmask.values.T
ds_oi_mask.coords['lat'].attrs['long_name'] = 'Latitude'
ds_oi_mask.coords['lat'].attrs['units'] = 'degrees_north'
ds_oi_mask.coords['lat'].attrs['grids'] = 'Uniform grid from -89.875 to 89.875 by 0.25'
ds_oi_mask.coords['lon'].attrs = {'long_name': 'Longitude', 'units': 'degrees_east', 'grids': 'Uniform grid from 0.125 to 359.875 by 0.25'}
da_lat_oi = ds_oi_mask.coords['lat']  # 720: -89.875, -89.625, -89.375, ...,  89.375,  89.625,  89.875
da_lon_oi = ds_oi_mask.coords['lon']  # 1440: 0.125 0.375 0.625 ... 359.625 359.875
N_LAT_OI = 720
N_LON_OI = 1440

day_or_night = np.arange(start=0, stop=2, step=1, dtype=np.int8)
sst_stat = np.arange(start=0, stop=6, step=1, dtype=np.int8)
# da_day_or_night = xr.DataArray(data=np.arange(start=0, stop=2, step=1, dtype=np.int8  \
#     , dims=['day_or_night'], coords={'day_or_night': da_day_or_night}, name='day_or_night'  \
#     , attrs={'comments':'0 for daytime and 1 for nighttime'})

da_day_or_night = xr.DataArray(data=day_or_night  \
    , dims=['day_or_night'], coords={'day_or_night': day_or_night}, name='day_or_night'  \
    , attrs={'comments':'0 for daytime and 1 for nighttime'})
da_sst_stat = xr.DataArray(data=sst_stat, dims=['sst_stat'], coords={'sst_stat': sst_stat}, name='sst_stat'  \
    , attrs={'comments':'num, mean, median, lowest, highest, std'})


np.set_printoptions(threshold=np.inf)

daily_sst_ac2oi = np.zeros((2000, 1440, 720, 2), dtype=np.float32, order='F')  # day (0) and night (1), max sst number per grid point 50;    
daily_sst_stat_ac2oi = np.zeros((1440, 720, 6, 2), dtype=np.float32, order='F') # stats: num; mean; median; lowest; highest; std; 
mask_10thBit = np.int16(2**9)

jday_19840101 = datetime.date(1984, 1, 1 )
jday_19840131 = datetime.date(1984, 1, 31)

jday = jday_19840101
while jday <= jday_19840131:
    str_date = jday.strftime('%Y%m%d')

    fns = sorted( glob.glob(pathname=DIR_AC+'/'+str_date+'*-STAR-L3U_GHRSST-SSTsubskin-AVHRRG_N07-ACSPO_V2.81-v02.0-fv01.0.nc') )
    print('\n\n\n current date: ', str_date, ', len(fns): ', len(fns))

    daily_sst_ac2oi.fill(0)  # In-place operation
    daily_sst_stat_ac2oi.fill(0)

    for (id_fn, fn) in enumerate(fns, start=0):
        print('\n id_fn=', str(id_fn).zfill(3), ', fn: ', fn)

        ds = xr.open_dataset(filename_or_obj=fn, mask_and_scale=True, decode_times=True  \
            , drop_variables=['or_number_of_pixels', 'dt_analysis', 'satellite_zenith_angle', 'sses_standard_deviation'  \
            , 'wind_speed', 'sst_dtime', 'crs']).isel(time=0)

        ds = ds.isel( lat=slice(ds.attrs['row_start'], ds.attrs['row_start']+ds.attrs['row_count'])  \
            , lon=slice(ds.attrs['col_start'], ds.attrs['col_start']+ds.attrs['col_count']) )

        ds['quality_level'] = ds.quality_level.astype(np.int8)  # convert back to type byte.
        ds['sea_surface_temperature'] = xr.where(ds.quality_level==5, ds.sea_surface_temperature, np.nan)
        ds['sea_surface_temperature'] = ds['sea_surface_temperature'] - ds['sses_bias']  # actual sst
      # ds['sea_surface_temperature'] = xr.where(ds['sea_surface_temperature'] < 250, np.nan, ds['sea_surface_temperature'])
      # ds['sea_surface_temperature'] = xr.where(ds['sea_surface_temperature'] > 350, np.nan, ds['sea_surface_temperature'])
        # HAVE TO use assignment! fillna() is NOT an in-place operation!
      # ds['sea_surface_temperature'] = ds['sea_surface_temperature'].fillna(0) # then no longer nan!

        ds.coords['lon'] = xr.where(ds.coords['lon']<0, 360+ds.coords['lon'], ds.coords['lon'])
        ds.sortby(ds.coords['lon'])
        ds['l2p_flags'] = ds.l2p_flags.astype(np.int16)

        ds_stacked = ds.stack(loc=('lat', 'lon'))
        ds_actual_1d = ds_stacked.where(ds_stacked['sea_surface_temperature'].notnull(), drop=True)
        ds_actual_1d['l2p_flags'] = ds_actual_1d.l2p_flags.astype(np.int16)
        is_daytime = mask_10thBit & ds_actual_1d['l2p_flags'].data
        is_daytime = is_daytime.astype(np.int16)
        print('ds_actual_1d.sea_surface_temperature.size = ', ds_actual_1d.sea_surface_temperature.size)
    
        fort.sst_acspo_l3u2oi_1rec(ds_actual_1d['sea_surface_temperature'].data  \
            , ds_actual_1d.coords['lon'].data, ds_actual_1d.coords['lat'].data, daily_sst_ac2oi  \
            , daily_sst_stat_ac2oi, oi_mask_fort, is_daytime)

  # print('\n\n\n daily_sst_stat_ac2oi: ', daily_sst_stat_ac2oi[:, 600, 0, 0])      
    fort.calc_sst_stat_acspo_l3u2oi_1day(daily_sst_ac2oi, daily_sst_stat_ac2oi, oi_mask_fort)    
  # print('\n\n\n sst num : ', daily_sst_stat_ac2oi[:, 600, 0, 0])
  # print('\n\n\n sst mean: ', daily_sst_stat_ac2oi[:, 600, 1, 0])   

    for i in range(1, 6): 
        daily_sst_stat_ac2oi[:, :, i, :] = np.where(daily_sst_stat_ac2oi[:, :, 0, :]==0, -999., daily_sst_stat_ac2oi[:, :, i, :])

    da_sst_num = xr.DataArray(data=daily_sst_stat_ac2oi[:, :, 0, :].T  \
        , dims=['day_or_night', 'lat', 'lon'], coords={'day_or_night': da_day_or_night, 'lat': da_lat_oi, 'lon': da_lon_oi}, name='sst_num'  \
        , attrs={'_FillValue':0})
    da_sst_mean = xr.DataArray(data=daily_sst_stat_ac2oi[:, :, 1, :].T  \
        , dims=['day_or_night', 'lat', 'lon'], coords={'day_or_night': da_day_or_night, 'lat': da_lat_oi, 'lon': da_lon_oi}, name='sst_mean'  \
        , attrs={'units':'Kelvin', '_FillValue':-999.})
    da_sst_median = xr.DataArray(data=daily_sst_stat_ac2oi[:, :, 2, :].T  \
        , dims=['day_or_night', 'lat', 'lon'], coords={'day_or_night': da_day_or_night, 'lat': da_lat_oi, 'lon': da_lon_oi}, name='sst_median'  \
        , attrs={'units':'Kelvin', '_FillValue':-999.})
    da_sst_lowest = xr.DataArray(data=daily_sst_stat_ac2oi[:, :, 3, :].T  \
        , dims=['day_or_night', 'lat', 'lon'], coords={'day_or_night': da_day_or_night, 'lat': da_lat_oi, 'lon': da_lon_oi}, name='sst_lowest'  \
        , attrs={'units':'Kelvin', '_FillValue':-999.})
    da_sst_highest = xr.DataArray(data=daily_sst_stat_ac2oi[:, :, 4, :].T  \
        , dims=['day_or_night', 'lat', 'lon'], coords={'day_or_night': da_day_or_night, 'lat': da_lat_oi, 'lon': da_lon_oi}, name='sst_highest'  \
        , attrs={'units':'Kelvin', '_FillValue':-999.})
    da_sst_std = xr.DataArray(data=daily_sst_stat_ac2oi[:, :, 5, :].T  \
        , dims=['day_or_night', 'lat', 'lon'], coords={'day_or_night': da_day_or_night, 'lat': da_lat_oi, 'lon': da_lon_oi}, name='sst_std'  \
        , attrs={'units':'Kelvin', '_FillValue':-999.})

    ds_sst_stat = xr.merge([da_sst_num, da_sst_mean, da_sst_median, da_sst_lowest, da_sst_highest, da_sst_std])

    fn_sst_StatOnOi_daily = str_date + '_sst_stat_ACSPO_L3U2OI.nc'
    ds_sst_stat.to_netcdf(DIR_AC + '/sst_StatOnOi_daily/' + fn_sst_StatOnOi_daily  \
        , encoding={'lat': {'_FillValue': None}, 'lon': {'_FillValue': None}})

    jday += datetime.timedelta(days=1)
