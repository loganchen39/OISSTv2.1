#!/glade/u/apps/ch/opt/python/3.7.9/gnu/9.1.0/pkg-library/20201220/bin/python3

'''
Description: python code to compute ACSPO_L3U daily (day/night) SST from 10-minute time resolution.
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


DIR_AC = '/glade/scratch/lgchen/data/ACSPO_L3U/NOAA07_1984-01-17'
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
# oi_mask_fort = np.asfortranarray(ds_oi_mask.landmask.values)
oi_mask_fort = ds_oi_mask.landmask.values.T
lat_oi = ds_oi_mask.coords['lat']  # 720: -89.875, -89.625, -89.375, ...,  89.375,  89.625,  89.875
lon_oi = ds_oi_mask.coords['lon']  # 1440: 0.125 0.375 0.625 ... 359.625 359.875
N_LAT_OI = 720
N_LON_OI = 1440

day = np.arange(start=0, stop=2, step=1, dtype=np.int8)
stat = np.arange(start=0, stop=6, step=1, dtype=np.int8)

np.set_printoptions(threshold=np.inf)

daily_sst_ac2oi = np.zeros((3000, 1440, 720, 2), dtype=np.float32, order='F')  # day (0) and night (1), max sst number per grid point 50;    
# daily_sst_stat_ac2oi = np.zeros((6, 1440, 720, 2), dtype=np.float32, order='F') # stats: num; mean; median; lowest; highest; std; 
daily_sst_stat_ac2oi = np.zeros((6, 1440, 720, 2), dtype=np.float32, order='F') # stats: num; mean; median; lowest; highest; std; 
mask_10thBit = np.int16(2**9)
print('mask_10thBit = ', mask_10thBit)

jday_19840117 = datetime.date(1984, 1 , 17)

jday = jday_19840117
while jday <= jday_19840117:
    str_date = jday.strftime('%Y%m%d')
    print('\n\n\n current date: ', str_date)

    fns = sorted( glob.glob(pathname=DIR_AC+'/'+str_date+'*-STAR-L3U_GHRSST-SSTsubskin-AVHRRG_N07-ACSPO_V2.81-v02.0-fv01.0.nc') )

    daily_sst_ac2oi.fill(0)  # In-place operation
    daily_sst_stat_ac2oi.fill(0)

    for (id_fn, fn) in enumerate(fns, start=0):
        print('\n\n\n id_fn=', str(id_fn).zfill(3), ', fn: ', fn)

        ds = xr.open_dataset(filename_or_obj=fn, mask_and_scale=True, decode_times=True  \
            , drop_variables=['or_number_of_pixels', 'dt_analysis', 'satellite_zenith_angle', 'sses_standard_deviation'  \
            , 'wind_speed', 'sst_dtime', 'crs']).isel(time=0)

        ds = ds.isel( lat=slice(ds.attrs['row_start'], ds.attrs['row_start']+ds.attrs['row_count'])  \
            , lon=slice(ds.attrs['col_start'], ds.attrs['col_start']+ds.attrs['col_count']) )
      # print('after ds.isel(), ds: ')
      # print(ds)

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
      # print('ds.coords[lon]: ', ds.coords['lon'].values)
      # print('ds: ', ds)

        ds_stacked = ds.stack(loc=('lat', 'lon'))
        ds_actual_1d = ds_stacked.where(ds_stacked['sea_surface_temperature'].notnull(), drop=True)
        ds_actual_1d['l2p_flags'] = ds_actual_1d.l2p_flags.astype(np.int16)
        is_daytime = mask_10thBit & ds_actual_1d['l2p_flags'].data
        is_daytime = is_daytime.astype(np.int16)
        print('ds_actual_1d.sea_surface_temperature.size = ', ds_actual_1d.sea_surface_temperature.size)
    
      # print('Start processing SST, current time: ', datetime.datetime.now().strftime('%H:%M:%S'))
      # daily_sst_ac2oi = fort.sst_acspo_l3u2oi_1rec(ds_actual_1d['sea_surface_temperature'].data  \
      #     , ds_actual_1d.coords['lon'].data, ds_actual_1d.coords['lat'].data, daily_sst_stat_ac2oi  \
      #     , oi_mask_fort, is_daytime)

        fort.sst_acspo_l3u2oi_1rec(ds_actual_1d['sea_surface_temperature'].data  \
            , ds_actual_1d.coords['lon'].data, ds_actual_1d.coords['lat'].data, daily_sst_ac2oi  \
            , daily_sst_stat_ac2oi, oi_mask_fort, is_daytime)

      # num_nonzero = np.count_nonzero(daily_sst_ac2oi)
      # print('id_fn=', id_fn, ', num_nonzero=', num_nonzero)
      # if (id_fn == 5):
      #     break  # temperary for debug
      # print('End processing SST, current time: ', datetime.datetime.now().strftime('%H:%M:%S'))


  # print('\n\n\n daily_sst_stat_ac2oi: ', daily_sst_stat_ac2oi[0, :, 600, 0])      
    fort.calc_sst_stat_acspo_l3u2oi_1day(daily_sst_ac2oi, daily_sst_stat_ac2oi, oi_mask_fort)    
    print('\n\n\n sst num : ', daily_sst_stat_ac2oi[0, :, 600, 0])      
    print('\n\n\n sst mean: ', daily_sst_stat_ac2oi[1, :, 600, 0])   

  # da_daily_sst_stat_day = xr.DataArray(data=daily_sst_stat_ac2oi.T  \
  #     , dims=['day', 'lat', 'lon', 'stat'], coords={'day': day, 'lat': lat_oi, 'lon': lon_oi, 'stat':stat}, name='sst_stat') 

    da_sst_num = xr.DataArray(data=daily_sst_stat_ac2oi[0, :, :, :].T  \
        , dims=['day', 'lat', 'lon'], coords={'day': day, 'lat': lat_oi, 'lon': lon_oi}, name='sst_num'  \
        )
    da_sst_mean = xr.DataArray(data=daily_sst_stat_ac2oi[1, :, :, :].T  \
        , dims=['day', 'lat', 'lon'], coords={'day': day, 'lat': lat_oi, 'lon': lon_oi}, name='sst_mean'  \
        , attrs={'units':'Kelvin', '_FillValue':0})
    da_sst_median = xr.DataArray(data=daily_sst_stat_ac2oi[2, :, :, :].T  \
        , dims=['day', 'lat', 'lon'], coords={'day': day, 'lat': lat_oi, 'lon': lon_oi}, name='sst_median'  \
        , attrs={'units':'Kelvin', '_FillValue':0})
    da_sst_lowest = xr.DataArray(data=daily_sst_stat_ac2oi[3, :, :, :].T  \
        , dims=['day', 'lat', 'lon'], coords={'day': day, 'lat': lat_oi, 'lon': lon_oi}, name='sst_lowest'  \
        , attrs={'units':'Kelvin', '_FillValue':0})
    da_sst_highest = xr.DataArray(data=daily_sst_stat_ac2oi[4, :, :, :].T  \
        , dims=['day', 'lat', 'lon'], coords={'day': day, 'lat': lat_oi, 'lon': lon_oi}, name='sst_highest'  \
        , attrs={'units':'Kelvin', '_FillValue':0})
    da_sst_std = xr.DataArray(data=daily_sst_stat_ac2oi[5, :, :, :].T  \
        , dims=['day', 'lat', 'lon'], coords={'day': day, 'lat': lat_oi, 'lon': lon_oi}, name='sst_std'  \
        , attrs={'units':'Kelvin', '_FillValue':0})

    ds_sst_stat = xr.merge([da_sst_num, da_sst_mean, da_sst_median, da_sst_lowest, da_sst_highest, da_sst_std])
    ds_sst_stat.to_netcdf('./sst_stat.nc')


    jday += datetime.timedelta(days=1)    


