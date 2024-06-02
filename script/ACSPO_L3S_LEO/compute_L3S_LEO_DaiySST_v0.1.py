#!/glade/u/apps/ch/opt/python/3.7.9/gnu/9.1.0/pkg-library/20201220/bin/python3

'''
Description: python code to compute L3S_LEO daily SST.
Author: Ligang Chen
Date created: 10/22/2021
Date last modified: 10/22/2021 
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


DIR_L3S = '/glade/scratch/lgchen/data/L3S_LEO'
FN_L3S_TEST = DIR_L3S+"/test/20150101120000-STAR-L3S_GHRSST-SSTsubskin-LEO_AM_D-ACSPO_V2.80-v02.0-fv01.0.nc"
ds_l3s = xr.open_dataset(filename_or_obj=FN_L3S_TEST)
lat_l3s = ds_l3s.coords['lat']  #  9000: 89.99, 89.97, ..., -89.97, -89.99;
lon_l3s = ds_l3s.coords['lon']  # 18000: -179.99, -179.97, ..., 179.97, 179.99;
# lat_l3s = np.arange(start=89.99, stop=-90.0, step=-0.02, dtype=np.float64)  #  9000: 89.99, 89.97, ..., -89.97, -89.99;
# lon_l3s = np.arange(start=-179.99, stop=180.0, step=0.02, dtype=np.float64)  # 18000: -179.99, -179.97, ..., 179.97, 179.99;
# print('type(lat_l3s): ', type(lat_l3s))
# print('lat_l3s: ', lat_l3s)
# print('lon_l3s: ', lon_l3s)
# quit()

DIR_OISST = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master'
FN_OISST_QUARTER_MASK = DIR_OISST + '/quarter-mask-extend_Ligang.nc'
ds_oisst_quarter_mask = xr.open_dataset(filename_or_obj=FN_OISST_QUARTER_MASK)
lat_oi = ds_oisst_quarter_mask.coords['lat']  # 720: -89.875, -89.625, -89.375, ...,  89.375,  89.625,  89.875
lon_oi = ds_oisst_quarter_mask.coords['lon']  # 1440: 0.125 0.375 0.625 ... 359.625 359.875
# lat_oi = np.arange(start=-89.875, stop=90.0, step=0.25, dtype=np.float64)  # 720: -89.875, -89.625, -89.375, ...,  89.375,  89.625,  89.875
# lon_oi = np.arange(start=0.125, stop=360.0, step=0.25, dtype=np.float64)  # 1440: 0.125 0.375 0.625 ... 359.625 359.875
# print('type(lat_oi): ', type(lat_oi))
# print('lat_oi: ', lat_oi)
# print('lon_oi: ', lon_oi)


mask_10thBit = (np.int16)(2**9)
print('mask_10thBit: ', mask_10thBit)

daily_sst_avg_l3s2oi = np.zeros((3, 720, 1440), dtype=np.float32)  # 3 types of night, day, day and night; 
daily_sst_num_l3s2oi = np.zeros((3, 720, 1440), dtype=np.uint8)  # for binned to avg

# da_daily_sst_avg_l3s2oi = xr.DataArray(data=np.zeros((3, 720, 1440), dtype=np.float32), name='da_daily_sst_avg_l3s2oi', dims=['lat', 'lon']  \
#     , coords={'lat': lat_oi, 'lon': lon_oi})  # attrs=dict(_FillValue=-9999)
# da_daily_sst_num_l3s2oi = xr.DataArray(data=np.zeros((3, 720, 1440), dtype=np.uint8)  , name='da_daily_sst_num_l3s2oi', dims=['lat', 'lon']  \
#     , coords={'lat': lat_oi, 'lon': lon_oi})  # attrs=dict(_FillValue=-9999)

jday_20150101 = datetime.date(2015, 1 , 1 )
jday_20150131 = datetime.date(2020, 1 , 31)
jday_20201231 = datetime.date(2020, 12, 31)

jday = jday_20150101
while jday <= jday_20150101:
    str_date = jday.strftime('%Y%m%d')
    print('current date: ', str_date)
    fns = glob.glob(pathname=DIR_L3S+'/link/'+str_date+'*_N-ACSPO_*.nc')
    print('fns: ', fns)

    daily_sst_avg_l3s2oi.fill(0)
    daily_sst_num_l3s2oi.fill(0)
    for (id_fn, fn) in enumerate(fns, start=1):
        print(str(id_fn).zfill(3)+', current file: ', fn)
        ds = xr.open_dataset(filename_or_obj=fn, mask_and_scale=True, decode_times=True  \
            , drop_variables=['sst_dtime', 'sses_standard_deviation', 'l3s_sst_reference', 'dt_analysis', 'sst_count', 'sst_source'  \
            , 'satellite_zenith_angle', 'wind_speed', 'crs', 'sst_gradient_magnitude', 'sst_front_position']).isel(time=0)
      # print(ds)
      # quit()

        ds['quality_level'] = ds.quality_level.astype(np.int8)  # convert back to type byte.
        ds['sea_surface_temperature'] = xr.where(ds.quality_level==5, ds.sea_surface_temperature, np.nan)
        ds.coords['lon'] = xr.where(ds.coords['lon']<0, 360+ds.coords['lon'], ds.coords['lon'])
        ds.sortby(ds.coords['lon'])
      # print('\n\n\n ds: ', ds)

        ds['l2p_flags'] = ds.l2p_flags.astype(np.int16)  

        ds_stacked = ds.stack(loc=('lat', 'lon'))
        print('\n\n\n ds_stacked: ', ds_stacked)
        ds_actual_1d = ds_stacked.where(ds_stacked['sea_surface_temperature'].notnull(), drop=True)
        ds_actual_1d['l2p_flags'] = ds_actual_1d.l2p_flags.astype(np.int16)  
        print('\n\n\n ds_actual_1d: ', ds_actual_1d)

      # print('ds_actual_1d.dims[loc]: ', ds_actual_1d.dims['loc'])
      # quit()

        # no need to use for-loop? use numpy array? what about the lat/lon info for interpolation? 
        for (id_loc, loc_l3s) in enumerate(ds_actual_1d.coords['loc'].values):
            print('id_loc = ', str(id_loc).zfill(9))

            sst_l3s = np.float32(ds_actual_1d['sea_surface_temperature'][id_loc])
            if 268.15 < sst_l3s and sst_l3s < 323.15:
                lat_l3s = loc_l3s[0]
                lon_l3s = loc_l3s[1]
              # if lon_l3s < 0 or lon_l3s > 360: print("ERROR: lon_l3u out of range!")
                j_lat_oi = int(round(4*(lat_l3s+89.875)))
              # if j_lat_oi < 0: j_lat_oi = 0
              # if j_lat_oi >= 720: j_lat_oi = 719
                i_lon_oi = int(round(4*(lon_l3s-0.125)))
              # if i_lon_oi < 0: i_lon_oi = 0
              # if i_lon_oi >= 1440: i_lon_oi = 1439

              # print('lat_l3s=', lat_l3s, ', lon_l3s=', lon_l3s)
              # print('lat_oi =', lat_oi[j_lat_oi]  , ', lon_oi = ', lon_oi[i_lon_oi])
                
                # day and night? currently only nighttime
                sst_bias_l3s = ds_actual_1d['sses_bias'][id_loc]
                daily_sst_avg_l3s2oi[2, j_lat_oi, i_lon_oi] += sst_l3s - sst_bias_l3s
                daily_sst_num_l3s2oi[2, j_lat_oi, i_lon_oi] += 1

              # if ds_actual_1d['l2p_flags'][id_loc] & mask_10thBit:  # daytime
              #     daily_sst_avg_l3s2oi[1, j_lat_oi, i_lon_oi] += sst_l3s - sst_bias_l3s
              #     daily_sst_num_l3s2oi[1, j_lat_oi, i_lon_oi] += 1
              # else:  # nighttime
              #     daily_sst_avg_l3s2oi[2, j_lat_oi, i_lon_oi] += sst_l3s - sst_bias_l3s
              #     daily_sst_num_l3s2oi[2, j_lat_oi, i_lon_oi] += 1

        if id_fn==0: 
            print('Finished the 1st file. \n')
            break
         
    daily_sst_avg_l3s2oi[2, :, :] = np.divide(daily_sst_avg_l3s2oi, daily_sst_num_l3s2oi, where=(daily_sst_num_l3s2oi != 0))
    daily_sst_avg_l3s2oi[2, :, :] = xr.where(daily_sst_num_l3s2oi==0, np.nan, daily_sst_avg_l3s2oi)
    daily_sst_num_l3s2oi[2, :, :] = xr.where(daily_sst_num_l3s2oi==0, np.nan, daily_sst_num_l3s2oi)

    da_daily_sst_avg_all = xr.DataArray(data=np.float32(daily_sst_avg_l3s2oi[0, :, :]), dims=['lat', 'lon'], coords={'lat': lat_oi, 'lon': lon_oi}, name='sst_avg_daily', attrs={'units':'Kelvin', '_FillValue':0})
    da_daily_sst_avg_day = xr.DataArray(data=np.float32(daily_sst_avg_l3s2oi[1, :, :]), dims=['lat', 'lon'], coords={'lat': lat_oi, 'lon': lon_oi}, name='sst_avg_daytime', attrs={'units':'Kelvin', '_FillValue':0})
    da_daily_sst_avg_nit = xr.DataArray(data=np.float32(daily_sst_avg_l3s2oi[2, :, :]), dims=['lat', 'lon'], coords={'lat': lat_oi, 'lon': lon_oi}, name='sst_avg_nighttime', attrs={'units':'Kelvin', '_FillValue':0})

    da_daily_sst_num_all = xr.DataArray(data=np.uint8(daily_sst_num_l3s2oi[0, :, :]), dims=['lat', 'lon'], coords={'lat': lat_oi, 'lon': lon_oi}, name='sst_num_daily', attrs=dict(_FillValue=0))
    da_daily_sst_num_day = xr.DataArray(data=np.uint8(daily_sst_num_l3s2oi[1, :, :]), dims=['lat', 'lon'], coords={'lat': lat_oi, 'lon': lon_oi}, name='sst_num_daytime', attrs=dict(_FillValue=-1))
    da_daily_sst_num_nit = xr.DataArray(data=np.uint8(daily_sst_num_l3s2oi[2, :, :]), dims=['lat', 'lon'], coords={'lat': lat_oi, 'lon': lon_oi}, name='sst_num_nighttime', attrs=dict(_FillValue=-1))

    ds_daily = xr.merge([da_daily_sst_avg_all, da_daily_sst_num_all, da_daily_sst_avg_day, da_daily_sst_num_day, da_daily_sst_avg_nit, da_daily_sst_num_nit])
    fn_daily_sst = str_date+'-STAR-L3S_GHRSST-SSTsubskin-LEO_AM_N-ACSPO_V2.80-v02.0-fv01.0.nc'
    ds_daily.to_netcdf(DIR_L3S+'/l3s2oi/'+fn_daily_sst)
    quit()

    jday += datetime.timedelta(days=1)
