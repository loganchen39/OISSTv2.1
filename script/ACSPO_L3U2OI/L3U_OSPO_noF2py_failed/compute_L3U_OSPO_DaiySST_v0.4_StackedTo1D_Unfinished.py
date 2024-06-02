#!/glade/u/apps/ch/opt/python/3.7.9/gnu/9.1.0/pkg-library/20201220/bin/python3

'''
Description: python code to compute L3U_OSPO daily SST.
Author: Ligang Chen
Date created: 10/11/2021
Date last modified: 10/11/2021 
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


DIR_L3U = '/glade/scratch/lgchen/data/L3U_OSPO_v2.61_SST_from_PO.DAAC'
FN_L3U_TEST = DIR_L3U+"/test/20150702112000-OSPO-L3U_GHRSST-SSTsubskin-VIIRS_NPP-ACSPO_V2.61-v02.0-fv01.0.nc"
ds_l3u = xr.open_dataset(filename_or_obj=FN_L3U_TEST)
# lat_l3u = ds_l3u.coords['lat']  #  9000: 89.99, 89.97, ..., -89.97, -89.99;
# lon_l3u = ds_l3u.coords['lon']  # 18000: -179.99, -179.97, ..., 179.97, 179.99;
lat_l3u = np.arange(start=89.99, stop=-90.0, step=-0.02, dtype=np.float64)  #  9000: 89.99, 89.97, ..., -89.97, -89.99;
lon_l3u = np.arange(start=-179.99, stop=180.0, step=0.02, dtype=np.float64)  # 18000: -179.99, -179.97, ..., 179.97, 179.99;
# print('type(lat_l3u): ', type(lat_l3u))
# print('lat_l3u: ', lat_l3u)
# print('lon_l3u: ', lon_l3u)
# quit()

DIR_OISST = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master'
FN_OISST_QUARTER_MASK = DIR_OISST + '/quarter-mask-extend_Ligang.nc'
ds_oisst_quarter_mask = xr.open_dataset(filename_or_obj=FN_OISST_QUARTER_MASK)
# lat_oi = ds_oisst_quarter_mask.coords['lat']  # 720: -89.875, -89.625, -89.375, ...,  89.375,  89.625,  89.875
# lon_oi = ds_oisst_quarter_mask.coords['lon']  # 1440: 0.125 0.375 0.625 ... 359.625 359.875
lat_oi = np.arange(start=-89.875, stop=90.0, step=0.25, dtype=np.float64)  # 720: -89.875, -89.625, -89.375, ...,  89.375,  89.625,  89.875
lon_oi = np.arange(start=0.125, stop=360.0, step=0.25, dtype=np.float64)  # 1440: 0.125 0.375 0.625 ... 359.625 359.875
# print('type(lat_oi): ', type(lat_oi))
# print('lat_oi: ', lat_oi)
# print('lon_oi: ', lon_oi)


mask_10thBit = (np.int16)(2**9)
print('mask_10thBit: ', mask_10thBit)

daily_sst_avg_l3u2oi = np.zeros((3, 720, 1440), dtype=np.float32)  # 3 types of day and night, day, night; 
daily_sst_num_l3u2oi = np.zeros((3, 720, 1440), dtype=np.uint8)

# da_daily_sst_avg_l3u2oi = xr.DataArray(data=np.zeros((3, 720, 1440), dtype=np.float32), name='da_daily_sst_avg_l3u2oi', dims=['lat', 'lon']  \
#     , coords={'lat': lat_oi, 'lon': lon_oisst})  # attrs=dict(_FillValue=-9999)
# da_daily_sst_num_l3u2oi = xr.DataArray(data=np.zeros((3, 720, 1440), dtype=np.uint8)  , name='da_daily_sst_num_l3u2oi', dims=['lat', 'lon']  \
#     , coords={'lat': lat_oi, 'lon': lon_oisst})  # attrs=dict(_FillValue=-9999)

jday_20150101 = datetime.date(2015, 1 , 1 )
jday_20150131 = datetime.date(2020, 1 , 31)
jday_20201231 = datetime.date(2020, 12, 31)

jday = jday_20150101
while jday <= jday_20150101:
    str_date = jday.strftime('%Y%m%d')
    print('current date: ', str_date)
    fns = glob.glob(pathname=DIR_L3U+'/link/'+str_date+'*.nc')
  # print('fns: ', fns)

    daily_sst_avg_l3u2oi.fill(0)
    daily_sst_num_l3u2oi.fill(0)
    for (id, fn) in enumerate(fns, start=1):
        print(str(id).zfill(3)+', current file: ', fn)
        ds = xr.open_dataset(filename_or_obj=fn, mask_and_scale=True, decode_times=True  \
            , drop_variables=['or_number_of_pixels', 'dt_analysis', 'satellite_zenith_angle', 'sses_standard_deviation'  \
            , 'wind_speed', 'sst_dtime', 'crs']).isel(time=0)

        ds_cropped = ds.sel(lat=slice(ds.attrs['geospatial_lat_max'], ds.attrs['geospatial_lat_min'])  \
            , lon=(slice(ds.attrs['geospatial_lon_min'], ds.attrs['geospatial_lon_max'])))
        ds_cropped['quality_level'] = ds_cropped.quality_level.astype(np.int8)  # convert back to type byte.
        ds_cropped['sea_surface_temperature'] = xr.where(ds_cropped.quality_level==5, ds_cropped.sea_surface_temperature, np.nan)
        ds_cropped.coords['lon'] = xr.where(ds_cropped.coords['lon']<0, 360+ds_cropped.coords['lon'], ds_cropped.coords['lon'])
        ds_cropped.sortby(ds_cropped.lon)
      # print('\n\n\n ds_cropped: ', ds_cropped)

        ds_stacked = ds_cropped.stack(loc=('lat', 'lon'))
      # print('\n\n\n ds_stacked: ', ds_stacked)
        ds_actual_1d = ds_stacked.where(ds_stacked['sea_surface_temperature'].notnull(), drop=True)
        ds_actual_1d['l2p_flags'] = ds_actual_1d.l2p_flags.astype(np.int16)  
      # print('\n\n\n ds_actual_1d: ', ds_actual_1d)

        print('ds_actual_1d.dims[loc]: ', ds_actual_1d.dims['loc'])
        # no need to use for-loop? use numpy array? what about the lat/lon info for interpolation? 
        for (idx, loc_l3u) in enumerate(ds_actual_1d.coords['loc'].values):
            print('idx = ', str(idx).zfill(7))

            sst_l3u = np.float32(ds_actual_1d['sea_surface_temperature'][idx])
            if 268.15 < sst_l3u and sst_l3u < 323.15:
                lat_l3u = loc_l3u[0]
                lon_l3u = loc_l3u[1]
              # if lon_l3u < 0 or lon_l3u > 360: print("ERROR: lon_l3u out of range!")
                j_lat_oi = int(round(4*(lat_l3u+89.875)))
              # if j_lat_oi < 0: j_lat_oi = 0
              # if j_lat_oi >= 720: j_lat_oi = 719
                i_lon_oi = int(round(4*(lon_l3u-0.125)))
              # if i_lon_oi < 0: i_lon_oi = 0
              # if i_lon_oi >= 1440: i_lon_oi = 1439

              # print('lat_l3u=', lat_l3u, ', lon_l3u=', lon_l3u)
              # print('lat_oi =', lat_oi[j_lat_oi]  , ', lon_oi = ', lon_oi[i_lon_oi])
                
                # day and night
                sst_bias_l3u = ds_actual_1d['sses_bias'][idx]
                daily_sst_avg_l3u2oi[0, j_lat_oi, i_lon_oi] += sst_l3u - sst_bias_l3u
                daily_sst_num_l3u2oi[0, j_lat_oi, i_lon_oi] += 1

                if ds_actual_1d['l2p_flags'][idx] & mask_10thBit:  # daytime
                    daily_sst_avg_l3u2oi[1, j_lat_oi, i_lon_oi] += sst_l3u - sst_bias_l3u
                    daily_sst_num_l3u2oi[1, j_lat_oi, i_lon_oi] += 1
                else:  # nighttime
                    daily_sst_avg_l3u2oi[2, j_lat_oi, i_lon_oi] += sst_l3u - sst_bias_l3u
                    daily_sst_num_l3u2oi[2, j_lat_oi, i_lon_oi] += 1

        if id==0: 
            break
         
    daily_sst_avg_l3u2oi = np.divide(daily_sst_avg_l3u2oi, daily_sst_num_l3u2oi, where=(daily_sst_num_l3u2oi != 0))
    daily_sst_avg_l3u2oi = xr.where(daily_sst_num_l3u2oi==0, np.nan, daily_sst_avg_l3u2oi)
    daily_sst_num_l3u2oi = xr.where(daily_sst_num_l3u2oi==0, np.nan, daily_sst_num_l3u2oi)

    da_daily_sst_avg_all = xr.DataArray(data=np.float32(daily_sst_avg_l3u2oi[0, :, :]), dims=['lat', 'lon'], coords={'lat': lat_oi, 'lon': lon_oi}, name='sst_avg_daily', attrs={'units':'Kelvin', '_FillValue':0})
    da_daily_sst_avg_day = xr.DataArray(data=np.float32(daily_sst_avg_l3u2oi[1, :, :]), dims=['lat', 'lon'], coords={'lat': lat_oi, 'lon': lon_oi}, name='sst_avg_daytime', attrs={'units':'Kelvin', '_FillValue':0})
    da_daily_sst_avg_nit = xr.DataArray(data=np.float32(daily_sst_avg_l3u2oi[2, :, :]), dims=['lat', 'lon'], coords={'lat': lat_oi, 'lon': lon_oi}, name='sst_avg_nighttime', attrs={'units':'Kelvin', '_FillValue':0})

    da_daily_sst_num_all = xr.DataArray(data=np.uint8(daily_sst_num_l3u2oi[0, :, :]), dims=['lat', 'lon'], coords={'lat': lat_oi, 'lon': lon_oi}, name='sst_num_daily', attrs=dict(_FillValue=0))
    da_daily_sst_num_day = xr.DataArray(data=np.uint8(daily_sst_num_l3u2oi[1, :, :]), dims=['lat', 'lon'], coords={'lat': lat_oi, 'lon': lon_oi}, name='sst_num_daytime', attrs=dict(_FillValue=-1))
    da_daily_sst_num_nit = xr.DataArray(data=np.uint8(daily_sst_num_l3u2oi[2, :, :]), dims=['lat', 'lon'], coords={'lat': lat_oi, 'lon': lon_oi}, name='sst_num_nighttime', attrs=dict(_FillValue=-1))

    ds_daily = xr.merge([da_daily_sst_avg_all, da_daily_sst_num_all, da_daily_sst_avg_day, da_daily_sst_num_day, da_daily_sst_avg_nit, da_daily_sst_num_nit])
    fn_daily_sst = str_date+'-OSPO-L3U_GHRSST-SSTsubskin-VIIRS_NPP-ACSPO_V2.61-v02.0-fv01.0.nc'
    ds_daily.to_netcdf(DIR_L3U+'/sst_day_night/l3u2oi/'+fn_daily_sst)
    quit()

    jday += datetime.timedelta(days=1)
