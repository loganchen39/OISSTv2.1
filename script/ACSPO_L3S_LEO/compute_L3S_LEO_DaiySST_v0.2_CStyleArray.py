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

import fort


DIR_L3S = '/glade/scratch/lgchen/data/L3S_LEO'
FN_L3S_TEST = DIR_L3S+"/test/20150101120000-STAR-L3S_GHRSST-SSTsubskin-LEO_AM_N-ACSPO_V2.80-v02.0-fv01.0.nc"
ds_l3s_test = xr.open_dataset(filename_or_obj=FN_L3S_TEST)
#  has to be np.float64, 9000: 89.99, 89.97, ..., -89.97, -89.99;
# lat_l3s = np.arange(start=89.99, stop=-90.0, step=-0.02, dtype=np.float64)  
# lon_l3s = np.arange(start=-179.99, stop=180.0, step=0.02, dtype=np.float64)  # 18000: -179.99, -179.97, ..., 179.97, 179.99;
ds_l3s_test.coords['lon'] = xr.where(ds_l3s_test.coords['lon']<0, 360+ds_l3s_test.coords['lon'], ds_l3s_test.coords['lon'])
ds_l3s_test.sortby(ds_l3s_test.coords['lon'])
lat_l3s = ds_l3s_test.coords['lat']  #  9000: 89.99, 89.97, ..., -89.97, -89.99;
lon_l3s = ds_l3s_test.coords['lon']  # 18000: -179.99, -179.97, ..., 179.97, 179.99;
N_LAT_L3S = 9000
N_LON_L3S = 18000

DIR_OI = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master'
FN_OI_MASK = DIR_OI + '/quarter-mask-extend_Ligang.nc'
ds_oi_mask = xr.open_dataset(filename_or_obj=FN_OI_MASK)
ds_oi_mask['landmask'] = ds_oi_mask.landmask.astype(np.int32)
lat_oi = ds_oi_mask.coords['lat']  # 720: -89.875, -89.625, -89.375, ...,  89.375,  89.625,  89.875
lon_oi = ds_oi_mask.coords['lon']  # 1440: 0.125 0.375 0.625 ... 359.625 359.875
# lat_oi = np.arange(start=-89.875, stop=90.0, step=0.25, dtype=np.float64)  # 720: -89.875, -89.625, -89.375, ...,  89.375,  89.625,  89.875
# lon_oi = np.arange(start=0.125, stop=360.0, step=0.25, dtype=np.float64)  # 1440: 0.125 0.375 0.625 ... 359.625 359.875
N_LAT_OI = 720
N_LON_OI = 1440

np.set_printoptions(threshold=np.inf)

# idx_lat_l3s2oi = np.zeros((9000 , ), dtype=np.int32)
# idx_lon_l3s2oi = np.zeros((18000, ), dtype=np.int32)
idx_lat_l3s2oi = np.around(4*(ds_l3s_test.coords['lat'].values+89.875)).astype(np.int32)
idx_lon_l3s2oi = np.around(4*(ds_l3s_test.coords['lon'].values-0.125 )).astype(np.int32)
print('type(idx_lat_l3s2oi): ', type(idx_lat_l3s2oi))
print('type(idx_lon_l3s2oi): ', type(idx_lon_l3s2oi))

idx_lat_oi2l3s = np.zeros((720 , 2), dtype=np.int32)
idx_lon_oi2l3s = np.zeros((1440, 2), dtype=np.int32)
# For each oi lat point, calculate its corresponding start and end indices in l3s
idx_lat_oi_curr = idx_lat_l3s2oi[0]
idx_lat_oi2l3s[idx_lat_oi_curr, 0] = 0
for i_lat_l3s in range(1, 9000): 
    if idx_lat_l3s2oi[i_lat_l3s] != idx_lat_oi_curr:
        idx_lat_oi2l3s[idx_lat_oi_curr, 1] = i_lat_l3s
        idx_lat_oi_curr = idx_lat_l3s2oi[i_lat_l3s]
        idx_lat_oi2l3s[idx_lat_oi_curr, 0] = i_lat_l3s
else:
    idx_lat_oi2l3s[idx_lat_oi_curr, 1] = 9000

# For each oi lon point, calculate its corresponding start and end indices in l3s
idx_lon_oi_curr = idx_lon_l3s2oi[0]
idx_lon_oi2l3s[idx_lon_oi_curr, 0] = 0
for i_lon_l3s in range(1, 18000): 
    if idx_lon_l3s2oi[i_lon_l3s] != idx_lon_oi_curr:
        idx_lon_oi2l3s[idx_lon_oi_curr, 1] = i_lon_l3s
        idx_lon_oi_curr = idx_lon_l3s2oi[i_lon_l3s]
        idx_lon_oi2l3s[idx_lon_oi_curr, 0] = i_lon_l3s
else:
    idx_lon_oi2l3s[idx_lon_oi_curr, 1] = 18000

# for fortran index convention
idx_lat_oi2l3s += 1
idx_lon_oi2l3s += 1
# print('\n\n\n idx_lat_oi2l3s[:, :]: ', idx_lat_oi2l3s)
# print('\n\n\n idx_lon_oi2l3s[:, :]: ', idx_lon_oi2l3s)
# quit()

# 1 record, could be either daytime or nighttime;
daily_sst_avg_l3s2oi = np.zeros((720, 1440), dtype=np.float32)   
daily_sst_num_l3s2oi = np.zeros((720, 1440), dtype=np.int32)  # for binned to avg

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
    for (id_fn, fn) in enumerate(fns, start=0):
        print(str(id_fn).zfill(3)+', current file: ', fn)
        ds = xr.open_dataset(filename_or_obj=fn, mask_and_scale=True, decode_times=True  \
            , drop_variables=['sst_dtime', 'sses_standard_deviation', 'l3s_sst_reference', 'dt_analysis', 'sst_count', 'sst_source'  \
            , 'satellite_zenith_angle', 'wind_speed', 'crs', 'sst_gradient_magnitude', 'sst_front_position']).isel(time=0)

        ds['quality_level'] = ds.quality_level.astype(np.int8)  # convert back to type byte.
        ds['sea_surface_temperature'] = xr.where(ds.quality_level==5, ds.sea_surface_temperature, np.nan)
        ds.coords['lon'] = xr.where(ds.coords['lon']<0, 360+ds.coords['lon'], ds.coords['lon'])
        ds.sortby(ds.coords['lon'])
        ds['l2p_flags'] = ds.l2p_flags.astype(np.int16)
        ds['sea_surface_temperature'] = ds['sea_surface_temperature'] - ds['sses_bias']  # actual sst
      # print('ds.coords[lon]: ', ds.coords['lon'].values)
      # print('\n\n\n ds: ', ds)

        print('Start processing SST, current time: ', datetime.datetime.now().strftime('%H:%M:%S'))
        sst_fort = np.asfortranarray(ds['sea_surface_temperature'].values)
        oi_mask_fort = np.asfortranarray(ds_oi_mask.landmask.values)
        fort.sst_acspo2oi_1rec(ds['sea_surface_temperature'].values, daily_sst_avg_l3s2oi, daily_sst_num_l3s2oi  \
            , ds_oi_mask.landmask.values, idx_lat_oi2l3s, idx_lon_oi2l3s)
        print('End processing SST, current time: ', datetime.datetime.now().strftime('%H:%M:%S'))

        print('daily_sst_avg_l3s2oi: ', daily_sst_avg_l3s2oi[350, :])
        print('daily_sst_num_l3s2oi: ', daily_sst_num_l3s2oi[350, :])
        quit()

         
    daily_sst_avg_l3s2oi = np.divide(daily_sst_avg_l3s2oi, daily_sst_num_l3s2oi, where=(daily_sst_num_l3s2oi != 0))
    daily_sst_avg_l3s2oi = xr.where(daily_sst_num_l3s2oi==0, np.nan, daily_sst_avg_l3s2oi)
    daily_sst_num_l3s2oi = xr.where(daily_sst_num_l3s2oi==0, np.nan, daily_sst_num_l3s2oi)

    da_daily_sst_avg_nit = xr.DataArray(data=np.float32(daily_sst_avg_l3s2oi), dims=['lat', 'lon'], coords={'lat': lat_oi, 'lon': lon_oi}, name='sst_avg_nighttime', attrs={'units':'Kelvin', '_FillValue':0})
    da_daily_sst_num_nit = xr.DataArray(data=np.uint8  (daily_sst_num_l3s2oi), dims=['lat', 'lon'], coords={'lat': lat_oi, 'lon': lon_oi}, name='sst_num_nighttime', attrs=dict(_FillValue=-1))

    ds_daily = xr.merge([da_daily_sst_avg_nit, da_daily_sst_num_nit])
    fn_daily_sst = str_date+'-STAR-L3S_GHRSST-SSTsubskin-LEO_AM_N-ACSPO_V2.80-v02.0-fv01.0.nc'
    ds_daily.to_netcdf(DIR_L3S+'/l3s2oi/test/'+fn_daily_sst)
    quit()

    jday += datetime.timedelta(days=1)
