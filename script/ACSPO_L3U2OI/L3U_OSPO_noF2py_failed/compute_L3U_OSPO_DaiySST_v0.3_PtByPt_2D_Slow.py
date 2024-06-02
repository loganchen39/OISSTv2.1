#!/glade/u/apps/ch/opt/python/3.7.9/gnu/9.1.0/pkg-library/20201220/bin/python3

# #!/glade/u/apps/ch/opt/python/3.6.8/gnu/8.3.0/pkg-library/20190627/bin/python3

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
# lat_oisst = ds_oisst_quarter_mask.coords['lat']  # 720: -89.875, -89.625, -89.375, ...,  89.375,  89.625,  89.875
# lon_oisst = ds_oisst_quarter_mask.coords['lon']  # 1440: 0.125 0.375 0.625 ... 359.625 359.875
lat_oisst = np.arange(start=-89.875, stop=90.0, step=0.25, dtype=np.float64)  # 720: -89.875, -89.625, -89.375, ...,  89.375,  89.625,  89.875
lon_oisst = np.arange(start=0.125, stop=360.0, step=0.25, dtype=np.float64)  # 1440: 0.125 0.375 0.625 ... 359.625 359.875
# print('type(lat_oisst): ', type(lat_oisst))
# print('lat_oisst: ', lat_oisst)
# print('lon_oisst: ', lon_oisst)


mask_10thBit = (np.int16)(2**9)
print('mask_10thBit: ', mask_10thBit)

daily_sst_avg_l3u2oi = np.zeros((3, 720, 1440), dtype=np.float32)  # 3 types of day and night, day, night; 
daily_sst_num_l3u2oi = np.zeros((3, 720, 1440), dtype=np.uint8)

# da_daily_sst_avg_l3u2oi = xr.DataArray(data=np.zeros((3, 720, 1440), dtype=np.float32), name='da_daily_sst_avg_l3u2oi', dims=['lat', 'lon']  \
#     , coords={'lat': lat_oisst, 'lon': lon_oisst})  # attrs=dict(_FillValue=-9999)
# da_daily_sst_num_l3u2oi = xr.DataArray(data=np.zeros((3, 720, 1440), dtype=np.uint8)  , name='da_daily_sst_num_l3u2oi', dims=['lat', 'lon']  \
#     , coords={'lat': lat_oisst, 'lon': lon_oisst})  # attrs=dict(_FillValue=-9999)

jday_20150101 = datetime.date(2015, 1 , 1 )
jday_20150131 = datetime.date(2020, 1 , 31)
jday_20201231 = datetime.date(2020, 12, 31)

jday = jday_20150101
while jday <= jday_20150101:
    str_date = jday.strftime('%Y%m%d')
    print('current date: ', str_date)
    fns = glob.glob(pathname=DIR_L3U+'/link/'+str_date+'*.nc')
  # print('fns: ', fns)

  # daily_sst_avg = 0.0  # TypeError: 'float' object is not subscriptable
  # daily_sst_num = 0*daily_sst_num  # it becomes an integer type number, not an ndarray anymore! Re-assign memory like new array?

    daily_sst_avg_l3u2oi.fill(0)
    daily_sst_num_l3u2oi.fill(0)
  # da_daily_sst_avg_l3u2oi.data.fill(0)
  # da_daily_sst_num_l3u2oi.data.fill(0)
    for (id, fn) in enumerate(fns, start=1):
        print(str(id).zfill(3)+', current file: ', fn)
        ds = xr.open_dataset(filename_or_obj=fn, mask_and_scale=True, decode_times=True).isel(time=0)
      # ds['sea_surface_temperature'] = xr.where(ds['quality_level']==5, ds['sea_surface_temperature'], np.nan)
        da_sst = xr.where(ds['quality_level']==5, ds['sea_surface_temperature'], np.nan)

        lat_min = ds.attrs['geospatial_lat_min']
        lat_max = ds.attrs['geospatial_lat_max']
        lon_min = ds.attrs['geospatial_lon_min']
        lon_max = ds.attrs['geospatial_lon_max']

        j_lat_min = int((89.99-lat_min)*50)
        j_lat_max = int((89.99-lat_max)*50)
        i_lon_min = int((179.99+lon_min)*50)
        i_lon_max = int((179.99+lon_max)*50)

        print('(lat_min, lon_min, lat_max, lon_max): ', lat_min, lon_min, lat_max, lon_max)
        print('(j_lat_min, i_lon_min, j_lat_max, i_lon_max): ', j_lat_min, i_lon_min, j_lat_max, i_lon_max)
      # quit()

        # no need to use for-loop? use numpy array? what about the lat/lon info for interpolation? 
      # for j_lat in range(start=j_lat_min, stop=j_lat_max, step=-1):  # TypeError: range() takes no keyword arguments
        for j_lat in range(j_lat_min, j_lat_max, -1):
            for i_lon in range(i_lon_min, i_lon_max, 1):
          # for i_lon in range(start=i_lon_min, stop=i_lon_max, step=1):
              # print('j_lat=', j_lat, ', i_lon=', i_lon, ', sst=', da_sst[j_lat, i_lon].data) # sst= <xarray.DataArray ()> if only da_sst[j_lat, i_lon]
              # print('j_lat=', j_lat, ', i_lon=', i_lon, ', sst=', da_sst.data[j_lat, i_lon])
              # time.sleep(2)

                # sst-sst_bias is more accurate!
              # if da_sst.data[j_lat, i_lon] != np.nan and 273.15-5 < da_sst.data[j_lat, i_lon] and da_sst.data[j_lat, i_lon] < 273.15+50:
                if da_sst[j_lat, i_lon] != np.nan and 273.15-5 < da_sst[j_lat, i_lon] and da_sst[j_lat, i_lon] < 273.15+50:
                  # print('Inside if, j_lat=', j_lat, ', i_lon=', i_lon, ', sst=', da_sst.data[j_lat, i_lon])
                  # time.sleep(2)

                    lat_curr = lat_l3u[j_lat]
                    lon_curr = lon_l3u[i_lon]
                    if lon_curr < 0: lon_curr = 360 + lon_curr  # [-180, 180] to [0, 360]
                    j_lat_oisst = int(round(4*(lat_curr+89.875)))
                    if j_lat_oisst < 0: j_lat_oisst = 0
                    if j_lat_oisst >= 720: j_lat_oisst = 719
                    i_lon_oisst = int(round(4*(lon_curr-0.125)))
                    if i_lon_oisst < 0: i_lon_oisst = 0
                    if i_lon_oisst >= 1440: i_lon_oisst = 1439

                  # print('lat_l3u=', lat_curr, ', lon_l3u=', lon_curr)
                  # print('lat_oi =', lat_oisst[j_lat_oisst]  , ', lon_oi = ', lon_oisst[i_lon_oisst])
                    
                    # day and night
                    daily_sst_avg_l3u2oi[0, j_lat_oisst, i_lon_oisst] += da_sst[j_lat, i_lon] - ds['sses_bias'][j_lat, i_lon]
                    daily_sst_num_l3u2oi[0, j_lat_oisst, i_lon_oisst] += 1

                    if ds['l2p_flags'][j_lat, i_lon] & mask_10thBit:  # daytime
                        daily_sst_avg_l3u2oi[1, j_lat_oisst, i_lon_oisst] += da_sst[j_lat, i_lon] - ds['sses_bias'][j_lat, i_lon]
                        daily_sst_num_l3u2oi[1, j_lat_oisst, i_lon_oisst] += 1
                    else:  # nighttime
                        daily_sst_avg_l3u2oi[2, j_lat_oisst, i_lon_oisst] += da_sst[j_lat, i_lon] - ds['sses_bias'][j_lat, i_lon]
                        daily_sst_num_l3u2oi[2, j_lat_oisst, i_lon_oisst] += 1

        if id==2: 
            break
         
  # daily_sst_avg = np.where(daily_sst_num==0, np.nan, daily_sst_avg/daily_sst_num)  # doesn't work this way
  # daily_sst_avg = np.divide(daily_sst_avg, daily_sst_num, where=(daily_sst_num!=0))
    daily_sst_avg_l3u2oi = np.divide(daily_sst_avg_l3u2oi, daily_sst_num_l3u2oi, where=(daily_sst_num_l3u2oi != 0))
    daily_sst_avg_l3u2oi = xr.where(daily_sst_avg_l3u2oi==0, np.nan, daily_sst_avg_l3u2oi)
    daily_sst_num_l3u2oi = xr.where(daily_sst_num_l3u2oi==0, np.nan, daily_sst_num_l3u2oi)

    da_daily_sst_avg_all = xr.DataArray(data=np.float32(daily_sst_avg_l3u2oi[0, :, :]), dims=['lat', 'lon'], coords={'lat': lat_oisst, 'lon': lon_oisst}, name='sst_avg_daily', attrs={'units':'Kelvin', '_FillValue':0})
    da_daily_sst_avg_day = xr.DataArray(data=np.float32(daily_sst_avg_l3u2oi[1, :, :]), dims=['lat', 'lon'], coords={'lat': lat_oisst, 'lon': lon_oisst}, name='sst_avg_daytime', attrs={'units':'Kelvin', '_FillValue':0})
    da_daily_sst_avg_nit = xr.DataArray(data=np.float32(daily_sst_avg_l3u2oi[2, :, :]), dims=['lat', 'lon'], coords={'lat': lat_oisst, 'lon': lon_oisst}, name='sst_avg_nighttime', attrs={'units':'Kelvin', '_FillValue':0})

    da_daily_sst_num_all = xr.DataArray(data=np.uint8(daily_sst_num_l3u2oi[0, :, :]), dims=['lat', 'lon'], coords={'lat': lat_oisst, 'lon': lon_oisst}, name='sst_num_daily', attrs=dict(_FillValue=0))
    da_daily_sst_num_day = xr.DataArray(data=np.uint8(daily_sst_num_l3u2oi[1, :, :]), dims=['lat', 'lon'], coords={'lat': lat_oisst, 'lon': lon_oisst}, name='sst_num_daytime', attrs=dict(_FillValue=0))
    da_daily_sst_num_nit = xr.DataArray(data=np.uint8(daily_sst_num_l3u2oi[2, :, :]), dims=['lat', 'lon'], coords={'lat': lat_oisst, 'lon': lon_oisst}, name='sst_num_nighttime', attrs=dict(_FillValue=0))

    ds_daily = xr.merge([da_daily_sst_avg_all, da_daily_sst_num_all, da_daily_sst_avg_day, da_daily_sst_num_day, da_daily_sst_avg_nit, da_daily_sst_num_nit])
    fn_daily_sst = str_date+'-OSPO-L3U_GHRSST-SSTsubskin-VIIRS_NPP-ACSPO_V2.61-v02.0-fv01.0.nc'
    ds_daily.to_netcdf(DIR_L3U+'/sst_day_night/l3u2oi/'+fn_daily_sst)
    quit()

    jday += datetime.timedelta(days=1)
