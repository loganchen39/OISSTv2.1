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


DIR_L3U = '/glade/scratch/lgchen/data/L3U_OSPO_v2.61_SST_from_PO.DAAC/link'
# FN_L3U = DIR_L3U+"/test/20150702112000-OSPO-L3U_GHRSST-SSTsubskin-VIIRS_NPP-ACSPO_V2.61-v02.0-fv01.0.nc"

mask_10thBit = (np.int16)(2**9)
print('mask_10thBit: ', mask_10thBit)
# quit()

jday_20150101 = datetime.date(2015, 1 , 1 )
jday_20150102 = datetime.date(2020, 1 , 2 )
jday_20201231 = datetime.date(2020, 12, 31)

daily_sst_avg = np.zeros((3, 9000, 18000))  # 3 types of day, night, day and night; 
daily_sst_num = np.zeros((3, 9000, 18000), dtype=int)

jday = jday_20150101
while jday <= jday_20150101:
    str_date = jday.strftime('%Y%m%d')
    print('current date: ', str_date)
    ds = xr.open_mfdataset(paths=DIR_L3U+'/'+str_date+'*.nc')
    ds['sea_surface_temperature'] = xr.where(ds['quality_level']==5, ds['sea_surface_temperature'], np.nan)

 #  da_daily_sst = ds['sea_surface_temperature'].copy(deep=True)
 #  da_daily_sst_avg = da_daily_sst.mean(dim='time', skipna=True, keep_attrs=True)
 #  da_daily_sst_num = xr.where(da_daily_sst.isnull(), 0, 1.0)
 #  da_daily_sst_num = da_daily_sst_num.sum(dim='time', skipna=True, keep_attrs=True)
 ## print("da_daily_sst_avg.sizes: ", da_daily_sst_avg.sizes)  # reduced to 9000x18000
 ## print("da_daily_sst_num.sizes: ", da_daily_sst_num.sizes)  # reduced to 9000x18000

    print('bit operation & start ...')
    # with "/ 512", TypeError: ufunc 'bitwise_and' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''
    mask_day = ds['l2p_flags'].values & mask_10thBit  # / 512  # worked with mem=109G
  # mask_day = mask_day / np.int16(512)  # MemoryError: Unable to allocate 174. GiB for an array with shape (144, 9000, 18000) and data type float64
    print('type(mask_day): ', type(mask_day))
    print(mask_day)
    mask_day[mask_day!=0] = 1
  # mask_day = np.int16(np.where(mask_day!=0, 1, 0))  # Unable to allocate 174. GiB for an array with shape (144, 9000, 18000) and data type int64
  # mask_day = mask_10thBit & ds['l2p_flags'].to_numpy()
  # mask_day = mask_10thBit & np.array(ds['l2p_flags'])

    mask_day_sum = np.sum(mask_day, axis=(1, 2), keepdims=False)
    print('type(mask_day_sum): ', type(mask_day_sum))
    print(mask_day_sum)
    quit()

    mask_day = xr.DataArray(data=mask_day, coords=ds['l2p_flags'].coords, dims=ds['l2p_flags'].dims, name='mask_day')
  # mask_day = xr.where(mask_day!=0, 1, 0)  # Unable to allocate 174. GiB for an array with shape (144, 9000, 18000) and data type int64
    print('type(mask_day): ', type(mask_day))
    print(mask_day)

    # MemoryError: Unable to allocate 43.5 GiB for an array with shape (144, 9000, 18000) and data type int16
    mask_day_sum = mask_day.sum(dim=('lat', 'lon'), skipna=True)  
    print('type(mask_day_sum): ', type(mask_day_sum))
    print(mask_day_sum)

 
    da_daily_sst_day = xr.where(mask_day!=0, ds['sea_surface_temperature'], np.nan)
    print('type(da_daily_sst_day): ', da_daily_sst_day)
    print(da_daily_sst_day)
    quit()
    da_daily_sst_day_avg = da_daily_sst_day.mean(dim='time', skipna=True, keep_attrs=True)
    da_daily_sst_day_num = xr.where(da_daily_sst_day.isnull(), 0, 1.0)
    da_daily_sst_day_num = da_daily_sst_day_num.sum(dim='time', skipna=True, keep_attrs=True)

    ds_daily = xr.merge([da_daily_sst_day_avg, da_daily_sst_day_num])
    ds_daily.to_netcdf('./ds_daily.nc')
   
  # da_daily_sst_nit = xr.where(mask_day==0, ds['sea_surface_temperature'], np.nan)
  # da_daily_sst_nit_avg = da_daily_sst_nit.mean(dim='time', skipna=True, keep_attrs=True)
  # da_daily_sst_nit_num = xr.where(da_daily_sst_nit.isnull(), 0, 1.0)
  # da_daily_sst_nit_num = da_daily_sst_nit_num.sum(dim='time', skipna=True, keep_attrs=True)   

    quit()

    jday += datetime.timedelta(days=1)




# ds = xr.open_dataset(filename_or_obj=FN_L3U, mask_and_scale=True, decode_times=True).isel(time=0)
# da_sst = ds['sea_surface_temperature'] - 273.15
# str_time = ds['time'].dt.strftime("%Y-%m-%d %H:%M:%S").data
# print(str_time)
# 
# # for plotting
# fig  = plt.figure(figsize=(12, 8))
# fig.suptitle('L3U OSPO SST', fontsize=18, y=0.93)
# 
# proj = ccrs.PlateCarree()
# ax = plt.axes(projection=proj)
# ax.set_global()
# ax.coastlines(linewidth=0.5)
#  
# cmap = gvcmaps.BlueYellowRed
# cbar_kw = {'orientation':'horizontal', 'shrink':0.5, 'pad':0.08, 'extend':'both', 'extendrect':False, 'drawedges':False, 'label':''}
# ct = da_sst.plot.contourf(ax=ax, vmin=0.0, vmax=36, levels=10, cmap=cmap, add_colorbar=True, add_labels=False, cbar_kwargs=cbar_kw)
# # ct = da_sst.plot.contourf(ax=ax, vmin=18.0, vmax=28, levels=6, cmap=cmap, add_colorbar=True, extend='both', add_labels=False, cbar_kwargs=cbar_kw)
# 
# ax.set_title(label='SST', loc='left', fontsize=10, y=1.0, pad=6.0)
# ax.set_title(label='Celsius', loc='right', fontsize=10, y=1.0, pad=6.0)
# ax.set_title(label=str_time, loc='center', fontsize=10, y=1.0, pad=6.0)
# gvutil.set_axes_limits_and_ticks(ax=ax, xticks=np.linspace(-180, 180, 13), yticks=np.linspace(-90, 90, 7))
# gvutil.add_lat_lon_ticklabels(ax)
# ax.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol=''))
# ax.yaxis.set_major_formatter(LatitudeFormatter(degree_symbol=''))
# 
# plt.show()
