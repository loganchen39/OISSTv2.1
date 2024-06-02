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


DIR_L3U = '/glade/scratch/lgchen/data/L3U_OSPO_v2.61_SST_from_PO.DAAC'
# FN_L3U = DIR_L3U+"/test/20150702112000-OSPO-L3U_GHRSST-SSTsubskin-VIIRS_NPP-ACSPO_V2.61-v02.0-fv01.0.nc"

mask_10thBit = (np.int16)(2**9)
print('mask_10thBit: ', mask_10thBit)

ds = xr.open_dataset(filename_or_obj=DIR_L3U+"/test/20150702112000-OSPO-L3U_GHRSST-SSTsubskin-VIIRS_NPP-ACSPO_V2.61-v02.0-fv01.0.nc")
lat = ds.coords['lat']
lon = ds.coords['lon']

daily_sst_avg = np.zeros((3, 9000, 18000), dtype=np.float32)  # 3 types of day and night, day, night; 
daily_sst_num = np.zeros((3, 9000, 18000), dtype=np.int)

jday_20150101 = datetime.date(2015, 1 , 1 )
jday_20150131 = datetime.date(2020, 1 , 31)
jday_20201231 = datetime.date(2020, 12, 31)

jday = jday_20150101
while jday <= jday_20150131:
    str_date = jday.strftime('%Y%m%d')
    print('current date: ', str_date)
    fns = glob.glob(pathname=DIR_L3U+'/link/'+str_date+'*.nc')
  # print('fns: ', fns)

  # daily_sst_avg = 0.0  # TypeError: 'float' object is not subscriptable
  # daily_sst_num = 0  # it becomes an integer type number, not an ndarray anymore!

    daily_sst_avg = 0.0*daily_sst_avg
    daily_sst_num = 0*daily_sst_num
    for (id, fn) in enumerate(fns, start=1):
        print(str(id).zfill(3)+', current file: ', fn)
        ds = xr.open_dataset(filename_or_obj=fn, mask_and_scale=True, decode_times=True).isel(time=0)
      # ds['sea_surface_temperature'] = xr.where(ds['quality_level']==5, ds['sea_surface_temperature'], np.nan)
        ds['sea_surface_temperature'] = xr.where(ds['quality_level']==4, ds['sea_surface_temperature'], np.nan)

        # day and night
        da_sst = ds['sea_surface_temperature'].copy(deep=True)
        da_sst = xr.where(da_sst.isnull(), 0, da_sst)
        daily_sst_avg[0, :, :] = daily_sst_avg[0, :, :] + da_sst
        da_sst = xr.where(da_sst!=0, 1, 0)
        daily_sst_num[0, :, :] = daily_sst_num[0, :, :] + da_sst

        # daytime
        da_sst = ds['sea_surface_temperature'].copy(deep=True)
        mask_day = ds['l2p_flags'].values & mask_10thBit
        mask_day = np.where(mask_day!=0, 1, 0)
      # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
      # da_sst = xr.where(mask_day!=0 and da_sst.notnull(), da_sst, 0)  
        da_sst = xr.where(da_sst.notnull(), da_sst, 0)
        da_sst = xr.where(mask_day!=0, da_sst, 0)
        daily_sst_avg[1, :, :] = daily_sst_avg[1, :, :] + da_sst
        da_sst = xr.where(da_sst!=0, 1, 0)
        daily_sst_num[1, :, :] = daily_sst_num[1, :, :] + da_sst

        # nighttime
        da_sst = ds['sea_surface_temperature'].copy(deep=True)
        mask_day = ds['l2p_flags'].values & mask_10thBit
        mask_day = np.where(mask_day!=0, 1, 0)
      # da_sst = xr.where(mask_day==0 and da_sst.notnull(), da_sst, 0)  # doesn't seem to work
        da_sst = xr.where(da_sst.notnull(), da_sst, 0)
        da_sst = xr.where(mask_day==0, da_sst, 0)
        daily_sst_avg[2, :, :] = daily_sst_avg[2, :, :] + da_sst
        da_sst = xr.where(da_sst!=0, 1, 0)
        daily_sst_num[2, :, :] = daily_sst_num[2, :, :] + da_sst

      # if id==5:
      #     break
        
  # daily_sst_avg = np.where(daily_sst_num==0, np.nan, daily_sst_avg/daily_sst_num)
    daily_sst_avg = np.divide(daily_sst_avg, daily_sst_num, where=(daily_sst_num!=0))

    da_daily_sst_avg_all = xr.DataArray(data=np.float32(daily_sst_avg[0, :, :]), dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}, name='daily_sst_avg', attrs={'units':'Kelvin', '_FillValue':0})
    da_daily_sst_avg_day = xr.DataArray(data=np.float32(daily_sst_avg[1, :, :]), dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}, name='daytime_sst_avg', attrs={'units':'Kelvin', '_FillValue':0})
    da_daily_sst_avg_nit = xr.DataArray(data=np.float32(daily_sst_avg[2, :, :]), dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}, name='nighttime_sst_avg', attrs={'units':'Kelvin', '_FillValue':0})

    da_daily_sst_num_all = xr.DataArray(data=np.uint8(daily_sst_num[0, :, :]), dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}, name='daily_sst_num', attrs=dict(_FillValue=0))
    da_daily_sst_num_day = xr.DataArray(data=np.uint8(daily_sst_num[1, :, :]), dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}, name='daytime_sst_num', attrs=dict(_FillValue=0))
    da_daily_sst_num_nit = xr.DataArray(data=np.uint8(daily_sst_num[2, :, :]), dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}, name='nighttime_sst_num', attrs=dict(_FillValue=0))

    ds_daily = xr.merge([da_daily_sst_avg_all, da_daily_sst_num_all, da_daily_sst_avg_day, da_daily_sst_num_day, da_daily_sst_avg_nit, da_daily_sst_num_nit])
    fn_daily_sst = str_date+'-OSPO-L3U_GHRSST-SSTsubskin-VIIRS_NPP-ACSPO_V2.61-v02.0-fv01.0.nc'
    ds_daily.to_netcdf(DIR_L3U+'/sst_day_night/'+fn_daily_sst)
  # quit()

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
