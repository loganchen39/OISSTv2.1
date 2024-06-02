#!/glade/u/apps/ch/opt/python/3.7.9/gnu/9.1.0/pkg-library/20201220/bin/python3

'''
Description: python code to plot ACSPO L3S_LEO daily SST.
Author: Ligang Chen
Date created: 10/27/2021
Date last modified: 10/27/2021 
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

# import fort


DIR_L3S = '/glade/scratch/lgchen/data/L3S_LEO'
# keywordargs = { 'drop_variables' : ['sst_dtime', 'sses_standard_deviation', 'l3s_sst_reference', 'dt_analysis', 'sst_count', 'sst_source'  \
#     , 'satellite_zenith_angle', 'wind_speed', 'crs', 'sst_gradient_magnitude', 'sst_front_position'] }
# ds_l3s = xr.open_mfdataset(paths=DIR_L3S+'/link/20150101*.nc', kwargs=keywordargs)
# ds_l3s = xr.open_mfdataset(paths=DIR_L3S+'/link/20150101*.nc', backend_kwargs=keywordargs)

ds_l3s = xr.open_mfdataset( paths=DIR_L3S+'/link/20150701*_D-ACSPO*.nc', concat_dim='time', combine='nested'  \
    , drop_variables=['sst_dtime', 'sses_standard_deviation', 'l3s_sst_reference', 'dt_analysis', 'sst_count', 'sst_source'  \
    , 'satellite_zenith_angle', 'wind_speed', 'crs', 'sst_gradient_magnitude', 'sst_front_position']  )  # worked with 'combine'
print(ds_l3s)
# ds_l3s.coords['lon'] = xr.where(ds_l3s.coords['lon']<0, 360+ds_l3s.coords['lon'], ds_l3s.coords['lon'])
# ds_l3s.sortby(ds_l3s.coords['lon'])

ds_l3s['quality_level'] = ds_l3s.quality_level.astype(np.int8)
ds_l3s['sea_surface_temperature'] = xr.where(ds_l3s.quality_level==5, ds_l3s.sea_surface_temperature, np.nan)
ds_l3s['sea_surface_temperature'] = ds_l3s['sea_surface_temperature'] - ds_l3s['sses_bias']
ds_l3s['sea_surface_temperature'] = xr.where(ds_l3s['sea_surface_temperature'] < 250, np.nan, ds_l3s['sea_surface_temperature'])
ds_l3s['sea_surface_temperature'] = xr.where(ds_l3s['sea_surface_temperature'] > 350, np.nan, ds_l3s['sea_surface_temperature'])

print('copy ds_l3s.sst ...')
da_sst_copy = ds_l3s['sea_surface_temperature'].copy(deep=True)
da_sst_num  = xr.where(da_sst_copy.isnull(), 0, 1)
da_sst_copy = da_sst_copy.fillna(0)  # has to assign, fillna() NOT in-place function!

da_sst_avg_day = da_sst_copy[0, :, :] + da_sst_copy[1, :, :]
da_sst_num_day = da_sst_num[0, :, :] + da_sst_num[1, :, :]
da_sst_avg_day.values = np.divide(da_sst_avg_day.values, da_sst_num_day.values, where=(da_sst_num_day>0.9))
da_sst_avg_day = xr.where(da_sst_num_day==0, np.nan, da_sst_avg_day)

# da_sst_avg_day = da_sst_copy[0, :, :] + da_sst_copy[2, :, :]
# da_sst_num_day = da_sst_num[0, :, :] + da_sst_num[2, :, :]
# da_sst_avg_day = np.divide(da_sst_avg_day.values, da_sst_num_day.values, where=(da_sst_num_day>0.9))

da_sst_avg_day = xr.where(da_sst_avg_day<250, np.nan, da_sst_avg_day)
da_sst_avg_day = xr.where(da_sst_avg_day>350, np.nan, da_sst_avg_day)
# da_sst_avg_day = xr.where(da_sst_avg_day<200, np.nan, da_sst_avg_day)
da_sst_avg_day -= 273.15
# da_sst_avg_day -= 273.15

ds_l3s['sea_surface_temperature'] = ds_l3s['sea_surface_temperature'] - 273.15  # Kelvin to Celsius

print('reading oi data ...')
ds_oi = xr.open_dataset(filename_or_obj=DIR_L3S+'/l3s2oi/test/20150701-STAR-L3S_GHRSST-SSTsubskin-LEO_AM_N-ACSPO_V2.80-v02.0-fv01.0.nc'  \
    , mask_and_scale=True)
# print(ds_oi)
ds_oi['sst_avg_daytime'] -= 273.15

# plot
print('start plotting ...')
fig = plt.figure(figsize=(12, 12))
fig.suptitle('ACSPO(0.02d) binned to OI(0.25d) 2015-07-01', fontsize=18, y=0.73)  # 0.95
grid = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[0.55, 0.45], hspace=-0.25, wspace=0.125) # hspace=-0.05

proj = ccrs.PlateCarree()
cmap = gvcmaps.BlueYellowRed

title = ['ACSPO AM day', 'ACSPO AM+PM day', 'ACSPO PM day', 'OI-grid AM+PM day']

ax = []
for i in range(4):
    ax.append(fig.add_subplot(grid[i], projection=proj))
    ax[i].set_global()
    ax[i].coastlines(linewidth=0.5)
    ax[i].set_xlabel('', fontsize=10)
    ax[i].set_ylabel('', fontsize=10)
    ax[i].set_title(label=title[i], loc='center', fontsize=10, y=1.0, pad=6.0)
    ax[i].set_title(label='SST', loc='left', fontsize=10, y=1.0, pad=6.0)
    ax[i].set_title(label='Celsius', loc='right', fontsize=10, y=1.0, pad=6.0)

  # gvutil.set_axes_limits_and_ticks(ax=ax[i], xticks=np.linspace(0, 360, 13), yticks=np.linspace(-90, 90, 7))
    gvutil.set_axes_limits_and_ticks(ax=ax[i], xlim=(-180, 180), ylim=(-90, 90)  \
        , xticks=np.linspace(-120, 120, 5), yticks=np.linspace(-60, 60, 5))
    gvutil.add_lat_lon_ticklabels(ax[i])
    ax[i].xaxis.set_major_formatter(LongitudeFormatter(degree_symbol=''))
    ax[i].yaxis.set_major_formatter(LatitudeFormatter(degree_symbol=''))

# plot data
ct = [0]*4
ct[0] = ds_l3s['sea_surface_temperature'][0, :, :].plot.contourf(ax=ax[0], vmin=0.0, vmax=32, levels=9, cmap=cmap, add_colorbar=False, transform=proj, add_labels=False)
ct[2] = ds_l3s['sea_surface_temperature'][1, :, :].plot.contourf(ax=ax[2], vmin=0.0, vmax=32, levels=9, cmap=cmap, add_colorbar=False, transform=proj, add_labels=False)
ct[1] = da_sst_avg_day.plot.contourf(ax=ax[1], vmin=0.0, vmax=32, levels=9, cmap=cmap, add_colorbar=False, transform=proj, add_labels=False)
ct[3] = ds_oi['sst_avg_daytime'].plot.contourf(ax=ax[3], vmin=0.0, vmax=32, levels=9, cmap=cmap, add_colorbar=False, transform=proj, add_labels=False)

# cbar = plt.colorbar(ct[3], ax=ax, orientation='horizontal', shrink=0.5, pad=0.08, extend='both', extendrect=False, drawedges=False)
cbar = plt.colorbar(ct[3], ax=ax, orientation='horizontal', shrink=0.5, pad=0.05, extend='both', extendrect=False, extendfrac='auto', drawedges=False)
cbar.ax.tick_params(labelsize=8)

# plt.show()
plt.savefig('./figure_20150701_day_01.png')
