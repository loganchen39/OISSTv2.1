#!/glade/u/apps/ch/opt/python/3.7.9/gnu/9.1.0/pkg-library/20201220/bin/python3

'''
Description: python code to plot iQuam2OI daily SST.
Author: Ligang Chen
Date created: 12/06/2021
Date last modified: 12/06/2021 
'''

import numpy as np
import pandas as pd
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


DIR_iQuam = '/glade/scratch/lgchen/data/iQuam_InSitu_SST_obs'

period = pd.period_range(start='2020-01', end='2020-01', freq='M')
prTups = tuple([(pr.year, pr.month) for pr in period])
print(prTups)
for (year, month) in prTups:
    str_ym = str(year) + str(month).zfill(2)
    print('processing ' + str_ym + '...')
  # dofm = calendar.monthrange(year, month)[1]
  # day = np.arange(start=1, stop=dofm+1, step=1, dtype=np.int8)

    fn_iq2oi = str_ym + '-iQuam2OI.nc'
    ds_iq2oi = xr.open_dataset(filename_or_obj=DIR_iQuam+'/iQuam2OI/'+fn_iq2oi, mask_and_scale=True)
    ds_iq2oi['sst'] = ds_iq2oi['sst'] - 273.15
    ds_iq2oi['sst_daytime'] = ds_iq2oi['sst_daytime'] - 273.15
    ds_iq2oi['sst_nighttime'] = ds_iq2oi['sst_nighttime'] - 273.15

    # plot
    print('start plotting ...')
    fig = plt.figure(figsize=(12, 12))
    fig.suptitle('iQuam in situ sst to OI(0.25d) 2020-01', fontsize=18, y=0.92)  # y initially 0.73
  # grid = fig.add_gridspec(nrows=3, ncols=3, height_ratios=[0.55, 0.45], hspace=-0.25, wspace=0.125)
  # grid = fig.add_gridspec(nrows=3, ncols=2, hspace=-0.25, wspace=0.125)
    grid = fig.add_gridspec(nrows=3, ncols=2)
    
    proj = ccrs.PlateCarree()
    cmap = gvcmaps.BlueYellowRed
  # cmap = gvcmaps.gui_default
  # cmap = gvcmaps.BlWhRe
    
  # title = ['ACSPO AM night', 'ACSPO AM+PM night', 'ACSPO PM night', 'OI-grid AM+PM night']
    title = ['sst', 'sst_num', 'sst_daytime', 'sst_num_daytime', 'sst_nighttime', 'sst_num_nighttime']
    
    ax = []  # 1st: just setup the axes
    for i in range(6):
        ax.append(fig.add_subplot(grid[i], projection=proj))
        ax[i].set_global()
        ax[i].coastlines(linewidth=0.5)
        ax[i].set_xlabel('', fontsize=10)
        ax[i].set_ylabel('', fontsize=10)
        ax[i].set_title(label=title[i], loc='center', fontsize=10, y=1.0, pad=6.0)
        if i%2==0:
            ax[i].set_title(label='SST', loc='left', fontsize=10, y=1.0, pad=6.0)
            ax[i].set_title(label='Celsius', loc='right', fontsize=10, y=1.0, pad=6.0)
        else:
            ax[i].set_title(label='Number', loc='left', fontsize=10, y=1.0, pad=6.0)
            ax[i].set_title(label=''      , loc='right', fontsize=10, y=1.0, pad=6.0)
    
      # even if ds_iq2oi's lon is 0~360, you should still set as follows.
        gvutil.set_axes_limits_and_ticks(ax=ax[i], xlim=(-180, 180), ylim=(-90, 90)  \
            , xticks=np.linspace(-120, 120, 5), yticks=np.linspace(-60, 60, 5))
        gvutil.add_lat_lon_ticklabels(ax[i])
        ax[i].xaxis.set_major_formatter(LongitudeFormatter(degree_symbol=''))
        ax[i].yaxis.set_major_formatter(LatitudeFormatter(degree_symbol=''))
    
    # 2nd: plot data
    ct = [0]*6
    ct[0] = ds_iq2oi['sst'][0, :, :].plot.contourf(ax=ax[0], vmin=0.0, vmax=32  \
        , levels=9, cmap=cmap, add_colorbar=False, transform=proj, add_labels=False)
    ct[2] = ds_iq2oi['sst_daytime'][0, :, :].plot.contourf(ax=ax[2], vmin=0.0, vmax=32  \
        , levels=9, cmap=cmap, add_colorbar=False, transform=proj, add_labels=False)
    ct[4] = ds_iq2oi['sst_nighttime'][0, :, :].plot.contourf(ax=ax[4], vmin=0.0, vmax=32  \
        , levels=9, cmap=cmap, add_colorbar=False, transform=proj, add_labels=False)

    ct[1] = ds_iq2oi['sst_num'][0, :, :].plot.contourf(ax=ax[1], vmin=1, vmax=5, levels=5, cmap=cmap  \
        , add_colorbar=False, transform=proj, add_labels=False)
    ct[3] = ds_iq2oi['sst_num_daytime'][0, :, :].plot.contourf(ax=ax[3], vmin=1, vmax=5, levels=5  \
        , cmap=cmap, add_colorbar=False, transform=proj, add_labels=False)
    ct[5] = ds_iq2oi['sst_num_nighttime'][0, :, :].plot.contourf(ax=ax[5], vmin=1, vmax=5, levels=5  \
        , cmap=cmap, add_colorbar=False, transform=proj, add_labels=False)   
 
    cbar1 = plt.colorbar(ct[4], ax=[ax[0], ax[2], ax[4]], orientation='horizontal', shrink=1.0, pad=0.05, extend='both'  \
        , extendrect=False, extendfrac='auto', drawedges=False) # shrink=0.5
    cbar1.ax.tick_params(labelsize=8)

    cbar2 = plt.colorbar(ct[5], ax=[ax[1], ax[3], ax[5]], orientation='horizontal', shrink=1.0, pad=0.05, extend='both'  \
        , extendrect=False, extendfrac='auto', drawedges=False)
    cbar2.ax.tick_params(labelsize=8)
    
    # plt.show()
    plt.savefig('./Figure_iQuam2OI_20200101_03.png')
