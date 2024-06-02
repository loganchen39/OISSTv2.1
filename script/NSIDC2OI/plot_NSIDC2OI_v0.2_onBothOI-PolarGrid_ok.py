'''
Description: python code to plot NSIDC2OI daily SIC.
Author: Ligang Chen
Date created: 03/16/2022
Date last modified: 03/16/2022 
'''

import numpy as np
# import pandas as pd
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


DIR_NS = '/glade/scratch/lgchen/data/SeaIceConcentration_NSIDC/G02202_V4'

fig = plt.figure(figsize=(12, 12))
# grid = fig.add_gridspec(nrows=4, ncols=3, height_ratios=[0.55, 0.45], hspace=-0.08, wspace=0.125)
grid = fig.add_gridspec(nrows=4, ncols=3)

cmap = gvcmaps.BlueYellowRed

proj_pc = ccrs.PlateCarree()
proj_nps = ccrs.NorthPolarStereo()
proj_sps = ccrs.SouthPolarStereo()

dx = dy = 25000
kw_nps = dict(central_latitude=90 , central_longitude=-45, true_scale_latitude=70 )
# kw_nps = dict(central_latitude=90 , central_longitude=0, true_scale_latitude=70 )
x_nps  = np.arange(-3837500, 3750000 , +dx)
y_nps  = np.arange(5837500 , -5350000, -dy)

kw_sps = dict(central_latitude=-90, central_longitude=0  , true_scale_latitude=-70)
x_sps  = np.arange(-3937500, 3950000 , +dx)
y_sps  = np.arange(4337500 , -3950000, -dy)


year = [1991, 2001, 2011, 2021]
# year = [1981]
dofy = [1, 91, 182, 274]
title_dofy = ['Jan 1', 'Apr 1', 'Jul 1', 'Oct 1']

for i_yr in range(len(year)):
    str_year = str(year[i_yr])
    print('Processing ' + str_year + ' ...')
    fig.suptitle('NSIDC SIC to OI-grid ' + str_year, fontsize=16, y=0.92)  # y initially 0.92

    fn_ns2oi = str_year + '_sic_nsidc2oi.nc'
    # OI zero masked
  # ds_ns2oi = xr.open_dataset(filename_or_obj=DIR_NS+'/NSIDC2OI/interpByCDO/'+fn_ns2oi, mask_and_scale=True)
    ds_ns2oi = xr.open_dataset(filename_or_obj=DIR_NS+'/NSIDC2OI/interpByCDO/zeroNotMasked/'+fn_ns2oi, mask_and_scale=True)

    fn_ns_nh = 'seaice_conc_daily_nh_' + str_year + '_v04r00.nc'
    ds_ns_nh = xr.open_dataset(filename_or_obj=DIR_NS+'/north/aggregate/'+fn_ns_nh, mask_and_scale=True)
    ds_ns_nh['cdr_seaice_conc'] = xr.where(ds_ns_nh['cdr_seaice_conc']>1.1, np.nan, ds_ns_nh['cdr_seaice_conc'])

    fn_ns_sh = 'seaice_conc_daily_sh_' + str_year + '_v04r00.nc'
    ds_ns_sh = xr.open_dataset(filename_or_obj=DIR_NS+'/south/aggregate/'+fn_ns_sh, mask_and_scale=True)
    ds_ns_sh['cdr_seaice_conc'] = xr.where(ds_ns_sh['cdr_seaice_conc']>1.1, np.nan, ds_ns_sh['cdr_seaice_conc'])

    ax = []  # 1st: just setup the axes
  # for i_dy in range(len(dofy)):
    for i_pt in range(12):
        print('i_pt = ', i_pt)

        if i_pt % 3 == 0:
            ax.append(fig.add_subplot(grid[i_pt], projection=proj_nps))
            ax[i_pt].set_extent([-180,180,50,90], crs=ccrs.PlateCarree())
            ax[i_pt].gridlines()
            ax[i_pt].set_title(label='NH', loc='right', fontsize=10, y=1.0, pad=6.0)
        elif i_pt % 3 == 1:
            ax.append(fig.add_subplot(grid[i_pt], projection=proj_pc ))
            ax[i_pt].set_global()
          # even if ds_ns2oi's lon is 0~360, you should still set as follows.
            gvutil.set_axes_limits_and_ticks(ax=ax[i_pt], xlim=(-180, 180), ylim=(-90, 90)  \
                , xticks=np.linspace(-120, 120, 5), yticks=np.linspace(-60, 60, 5))
            gvutil.add_lat_lon_ticklabels(ax[i_pt])
            ax[i_pt].xaxis.set_major_formatter(LongitudeFormatter(degree_symbol=''))
            ax[i_pt].yaxis.set_major_formatter(LatitudeFormatter(degree_symbol=''))
            ax[i_pt].set_title(label='', loc='right', fontsize=10, y=1.0, pad=6.0)
        else:
            ax.append(fig.add_subplot(grid[i_pt], projection=proj_sps))
            ax[i_pt].set_extent([-180,180,-90,-50], crs=ccrs.PlateCarree())
            ax[i_pt].gridlines()
            ax[i_pt].set_title(label='SH', loc='right', fontsize=10, y=1.0, pad=6.0)

        ax[i_pt].coastlines(resolution='110m', linewidth=0.5)
        ax[i_pt].set_xlabel('', fontsize=10)
        ax[i_pt].set_ylabel('', fontsize=10)

        ax[i_pt].set_title(label=title_dofy[int(i_pt/3)], loc='center', fontsize=10, y=1.0, pad=6.0)
        ax[i_pt].set_title(label='SIC', loc='left', fontsize=10, y=1.0, pad=6.0)

    
    # 2nd: plot data
    ct = [0]*12
    for i_dy in range(len(dofy)):
        ct[i_dy*3]   = ax[i_dy*3].pcolormesh(x_nps, y_nps, ds_ns_nh['cdr_seaice_conc'][dofy[i_dy], :, :], vmin=0.0  \
            , vmax=1.0, cmap=cmap, transform=ccrs.Stereographic(**kw_nps))
        # if OI zero masked, then vmin=0.9; initial levels=11
        ct[i_dy*3+1] = ds_ns2oi['cdr_seaice_conc'][dofy[i_dy], :, :].plot.contourf(ax=ax[i_dy*3+1], vmin=0.0, vmax=1.0  \
            , levels=6, cmap=cmap, add_colorbar=False, transform=ccrs.PlateCarree(), add_labels=False)
        ct[i_dy*3+2] = ax[i_dy*3+2].pcolormesh(x_sps, y_sps,ds_ns_sh['cdr_seaice_conc'][dofy[i_dy], :, :], cmap=cmap  \
            , transform=ccrs.Stereographic(**kw_sps))

    cbar_np = plt.colorbar(ct[9], ax=[ax[3*i] for i in range(4)], orientation='horizontal', shrink=0.8, aspect=16, pad=0.05, extend='both'  \
        , extendrect=False, extendfrac='auto', drawedges=False) # shrink=0.5
    cbar_np.ax.tick_params(labelsize=10)
 
    cbar_oi = plt.colorbar(ct[10], ax=[ax[3*i+1] for i in range(4)], orientation='horizontal', shrink=1.0, pad=0.05, extend='both'  \
        , extendrect=False, extendfrac='auto', drawedges=False) # shrink=1.0
    cbar_oi.ax.tick_params(labelsize=10) 

    cbar_sp = plt.colorbar(ct[11], ax=[ax[3*i+2] for i in range(4)], orientation='horizontal', shrink=0.8, aspect=16, pad=0.05, extend='both'  \
        , extendrect=False, extendfrac='auto', drawedges=False) # shrink=1.0
    cbar_sp.ax.tick_params(labelsize=10)

  # plt.show()
    fn_fig = 'NSIDC2OI_byCDOInterp_' + str(year[i_yr]) + '.png'
    plt.savefig('./' + fn_fig)

  # exit()

