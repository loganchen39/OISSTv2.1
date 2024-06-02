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
grid = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[0.55, 0.45], hspace=-0.08, wspace=0.125)

proj = ccrs.PlateCarree()
cmap = gvcmaps.BlueYellowRed

year = [1981, 1991, 2001, 2011, 2021]
dofy = [1, 91, 182, 274]
title_dofy = ['Jan 1', 'Apr 1', 'Jul 1', 'Oct 1']

for i_yr in range(len(year)):
    str_year = str(year[i_yr])
    print('Processing ' + str_year + ' ...')
    fig.suptitle('NSIDC SIC to OI-grid ' + str_year, fontsize=18, y=0.80)  # y initially 0.92

    fn_ns2oi = str_year + '_sic_nsidc2oi.nc'
    ds_ns2oi = xr.open_dataset(filename_or_obj=DIR_NS+'/NSIDC2OI/interp/'+fn_ns2oi, mask_and_scale=True)


    ax = []  # 1st: just setup the axes
    for i_dy in range(len(dofy)):
        ax.append(fig.add_subplot(grid[i_dy], projection=proj))
        ax[i_dy].set_global()
        ax[i_dy].coastlines(linewidth=0.5)
        ax[i_dy].set_xlabel('', fontsize=10)
        ax[i_dy].set_ylabel('', fontsize=10)
        ax[i_dy].set_title(label=title_dofy[i_dy], loc='center', fontsize=10, y=1.0, pad=6.0)
        ax[i_dy].set_title(label='SIC', loc='left', fontsize=10, y=1.0, pad=6.0)
        ax[i_dy].set_title(label='', loc='right', fontsize=10, y=1.0, pad=6.0)
    
      # even if ds_ns2oi's lon is 0~360, you should still set as follows.
        gvutil.set_axes_limits_and_ticks(ax=ax[i_dy], xlim=(-180, 180), ylim=(-90, 90)  \
            , xticks=np.linspace(-120, 120, 5), yticks=np.linspace(-60, 60, 5))
        gvutil.add_lat_lon_ticklabels(ax[i_dy])
        ax[i_dy].xaxis.set_major_formatter(LongitudeFormatter(degree_symbol=''))
        ax[i_dy].yaxis.set_major_formatter(LatitudeFormatter(degree_symbol=''))
    
    # 2nd: plot data
    ct = [0]*4
    for i_dy in range(len(dofy)):
        ct[i_dy] = ds_ns2oi['cdr_seaice_conc'][dofy[i_dy], :, :].plot.contourf(ax=ax[i_dy], vmin=0.9, vmax=1.0  \
            , levels=11, cmap=cmap, add_colorbar=False, transform=proj, add_labels=False)
    
    cbar = plt.colorbar(ct[3], ax=ax, orientation='horizontal', shrink=0.5, pad=0.05, extend='both'  \
        , extendrect=False, extendfrac='auto', drawedges=False) # shrink=1.0
    cbar.ax.tick_params(labelsize=10)
    
  # plt.show()
    fn_fig = 'NSIDC2OI_byCDOInterp_' + str(year[i_yr]) + '.png'
    plt.savefig('./' + fn_fig)

  # exit()



