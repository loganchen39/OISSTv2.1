
# #!/glade/u/apps/ch/opt/python/3.6.8/gnu/8.3.0/pkg-library/20190627/bin/python3

'''
Description: python code to plot and check OISSTv2.1 satelite sobs
Author: Ligang Chen
Date created: 09/10/2021
Date last modified: 09/10/2021 
'''

import numpy as np
import xarray as xr

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors

# import geocat.datafiles as gdf
from geocat.viz import cmaps as gvcmaps
from geocat.viz import util as gvutil

import os
import calendar


DIR_DATA = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master'
# lst_fn = ['dgrid.sst.20160117_satA_Ligang.nc', 'dsobs.20160117_satA_Ligang.nc', 'ngrid.sst.20160117_satA_Ligang.nc', 'nsobs.20160117_satA_Ligang.nc']
lst_title = ['day grid, total 59595', 'day sobs, total 59595', 'night grid, total 78311', 'night sobs, total 78311']  # for 17th

# for plotting
# ds = xr.open_mfdataset(paths=DIR_DATA + '/tmp_check_sat_sobs/*.nc', concat_dim='time', compat='no_conflicts', data_vars='all', coords='minimal', combine='nested')
ds = xr.open_mfdataset(paths=DIR_DATA + '/tmp_check_sat_sobs/*.nc', concat_dim='t', combine='nested')
da_sst = ds['sst']
print(da_sst.shape)

fig  = plt.figure(figsize=(12, 12))
grid = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[0.55, 0.45], hspace=-0.05, wspace=0.125)
proj = ccrs.PlateCarree()

divnorm = colors.TwoSlopeNorm(vmin=-15, vcenter=0, vmax=40)
cmap = gvcmaps.BlueRed
levels = np.arange(0, 36, 6)  # Specify levels for contours

fig.suptitle('OISSTv2.1 satellite sobs check 01-17-2016', fontsize=18, y=1.20)

ax = []
ct = []
for i in range(4):
    ax.append(fig.add_subplot(grid[i], projection=proj))
    ax[i].coastlines(linewidth=0.5)
    ax[i].set_title(label=lst_title[i], loc='center', fontsize=10, y=1.0, pad=6.0)
    ax[i].set_title(label='SST', loc='left', fontsize=10, y=1.0, pad=6.0)
    ax[i].set_title(label='Celsius', loc='right', fontsize=10, y=1.0, pad=6.0)
  # gvutil.set_axes_limits_and_ticks(ax=ax[i], xlim=(0   , 360), ylim=(-90, 90), xticks=np.arange(60  , 301, 60), yticks=np.arange(-60, 61, 30)) # failed
    gvutil.set_axes_limits_and_ticks(ax=ax[i], xlim=(-180, 180), ylim=(-90, 90), xticks=np.arange(-120, 121, 60), yticks=np.arange(-60, 61, 30))
    gvutil.add_lat_lon_ticklabels(ax[i])
    ax[i].yaxis.set_major_formatter(LatitudeFormatter(degree_symbol=''))
    ax[i].xaxis.set_major_formatter(LongitudeFormatter(degree_symbol=''))
    ct_tmp = ax[i].contourf(da_sst['lon'], da_sst['lat'], da_sst[i, :, :].data, cmap=cmap, norm=divnorm, levels=levels, extend='both')
    ct.append(ct_tmp)

cbar = fig.colorbar(ct[3], ax=ax, ticks=np.linspace(0, 36, 7), orientation='horizontal', drawedges=True, 
    extendrect=False, extendfrac='auto', shrink=0.6, aspect=20,  pad=0.05)
cbar.ax.tick_params(labelsize=8)

plt.show()
