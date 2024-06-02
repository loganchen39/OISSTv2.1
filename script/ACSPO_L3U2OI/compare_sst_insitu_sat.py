
# #!/glade/u/apps/ch/opt/python/3.6.8/gnu/8.3.0/pkg-library/20190627/bin/python3

'''
Description: python code to compare iQuam in situ SST with L3U OSPO satellite SST, etc.
Author: Ligang Chen
Date created: 10/05/2021
Date last modified: 10/05/2021 
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


DIR_L3U = '/glade/scratch/lgchen/data/L3U_OSPO_v2.61_SST_from_PO.DAAC'
FN_L3U = DIR_L3U+"/test/20150702112000-OSPO-L3U_GHRSST-SSTsubskin-VIIRS_NPP-ACSPO_V2.61-v02.0-fv01.0.nc"
# FN_L3U = DIR_L3U+"/2015_NetCDF/20150702164000-OSPO-L3U_GHRSST-SSTsubskin-VIIRS_NPP-ACSPO_V2.61-v02.0-fv01.0.nc"

ds = xr.open_dataset(filename_or_obj=FN_L3U, mask_and_scale=True, decode_times=True).isel(time=0)
da_sst = ds['sea_surface_temperature'] - 273.15
str_time = ds['time'].dt.strftime("%Y-%m-%d %H:%M:%S").data
print(str_time)

# for plotting
fig  = plt.figure(figsize=(12, 8))
fig.suptitle('L3U OSPO SST', fontsize=18, y=0.93)

proj = ccrs.PlateCarree()
ax = plt.axes(projection=proj)
ax.set_global()
ax.coastlines(linewidth=0.5)
 
cmap = gvcmaps.BlueYellowRed
cbar_kw = {'orientation':'horizontal', 'shrink':0.5, 'pad':0.08, 'extend':'both', 'extendrect':False, 'drawedges':False, 'label':''}
ct = da_sst.plot.contourf(ax=ax, vmin=0.0, vmax=36, levels=10, cmap=cmap, add_colorbar=True, add_labels=False, cbar_kwargs=cbar_kw)
# ct = da_sst.plot.contourf(ax=ax, vmin=18.0, vmax=28, levels=6, cmap=cmap, add_colorbar=True, extend='both', add_labels=False, cbar_kwargs=cbar_kw)

ax.set_title(label='SST', loc='left', fontsize=10, y=1.0, pad=6.0)
ax.set_title(label='Celsius', loc='right', fontsize=10, y=1.0, pad=6.0)
ax.set_title(label=str_time, loc='center', fontsize=10, y=1.0, pad=6.0)
gvutil.set_axes_limits_and_ticks(ax=ax, xticks=np.linspace(-180, 180, 13), yticks=np.linspace(-90, 90, 7))
gvutil.add_lat_lon_ticklabels(ax)
ax.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol=''))
ax.yaxis.set_major_formatter(LatitudeFormatter(degree_symbol=''))

plt.show()
