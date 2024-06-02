#!/glade/u/apps/ch/opt/python/3.7.9/gnu/9.1.0/pkg-library/20201220/bin/python3

'''
Description: python code to compute ACSPO_L3U daily (day/night) SST from 10-minute time resolution.
Author: Ligang Chen
Date created: 12/14/2021
Date last modified: 12/14/2021 
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


DIR_AC = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/ACSPO_L3S_LEO'
FN_AC = DIR_AC+"/sst_stat.nc"
ds = xr.open_dataset(filename_or_obj=FN_AC)

# ds['sst_stat'].transpose('day', 'stat', 'lat', 'lon')
# ds['sst_stat'].to_netcdf('./sst_stat_reordered.nc')

print(ds)
ds = ds.transpose("day", "stat", "lat", "lon")
print(ds)
ds.to_netcdf('./sst_stat_reordered.nc')
