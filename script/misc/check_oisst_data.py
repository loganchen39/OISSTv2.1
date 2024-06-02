#!/glade/u/apps/ch/opt/python/3.6.8/gnu/8.3.0/pkg-library/20190627/bin/python3

'''
Description: python code to check oisst for Garrett's group as someone found it's not consistency.
Author: Ligang Chen
Date created: 02/16/2022
Date last modified: 02/16/2022 
'''

import numpy as np
import xarray as xr

import cartopy.crs as ccrs
import cartopy.feature as cfeature
# from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors

# import geocat.datafiles as gdf
# from geocat.viz import cmaps as gvcmaps
# from geocat.viz import util as gvutil

import os
import calendar
import datetime


DIR_ROOT = '/glade/scratch/lgchen/data/oisstv2.1_check_tmp'
FN_OF = DIR_ROOT + "/OISSTV2.1_Official/oisst-avhrr-v02r01.19860830.nc"
FN_CO = DIR_ROOT + "/2022-03-04/oisst-avhrr-v02r01.20180830.nc"

ds_of = xr.open_dataset(filename_or_obj=FN_OF, mask_and_scale=True, decode_times=True).isel(time=0)
ds_co = xr.open_dataset(filename_or_obj=FN_CO, mask_and_scale=True, decode_times=True).isel(time=0)

VAR_NM = list(ds_of.coords) + list(ds_of.keys())
for var in VAR_NM:
    da_diff = ds_of[var] - ds_co[var]
    print('var: ' + var + ', max_abs_diff = ' + str(np.abs(da_diff.max().data)))

