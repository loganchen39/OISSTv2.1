'''
Description: Convert bedmacmask in 500m in NetCDF to binary format with signed byte.
Author: Ligang Chen
Date created: 05/16/2022
Date last modified: 05/16/2022
'''

import numpy as np
import xarray as xr

import os
import calendar
import datetime
import glob


DIR_NS = '/glade/scratch/lgchen/data/SeaIceConcentration_NSIDC/mask_fromJim/fromDT'

# np.set_printoptions(threshold=np.inf) # To print all numpy array elements.
check_bin = True 

fn_mask_oi = 'mask_oi.nc'
ds_mask_oi = xr.open_dataset(filename_or_obj=DIR_NS+'/'+fn_mask_oi, mask_and_scale=False, decode_times=True)

fn_mask_oi_bin = 'mask_oi.bin'
ds_mask_oi['mask'].values.tofile(DIR_NS+'/'+fn_mask_oi_bin)


if check_bin:
    QUART_DEG_LON_DIM = 1440
    QUART_DEG_LAT_DIM = 720
    LON_START         = 0.125
    LAT_START         = -89.875
    GRID_SIZE         = 0.25

    lon = np.arange(start=LON_START, stop=360., step=GRID_SIZE, dtype='f')
    lat = np.arange(start=LAT_START, stop=90. , step=GRID_SIZE, dtype='f')

    fn_bin = DIR_NS + '/' + 'mask_oi.bin'
    f = open(fn_bin, 'rb')
    data = np.fromfile(f, dtype='<i1', count=QUART_DEG_LON_DIM*QUART_DEG_LAT_DIM)
    mask = np.reshape(data, (QUART_DEG_LAT_DIM, QUART_DEG_LON_DIM), order='C')

    da_mask = xr.DataArray(mask, name='mask', dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon}  \
        , attrs={'_FillValue':100})
    da_mask.to_dataset(name='mask').to_netcdf('./mask_oi_fromBin.nc')


