'''
Description: Convert SIC in NetCDF to binary format with unsigned byte first.
Author: Ligang Chen
Date created: 04/04/2022
Date last modified: 04/04/2022
'''

import numpy as np
import xarray as xr
# import pandas as pd

import os
import calendar
import datetime
import glob

# import fort


DIR_NS = '/glade/scratch/lgchen/data/SeaIceConcentration_NSIDC/G02202_V4'

# np.set_printoptions(threshold=np.inf) # To print all numpy array elements.
check_bin = False

START_YEAR = 1978
END_YEAR   = 2021

for year in range(START_YEAR, END_YEAR+1):
    str_year = str(year)
    print('Processing ' + str_year + ' ...')

    fn_oi = str_year + '_sic_nsidc2oi.nc'
    ds_oi = xr.open_dataset(filename_or_obj=DIR_NS+'/NSIDC2OI/interpByCDO/'+fn_oi, mask_and_scale=False, decode_times=True)

    fn_oi_bin = str_year + '_sic_nsidc2oi.bin'
  # np_arr_fort = ds_oi['cdr_seaice_conc'].values.T
  # print(ds_oi['cdr_seaice_conc'].values[0, 679, :])
    ds_oi['cdr_seaice_conc'].values.tofile(DIR_NS+'/NSIDC2OI/interpByCDO/bin/'+fn_oi_bin)


if check_bin:
    QUART_DEG_LON_DIM = 1440
    QUART_DEG_LAT_DIM = 720
    LON_START         = 0.125
    LAT_START         = -89.875
    GRID_SIZE         = 0.25

    day = np.arange(1, 365+1, dtype='i')
    lon = np.arange(start=LON_START, stop=360., step=GRID_SIZE, dtype='f')
    lat = np.arange(start=LAT_START, stop=90. , step=GRID_SIZE, dtype='f')

    fn_bin = DIR_NS + '/NSIDC2OI/interpByCDO/' + '1979_sic_nsidc2oi.bin'
    f = open(fn_bin, 'rb')
    data = np.fromfile(f, dtype='<u1', count=365*QUART_DEG_LON_DIM*QUART_DEG_LAT_DIM)
    sic = np.reshape(data, (365, QUART_DEG_LAT_DIM, QUART_DEG_LON_DIM), order='C')

    da_sic = xr.DataArray(sic, name='sic', dims=('day', 'lat', 'lon'), coords={'day': day, 'lat': lat, 'lon': lon}  \
        , attrs={'_FillValue':255})
    da_sic.to_dataset(name='sic').to_netcdf('./1979_sic.nc')


