'''
Description: Convert eot bias corrected satellite sst (e.g. ACSPO-L3C) from 1D binary to 2D netCDF. 
Author: Ligang Chen
Date created: 09/12/2022
Date last modified: 09/12/2022 
'''

import numpy as np
import xarray as xr

import os
import datetime
import glob
import struct

import sys
sys.path.append('/glade/u/home/lgchen/lib/fortran/f2py')

import fort  # using F2PY call Fortran subroutine


DIR_BIN1D = '/glade/scratch/lgchen/data/tmp/input_link/noaa09'
DIR_NC2D  = '/glade/scratch/lgchen/data/tmp/ACSPO_L3C_sobsc_bin1d2nc2d/noaa09'

DIR_OI = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master'
FN_OI_MASK = DIR_OI + '/bin2netCDF/quarter-mask-extend_Ligang.nc'
ds_oi_mask = xr.open_dataset(filename_or_obj=FN_OI_MASK)
ds_oi_mask['landmask'] = ds_oi_mask.landmask.astype(np.int32)
oi_mask_fort = ds_oi_mask.landmask.values.T  # instead this transpose works.
lat_oi = ds_oi_mask.coords['lat']  # 720: -89.875, -89.625, -89.375, ...,  89.375,  89.625,  89.875
lon_oi = ds_oi_mask.coords['lon']  # 1440: 0.125 0.375 0.625 ... 359.625 359.875

# To print all numpy array elements.
# np.set_printoptions(threshold=np.inf)

# 1 record/file, could be either daytime or nighttime;
sst_sobsc = -9999*np.ones((1440, 720), dtype=np.float32, order='F')   

day_nit = ('d', 'n')
jday_19850216 = datetime.date(1985, 2 , 16)
jday_19881102 = datetime.date(1988, 11, 2 )

jday = jday_19850216
while jday <= jday_19881102:
    str_date = jday.strftime('%Y%m%d')
    print('\n\n current date: ', str_date)

    for (id_fn, dn) in enumerate(day_nit, start=0):  # for day and nit
        fn = DIR_BIN1D+'/'+dn+'sobsc.'+str_date
        if not os.path.exists(fn):
            print('file not exist: ' + fn)
            continue

      # print(str(id_fn).zfill(5)+', current file: ', fn)
        sst_sobsc.fill(-9999)  # In-place operation

      # print('Start processing SST, current time: ', datetime.datetime.now().strftime('%H:%M:%S'))
        sst_sobsc = fort.convert_sobsc_1dto2d(fn, oi_mask_fort)
      # print('End processing SST, current time: ', datetime.datetime.now().strftime('%H:%M:%S'))

        da_sst = xr.DataArray(data=np.float32(sst_sobsc.T), dims=['lat', 'lon'], coords={'lat': lat_oi, 'lon': lon_oi}, name='sst'  \
            , attrs={'long_name':'bias-corrected satellite sst', 'units':'degC', '_FillValue':-9999}) 

        # to netCDF
        fn_nc = dn+'sobsc.'+str_date+'.nc'
        da_sst.to_netcdf(DIR_NC2D+'/'+fn_nc, encoding={'lat': {'_FillValue': None}, 'lon': {'_FillValue': None}})

    jday += datetime.timedelta(days=1)
