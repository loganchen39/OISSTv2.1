'''
Description: Compute interpolation result on OI 0.25-degree grid from NSIDC Sea Ice Concentration by CDO
Author: Ligang Chen
Date created: 03/21/2022
Date last modified: 03/21/2022
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

DIR_OI = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master'
FN_OI_MASK = DIR_OI + '/bin2netCDF/quarter-mask-extend_Ligang.nc'
ds_oi_mask = xr.open_dataset(filename_or_obj=FN_OI_MASK)
ds_oi_mask['landmask'] = ds_oi_mask.landmask.astype(np.int32)
landmask_oi_fort = ds_oi_mask.landmask.values.T
# oi_mask_fort = ds_oi_mask.landmask.values.T  # instead this transpose seems to work.
lat_oi = ds_oi_mask.coords['lat']  # 720: -89.875, -89.625, -89.375, ...,  89.375,  89.625,  89.875
lon_oi = ds_oi_mask.coords['lon']  # 1440: 0.125 0.375 0.625 ... 359.625 359.875

# np.set_printoptions(threshold=np.inf) # To print all numpy array elements.


# since combine "-remapbil,oigrid.txt -selname,cdr_seaice_conc" always have Segmentation fault or HDF error
#     now better to seperate them.

START_YEAR = 1978
END_YEAR   = 2021
years = [1981, 1991, 2001, 2011, 2021]

# for year in years:
for year in range(START_YEAR, END_YEAR+1):
    if year in years:
        continue

    str_year = str(year)
    print('Processing ' + str_year + ' ...')
    
    # nh then sh
    fn_ns_nh = 'seaice_conc_daily_nh_' + str_year + '_v04r00.nc'
    fn_sic_nh = str_year + '_sic_nh.nc'
    fn_oi_nh = str_year + '_sic_nsidc2oi_nh.nc'
    str_cmd = 'cdo -select,name=cdr_seaice_conc ' + DIR_NS + '/north/aggregate/' + fn_ns_nh + ' ' + DIR_NS + '/NSIDC2OI/interpByCDO/' + fn_sic_nh
    os.system(str_cmd)
    str_cmd = 'cdo remapbil,oigrid.txt -setgrid,nh_griddes.txt ' + DIR_NS + '/NSIDC2OI/interpByCDO/' + fn_sic_nh + ' ' + DIR_NS + '/NSIDC2OI/interpByCDO/' + fn_oi_nh
    os.system(str_cmd)

    fn_ns_sh = 'seaice_conc_daily_sh_' + str_year + '_v04r00.nc'
    fn_sic_sh = str_year + '_sic_sh.nc'
    fn_oi_sh = str_year + '_sic_nsidc2oi_sh.nc'
    str_cmd = 'cdo -select,name=cdr_seaice_conc ' + DIR_NS + '/south/aggregate/' + fn_ns_sh + ' ' + DIR_NS + '/NSIDC2OI/interpByCDO/' + fn_sic_sh
    os.system(str_cmd)
    str_cmd = 'cdo remapbil,oigrid.txt -setgrid,sh_griddes.txt  ' + DIR_NS + '/NSIDC2OI/interpByCDO/' + fn_sic_sh + ' ' + DIR_NS + '/NSIDC2OI/interpByCDO/' + fn_oi_sh
    os.system(str_cmd)

    # combine to one OI grid
    ds_oi_nh = xr.open_dataset(filename_or_obj=DIR_NS+'/NSIDC2OI/interpByCDO/'+fn_oi_nh, mask_and_scale=False, decode_times=True)
    ds_oi_sh = xr.open_dataset(filename_or_obj=DIR_NS+'/NSIDC2OI/interpByCDO/'+fn_oi_sh, mask_and_scale=False, decode_times=True)
    n_time = ds_oi_nh.dims['time']
    sic_attrs = ds_oi_nh['cdr_seaice_conc'].attrs
    time_attrs = ds_oi_nh['time'].attrs
    lat_attrs = ds_oi_nh['lat'].attrs
    lon_attrs = ds_oi_nh['lon'].attrs

    ds_oi_nh['cdr_seaice_conc'] = xr.where(ds_oi_nh.cdr_seaice_conc > 100, 255, ds_oi_nh.cdr_seaice_conc)
    for i_time in range(n_time): 
        ds_oi_nh['cdr_seaice_conc'][i_time, :, :] = xr.where(ds_oi_mask.landmask == 0, 255, ds_oi_nh.cdr_seaice_conc[i_time, :, :])
    
    ds_oi_sh['cdr_seaice_conc'] = xr.where(ds_oi_sh.cdr_seaice_conc > 100, 255, ds_oi_sh.cdr_seaice_conc)
    for i_time in range(n_time):
        ds_oi_sh['cdr_seaice_conc'][i_time, :, :] = xr.where(ds_oi_mask.landmask == 0, 255, ds_oi_sh.cdr_seaice_conc[i_time, :, :])

    ds_oi_nh['cdr_seaice_conc'] = xr.where(ds_oi_sh['cdr_seaice_conc'] == 0, 0, ds_oi_nh['cdr_seaice_conc'])
    ds_oi_sh['cdr_seaice_conc'] = xr.where(ds_oi_nh['cdr_seaice_conc'] == 0, 0, ds_oi_sh['cdr_seaice_conc'])

    ds_oi_nh['cdr_seaice_conc'] = ds_oi_nh['cdr_seaice_conc'] + ds_oi_sh['cdr_seaice_conc']
    ds_oi_nh['cdr_seaice_conc'] = xr.where(ds_oi_nh['cdr_seaice_conc'] == 0, 255, ds_oi_nh['cdr_seaice_conc']) # mask zero
    ds_oi_nh['cdr_seaice_conc'] = xr.where(ds_oi_nh['cdr_seaice_conc'] > 100, 255, ds_oi_nh['cdr_seaice_conc'])
    ds_oi_nh['cdr_seaice_conc'].attrs = sic_attrs
    ds_oi_nh['cdr_seaice_conc'].attrs['long_name'] = "NOAA/NSIDC Climate Data Record of Passive Microwave Daily Sea Ice Concentration interpolated to OISST grid"
    ds_oi_nh['cdr_seaice_conc'].attrs.pop('ancillary_variables')
    ds_oi_nh['cdr_seaice_conc'].encoding['_FillValue'] = None

    ds_oi_nh['time'].attrs = time_attrs
    ds_oi_nh['time'].encoding['_FillValue'] = None
    ds_oi_nh['lat'].attrs = lat_attrs
    ds_oi_nh['lat'].encoding['_FillValue'] = None
    ds_oi_nh['lon'].attrs = lon_attrs
    ds_oi_nh['lon'].encoding['_FillValue'] = None

    ds_oi_nh.attrs = {}
    
    fn_oi = str_year + '_sic_nsidc2oi.nc'
    ds_oi_nh.to_netcdf(DIR_NS + '/NSIDC2OI/interpByCDO/' + fn_oi)

