'''
Description: Compute binned/average result on OI 0.25-degree grid from NSIDC Sea Ice Concentration
Author: Ligang Chen
Date created: 03/14/2022
Date last modified: 03/14/2022
'''

import numpy as np
import xarray as xr
# import pandas as pd

import calendar
import datetime
import glob

import fort


DIR_NS = '/glade/scratch/lgchen/data/SeaIceConcentration_NSIDC/G02202_V4'

fn_anci_nh = 'G02202-cdr-ancillary-nh.nc'
fn_anci_sh = 'G02202-cdr-ancillary-sh.nc'
ds_anci_nh = xr.open_dataset(filename_or_obj=DIR_NS+'/ancillary/'+fn_anci_nh, mask_and_scale=True, decode_times=True)
ds_anci_sh = xr.open_dataset(filename_or_obj=DIR_NS+'/ancillary/'+fn_anci_sh, mask_and_scale=True, decode_times=True)

x_ns_nh = 304
y_ns_nh = 448
lat_ns_nh = ds_anci_nh.latitude  # (y, x)=(448, 304)
lon_ns_nh = ds_anci_nh.longitude # (y, x)=(448, 304)
idx_lat_nh_ns2oi = np.zeros((304, 448), dtype=np.int32, order='F')  # now (x, y)
idx_lon_nh_ns2oi = np.zeros((304, 448), dtype=np.int32, order='F')
for i_lat_ns in range(y_ns_nh):
    for i_lon_ns in range(x_ns_nh):
        lat_ns = lat_ns_nh[i_lat_ns, i_lon_ns]
        idx_lat_nh_ns2oi[i_lon_ns, i_lat_ns] = np.around(4*(lat_ns+89.875))
        if (idx_lat_nh_ns2oi[i_lon_ns, i_lat_ns] < 0):
            idx_lat_nh_ns2oi[i_lon_ns, i_lat_ns] = 0
        elif idx_lat_nh_ns2oi[i_lon_ns, i_lat_ns] > 719:
            idx_lat_nh_ns2oi[i_lon_ns, i_lat_ns] = 719

        lon_ns = lon_ns_nh[i_lat_ns, i_lon_ns]
        if lon_ns < 0:
            lon_ns += 360
        idx_lon_nh_ns2oi[i_lon_ns, i_lat_ns] =np.around(4*(lon_ns-0.125 ))
        if idx_lon_nh_ns2oi[i_lon_ns, i_lat_ns] < 0:
            idx_lon_nh_ns2oi[i_lon_ns, i_lat_ns] = 1439
        elif idx_lon_nh_ns2oi[i_lon_ns, i_lat_ns] > 1439:
            idx_lon_nh_ns2oi[i_lon_ns, i_lat_ns] = 0

idx_lat_nh_ns2oi += 1
idx_lon_nh_ns2oi += 1
print('idx_lat_nh_ns2oi.min() = ', idx_lat_nh_ns2oi.min())
print('idx_lat_nh_ns2oi.max() = ', idx_lat_nh_ns2oi.max())
print('idx_lon_nh_ns2oi.min() = ', idx_lon_nh_ns2oi.min())
print('idx_lon_nh_ns2oi.max() = ', idx_lon_nh_ns2oi.max())


x_ns_sh = 316
y_ns_sh = 332
lat_ns_sh = ds_anci_sh.latitude  # (y, x)=(332, 316)
lon_ns_sh = ds_anci_sh.longitude 
idx_lat_sh_ns2oi = np.zeros((316, 332), dtype=np.int32, order='F')  # now (x, y)
idx_lon_sh_ns2oi = np.zeros((316, 332), dtype=np.int32, order='F')
for i_lat_ns in range(y_ns_sh):
    for i_lon_ns in range(x_ns_sh):
        lat_ns = lat_ns_sh[i_lat_ns, i_lon_ns]
        idx_lat_sh_ns2oi[i_lon_ns, i_lat_ns] =np.around(4*(lat_ns+89.875))
        if idx_lat_sh_ns2oi[i_lon_ns, i_lat_ns] < 0:
            idx_lat_sh_ns2oi[i_lon_ns, i_lat_ns] = 0
        elif idx_lat_sh_ns2oi[i_lon_ns, i_lat_ns] > 719:
            idx_lat_sh_ns2oi[i_lon_ns, i_lat_ns] = 719

        lon_ns = lon_ns_sh[i_lat_ns, i_lon_ns]
        if lon_ns < 0:
            lon_ns += 360
        idx_lon_sh_ns2oi[i_lon_ns, i_lat_ns] =np.around(4*(lon_ns-0.125 ))
        if idx_lon_sh_ns2oi[i_lon_ns, i_lat_ns] < 0:
            idx_lon_sh_ns2oi[i_lon_ns, i_lat_ns] = 1439
        elif idx_lon_sh_ns2oi[i_lon_ns, i_lat_ns] > 1439:
            idx_lon_sh_ns2oi[i_lon_ns, i_lat_ns] = 0

idx_lat_sh_ns2oi += 1
idx_lon_sh_ns2oi += 1
print('idx_lat_sh_ns2oi.min() = ', idx_lat_sh_ns2oi.min())
print('idx_lat_sh_ns2oi.max() = ', idx_lat_sh_ns2oi.max())
print('idx_lon_sh_ns2oi.min() = ', idx_lon_sh_ns2oi.min())
print('idx_lon_sh_ns2oi.max() = ', idx_lon_sh_ns2oi.max())


DIR_OI = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master'
FN_OI_MASK = DIR_OI + '/bin2netCDF/quarter-mask-extend_Ligang.nc'
ds_oi_mask = xr.open_dataset(filename_or_obj=FN_OI_MASK)
ds_oi_mask['landmask'] = ds_oi_mask.landmask.astype(np.int32)
landmask_oi_fort = ds_oi_mask.landmask.values.T
# oi_mask_fort = ds_oi_mask.landmask.values.T  # instead this transpose seems to work.
lat_oi = ds_oi_mask.coords['lat']  # 720: -89.875, -89.625, -89.375, ...,  89.375,  89.625,  89.875
lon_oi = ds_oi_mask.coords['lon']  # 1440: 0.125 0.375 0.625 ... 359.625 359.875

np.set_printoptions(threshold=np.inf) # To print all numpy array elements.

# 1 year per daily file, no need and can not differentiate daytime or nighttime;
sic_avg_ns2oi_1year = np.zeros((1440, 720, 366), dtype=np.float32, order='F')
sic_num_ns2oi_1year = np.zeros((1440, 720, 366), dtype=np.int32  , order='F') 

sic_avg_ns2oi_1day = np.zeros((1440, 720), dtype=np.float32, order='F')
sic_num_ns2oi_1day = np.zeros((1440, 720), dtype=np.int32  , order='F')

dropped_vars = ['melt_onset_day_cdr_seaice_conc', 'nsidc_bt_seaice_conc', 'nsidc_nt_seaice_conc'  \
    , 'projection', 'spatial_interpolation_flag', 'temporal_interpolation_flag', 'xgrid', 'ygrid']
# kept_vars = ['cdr_seaice_conc', 'qa_of_cdr_seaice_conc', 'stdev_of_cdr_seaice_conc', 'time', 'latitude', 'longitude']


START_YEAR = 1979
END_YEAR   = 2014
for year in range(START_YEAR, END_YEAR+1):
    str_year = str(year)
    print('\n\n  Processing ' + str_year + ' ...')

    fn_ns_nh = 'seaice_conc_daily_nh_' + str_year + '_v04r00.nc'
    fn_ns_sh = 'seaice_conc_daily_sh_' + str_year + '_v04r00.nc'
    ds_ns_nh = xr.open_dataset(filename_or_obj=DIR_NS+'/north/aggregate/'+fn_ns_nh, mask_and_scale=True, decode_times=True  \
        , drop_variables=dropped_vars)
    ds_ns_sh = xr.open_dataset(filename_or_obj=DIR_NS+'/south/aggregate/'+fn_ns_sh, mask_and_scale=True, decode_times=True  \
        , drop_variables=dropped_vars)
    tdim = ds_ns_nh.dims['tdim']
    time = ds_ns_nh.time

    ds_ns_nh['cdr_seaice_conc'] = ds_ns_nh['cdr_seaice_conc'].fillna(2.55)
    ds_ns_sh['cdr_seaice_conc'] = ds_ns_sh['cdr_seaice_conc'].fillna(2.55)
    
    # test if the time dims are ok
    dofy = 365
    if calendar.isleap(year):
        dofy = 366

    if year != 1978: 
        if tdim != dofy or tdim != ds_ns_sh.dims['tdim']:
            print('Error: tdim=', tdim, ', dofy=', dofy, ', ds_ns_sh.dims.tdim=', ds_ns_sh.dims.tdim)
            exit()

    for itime in range(tdim):  # tdim
        print('Processing itime = ', itime)
        sic_avg_ns2oi_1day.fill(0)  # can NOT use sic_avg_ns2oi_1day = 0 ! 
        sic_num_ns2oi_1day.fill(0)

        sic_ns_fort = ds_ns_nh['cdr_seaice_conc'].values[itime, :, :].T
        fort.sic_nsidc2oi_nh_1rec(sic_ns_fort, sic_avg_ns2oi_1day, sic_num_ns2oi_1day, landmask_oi_fort, idx_lat_nh_ns2oi, idx_lon_nh_ns2oi)   
        sic_ns_fort = ds_ns_sh['cdr_seaice_conc'].values[itime, :, :].T
        fort.sic_nsidc2oi_sh_1rec(sic_ns_fort, sic_avg_ns2oi_1day, sic_num_ns2oi_1day, landmask_oi_fort, idx_lat_sh_ns2oi, idx_lon_sh_ns2oi) 

        # if no 'dtype=np.float32', the returned ndarray from np.divide will be np.float64!
        sic_avg_ns2oi_1day = np.divide(sic_avg_ns2oi_1day, sic_num_ns2oi_1day, where=(sic_num_ns2oi_1day > 0.9), dtype=np.float32)
        sic_avg_ns2oi_1day = xr.where(sic_num_ns2oi_1day==0, -999., sic_avg_ns2oi_1day)
       
        sic_avg_ns2oi_1year[:, :, itime] = sic_avg_ns2oi_1day
        sic_num_ns2oi_1year[:, :, itime] = sic_num_ns2oi_1day

      # print(sic_avg_ns2oi_1day.shape)
      # print(sic_avg_ns2oi_1day.dtype)
      # print(sic_avg_ns2oi_1day.size )


    da_sic_avg_ns2oi_1year = xr.DataArray(data=np.float32(sic_avg_ns2oi_1year[:, :, 0:tdim].T)  \
        , dims=['tdim', 'lat', 'lon'], coords={'tdim': time, 'lat': lat_oi, 'lon': lon_oi}, name='sic'  \
        , attrs={'units':'1', '_FillValue':-999})
    da_sic_num_ns2oi_1year = xr.DataArray(data=np.int8   (sic_num_ns2oi_1year[:, :, 0:tdim].T)  \
        , dims=['tdim', 'lat', 'lon'], coords={'tdim': time, 'lat': lat_oi, 'lon': lon_oi}, name='sic_num'  \
        , attrs=dict(_FillValue=0))

    ds_oi = xr.merge([da_sic_avg_ns2oi_1year, da_sic_num_ns2oi_1year])
    ds_oi['tdim'].encoding['_FillValue'] = None
    ds_oi.lat.encoding['_FillValue'] = None
    ds_oi.lon.encoding['_FillValue'] = None
    ds_oi.attrs = {}


    fn_oi = str_year+'-NSIDC2OI.nc'
    ds_oi.to_netcdf(DIR_NS+'/NSIDC2OI/'+fn_oi)

  # exit()
