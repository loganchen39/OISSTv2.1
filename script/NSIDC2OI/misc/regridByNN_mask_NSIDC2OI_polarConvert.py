'''
Description: Compute regrid result on OI 0.25-degree grid from NSIDC mask 
Author: Ligang Chen
Date created: 04/29/2022
Date last modified: 04/29/2022
'''

import numpy as np
import xarray as xr
# import xy2ll
import convert_polar_lonlat.polarstereo-lonlat-convert-py.polar_convert.polar_convert as polar_convert
# import pandas as pd

import calendar
import datetime
import glob

# import fort


np.set_printoptions(threshold=np.inf) # To print all numpy array elements.

DIR_NS = '/glade/scratch/lgchen/data/SeaIceConcentration_NSIDC/mask_fromJim'
fn_mask_ns = 'bedmacmask.nc'
ds_mask_ns = xr.open_dataset(filename_or_obj=DIR_NS+'/'+fn_mask_ns, mask_and_scale=True, decode_times=True)

[lat, lon] = polar_convert.polar_xy_to_lonlat(0.001*ds_mask_ns.x.data, 0.001*ds_mask_ns.y.values, -1)
print(lat.size)
print(lat)
print(lon.size)
print(lon)


# to_lat, to_lon = [], []
# for (x, y) in zip(ds_mask_ns.x.data, ds_mask_ns.y.values):
#     la, lo = xy2ll.xy2ll(x, y, -1, 0, 71)
#     print('x=', x, ', y=', y, ', la=', la, ', lo=', lo)
#     to_lat.append(la)
#     to_lon.append(lo)
# 
# print(to_lat.size)
# print(to_lat)
# print(to_lon.size)
# print(to_lon)


# fn_anci_nh = 'G02202-cdr-ancillary-nh.nc'
# fn_anci_sh = 'G02202-cdr-ancillary-sh.nc'
# ds_anci_nh = xr.open_dataset(filename_or_obj=DIR_NS+'/ancillary/'+fn_anci_nh, mask_and_scale=True, decode_times=True)
# ds_anci_sh = xr.open_dataset(filename_or_obj=DIR_NS+'/ancillary/'+fn_anci_sh, mask_and_scale=True, decode_times=True)
# 
# 
# DIR_OI = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master'
# FN_OI_MASK = DIR_OI + '/bin2netCDF/quarter-mask-extend_Ligang.nc'
# ds_oi_mask = xr.open_dataset(filename_or_obj=FN_OI_MASK)
# ds_oi_mask['landmask'] = ds_oi_mask.landmask.astype(np.int32)
# landmask_oi_fort = ds_oi_mask.landmask.values.T
# # oi_mask_fort = ds_oi_mask.landmask.values.T  # instead this transpose seems to work.
# lat_oi = ds_oi_mask.coords['lat']  # 720: -89.875, -89.625, -89.375, ...,  89.375,  89.625,  89.875
# lon_oi = ds_oi_mask.coords['lon']  # 1440: 0.125 0.375 0.625 ... 359.625 359.875
# 
# 
# # 1 year per daily file, no need and can not differentiate daytime or nighttime;
# sic_avg_ns2oi_1year = np.zeros((1440, 720, 366), dtype=np.float32, order='F')
# sic_num_ns2oi_1year = np.zeros((1440, 720, 366), dtype=np.int32  , order='F') 
# 
# sic_avg_ns2oi_1day = np.zeros((1440, 720), dtype=np.float32, order='F')
# sic_num_ns2oi_1day = np.zeros((1440, 720), dtype=np.int32  , order='F')



