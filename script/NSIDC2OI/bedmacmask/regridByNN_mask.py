'''
Description: 
Author: Ligang Chen
Date created: 05/10/2022
Date last modified: 05/10/2022 
'''

import numpy as np
import xarray as xr
# import nsidc0756.xy2ll as xy2ll
import fort


# np.set_printoptions(threshold=np.inf)

DIR_NS = '/homes/lchen2/project/OISSTv2.1_NOAA/ancillary/NSIDC_mask'
ds_mask_ns = xr.open_dataset(filename_or_obj=DIR_NS + '/' + 'bedmacmask.nc')
ds_lonlat_ns = xr.open_dataset(filename_or_obj=DIR_NS + '/' + 'lat_lon.nc')
ds_mask_oi = xr.open_dataset(filename_or_obj=DIR_NS + '/' + 'quarter-mask-extend_Ligang.nc')

lat_ns = ds_lonlat_ns['lat']
lon_ns = ds_lonlat_ns['lon']

lat_oi = ds_mask_oi.coords['lat']
lon_oi = ds_mask_oi.coords['lon']

# mask_oi = np.zeros((1440, 720), dtype=np.byte, order='F')
mask_oi = 100*np.ones((1440, 720), dtype=np.byte, order='F')

mask_oi = fort.remap_mask_nsidc2oi(ds_mask_ns['mask'].data.T, lat_ns.data.T, lon_ns.data.T, lat_oi.data, lon_oi.data)

da_mask_oi = xr.DataArray(data=mask_oi.T, dims=['lat', 'lon'], name='mask', attrs=dict(_FillValue=100))
fn_mask_oi = 'mask_oi.nc'
da_mask_oi.to_netcdf(DIR_NS + '/' + fn_mask_oi)
