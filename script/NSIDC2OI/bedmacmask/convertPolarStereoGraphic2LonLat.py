'''
Description: Convert the polar stereographic projection coordinates (x,y) with 500m resolution of 
    the mask of MEaSUREs BedMachine Antarctica, Version 1, to lon/lat coordinates using their xy2ll
    function.
Author: Ligang Chen
Date created: 05/09/2022
Date last modified: 05/09/2022 
'''

import numpy as np
import xarray as xr
import nsidc0756.xy2ll as xy2ll


# np.set_printoptions(threshold=np.inf)

DIR_NS = '/homes/lchen2/project/OISSTv2.1_NOAA/ancillary/NSIDC_mask'
FN_MASK_NS = "bedmacmask.nc"
ds_mask_ns = xr.open_dataset(filename_or_obj=DIR_NS + '/' + FN_MASK_NS)

xs = 1.0*ds_mask_ns.coords['x'].data  # have to convert from int to float, or it has nan.
ys = 1.0*ds_mask_ns.coords['y'].data

to_lat = np.zeros((13333, 13333), dtype=np.float32, order='C')
to_lon = np.zeros((13333, 13333), dtype=np.float32, order='C')

for j in range(13333): 
    print('j=', j)
    for i in range(13333):
        if ys[j]==0 and xs[i]==0: # the xy2ll function will fail for this special case.
            lat, lon = -89.999999, -45.0
        else:
            lat, lon = xy2ll.xy2ll(xs[i], ys[j], -1, 0, 71)

        to_lat[j, i] = lat
        to_lon[j, i] = lon

da_lat = xr.DataArray(data=to_lat, dims=['y', 'x'], name='lat')
da_lon = xr.DataArray(data=to_lon, dims=['y', 'x'], name='lon')
ds_tmp = xr.merge([da_lat, da_lon])
fn_tmp = 'lat_lon.nc'
ds_tmp.to_netcdf(DIR_NS+'/'+fn_tmp)
