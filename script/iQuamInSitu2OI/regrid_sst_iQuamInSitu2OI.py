# #!/glade/u/apps/ch/opt/python/3.7.9/gnu/9.1.0/pkg-library/20201220/bin/python3

'''
Description: Compute binned/average result on OI 0.25-degree grid from 1D iQuam InSitu SST.
Author: Ligang Chen
Date created: 12/01/2021
Date last modified: 12/01/2021 
'''

import numpy as np
import xarray as xr
import pandas as pd

import calendar
import datetime
import glob

import fort  # using F2PY call Fortran subroutine


DIR_IQ = '/glade/scratch/lgchen/data/iQuam_InSitu_SST_obs'
# FN_PF_IQ = '-STAR-L2i_GHRSST-SST-iQuam-V2.10-v01.0-fv01.0.nc'
FN_PF_IQ = '-STAR-L2i_GHRSST-SST-iQuam-V2.10-v01.0-'

DIR_OI = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master'
FN_OI_MASK = DIR_OI + '/bin2netCDF/quarter-mask-extend_Ligang.nc'
ds_oi_mask = xr.open_dataset(filename_or_obj=FN_OI_MASK)
ds_oi_mask['landmask'] = ds_oi_mask.landmask.astype(np.int32)
landmask_oi_fort = ds_oi_mask.landmask.values.T
lat_oi = ds_oi_mask.coords['lat']  # 720: -89.875, -89.625, -89.375, ...,  89.375,  89.625,  89.875
lon_oi = ds_oi_mask.coords['lon']  # 1440: 0.125 0.375 0.625 ... 359.625 359.875

# np.set_printoptions(threshold=np.inf) # To print all numpy array elements.

# 1 month per file, could be either daytime or nighttime;
daily_sst_avg_iq2oi = np.zeros((1440, 720, 3, 31), dtype=np.float32, order='F')  # 3 for daytime, nighttime and allday
daily_sst_num_iq2oi = np.zeros((1440, 720, 3, 31), dtype=np.int32  , order='F')  # for binned to avg

dropped_vars = ['year', 'month', 'hour', 'minute', 'second', 'platform_id', 'sst_ref1', 'sst_ref2', 'iquam_flags', 'optional_flags1'  \
    , 'optional_flags2', 'res_flag', 'deploy_id', 'manuf', 'depth', 'wind_speed', 'wind_direction', 'air_temperature'  \
    , 'air_pressure', 'cloud_coverage', 'dew_point_temperature', 'source']
# kept_vars = ['day', 'lat', 'lon', 'platform_type', 'sst', 'quality_level', 'sst_flags', 'time']
mask_2ndBit = np.int16(2)

period = pd.period_range(start='1981-09', end='2021-12', freq='M')
prTups = tuple([(pr.year, pr.month) for pr in period])
print(prTups)

for (year, month) in prTups:
    str_ym = str(year) + str(month).zfill(2)
    print('processing ' + str_ym + '...')
    dofm = calendar.monthrange(year, month)[1]
    day = np.arange(start=1, stop=dofm+1, step=1, dtype=np.int8)
  # print('day=', day)

    abs_fn_iq = glob.glob(pathname=DIR_IQ+'/Original/'+str_ym + FN_PF_IQ + '*.nc')[0]
    ds_iq = xr.open_dataset(filename_or_obj=abs_fn_iq, mask_and_scale=True, decode_times=True, drop_variables=dropped_vars)
    ds_iq['quality_level'] = ds_iq.quality_level.astype(np.int8)
  # print('before calling where function ...')
  # ds_iq = ds_iq.where(ds_iq.quality_level==5, drop=True)  # too slow
  # print('after calling where function ...')
    ds_iq['day'] = ds_iq.day.astype(np.int8)
    ds_iq['platform_type'] = ds_iq.platform_type.astype(np.int8)

    idx_lat_iq2oi = np.around(4*(ds_iq.coords['lat'].values+89.875)).astype(np.int32) + 1 # idx start from 1 for fortran
    ds_iq.coords['lon'] = xr.where(ds_iq.coords['lon']<0, 360+ds_iq.coords['lon'], ds_iq.coords['lon'])
    idx_lon_iq2oi = np.around(4*(ds_iq.coords['lon'].values-0.125 )).astype(np.int32) + 1 # idx start from 1 for fortran

    is_daytime = mask_2ndBit & ds_iq['sst_flags'].data
    is_daytime = is_daytime.astype(np.int16)

    (daily_sst_avg_iq2oi, daily_sst_num_iq2oi) = fort.sst_iquam2oi_1mon(ds_iq['sst'].data, landmask_oi_fort  \
        , ds_iq['day'].data, idx_lat_iq2oi, idx_lon_iq2oi, is_daytime, ds_iq['quality_level'].data)
  # num_negtive = np.sum(daily_sst_num_iq2oi[:, :, 2, 0]<0)
  # print('num_negtive=', num_negtive)
    daily_sst_avg_iq2oi = np.divide(daily_sst_avg_iq2oi, daily_sst_num_iq2oi, where=(daily_sst_num_iq2oi > 0.9))
    daily_sst_avg_iq2oi = xr.where(daily_sst_num_iq2oi==0, -9999, daily_sst_avg_iq2oi)

    da_daily_sst_avg_day = xr.DataArray(data=np.float32(daily_sst_avg_iq2oi[:, :, 0, 0:dofm].T)  \
        , dims=['day', 'lat', 'lon'], coords={'day': day, 'lat': lat_oi, 'lon': lon_oi}, name='sst_day'  \
        , attrs={'long_name':'daytime sst', 'units':'Kelvin', '_FillValue':-9999})
  # da_daily_sst_num_day = xr.DataArray(data=np.int8   (daily_sst_num_iq2oi[:, :, 0, 0:dofm].T)  \
  #     , dims=['day', 'lat', 'lon'], coords={'day': day, 'lat': lat_oi, 'lon': lon_oi}, name='sst_num_day'  \
  #     , attrs=dict(_FillValue=0))

    da_daily_sst_avg_nit = xr.DataArray(data=np.float32(daily_sst_avg_iq2oi[:, :, 1, 0:dofm].T)  \
        , dims=['day', 'lat', 'lon'], coords={'day': day, 'lat': lat_oi, 'lon': lon_oi}, name='sst_night'  \
        , attrs={'long_name':'nighttime sst', 'units':'Kelvin', '_FillValue':-9999})
  # da_daily_sst_num_nit = xr.DataArray(data=np.int8   (daily_sst_num_iq2oi[:, :, 1, 0:dofm].T)  \
  #     , dims=['day', 'lat', 'lon'], coords={'day': day, 'lat': lat_oi, 'lon': lon_oi}, name='sst_num_night'  \
  #     , attrs=dict(_FillValue=0))

    da_daily_sst_avg = xr.DataArray(data=np.float32(daily_sst_avg_iq2oi[:, :, 2, 0:dofm].T)  \
        , dims=['day', 'lat', 'lon'], coords={'day': day, 'lat': lat_oi, 'lon': lon_oi}, name='sst_daily'  \
        , attrs={'long_name':'daily sst', 'units':'Kelvin', '_FillValue':-9999})
  # da_daily_sst_num = xr.DataArray(data=np.int8   (daily_sst_num_iq2oi[:, :, 2, 0:dofm].T)  \
  #     , dims=['day', 'lat', 'lon'], coords={'day': day, 'lat': lat_oi, 'lon': lon_oi}, name='sst_num_daily'  \
  #     , attrs=dict(_FillValue=0))

  # ds_oi = xr.merge([da_daily_sst_avg_day, da_daily_sst_num_day, da_daily_sst_avg_nit  \
  #     , da_daily_sst_num_nit, da_daily_sst_avg, da_daily_sst_num])
    ds_oi = xr.merge([da_daily_sst_avg_day, da_daily_sst_avg_nit, da_daily_sst_avg])

    fn_oi = str_ym+'-iQuam2OI.nc'
    ds_oi.to_netcdf(DIR_IQ+'/iQuam2OI/'+fn_oi, encoding={'lat': {'_FillValue': None}, 'lon': {'_FillValue': None}})


