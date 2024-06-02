'''
Description: Compute binned/average result on OI 0.25-degree grid from Met Office Hadley Centre
    EN.4.2.2 quality controlled profile ocean data using the uppermost temperature as SST.
Author: Ligang Chen
Date created: 03/09/2022
Date last modified: 03/09/2022
'''

import numpy as np
import xarray as xr
import pandas as pd

import calendar
import datetime
import glob

# import fort


DIR_EN = '/glade/scratch/lgchen/data/EN4_QC_OceanData_HadleyCentre/2020'
FN_PF_EN = 'EN.4.2.2.f.profiles.g10.'

DIR_OI = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master'
FN_OI_MASK = DIR_OI + '/bin2netCDF/quarter-mask-extend_Ligang.nc'
ds_oi_mask = xr.open_dataset(filename_or_obj=FN_OI_MASK)
ds_oi_mask['landmask'] = ds_oi_mask.landmask.astype(np.int32)
landmask_oi_fort = ds_oi_mask.landmask.values.T
# oi_mask_fort = ds_oi_mask.landmask.values.T  # instead this transpose seems to work.
lat_oi = ds_oi_mask.coords['lat']  # 720: -89.875, -89.625, -89.375, ...,  89.375,  89.625,  89.875
lon_oi = ds_oi_mask.coords['lon']  # 1440: 0.125 0.375 0.625 ... 359.625 359.875

np.set_printoptions(threshold=np.inf) # To print all numpy array elements.

# 1 month per file, could be either daytime or nighttime;
daily_temp_avg_en2oi = np.zeros((1440, 720, 31), dtype=np.float32, order='F')  # daily
daily_temp_num_en2oi = np.zeros((1440, 720, 31), dtype=np.int32  , order='F') 

dropped_vars = ['CALIBRATION_DATE', 'CYCLE_NUMBER', 'DATA_CENTRE', 'DATA_MODE', 'DATA_STATE_INDICATOR', 'DATA_TYPE'  \
    , 'DATE_CREATION', 'DATE_UPDATE', 'DC_REFERENCE', 'DIRECTION', 'FORMAT_VERSION', 'HANDBOOK_VERSION'  \
    , 'HISTORY_ACTION', 'HISTORY_DATE', 'HISTORY_INSTITUTION', 'HISTORY_PARAMETER', 'HISTORY_PREVIOUS_VALUE'  \
    , 'HISTORY_QCTEST', 'HISTORY_SOFTWARE', 'HISTORY_SOFTWARE_RELEASE', 'HISTORY_START_DEPH', 'HISTORY_STOP_DEPH'  \
    , 'INST_REFERENCE', 'JULD_LOCATION', 'PARAMETER', 'PI_NAME', 'PLATFORM_NUMBER', 'POSITIONING_SYSTEM'  \
    , 'POTM_CORRECTED', 'POTM_CORRECTED_QC', 'PROFILE_POTM_QC', 'PROFILE_PSAL_QC', 'PROJECT_NAME', 'PSAL_CORRECTED'  \
    , 'PSAL_CORRECTED_QC', 'REFERENCE_DATE_TIME', 'SCIENTIFIC_CALIB_COEFFICIENT', 'SCIENTIFIC_CALIB_COMMENT'  \
    , 'SCIENTIFIC_CALIB_EQUATION', 'STATION_PARAMETERS', 'WMO_INST_TYPE', 'instrument_type', 'thermal_corrections']
# kept_vars = ['DEPH_CORRECTED', 'DEPH_CORRECTED_QC', 'JULD', 'JULD_QC', 'LATITUDE', 'LONGITUDE', 'PROFILE_DEPH_QC'  \
#     , 'QC_FLAGS_LEVELS', 'QC_FLAGS_PROFILES', 'TEMP', 'depth_corrections']
mask_2ndBit = np.int16(2)

period = pd.period_range(start='2015-01', end='2020-12', freq='M')
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
    daily_sst_avg_iq2oi = xr.where(daily_sst_num_iq2oi==0, 0, daily_sst_avg_iq2oi)

    da_daily_sst_avg_day = xr.DataArray(data=np.float32(daily_sst_avg_iq2oi[:, :, 0, 0:dofm].T)  \
        , dims=['day', 'lat', 'lon'], coords={'day': day, 'lat': lat_oi, 'lon': lon_oi}, name='sst_daytime'  \
        , attrs={'units':'Kelvin', '_FillValue':0})
  # da_daily_sst_num_day = xr.DataArray(data=np.int8   (daily_sst_num_iq2oi[:, :, 0, 0:dofm].T)  \
  #     , dims=['day', 'lat', 'lon'], coords={'day': day, 'lat': lat_oi, 'lon': lon_oi}, name='sst_num_daytime'  \
  #     , attrs=dict(_FillValue=0))

    da_daily_sst_avg_nit = xr.DataArray(data=np.float32(daily_sst_avg_iq2oi[:, :, 1, 0:dofm].T)  \
        , dims=['day', 'lat', 'lon'], coords={'day': day, 'lat': lat_oi, 'lon': lon_oi}, name='sst_nighttime'  \
        , attrs={'units':'Kelvin', '_FillValue':0})
  # da_daily_sst_num_nit = xr.DataArray(data=np.int8   (daily_sst_num_iq2oi[:, :, 1, 0:dofm].T)  \
  #     , dims=['day', 'lat', 'lon'], coords={'day': day, 'lat': lat_oi, 'lon': lon_oi}, name='sst_num_nighttime'  \
  #     , attrs=dict(_FillValue=0))

    da_daily_sst_avg = xr.DataArray(data=np.float32(daily_sst_avg_iq2oi[:, :, 2, 0:dofm].T)  \
        , dims=['day', 'lat', 'lon'], coords={'day': day, 'lat': lat_oi, 'lon': lon_oi}, name='sst'  \
        , attrs={'units':'Kelvin', '_FillValue':0})
  # da_daily_sst_num = xr.DataArray(data=np.int8   (daily_sst_num_iq2oi[:, :, 2, 0:dofm].T)  \
  #     , dims=['day', 'lat', 'lon'], coords={'day': day, 'lat': lat_oi, 'lon': lon_oi}, name='sst_num'  \
  #     , attrs=dict(_FillValue=0))

  # ds_oi = xr.merge([da_daily_sst_avg_day, da_daily_sst_num_day, da_daily_sst_avg_nit  \
  #     , da_daily_sst_num_nit, da_daily_sst_avg, da_daily_sst_num])
    ds_oi = xr.merge([da_daily_sst_avg_day, da_daily_sst_avg_nit, da_daily_sst_avg])

    fn_oi = str_ym+'-iQuam2OI.nc'
    ds_oi.to_netcdf(DIR_IQ+'/iQuam2OI/'+fn_oi)

