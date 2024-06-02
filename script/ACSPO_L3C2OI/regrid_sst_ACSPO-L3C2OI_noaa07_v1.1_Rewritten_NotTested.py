'''
Description: Compute binned/average result on OI 0.25-degree grid from 0.02-degree ACSPO L3C daily SST.
    ACSPO L3C grid is the same as L3S, this script is modified from L3S script.
Author: Ligang Chen
Date created: 07/26/2022
Date last modified: 07/26/2022 
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


DIR_L3C = '/glade/scratch/lgchen/data/ACSPO_L3C'
FN_L3C_TEST = DIR_L3C+"/test/19810901120000-STAR-L3C_GHRSST-SSTsubskin-AVHRRG_N07_N-ACSPO_V2.81-v02.0-fv01.0.nc"
ds_l3c_test = xr.open_dataset(filename_or_obj=FN_L3C_TEST)
ds_l3c_test.coords['lon'] = xr.where(ds_l3c_test.coords['lon']<0, 360+ds_l3c_test.coords['lon']  \
    , ds_l3c_test.coords['lon'])
ds_l3c_test.sortby(ds_l3c_test.coords['lon'])
lat_l3c = ds_l3c_test.coords['lat']  #  9000         : 89.99, 89.97, ..., -89.97, -89.99;
lon_l3c = ds_l3c_test.coords['lon']  # 18000(initial): -179.99, -179.97, ..., 179.97, 179.99;

DIR_OI = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master'
FN_OI_MASK = DIR_OI + '/bin2netCDF/quarter-mask-extend_Ligang.nc'
ds_oi_mask = xr.open_dataset(filename_or_obj=FN_OI_MASK)
ds_oi_mask['landmask'] = ds_oi_mask.landmask.astype(np.int32)
# the following seems not working, what exactly does it do?
# oi_mask_fort = np.asfortranarray(ds_oi_mask.landmask.values) 
oi_mask_fort = ds_oi_mask.landmask.values.T  # instead this transpose works.
lat_oi = ds_oi_mask.coords['lat']  # 720: -89.875, -89.625, -89.375, ...,  89.375,  89.625,  89.875
lon_oi = ds_oi_mask.coords['lon']  # 1440: 0.125 0.375 0.625 ... 359.625 359.875

# To print all numpy array elements.
# np.set_printoptions(threshold=np.inf)

# lat: 9000: 719 719 ... 0 0; lon: 720 720 ... 1439 1439 0 0 ... 719 719.
idx_lat_l3c2oi = np.around(4*(ds_l3c_test.coords['lat'].values+89.875)).astype(np.int32) 
idx_lon_l3c2oi = np.around(4*(ds_l3c_test.coords['lon'].values-0.125 )).astype(np.int32)

# the index should still start with 0?
idx_lat_oi2l3c = np.zeros((2, 720 ), dtype=np.int32, order='F') 
idx_lon_oi2l3c = np.zeros((2, 1440), dtype=np.int32, order='F')

# For each oi lat point, calculate its corresponding start and end indices in l3c
# result, [[8987 9000], [8975 8987], ..., [  12   25], [   0   12]]
idx_lat_oi_curr = idx_lat_l3c2oi[0]
idx_lat_oi2l3c[0, idx_lat_oi_curr] = 0
for i_lat_l3c in range(1, 9000): 
    if idx_lat_l3c2oi[i_lat_l3c] != idx_lat_oi_curr:
        idx_lat_oi2l3c[1, idx_lat_oi_curr] = i_lat_l3c - 1
        idx_lat_oi_curr = idx_lat_l3c2oi[i_lat_l3c]
        idx_lat_oi2l3c[0, idx_lat_oi_curr] = i_lat_l3c
else:
    idx_lat_oi2l3c[1, idx_lat_oi_curr] = 9000 - 1

# For each oi lon point, calculate its corresponding start and end indices in l3c
# result, [[9000 9013], [9013 9025],...,[17988 18000], [0 13], ..., [8975 8988], [8988 9000]]
idx_lon_oi_curr = idx_lon_l3c2oi[0]
idx_lon_oi2l3c[0, idx_lon_oi_curr] = 0
for i_lon_l3c in range(1, 18000): 
    if idx_lon_l3c2oi[i_lon_l3c] != idx_lon_oi_curr:
        idx_lon_oi2l3c[1, idx_lon_oi_curr] = i_lon_l3c - 1
        idx_lon_oi_curr = idx_lon_l3c2oi[i_lon_l3c]
        idx_lon_oi2l3c[0, idx_lon_oi_curr] = i_lon_l3c
else:
    idx_lon_oi2l3c[1, idx_lon_oi_curr] = 18000 - 1

# for fortran index convention, maybe not here
idx_lat_oi2l3c += 1
idx_lon_oi2l3c += 1

# 1 record/file, could be either daytime or nighttime;
daily_sst_avg_l3c2oi_1rec = np.zeros((1440, 720), dtype=np.float32, order='F')   
daily_sst_num_l3c2oi_1rec = np.zeros((1440, 720), dtype=np.int32  , order='F')  # for binned to avg

jday_19810901 = datetime.date(1981, 9 , 1 )
jday_19850202 = datetime.date(1985, 2 , 2 )

jday = jday_19810901
while jday <= jday_19850202:
    str_date = jday.strftime('%Y%m%d')
    print('\n\n current date: ', str_date)

    # day and night
    fn_day = DIR_L3C+'/noaa07/'+str_date+'120000-STAR-L3C_GHRSST-SSTsubskin-AVHRRG_N07_D-ACSPO_V2.81-v02.0-fv01.0.nc'
    fn_nit = DIR_L3C+'/noaa07/'+str_date+'120000-STAR-L3C_GHRSST-SSTsubskin-AVHRRG_N07_N-ACSPO_V2.81-v02.0-fv01.0.nc'
    fns = (fn_day, fn_nit)

    for (id_fn, fn) in enumerate(fns, start=0):  # 0: day; 1: night
        if not os.path.exists(fn):
            print('file not exist: ' + fn)
            continue

        print(str(id_fn).zfill(5)+', current file: ', fn)
        daily_sst_avg_l3c2oi_1rec.fill(0)  # In-place operation
        daily_sst_num_l3c2oi_1rec.fill(0)

        ds = xr.open_dataset(filename_or_obj=fn, mask_and_scale=True, decode_times=True  \
            , drop_variables=['sst_dtime', 'sses_standard_deviation', 'l3s_sst_reference'  \
            , 'dt_analysis', 'sst_count', 'sst_source', 'satellite_zenith_angle', 'wind_speed'  \
            , 'crs']).isel(time=0)

        ds['quality_level'] = ds.quality_level.astype(np.int8)  # convert back to type byte.
        ds['sea_surface_temperature'] = xr.where(ds.quality_level==5, ds.sea_surface_temperature, np.nan)
        ds['sea_surface_temperature'] = ds['sea_surface_temperature'] - ds['sses_bias']  # actual sst
        ds['sea_surface_temperature'] = xr.where(ds['sea_surface_temperature'] < 250, 0, ds['sea_surface_temperature'])
        ds['sea_surface_temperature'] = xr.where(ds['sea_surface_temperature'] > 350, 0, ds['sea_surface_temperature'])
        # HAVE TO use assignment! DataArray.fillna() itself is NOT an in-place operation!
        ds['sea_surface_temperature'] = ds['sea_surface_temperature'].fillna(0) # need to set 0 first for later computation

        ds.coords['lon'] = xr.where(ds.coords['lon']<0, 360+ds.coords['lon'], ds.coords['lon'])
        ds.sortby(ds.coords['lon'])
        ds['l2p_flags'] = ds.l2p_flags.astype(np.int16)
      # print('ds.coords[lon]: ', ds.coords['lon'].values)

      # print('Start processing SST, current time: ', datetime.datetime.now().strftime('%H:%M:%S'))
        sst_fort = ds['sea_surface_temperature'].values.T  # This seems to work.
        (daily_sst_avg_l3c2oi_1rec, daily_sst_num_l3c2oi_1rec) = fort.sst_acspo2oi_1rec(sst_fort  \
            , oi_mask_fort, idx_lat_oi2l3c, idx_lon_oi2l3c) 
      # print('End processing SST, current time: ', datetime.datetime.now().strftime('%H:%M:%S'))

        daily_sst_avg_l3c2oi_1rec  = np.divide(daily_sst_avg_l3c2oi_1rec , daily_sst_num_l3c2oi_1rec   \
            , where=(daily_sst_num_l3c2oi_1rec  > 0.9))  # or use '!=0'
        daily_sst_avg_l3c2oi_1rec  = xr.where(daily_sst_num_l3c2oi_1rec ==0, -9999, daily_sst_avg_l3c2oi_1rec -273.15)

        da_daily_sst_avg = xr.DataArray(data=np.float32(daily_sst_avg_l3c2oi_1rec.T)  \
            , dims=['lat', 'lon'], coords={'lat': lat_oi, 'lon': lon_oi}, name='sst'  \
            , attrs={'long_name':'sst binned average', 'units':'degC', '_FillValue':-9999}) 
        da_daily_sst_num = xr.DataArray(data=np.int32  (daily_sst_num_l3c2oi_1rec.T)  \
            , dims=['lat', 'lon'], coords={'lat': lat_oi, 'lon': lon_oi}, name='sst_num'  \
            , attrs=dict(long_name='number of satellite sst obs binned or averaged to OI grid box', _FillValue=0))


        # to netCDF
        ds_daily = xr.merge([da_daily_sst_avg, da_daily_sst_num])
        ds_daily.attrs = {}  # or it'll have unwanted attr from da_daily_sst_avg
        if id_fn == 0:
            fn_daily = str_date+'_N07_D_ACSPO-L3C2OI.nc'
        else:
            fn_daily = str_date+'_N07_N_ACSPO-L3C2OI.nc'

        ds_daily.to_netcdf(DIR_L3C+'/l3c2oi/noaa07/'+fn_daily, encoding={'lat': {'_FillValue': None}, 'lon': {'_FillValue': None}})


        # to 2D OI-grid binary
        if id_fn == 0:
            fn_daily_bin_2d = str_date+'_N07_D_ACSPO-L3C2OI_2D.bin'
        else:
            fn_daily_bin_2d = str_date+'_N07_N_ACSPO-L3C2OI_2D.bin'

        fl = open(DIR_L3C+'/l3c2oi/noaa07/'+fn_daily_bin_2d, 'wb')
        fl.write(struct.pack('>i', 4147212))
        fl.write(struct.pack('>iii', jday.year, jday.month, jday.day))
        fl.write(np.float32(daily_sst_avg_l3c2oi_1rec.T).byteswap())
        fl.write(struct.pack('>i', 4147212))
        fl.close()

        # to 1D super-obs
        if id_fn == 0:
            fn_daily_bin_1d = str_date+'_N07_D_ACSPO-L3C2OI_1D.bin'
        else:
            fn_daily_bin_1d = str_date+'_N07_N_ACSPO-L3C2OI_1D.bin'

        ctype = 'dsat07  '
        fort.write_1d_sat_sst_super_obs(DIR_L3C+'/l3c2oi/noaa07/'+fn_daily_bin_1d  \
            , np.float32(daily_sst_avg_l3c2oi_1rec), np.int32(daily_sst_num_l3c2oi_1rec), oi_mask_fort, jday.year  \
            , jday.month, jday.day, ctype)


    jday += datetime.timedelta(days=1)