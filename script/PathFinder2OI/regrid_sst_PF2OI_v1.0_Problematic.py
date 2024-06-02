# #!/glade/u/apps/ch/opt/python/3.7.9/gnu/9.1.0/pkg-library/20201220/bin/python3


'''
Description: Compute binned/average result on OI 0.25-degree grid from 0.0417-degree Pathfinder L3C daily SST.
Author: Ligang Chen
Date created: 11/09/2021
Date last modified: 08/03/2022 
'''

import numpy as np
import xarray as xr

import os
import datetime
import glob

import sys
sys.path.append('/glade/u/home/lgchen/lib/fortran/f2py')

import fort


# To print all numpy array elements.
np.set_printoptions(threshold=np.inf)

DIR_PF = '/glade/scratch/lgchen/data/PathfinderV5.3'
FN_PF_TEST = DIR_PF+"/test/19810825023019-NCEI-L3C_GHRSST-SSTskin-AVHRR_Pathfinder-PFV5.3_NOAA07_G_1981237_night-v02.0-fv01.0.nc"
ds_pf_test = xr.open_dataset(filename_or_obj=FN_PF_TEST)
# lon, 8640: -179.9792, -179.9375, ..., 179.9375, 179.9792 (initial)
ds_pf_test.coords['lon'] = xr.where(ds_pf_test.coords['lon']<0, 360+ds_pf_test.coords['lon'], ds_pf_test.coords['lon']) 
ds_pf_test.sortby(ds_pf_test.coords['lon'])
lat_pf = ds_pf_test.coords['lat']  # 4320: 89.97917, 89.9375, ..., -89.9375, -89.97916;
lon_pf = ds_pf_test.coords['lon']  # 8640: 180.02083, 180.0625, ..., 179.9375 , 179.97916
# print("lon_pf: ", lon_pf) 

DIR_OI = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master'
FN_OI_MASK = DIR_OI + '/bin2netCDF/quarter-mask-extend_Ligang.nc'
ds_oi_mask = xr.open_dataset(filename_or_obj=FN_OI_MASK)
ds_oi_mask['landmask'] = ds_oi_mask.landmask.astype(np.int32)
# the following seems not working, what exactly does it do?
# oi_mask_fort = np.asfortranarray(ds_oi_mask.landmask.values) 
oi_mask_fort = ds_oi_mask.landmask.values.T  # instead this transpose seems to work.
lat_oi = ds_oi_mask.coords['lat']  # 720: -89.875, -89.625, -89.375, ...,  89.375,  89.625,  89.875
lon_oi = ds_oi_mask.coords['lon']  # 1440: 0.125 0.375 0.625 ... 359.625 359.875


# lat(4320): 719 719 ... 0 0; 
# lon(8640): 720 720 ... 1439 1439 0 0 ... 719 719.
idx_lat_pf2oi = np.around(4*(ds_pf_test.coords['lat'].values+89.875)).astype(np.int32) 
idx_lon_pf2oi = np.around(4*(ds_pf_test.coords['lon'].values-0.125 )).astype(np.int32)
# print('\n\n\n idx_lat_pf2oi, ', idx_lat_pf2oi)
# print('\n\n\n idx_lon_pf2oi, ', idx_lon_pf2oi)

# the index should still start with 0?
idx_lat_oi2pf = np.zeros((2, 720 ), dtype=np.int32, order='F') 
idx_lon_oi2pf = np.zeros((2, 1440), dtype=np.int32, order='F')

# For each oi lat point, calculate its corresponding start and end indices in pf
# result, [[4314 4319], [4308 4313], ..., [6 11], [   0   5]]
idx_lat_oi_curr = idx_lat_pf2oi[0]
idx_lat_oi2pf[0, idx_lat_oi_curr] = 0
for i_lat_pf in range(1, 4320): 
    if idx_lat_pf2oi[i_lat_pf] != idx_lat_oi_curr:
        idx_lat_oi2pf[1, idx_lat_oi_curr] = i_lat_pf - 1
        idx_lat_oi_curr = idx_lat_pf2oi[i_lat_pf]
        idx_lat_oi2pf[0, idx_lat_oi_curr] = i_lat_pf
else:
    idx_lat_oi2pf[1, idx_lat_oi_curr] = 4320 - 1

# For each oi lon point, calculate its corresponding start and end indices in pf
# result, [[4320 4325], [4326 4331],...,[8634 8639], [0 5], ..., [4308 4313], [4314 4319]]
idx_lon_oi_curr = idx_lon_pf2oi[0]
idx_lon_oi2pf[0, idx_lon_oi_curr] = 0
for i_lon_pf in range(1, 8640): 
    if idx_lon_pf2oi[i_lon_pf] != idx_lon_oi_curr:
        idx_lon_oi2pf[1, idx_lon_oi_curr] = i_lon_pf - 1
        idx_lon_oi_curr = idx_lon_pf2oi[i_lon_pf]
        idx_lon_oi2pf[0, idx_lon_oi_curr] = i_lon_pf
else:
    idx_lon_oi2pf[1, idx_lon_oi_curr] = 8640 - 1

# print('\n\n\n idx_lat_oi2pf, ', idx_lat_oi2pf)
# print('\n\n\n idx_lon_oi2pf, ', idx_lon_oi2pf)

# for fortran index convention, maybe not here
idx_lat_oi2pf += 1
idx_lon_oi2pf += 1

# 1 record/file, daytime or nighttime; binned to avg
daily_sst_avg_pf2oi_1rec = np.zeros((1440, 720), dtype=np.float32, order='F')   
daily_sst_num_pf2oi_1rec = np.zeros((1440, 720), dtype=np.int32  , order='F')

daily_sst_avg_pf2oi_final = np.zeros((1440, 720, 2), dtype=np.float32, order='F')  # day and night   
daily_sst_num_pf2oi_final = np.zeros((1440, 720, 2), dtype=np.int32  , order='F')

jday_19810825 = datetime.date(1981, 8 , 25)
jday_20200630 = datetime.date(2020, 6 , 30)
jday_20211231 = datetime.date(2021, 12, 31)

jday = jday_20200630
while jday <= jday_20211231:
    str_date = jday.strftime('%Y%m%d')
    print('\n\n current date: ', str_date)

    # day and night
    fn_day = glob.glob(pathname=DIR_PF+'/PFV5.3/'+str_date+'*_day-*.nc')
    fn_nit = glob.glob(pathname=DIR_PF+'/PFV5.3/'+str_date+'*_night-*.nc')

    n = len(fn_day)
    if n >= 1:
        fn_day = fn_day[0]
        if n > 1:
            print('Warning: ', str_date, ' ', n, ' day files!')
    else:
        print('Warning: ', str_date, ' no day file!')
        fn_day = ''

    n = len(fn_nit)
    if n >= 1:
        fn_nit = fn_nit[0]
        if n > 1:
            print('Warning: ', str_date, ' ', n, ' nit files!')
    else:
        print('Warning: ', str_date, ' no nit file!')
        fn_nit = ''
    
    fns = (fn_day, fn_nit)

    daily_sst_avg_pf2oi_final.fill(0)  # In-place operation
    daily_sst_num_pf2oi_final.fill(0)

    for (id_fn, fn) in enumerate(fns, start=0):
        if not os.path.exists(fn):
            print('file not exist: ' + fn)
            continue

        print(str(id_fn).zfill(5)+', current file: ', fn)
        daily_sst_avg_pf2oi_1rec.fill(0)
        daily_sst_num_pf2oi_1rec.fill(0)

        ds = xr.open_dataset(filename_or_obj=fn, mask_and_scale=True, decode_times=True  \
            , drop_variables=['time_bounds', 'lat_bounds', 'lon_bounds', 'crs', 'sst_dtime'  \
            , 'sses_standard_deviation', 'dt_analysis', 'wind_speed', 'sea_ice_fraction'  \
            , 'aerosol_dynamic_indicator']).isel(time=0)

        ds['quality_level'] = ds.quality_level.astype(np.int8)  # convert back to type byte.
        ds['sea_surface_temperature'] = xr.where(ds.quality_level==5, ds.sea_surface_temperature, np.nan)
        # in this version the sses_bias is "empty", so no need to subtract.
        # ds['sea_surface_temperature'] = ds['sea_surface_temperature'] - ds['sses_bias']
        ds['sea_surface_temperature'] = xr.where(ds['sea_surface_temperature'] < 250, 0  \
            , ds['sea_surface_temperature'])
        ds['sea_surface_temperature'] = xr.where(ds['sea_surface_temperature'] > 350, 0  \
            , ds['sea_surface_temperature'])
        # HAVE TO use assignment! DataArray.fillna() itself is NOT an in-place operation!
        # Need to fillna as 0 for later computation!
        ds['sea_surface_temperature'] = ds['sea_surface_temperature'].fillna(0)

        ds.coords['lon'] = xr.where(ds.coords['lon']<0, 360+ds.coords['lon'], ds.coords['lon'])
        ds.sortby(ds.coords['lon'])
        ds['l2p_flags'] = ds.l2p_flags.astype(np.int16)
        # print('ds.coords[lon]: ', ds.coords['lon'].values)

        # print('Start processing SST, current time: ', datetime.datetime.now().strftime('%H:%M:%S'))
        sst_fort = ds['sea_surface_temperature'].values.T  # This seems to work.
        (daily_sst_avg_pf2oi_1rec, daily_sst_num_pf2oi_1rec) = fort.sst_pf2oi_1rec(sst_fort  \
            , oi_mask_fort, idx_lat_oi2pf, idx_lon_oi2pf)
        # print('End processing SST, current time: ', datetime.datetime.now().strftime('%H:%M:%S'))

        daily_sst_avg_pf2oi_final[:, :, id_fn] += daily_sst_avg_pf2oi_1rec
        daily_sst_num_pf2oi_final[:, :, id_fn] += daily_sst_num_pf2oi_1rec
         
    daily_sst_avg_pf2oi_final = np.divide(daily_sst_avg_pf2oi_final, daily_sst_num_pf2oi_final  \
        , where=(daily_sst_num_pf2oi_final > 0.9))  # or use '!=0'
    daily_sst_avg_pf2oi_final = xr.where(daily_sst_num_pf2oi_final==0, -9999, daily_sst_avg_pf2oi_final-273.15)

    da_daily_sst_avg_day = xr.DataArray(data=np.float32(daily_sst_avg_pf2oi_final[:, :, 0].T)  \
        , dims=['lat', 'lon'], coords={'lat': lat_oi, 'lon': lon_oi}, name='sst_avg_day'  \
        , attrs={'long_name':'daytime sst binned average', 'units':'degC', '_FillValue':-9999})
    da_daily_sst_num_day = xr.DataArray(data=np.int32  (daily_sst_num_pf2oi_final[:, :, 0].T)  \
        , dims=['lat', 'lon'], coords={'lat': lat_oi, 'lon': lon_oi}, name='sst_num_day'  \
        , attrs=dict(long_name='number of daytime satellite sst obs binned to OI grid box', _FillValue=0))

    # to netCDF
    ds_daily_day = xr.merge([da_daily_sst_avg_day, da_daily_sst_num_day])
    ds_daily_day.attrs = {}
    fn_daily_day = str_date+'_day_PFV5.3-2OI.nc'
    ds_daily_day.to_netcdf(DIR_PF+'/PF2OI/'+fn_daily_day, encoding={'lat': {'_FillValue': None}, 'lon': {'_FillValue': None}})


    da_daily_sst_avg_nit = xr.DataArray(data=np.float32(daily_sst_avg_pf2oi_final[:, :, 1].T)  \
        , dims=['lat', 'lon'], coords={'lat': lat_oi, 'lon': lon_oi}, name='sst_avg_nit'  \
        , attrs={'long_name':'nighttime sst binned average', 'units':'degC', '_FillValue':-9999})
    da_daily_sst_num_nit = xr.DataArray(data=np.int32  (daily_sst_num_pf2oi_final[:, :, 1].T)  \
        , dims=['lat', 'lon'], coords={'lat': lat_oi, 'lon': lon_oi}, name='sst_num_nit'  \
        , attrs=dict(long_name='number of nighttime satellite sst obs binned to OI grid box', _FillValue=0))

    # to netCDF
    ds_daily_nit = xr.merge([da_daily_sst_avg_nit, da_daily_sst_num_nit])
    ds_daily_nit.attrs = {}
    fn_daily_nit = str_date+'_night_PFV5.3-2OI.nc'
    ds_daily_nit.to_netcdf(DIR_PF+'/PF2OI/'+fn_daily_nit, encoding={'lat': {'_FillValue': None}, 'lon': {'_FillValue': None}})


    jday += datetime.timedelta(days=1)
