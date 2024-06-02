# #!/glade/u/apps/ch/opt/python/3.7.9/gnu/9.1.0/pkg-library/20201220/bin/python3

'''
Description: Compute binned/average result on OI 0.25-degree grid from 0.02-degree ACSPO L3S_LEO daily SST.
Author: Ligang Chen
Date created: 10/22/2021
Date last modified: 03/04/2022 
'''

import numpy as np
import xarray as xr

import datetime
import glob

import fort  # using F2PY call Fortran subroutine


DIR_L3S = '/glade/scratch/lgchen/data/L3S_LEO'
FN_L3S_TEST = DIR_L3S+"/test/20150101120000-STAR-L3S_GHRSST-SSTsubskin-LEO_AM_N-ACSPO_V2.80-v02.0-fv01.0.nc"
ds_l3s_test = xr.open_dataset(filename_or_obj=FN_L3S_TEST)
ds_l3s_test.coords['lon'] = xr.where(ds_l3s_test.coords['lon']<0, 360+ds_l3s_test.coords['lon']  \
    , ds_l3s_test.coords['lon'])
ds_l3s_test.sortby(ds_l3s_test.coords['lon'])
lat_l3s = ds_l3s_test.coords['lat']  #  9000         : 89.99, 89.97, ..., -89.97, -89.99;
lon_l3s = ds_l3s_test.coords['lon']  # 18000(initial): -179.99, -179.97, ..., 179.97, 179.99;

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
np.set_printoptions(threshold=np.inf)

# lat: 9000: 719 719 ... 0 0; lon: 720 720 ... 1439 1439 0 0 ... 719 719.
idx_lat_l3s2oi = np.around(4*(ds_l3s_test.coords['lat'].values+89.875)).astype(np.int32) 
idx_lon_l3s2oi = np.around(4*(ds_l3s_test.coords['lon'].values-0.125 )).astype(np.int32)

# the index should still start with 0?
idx_lat_oi2l3s = np.zeros((2, 720 ), dtype=np.int32, order='F') 
idx_lon_oi2l3s = np.zeros((2, 1440), dtype=np.int32, order='F')

# For each oi lat point, calculate its corresponding start and end indices in l3s
# result, [[8987 9000], [8975 8987], ..., [  12   25], [   0   12]]
idx_lat_oi_curr = idx_lat_l3s2oi[0]
idx_lat_oi2l3s[0, idx_lat_oi_curr] = 0
for i_lat_l3s in range(1, 9000): 
    if idx_lat_l3s2oi[i_lat_l3s] != idx_lat_oi_curr:
        idx_lat_oi2l3s[1, idx_lat_oi_curr] = i_lat_l3s - 1
        idx_lat_oi_curr = idx_lat_l3s2oi[i_lat_l3s]
        idx_lat_oi2l3s[0, idx_lat_oi_curr] = i_lat_l3s
else:
    idx_lat_oi2l3s[1, idx_lat_oi_curr] = 9000 - 1

# For each oi lon point, calculate its corresponding start and end indices in l3s
# result, [[9000 9013], [9013 9025],...,[17988 18000], [0 13], ..., [8975 8988], [8988 9000]]
idx_lon_oi_curr = idx_lon_l3s2oi[0]
idx_lon_oi2l3s[0, idx_lon_oi_curr] = 0
for i_lon_l3s in range(1, 18000): 
    if idx_lon_l3s2oi[i_lon_l3s] != idx_lon_oi_curr:
        idx_lon_oi2l3s[1, idx_lon_oi_curr] = i_lon_l3s - 1
        idx_lon_oi_curr = idx_lon_l3s2oi[i_lon_l3s]
        idx_lon_oi2l3s[0, idx_lon_oi_curr] = i_lon_l3s
else:
    idx_lon_oi2l3s[1, idx_lon_oi_curr] = 18000 - 1

# for fortran index convention, maybe not here
idx_lat_oi2l3s += 1
idx_lon_oi2l3s += 1

# 1 record/file, could be either daytime or nighttime;
daily_sst_avg_l3s2oi_1rec = np.zeros((1440, 720), dtype=np.float32, order='F')   
daily_sst_num_l3s2oi_1rec = np.zeros((1440, 720), dtype=np.int32  , order='F')  # for binned to avg

daily_sst_avg_l3s2oi_final = np.zeros((1440, 720, 2), dtype=np.float32, order='F')  # 0-day and 1-night   
daily_sst_num_l3s2oi_final = np.zeros((1440, 720, 2), dtype=np.int32  , order='F')

jday_20160101 = datetime.date(2016, 1 , 1 )
jday_20161231 = datetime.date(2016, 12, 31)

jday = jday_20160101
while jday <= jday_20161231:
    str_date = jday.strftime('%Y%m%d')
    print('\n\n current date: ', str_date)

    # day and night
    fns_day = glob.glob(pathname=DIR_L3S+'/link/'+str_date+'*_D-ACSPO_*.nc')
    fns_nit = glob.glob(pathname=DIR_L3S+'/link/'+str_date+'*_N-ACSPO_*.nc')

    daily_sst_avg_l3s2oi_final.fill(0)  # In-place operation
    daily_sst_num_l3s2oi_final.fill(0)
    for (id_fns, fns) in enumerate((fns_day, fns_nit), start=0):
        print('id_fns=', id_fns, 'fns: ', fns)

        for (id_fn, fn) in enumerate(fns, start=0):
            print(str(id_fn).zfill(5)+', current file: ', fn)
            daily_sst_avg_l3s2oi_1rec.fill(0)
            daily_sst_num_l3s2oi_1rec.fill(0)

            ds = xr.open_dataset(filename_or_obj=fn, mask_and_scale=True, decode_times=True  \
                , drop_variables=['sst_dtime', 'sses_standard_deviation', 'l3s_sst_reference'  \
                , 'dt_analysis', 'sst_count', 'sst_source', 'satellite_zenith_angle', 'wind_speed'  \
                , 'crs', 'sst_gradient_magnitude', 'sst_front_position']).isel(time=0)

            ds['quality_level'] = ds.quality_level.astype(np.int8)  # convert back to type byte.
            ds['sea_surface_temperature'] = xr.where(ds.quality_level==5, ds.sea_surface_temperature, np.nan)
            ds['sea_surface_temperature'] = ds['sea_surface_temperature'] - ds['sses_bias']  # actual sst
            ds['sea_surface_temperature'] = xr.where(ds['sea_surface_temperature'] < 250, 0, ds['sea_surface_temperature'])
            ds['sea_surface_temperature'] = xr.where(ds['sea_surface_temperature'] > 350, 0, ds['sea_surface_temperature'])
            # HAVE TO use assignment! DataArray.fillna() itself is NOT an in-place operation!
            ds['sea_surface_temperature'] = ds['sea_surface_temperature'].fillna(0)

            ds.coords['lon'] = xr.where(ds.coords['lon']<0, 360+ds.coords['lon'], ds.coords['lon'])
            ds.sortby(ds.coords['lon'])
            ds['l2p_flags'] = ds.l2p_flags.astype(np.int16)
          # print('ds.coords[lon]: ', ds.coords['lon'].values)

          # print('Start processing SST, current time: ', datetime.datetime.now().strftime('%H:%M:%S'))
          # sst_fort = np.asfortranarray(ds['sea_surface_temperature'].values)  # doesn't work!
            sst_fort = ds['sea_surface_temperature'].values.T  # This seems to work.
            (daily_sst_avg_l3s2oi_1rec, daily_sst_num_l3s2oi_1rec) = fort.sst_acspo2oi_1rec(sst_fort  \
                , oi_mask_fort, idx_lat_oi2l3s, idx_lon_oi2l3s) 
          # print('End processing SST, current time: ', datetime.datetime.now().strftime('%H:%M:%S'))

            daily_sst_avg_l3s2oi_final[:, :, id_fns] += daily_sst_avg_l3s2oi_1rec
            daily_sst_num_l3s2oi_final[:, :, id_fns] += daily_sst_num_l3s2oi_1rec

    # 1. may need to check above if the file exist, if not, below will generate files with value zero
    #     which is not what we want.         
    daily_sst_avg_l3s2oi_final = np.divide(daily_sst_avg_l3s2oi_final, daily_sst_num_l3s2oi_final  \
        , where=(daily_sst_num_l3s2oi_final > 0.9))  # or use '!=0'
    daily_sst_avg_l3s2oi_final = xr.where(daily_sst_num_l3s2oi_final==0, 0, daily_sst_avg_l3s2oi_final)

    da_daily_sst_avg_day = xr.DataArray(data=np.float32(daily_sst_avg_l3s2oi_final[:, :, 0].T)  \
        , dims=['lat', 'lon'], coords={'lat': lat_oi, 'lon': lon_oi}, name='sst_avg_daytime'  \
        , attrs={'units':'Kelvin', '_FillValue':0})
    da_daily_sst_num_day = xr.DataArray(data=np.int32  (daily_sst_num_l3s2oi_final[:, :, 0].T)  \
        , dims=['lat', 'lon'], coords={'lat': lat_oi, 'lon': lon_oi}, name='sst_num_daytime'  \
        , attrs=dict(_FillValue=0))

    da_daily_sst_avg_nit = xr.DataArray(data=np.float32(daily_sst_avg_l3s2oi_final[:, :, 1].T)  \
        , dims=['lat', 'lon'], coords={'lat': lat_oi, 'lon': lon_oi}, name='sst_avg_nighttime'  \
        , attrs={'units':'Kelvin', '_FillValue':0})
    da_daily_sst_num_nit = xr.DataArray(data=np.int32  (daily_sst_num_l3s2oi_final[:, :, 1].T)  \
        , dims=['lat', 'lon'], coords={'lat': lat_oi, 'lon': lon_oi}, name='sst_num_nighttime'  \
        , attrs=dict(_FillValue=0))

    ds_daily = xr.merge([da_daily_sst_avg_day, da_daily_sst_num_day, da_daily_sst_avg_nit  \
        , da_daily_sst_num_nit])
    fn_daily_sst = str_date+'-STAR-L3S_GHRSST-SSTsubskin-LEO_AM_PM_D_N-ACSPO2OI_V2.80-v02.0-fv01.0.nc'
    ds_daily.to_netcdf(DIR_L3S+'/l3s2oi/2016/'+fn_daily_sst, encoding={'lat': {'_FillValue': None}, 'lon': {'_FillValue': None}})

    jday += datetime.timedelta(days=1)
