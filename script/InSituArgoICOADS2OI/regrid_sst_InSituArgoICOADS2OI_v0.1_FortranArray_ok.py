'''
Description: Compute binned/average result on OI 0.25-degree grid from OISST InSitu ship/buoy daily SST.
Author: Ligang Chen
Date created: 04/12/2022
Date last modified: 04/12/2022 
'''

import numpy as np
import xarray as xr

import datetime
import glob

# import fort


DIR_IN = '/glade/scratch/lgchen/data/OISST_InSitu_Argo_ICOADS'

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


daily_sst_avg_in2oi = np.zeros((1440, 720, 2), dtype=np.float32, order='F')  # 0-ship and 1-buoy and everything else
daily_sst_num_in2oi = np.zeros((1440, 720, 2), dtype=np.ubyte  , order='F')

jday_20160101 = datetime.date(2016, 1 , 1 )
jday_20161231 = datetime.date(2016, 12, 31)

jday = jday_20160101
while jday <= jday_20161231:
    str_date = jday.strftime('%Y%m%d')
    print('\n\n current date: ', str_date)
  # print(datetime.datetime.now().strftime("%H:%M:%S"))

    fn_in = 'mq.' + str_date
    fl_in = open(DIR_IN + '/' + fn_in, 'rt')
    daily_sst_avg_in2oi.fill(0)
    daily_sst_num_in2oi.fill(0)
    line_num = 0

    while True:
        line_num = line_num + 1
        str_line = fl_in.readline()
        if not str_line: break

        lst_item = str_line.split(',')
      # year, month, day, hour, lat, lon, depth, ididid, cid, sst = int(lst_item[0]), int(lst_item[1]), int(lst_item[2]) \
      #     , int(lst_item[3])//100, 0.01*int(lst_item[4]), 0.01*int(lst_item[5]), int(lst_item[6]), int(lst_item[7]), lst_item[8] \
      #     , 0.001*int(lst_item[9]) 
      # print(year, month, day, hour, lat, lon, depth, ididid, cid, sst)

        lat, lon, ididid, cid, sst = 0.01*int(lst_item[4]), 0.01*int(lst_item[5]), int(lst_item[7]), lst_item[8], 0.001*int(lst_item[9])
        idx_lat_oi = np.around(4*(lat+89.875)).astype(np.int32)
        idx_lon_oi = np.around(4*(lon-0.125 )).astype(np.int32)

        if idx_lat_oi < 0:
            print('line_num=', line_num, ', idx_lat_oi=', idx_lat_oi) 
            idx_lat_oi = 0
        elif idx_lat_oi > 719:
            print('line_num=', line_num, ', idx_lat_oi = ', idx_lat_oi) 
            idx_lat_oi = 719

        if idx_lon_oi < 0: 
            print('line_num=', line_num, ', idx_lon_oi = ', idx_lon_oi) 
            idx_lat_oi = 0
        elif idx_lon_oi > 1439:
            print('line_num=', line_num, ', idx_lon_oi = ', idx_lon_oi) 
            idx_lat_oi = 1439

        if oi_mask_fort[idx_lon_oi, idx_lat_oi] < 0.1:  # land
            continue
 
        if 0<=idx_lat_oi and idx_lat_oi<720 and 0<=idx_lon_oi and idx_lon_oi<1440:
            if ididid==1:
                daily_sst_avg_in2oi[idx_lon_oi, idx_lat_oi, 0] = daily_sst_avg_in2oi[idx_lon_oi, idx_lat_oi, 0] + sst
                daily_sst_num_in2oi[idx_lon_oi, idx_lat_oi, 0] = daily_sst_num_in2oi[idx_lon_oi, idx_lat_oi, 0] + 1
            else:
                daily_sst_avg_in2oi[idx_lon_oi, idx_lat_oi, 1] = daily_sst_avg_in2oi[idx_lon_oi, idx_lat_oi, 1] + sst
                daily_sst_num_in2oi[idx_lon_oi, idx_lat_oi, 1] = daily_sst_num_in2oi[idx_lon_oi, idx_lat_oi, 1] + 1      

    fl_in.close()

    daily_sst_avg_in2oi = np.divide(daily_sst_avg_in2oi, daily_sst_num_in2oi, where=(daily_sst_num_in2oi > 0.9))
    daily_sst_avg_in2oi = xr.where(daily_sst_num_in2oi==0, -999., daily_sst_avg_in2oi)

    da_daily_sst_avg_ship = xr.DataArray(data=np.float32(daily_sst_avg_in2oi[:, :, 0].T)  \
        , dims=['lat', 'lon'], coords={'lat': lat_oi, 'lon': lon_oi}, name='sst_ship'  \
        , attrs={'units':'Celcius', '_FillValue':-999.})
    da_daily_sst_num_ship = xr.DataArray(data=np.ubyte  (daily_sst_num_in2oi[:, :, 0].T)  \
        , dims=['lat', 'lon'], coords={'lat': lat_oi, 'lon': lon_oi}, name='sst_num_ship'  \
        , attrs=dict(_FillValue=0))

    da_daily_sst_avg_buoy = xr.DataArray(data=np.float32(daily_sst_avg_in2oi[:, :, 1].T)  \
        , dims=['lat', 'lon'], coords={'lat': lat_oi, 'lon': lon_oi}, name='sst_buoy'  \
        , attrs={'units':'Celcius', '_FillValue':-999.})
    da_daily_sst_num_buoy = xr.DataArray(data=np.ubyte  (daily_sst_num_in2oi[:, :, 1].T)  \
        , dims=['lat', 'lon'], coords={'lat': lat_oi, 'lon': lon_oi}, name='sst_num_buoy'  \
        , attrs=dict(_FillValue=0))

    ds_oi = xr.merge([da_daily_sst_avg_ship, da_daily_sst_num_ship, da_daily_sst_avg_buoy, da_daily_sst_num_buoy])
    fn_oi = str_date+'_InSitu2OI.nc'
    ds_oi.to_netcdf(DIR_IN+'/in2oi/'+fn_oi, encoding={'lat': {'_FillValue': None}, 'lon': {'_FillValue': None}})

    jday += datetime.timedelta(days=1)
