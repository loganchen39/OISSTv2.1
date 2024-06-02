#!/glade/u/apps/ch/opt/python/3.7.9/gnu/9.1.0/pkg-library/20201220/bin/python3

'''
Description: Mask binned/average ACSPO SST to Pathfinder resolution on OI 0.25-degree grid 
    for the OISST experiment.
Author: Ligang Chen
Date created: 11/09/2021
Date last modified: 11/09/2021 
'''

import numpy as np
import xarray as xr

import datetime
import glob

# import fort  # using F2PY call Fortran subroutine


DIR_L3S = '/glade/scratch/lgchen/data/L3S_LEO'
DIR_PF  = '/glade/scratch/lgchen/data/pathfinder'

DIR_OI = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master'
FN_OI_MASK = DIR_OI + '/bin2netCDF/quarter-mask-extend_Ligang.nc'
ds_oi_mask = xr.open_dataset(filename_or_obj=FN_OI_MASK)
ds_oi_mask['landmask'] = ds_oi_mask.landmask.astype(np.int32)
lat_oi = ds_oi_mask.coords['lat']  # 720: -89.875, -89.625, -89.375, ...,  89.375,  89.625,  89.875
lon_oi = ds_oi_mask.coords['lon']  # 1440: 0.125 0.375 0.625 ... 359.625 359.875
num_ocn = np.count_nonzero(ds_oi_mask.landmask.data)
# print('num_ocn=', num_ocn, 'ocn percentage=', 1.0*num_ocn/(720*1440))  # 691150, 66.66%

# To print all numpy array elements.
# np.set_printoptions(threshold=np.inf)

num_per_ocn_ac = []  # (day_num, night_num, day_per, night_per)
num_per_ocn_pf = []  # (day_num, night_num, day_per, night_per)

jday_20200101 = datetime.date(2020, 1 , 1 )
jday_20200105 = datetime.date(2020, 1 , 5 )
jday_20201231 = datetime.date(2020, 12, 31)

jday = jday_20200101
while jday <= jday_20201231:
    str_date = jday.strftime('%Y%m%d')
    print('\n\n current date: ', str_date)

    fn_ac = str_date + '-STAR-L3S_GHRSST-SSTsubskin-LEO_AM_PM_D_N-ACSPO2OI_V2.80-v02.0-fv01.0.nc'
    fn_pf = str_date + '-NCEI-L3C_GHRSST-SSTskin-AVHRR_Pathfinder-PFV5.3_NOAA19_G_2020001_day_night-v02.0-fv01.0.nc'
    ds_ac = xr.open_dataset(filename_or_obj=DIR_L3S+'/l3s2oi/2020/'+fn_ac, mask_and_scale=True, decode_times=True)
    ds_pf = xr.open_dataset(filename_or_obj=DIR_PF+'/pf2oi/2020/'+fn_pf, mask_and_scale=True, decode_times=True)
   
  # ds_ac = xr.where(ds_oi_mask.landmask==0, np.nan, ds_ac) # not needed
    num_day = np.count_nonzero(~np.isnan(ds_ac.sst_avg_daytime.data))  
    num_nit = np.count_nonzero(~np.isnan(ds_ac.sst_avg_nighttime.data)) 
    per_day = 1.0*num_day/691150
    per_nit = 1.0*num_nit/691150
    num_per_ocn_ac.append((num_day, num_nit, per_day, per_nit))
 
    num_day = np.count_nonzero(~np.isnan(ds_pf.sst_avg_daytime.data))  
    num_nit = np.count_nonzero(~np.isnan(ds_pf.sst_avg_nighttime.data)) 
    per_day = 1.0*num_day/691150
    per_nit = 1.0*num_nit/691150
    num_per_ocn_pf.append((num_day, num_nit, per_day, per_nit))

    ds_ac['sst_avg_daytime'] = xr.where(ds_pf['sst_avg_daytime'].isnull(), np.nan, ds_ac['sst_avg_daytime'])
    ds_ac['sst_num_daytime'] = xr.where(ds_pf['sst_avg_daytime'].isnull(), np.nan, ds_ac['sst_num_daytime'])
    ds_ac['sst_avg_nighttime'] = xr.where(ds_pf['sst_avg_nighttime'].isnull(), np.nan, ds_ac['sst_avg_nighttime'])
    ds_ac['sst_num_nighttime'] = xr.where(ds_pf['sst_avg_nighttime'].isnull(), np.nan, ds_ac['sst_num_nighttime'])

    fn_ac_masked = str_date + '-masked2pf-STAR-L3S_GHRSST-SSTsubskin-LEO_AM_PM_D_N-ACSPO2OI_V2.80-v02.0-fv01.0.nc'
    ds_ac.to_netcdf(DIR_L3S + '/l3s2oi/2020_AtPfResolution/' + fn_ac_masked)

    jday += datetime.timedelta(days=1)


fl = open(r'num_ocn.txt', 'w')

fl.write(format('day_num', '>8s') + ' ' + format('nit_num', '>8s') + ' ' + format('day_per', '>7s')  \
    + ' ' + format('nit_per', '>7s') + ' AC\n')
for (day_num, nit_num, day_per, nit_per) in num_per_ocn_ac:
    fl.write(format(day_num, '>8d') + ' ' + format(nit_num, '>8d') + ' ' + format(day_per, '>7.4f')  \
        + ' ' + format(nit_per, '>7.4f') + '\n')

fl.write(format('day_num', '>8s') + ' ' + format('nit_num', '>8s') + ' ' + format('day_per', '>7s')  \
    + ' ' + format('nit_per', '>7s') + ' PF\n')
for (day_num, nit_num, day_per, nit_per) in num_per_ocn_pf:
    fl.write(format(day_num, '>8d') + ' ' + format(nit_num, '>8d') + ' ' + format(day_per, '>7.4f')  \
        + ' ' + format(nit_per, '>7.4f') + '\n')

fl.close()

# calculate total average percentage
avg_ac = np.mean(num_per_ocn_ac, axis=0)
avg_pf = np.mean(num_per_ocn_pf, axis=0)
print('avg_ac=', avg_ac)
print('avg_pf=', avg_pf)




