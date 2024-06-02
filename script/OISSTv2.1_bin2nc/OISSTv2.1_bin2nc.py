#!/glade/u/apps/ch/opt/python/3.6.8/gnu/8.3.0/pkg-library/20190627/bin/python3
# #!/glade/u/apps/ch/opt/python/3.7.9/gnu/9.1.0/bin/python3

import os
import datetime

import numpy as np
import xarray as xr


MONTHS_IN_YEAR = 12

QUART_DEG_LON_DIM = 1440
QUART_DEG_LAT_DIM = 720
LON_START         = 0.125
LAT_START         = -89.875
GRID_SIZE         = 0.25

ONE_DEG_LON_DIM   = 360
ONE_DEG_LAT_DIM   = 180
ONE_DEG_LON_START = -0.5
ONE_DEG_LAT_START = -89.5
ONE_DEG_GRID_SIZE = 1.0

HALF_DEG_LON_DIM   = 720
HALF_DEG_LAT_DIM   = 360
HALF_DEG_LON_START = -0.25
HALF_DEG_LAT_START = -90.25
HALF_DEG_GRID_SIZE = 0.5

NUM_EOT_MODES = 130


check_mask = False
file_mask = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/common/static/quarter-mask-extend'

check_stdev = False
file_stdev  = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/common/static/stdev1d-coads3-fill'

check_fg = False
file_fg = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/prelim/out/oiout/2016/sst4-metopab-eot-intv2.20160116'

check_obs_buoy_grid = False
file_obs_buoy_grid = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/prelim/work/grid/buoyship/2016/buoy.20160117'

check_obs_argo_grid = False
file_obs_argo_grid = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/prelim/work/grid/buoyship/2016/argo.20160117'

check_obs_ship_grid = False
file_obs_ship_grid = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/prelim/work/grid/buoyship/2016/ship.20160117'

check_sobs_buoy = False
file_sobs_buoy = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/prelim/work/sobs/buoyship/2016/buoy.20160117'

check_sobs_argo = False
file_sobs_argo = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/prelim/work/sobs/buoyship/2016/argo.20160117'

check_sobs_ship = False
file_sobs_ship = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/prelim/work/sobs/buoyship/2016/shipc.20160117'

check_obs_sat_dgrid = True 
# file_obs_sat_dgrid = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/prelim/work/grid/metopa/2016/dgrid.sst.20160119'
file_obs_sat_dgrid = '/glade/scratch/lgchen/data/PathfinderV5.3/PF2OI/19810825_day_PFV5.3-2OI_2D.bin'

check_obs_sat_ngrid = True 
# file_obs_sat_ngrid = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/prelim/work/grid/metopa/2016/ngrid.sst.20160119'
# file_obs_sat_ngrid = '/glade/scratch/lgchen/data/ACSPO_L3C/l3c2oi/noaa07/19810905_N07_N_ACSPO-L3C2OI_2D.bin'
file_obs_sat_ngrid = '/glade/scratch/lgchen/data/PathfinderV5.3/PF2OI/19810825_night_PFV5.3-2OI_2D.bin'

check_dsobs_satA = True 
# file_dsobs_satA = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/prelim/work/sobs/metopa/2016/dsobs.20160117'
# file_dsobs_satA = '/glade/scratch/lgchen/data/ACSPO_L3C/l3c2oi/noaa07/19810905_N07_D_ACSPO-L3C2OI_1D.bin'
file_dsobs_satA = '/glade/scratch/lgchen/data/PathfinderV5.3/PF2OI/19810825_day_PFV5.3-2OI_1D.bin'

check_nsobs_satA = True 
# file_nsobs_satA = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/prelim/work/sobs/metopa/2016/nsobs.20160117'
# file_nsobs_satA = '/glade/scratch/lgchen/data/ACSPO_L3C/l3c2oi/noaa07/19810905_N07_N_ACSPO-L3C2OI_1D.bin'
file_nsobs_satA = '/glade/scratch/lgchen/data/PathfinderV5.3/PF2OI/19810825_night_PFV5.3-2OI_1D.bin'

check_icecon720x360 = False
file_icecon720x360 = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/prelim/work/grid/ice/con/2016/icecon720x360.20160117'

check_icecon = False
file_icecon = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/prelim/work/grid/ice/con/2016/icecon.20160117'

check_icemask = False
file_icemask = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/common/static/ice_flags_mask.dat'

check_frzpnt = False
file_frzpnt = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/common/fzsst/daily-fzsst.0117'

check_icecon7 = False
file_icecon7 = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/prelim/work/grid/ice/con/2016/icecon7.20160117'

check_iceConMed = False
file_iceConMed = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/prelim/work/grid/ice/con-med/2016/icecon-med.20160117'

check_icesst_grid = False
file_icesst_grid = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/prelim/work/grid/ice/ice-sst/2016/icesst.20160117'

check_icesst_sobs = False
file_icesst_sobs = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/prelim/work/sobs/ice/2016/icesst.20160117'

check_mask_twodeg = False
file_mask_twodeg = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/common/static/lstags.twodeg.dat'

check_climatology = False
file_climatology = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/common/static/clim.71.00.gdat'

check_eot_mode = False
file_eot_mode  = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/common/static/eot6.damp-zero.ev130.ano.dat'

check_eot_weight = False
# file_eot_weight = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/prelim/work/eotwt/metopb/2016/nwt.20160117'
file_eot_weight = '/glade/work/ggraham/oisst.v2.1-master/prelim/work/eotwt/noaa07/1981/dwt.19811128'

check_mode_variance = False
file_mode_variance = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/common/static/var-mode'

check_mode_variance = False
file_mode_variance = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/common/static/var-mode'

check_eot_bias = False
file_eot_bias  = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/prelim/work/eotbias/metopb/2016/nbias.20160117'

check_nsobsc_sat = False
file_nsobsc_sat = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/prelim/work/eotcor/metopb/2016/nsobsc.20160117'

check_correlation_scale = False
file_correlation_scale = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/common/static/cor4sm-stat-v2'

check_fg_err_variance = False
file_fg_err_variance = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/common/static/path-incr-var-stat-v2'

check_fg_correction = False
file_fg_correction = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/common/static/error-cor-stat-v2'

check_monthly_clim = False
file_monthly_clim = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/common/static/oiclm4.mon'

check_residual_bias = False
file_residual_bias = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/common/static/residual-stat-v2'

check_buoy4sm_nsr = False
file_buoy4sm_nsr = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/common/static/buoy4sm-nsr-stat-v2'

check_ship4sm_nsr = False
file_ship4sm_nsr = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/common/static/ship4sm-nsr-stat-v2'

check_day_path4sm_nsr = False
file_day_path4sm_nsr = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/common/static/day-path4sm-nsr-stat-v2'

check_nte_path4sm_nsr = False
file_nte_path4sm_nsr = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/common/static/nte-path4sm-nsr-stat-v2'

check_cice4sm_nsr = False
file_cice4sm_nsr = '/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/common/static/cice4sm-nsr-stat-v2'


if check_mask:
  with open(file_mask, 'rb') as f:
    header = np.fromfile(f, dtype='>i4', count=1)
    data = np.fromfile(f, dtype='>f', count=QUART_DEG_LON_DIM*QUART_DEG_LAT_DIM)
    mask = np.reshape(data, (QUART_DEG_LAT_DIM, QUART_DEG_LON_DIM), order='C')
    tail = np.fromfile(f, dtype='>i4', count=1)
    print('header=', header, ', tail=', tail)

  lon = np.arange(start=LON_START, stop=360., step=GRID_SIZE)
  lat = np.arange(start=LAT_START, stop=90. , step=GRID_SIZE)

  da_mask = xr.DataArray(mask, name='landmask', dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon})
  da_mask.to_dataset(name='landmask').to_netcdf('./quarter-mask-extend_Ligang.nc'  \
    , encoding={'lat': {'_FillValue': None}, 'lon': {'_FillValue': None}})


if check_stdev:
# month = range(1, MONTHS_IN_YEAR+1)
  month = np.arange(1, MONTHS_IN_YEAR+1, dtype='i')
  lon = np.arange(start=ONE_DEG_LON_START, stop=359., step=ONE_DEG_GRID_SIZE, dtype='f')
  lat = np.arange(start=ONE_DEG_LAT_START, stop=90. , step=ONE_DEG_GRID_SIZE, dtype='f')
  stdev = np.zeros((MONTHS_IN_YEAR, ONE_DEG_LAT_DIM, ONE_DEG_LON_DIM), dtype='f')

  f = open(file_stdev, 'rb')
  for i_mon in range(1, MONTHS_IN_YEAR+1):
    header = np.fromfile(f, dtype='>i4', count=1)
    base_year  = np.fromfile(f, dtype='>i4', count=1)
    base_month = np.fromfile(f, dtype='>i4', count=1)
    data       = np.fromfile(f, dtype='>f', count=ONE_DEG_LON_DIM*ONE_DEG_LAT_DIM)
    stdev[i_mon-1, :, :] = np.reshape(data, (ONE_DEG_LAT_DIM, ONE_DEG_LON_DIM), order='C')
    tail = np.fromfile(f, dtype='>i4', count=1)
    print('i_mon=', i_mon, 'header=', header, ', tail=', tail, 'base_year=', base_year, ', base_month=', base_month)

  da_stdev = xr.DataArray(stdev, name='sst_stdev', dims=('month', 'lat', 'lon')  \
    , coords={'month': month, 'lat': lat, 'lon': lon})
  da_stdev.to_dataset(name='sst_stdev').to_netcdf('./stdev1d-coads3-fill_Ligang.nc')


if check_fg:
  f = open(file_fg, 'rb')
  header = np.fromfile(f, dtype='>i', count=1)
  jyr = np.fromfile(f, dtype='>i', count=1)
  jmo = np.fromfile(f, dtype='>i', count=1)
  jda = np.fromfile(f, dtype='>i', count=1)  # if type f, then 2.2e-44==0
  data = np.fromfile(f, dtype='>f', count=QUART_DEG_LON_DIM*QUART_DEG_LAT_DIM)
  gsst = np.reshape(data, (QUART_DEG_LAT_DIM, QUART_DEG_LON_DIM), order='C')
  tail = np.fromfile(f, dtype='>i', count=1)
  print('header=', header, 'jyr=', jyr, ', jmo=', jmo, ', jda=', jda, ', tail=', tail)

# header2 = np.fromfile(f, dtype='>i', count=1)
# jyr = np.fromfile(f, dtype='>i', count=1)
# jmo = np.fromfile(f, dtype='>i', count=1)
# jda = np.fromfile(f, dtype='>i', count=1)  # if type f, then 2.2e-44==0
# data = np.fromfile(f, dtype='>f', count=QUART_DEG_LON_DIM*QUART_DEG_LAT_DIM)
# gsst = np.reshape(data, (QUART_DEG_LAT_DIM, QUART_DEG_LON_DIM), order='C')
# tail2 = np.fromfile(f, dtype='>i', count=1)
# print('header2=', header, 'jyr=', jyr, ', jmo=', jmo, ', jda=', jda, ', tail2=', tail2)

# header3 = np.fromfile(f, dtype='>i', count=1)
# jyr = np.fromfile(f, dtype='>i', count=1)
# jmo = np.fromfile(f, dtype='>i', count=1)
# jda = np.fromfile(f, dtype='>i', count=1)  # if type f, then 2.2e-44==0
# data = np.fromfile(f, dtype='>f', count=QUART_DEG_LON_DIM*QUART_DEG_LAT_DIM)
# gsst = np.reshape(data, (QUART_DEG_LAT_DIM, QUART_DEG_LON_DIM), order='C')
# tail3 = np.fromfile(f, dtype='>i', count=1)
# print('header3=', header, 'jyr=', jyr, ', jmo=', jmo, ', jda=', jda, ', tail3=', tail3)

# header4 = np.fromfile(f, dtype='>i', count=1)
# print('header4=', header4)  # printed 110; 

  lon = np.arange(start=LON_START, stop=360., step=GRID_SIZE)
  lat = np.arange(start=LAT_START, stop=90. , step=GRID_SIZE)

  da_gsst = xr.DataArray(gsst, name='sst', dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}  \
    , attrs=dict(_FillValue=-9999, units="degC") )
  da_gsst.to_dataset(name='sst').to_netcdf('./sst4-metopab-eot-intv2.20160116_Ligang.nc')


if check_obs_buoy_grid:  # [27.685, 29.8889]
  f = open(file_obs_buoy_grid, 'rb')
  header = np.fromfile(f, dtype='>i', count=1)
  jyr = np.fromfile(f, dtype='>i', count=1)
  jmo = np.fromfile(f, dtype='>i', count=1)
  jda = np.fromfile(f, dtype='>i', count=1)
  data = np.fromfile(f, dtype='>f', count=QUART_DEG_LON_DIM*QUART_DEG_LAT_DIM)
  sst_obs = np.reshape(data, (QUART_DEG_LAT_DIM, QUART_DEG_LON_DIM), order='C')
  tail = np.fromfile(f, dtype='>i', count=1)
  print('header=', header, ', jyr=', jyr, ', jmo=', jmo, ', jda=', jda, ', tail=', tail)

  lon = np.arange(start=LON_START, stop=360., step=GRID_SIZE)
  lat = np.arange(start=LAT_START, stop=90. , step=GRID_SIZE)

  da_sst_obs = xr.DataArray(sst_obs, name='sst', dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}  \
    , attrs=dict(_FillValue=-9999, units="degC") )
  da_sst_obs.to_dataset(name='sst').to_netcdf('./obs_buoy_grid.20160117_Ligang.nc') 

  da_tmp = xr.where(da_sst_obs == -9999, 0, 1)
  num_obs = da_tmp.sum().values
  print('For obs_buoy_grid, num_obs=', num_obs)  # printed 2995, which is 0.29%, for sea_ict=691150, it's 0.43%;


if check_obs_argo_grid:  # min and max are both 0
  f = open(file_obs_argo_grid, 'rb')
  header = np.fromfile(f, dtype='>i', count=1)
  jyr = np.fromfile(f, dtype='>i', count=1)
  jmo = np.fromfile(f, dtype='>i', count=1)
  jda = np.fromfile(f, dtype='>i', count=1)
  data = np.fromfile(f, dtype='>f', count=QUART_DEG_LON_DIM*QUART_DEG_LAT_DIM)
  sst_obs = np.reshape(data, (QUART_DEG_LAT_DIM, QUART_DEG_LON_DIM), order='C')
  tail= np.fromfile(f, dtype='>i', count=1)
  print('header=', header, ', jyr=', jyr, ', jmo=', jmo, ', jda=', jda, ', tail=', tail)

  lon = np.arange(start=LON_START, stop=360., step=GRID_SIZE)
  lat = np.arange(start=LAT_START, stop=90. , step=GRID_SIZE)

  da_sst_obs = xr.DataArray(sst_obs, name='sst', dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}  \
    , attrs=dict(_FillValue=-9999, units="degC") )
  da_sst_obs.to_dataset(name='sst').to_netcdf('./obs_argo_grid.20160117_Ligang.nc') 

  da_tmp = xr.where(da_sst_obs == -9999, 0, 1)
  num_obs = da_tmp.sum().values
  print('For obs_argo_grid, num_obs=', num_obs)  # printed 305, which is 0.0294%, for sea_ict=691150, it's 0.044%


if check_obs_ship_grid:  # both min and max are 29.4
  f = open(file_obs_ship_grid, 'rb')
  header = np.fromfile(f, dtype='>i', count=1)
  jyr = np.fromfile(f, dtype='>i', count=1)
  jmo = np.fromfile(f, dtype='>i', count=1)
  jda = np.fromfile(f, dtype='>i', count=1)
  data = np.fromfile(f, dtype='>f', count=QUART_DEG_LON_DIM*QUART_DEG_LAT_DIM)
  sst_obs = np.reshape(data, (QUART_DEG_LAT_DIM, QUART_DEG_LON_DIM), order='C')
  tail = np.fromfile(f, dtype='>i', count=1)
  print('header=', header, ', jyr=', jyr, ', jmo=', jmo, ', jda=', jda, ', tail=', tail)

  lon = np.arange(start=LON_START, stop=360., step=GRID_SIZE)
  lat = np.arange(start=LAT_START, stop=90. , step=GRID_SIZE)

  da_sst_obs = xr.DataArray(sst_obs, name='sst', dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}  \
    , attrs=dict(_FillValue=-9999, units="degC") )
  da_sst_obs.to_dataset(name='sst').to_netcdf('./obs_ship_grid.20160117_Ligang.nc') 

# da_tmp = xr.where(da_sst_obs.isnull(), 0, 1)  # isnull() does NOT work, FillValue of -9999 is not null!
  da_tmp = xr.where(da_sst_obs == -9999, 0, 1)  # works!
  num_obs = da_tmp.sum().values
  # printed 1729, for sea_ict=691150, it's 0.25%, which is the same as the log_debug output, all of these 3 are correct.
  print('For obs_ship, num_obs=', num_obs) 


if check_sobs_buoy:  # the result is the same as obs_buoy_grid, except 2D vs. 1D. 
  f = open(file_sobs_buoy, 'rb')
  header = np.fromfile(f, dtype='>i4', count=1)
  jyr = np.fromfile(f, dtype='>i4', count=1)
  jmo = np.fromfile(f, dtype='>i4', count=1)
  jda = np.fromfile(f, dtype='>i4', count=1)
# str_ctype = np.fromfile(f, dtype='>b', count=8).to_string()  # doesn't work this way
# str_ctype = str(np.fromfile(f, dtype='>U8', count=1)) # failed
# str_ctype = str(np.fromfile(f, dtype='S8', count=1))  # worked
  str_ctype = np.fromfile(f, dtype=np.uint8, count=8).tostring()  # worked
  tail = np.fromfile(f, dtype='>i4', count=1)
  print('header=', header, ', jyr=', jyr, ', jmo=', jmo, ', jda=', jda, ', str_ctype=', str_ctype, ', tail=', tail)

  header = np.fromfile(f, dtype='>i4', count=1)
  ict  = np.fromfile(f, dtype='>i4', count=1)
  tail = np.fromfile(f, dtype='>i4', count=1)
  print('header=', header, ', ict=', ict, ', tail=', tail)
  
  header = np.fromfile(f, dtype='>i4', count=1)
  wnum = np.fromfile(f, dtype='>f', count=int(ict))
  tail = np.fromfile(f, dtype='>i4', count=1)
  print('header=', header, ', wnum=', wnum, ', tail=', tail)

  header = np.fromfile(f, dtype='>i4', count=1)
  wsst = np.fromfile(f, dtype='>f', count=int(ict))
  tail = np.fromfile(f, dtype='>i4', count=1)
  print('header=', header, ', wsst=', wsst, ', tail=', tail)

  header = np.fromfile(f, dtype='>i4', count=1)
  wlat = np.fromfile(f, dtype='>f', count=int(ict))
  tail = np.fromfile(f, dtype='>i4', count=1)
  print('header=', header, ', wlat=', wlat, ', tail=', tail)

  header = np.fromfile(f, dtype='>i4', count=1)
  wlon = np.fromfile(f, dtype='>f', count=int(ict))
  tail = np.fromfile(f, dtype='>i4', count=1)
  print('header=', header, ', wlon=', wlon, ', tail=', tail)


if check_sobs_argo:
  f = open(file_sobs_argo, 'rb')
  header = np.fromfile(f, dtype='>i4', count=1)
  jyr = np.fromfile(f, dtype='>i4', count=1)
  jmo = np.fromfile(f, dtype='>i4', count=1)
  jda = np.fromfile(f, dtype='>i4', count=1)
# str_ctype = np.fromfile(f, dtype='>b', count=8).to_string()  # doesn't work this way
# str_ctype = str(np.fromfile(f, dtype='>U8', count=1)) # failed
# str_ctype = str(np.fromfile(f, dtype='S8', count=1))  # worked
  str_ctype = np.fromfile(f, dtype=np.uint8, count=8).tostring()  # worked
  tail = np.fromfile(f, dtype='>i4', count=1)
  print('header=', header, ', jyr=', jyr, ', jmo=', jmo, ', jda=', jda, ', str_ctype=', str_ctype, ', tail=', tail)

  header = np.fromfile(f, dtype='>i4', count=1)
  ict  = np.fromfile(f, dtype='>i4', count=1)
  tail = np.fromfile(f, dtype='>i4', count=1)
  print('header=', header, ', ict=', ict, ', tail=', tail)
  
  header = np.fromfile(f, dtype='>i4', count=1)
  wnum = np.fromfile(f, dtype='>f', count=int(ict))
  tail = np.fromfile(f, dtype='>i4', count=1)
  print('header=', header, ', wnum=', wnum, ', tail=', tail)

  header = np.fromfile(f, dtype='>i4', count=1)
  wsst = np.fromfile(f, dtype='>f', count=int(ict))
  tail = np.fromfile(f, dtype='>i4', count=1)
  print('header=', header, ', wsst=', wsst, ', tail=', tail)

  header = np.fromfile(f, dtype='>i4', count=1)
  wlat = np.fromfile(f, dtype='>f', count=int(ict))
  tail = np.fromfile(f, dtype='>i4', count=1)
  print('header=', header, ', wlat=', wlat, ', tail=', tail)

  header = np.fromfile(f, dtype='>i4', count=1)
  wlon = np.fromfile(f, dtype='>f', count=int(ict))
  tail = np.fromfile(f, dtype='>i4', count=1)
  print('header=', header, ', wlon=', wlon, ', tail=', tail)


if check_sobs_ship:
  f = open(file_sobs_ship, 'rb')
  header = np.fromfile(f, dtype='>i4', count=1)
  jyr = np.fromfile(f, dtype='>i4', count=1)
  jmo = np.fromfile(f, dtype='>i4', count=1)
  jda = np.fromfile(f, dtype='>i4', count=1)
# str_ctype = np.fromfile(f, dtype='>b', count=8).to_string()  # doesn't work this way
# str_ctype = str(np.fromfile(f, dtype='>U8', count=1)) # failed
# str_ctype = str(np.fromfile(f, dtype='S8', count=1))  # worked
  str_ctype = np.fromfile(f, dtype=np.uint8, count=8).tostring()  # worked
  tail = np.fromfile(f, dtype='>i4', count=1)
  print('header=', header, ', jyr=', jyr, ', jmo=', jmo, ', jda=', jda, ', str_ctype=', str_ctype, ', tail=', tail)

  header = np.fromfile(f, dtype='>i4', count=1)
  ict  = np.fromfile(f, dtype='>i4', count=1)
  tail = np.fromfile(f, dtype='>i4', count=1)
  print('header=', header, ', ict=', ict, ', tail=', tail)
  
  header = np.fromfile(f, dtype='>i4', count=1)
  wnum = np.fromfile(f, dtype='>f', count=int(ict))
  tail = np.fromfile(f, dtype='>i4', count=1)
  print('header=', header, ', wnum=', wnum, ', tail=', tail)

  header = np.fromfile(f, dtype='>i4', count=1)
  wsst = np.fromfile(f, dtype='>f', count=int(ict))
  tail = np.fromfile(f, dtype='>i4', count=1)
  print('header=', header, ', wsst=', wsst, ', tail=', tail)

  header = np.fromfile(f, dtype='>i4', count=1)
  wlat = np.fromfile(f, dtype='>f', count=int(ict))
  tail = np.fromfile(f, dtype='>i4', count=1)
  print('header=', header, ', wlat=', wlat, ', tail=', tail)

  header = np.fromfile(f, dtype='>i4', count=1)
  wlon = np.fromfile(f, dtype='>f', count=int(ict))
  tail = np.fromfile(f, dtype='>i4', count=1)
  print('header=', header, ', wlon=', wlon, ', tail=', tail)


if check_obs_sat_dgrid:
  f = open(file_obs_sat_dgrid, 'rb')
  header = np.fromfile(f, dtype='>i', count=1)
  jyr = np.fromfile(f, dtype='>i', count=1)
  jmo = np.fromfile(f, dtype='>i', count=1)
  jda = np.fromfile(f, dtype='>i', count=1)
  data = np.fromfile(f, dtype='>f', count=QUART_DEG_LON_DIM*QUART_DEG_LAT_DIM)
  sst_obs = np.reshape(data, (QUART_DEG_LAT_DIM, QUART_DEG_LON_DIM), order='C')
  tail= np.fromfile(f, dtype='>i', count=1)
  print('header=', header, ', jyr=', jyr, ', jmo=', jmo, ', jda=', jda, ', tail=', tail)

  lon = np.arange(start=LON_START, stop=360., step=GRID_SIZE)
  lat = np.arange(start=LAT_START, stop=90. , step=GRID_SIZE)

# da_sst_obs = xr.DataArray(sst_obs, name='sst', dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}, attrs=dict(_FillValue=-9999, units="degC") )
# da_sst_obs.to_dataset(name='sst').to_netcdf('./dgrid.sst.20160119_satA_Ligang.nc', encoding={'lat': {'_FillValue': None}, 'lon': {'_FillValue': None}}) 

  da_sst_obs = xr.DataArray(sst_obs, name='sst', dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}, attrs=dict(_FillValue=-9999, units="degC") )
  da_sst_obs.to_dataset(name='sst').to_netcdf('./day_2D.nc', encoding={'lat': {'_FillValue': None}, 'lon': {'_FillValue': None}})

  da_tmp = xr.where(da_sst_obs == -9999, 0, 1)
  num_obs = da_tmp.sum().values
  print('For obs_sat_dgrid, num_obs=', num_obs)  
  # printed 59595 for 17, (60261 for 18, 54972 for 19), for sea_ict=691150, it's 8.62%; 


if check_obs_sat_ngrid:
  f = open(file_obs_sat_ngrid, 'rb')
  header = np.fromfile(f, dtype='>i', count=1)
  jyr = np.fromfile(f, dtype='>i', count=1)
  jmo = np.fromfile(f, dtype='>i', count=1)
  jda = np.fromfile(f, dtype='>i', count=1)
  data = np.fromfile(f, dtype='>f', count=QUART_DEG_LON_DIM*QUART_DEG_LAT_DIM)
  sst_obs = np.reshape(data, (QUART_DEG_LAT_DIM, QUART_DEG_LON_DIM), order='C')
  tail= np.fromfile(f, dtype='>i', count=1)
  print('header=', header, ', jyr=', jyr, ', jmo=', jmo, ', jda=', jda, ', tail=', tail)

  lon = np.arange(start=LON_START, stop=360., step=GRID_SIZE)
  lat = np.arange(start=LAT_START, stop=90. , step=GRID_SIZE)

# da_sst_obs = xr.DataArray(sst_obs, name='sst', dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}, attrs=dict(_FillValue=-9999, units="degC") )
# da_sst_obs.to_dataset(name='sst').to_netcdf('./ngrid.sst.20160119_satA_Ligang.nc', encoding={'lat': {'_FillValue': None}, 'lon': {'_FillValue': None}}) 

  da_sst_obs = xr.DataArray(sst_obs, name='sst', dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}, attrs=dict(_FillValue=-9999, units="degC") )
  da_sst_obs.to_dataset(name='sst').to_netcdf('./night_2D.nc', encoding={'lat': {'_FillValue': None}, 'lon': {'_FillValue': None}}) 

  da_tmp = xr.where(da_sst_obs == -9999, 0, 1)
  num_obs = da_tmp.sum().values
  print('For obs_sat_ngrid, num_obs=', num_obs)  
  # printed 78311 for 17, (77998 for 18, 75086 for 19), for sea_ict=691150, it's 11.33%; 


if check_dsobs_satA:
  f = open(file_dsobs_satA, 'rb')
  header = np.fromfile(f, dtype='>i4', count=1)
  jyr = np.fromfile(f, dtype='>i4', count=1)
  jmo = np.fromfile(f, dtype='>i4', count=1)
  jda = np.fromfile(f, dtype='>i4', count=1)
# str_ctype = np.fromfile(f, dtype='>b', count=8).to_string()  # doesn't work this way
# str_ctype = str(np.fromfile(f, dtype='>U8', count=1)) # failed
# str_ctype = str(np.fromfile(f, dtype='S8', count=1))  # worked
  str_ctype = np.fromfile(f, dtype=np.uint8, count=8).tostring()  # worked
  tail = np.fromfile(f, dtype='>i4', count=1)
  print('header=', header, ', jyr=', jyr, ', jmo=', jmo, ', jda=', jda, ', str_ctype=', str_ctype, ', tail=', tail)

  # Now write out the 2D grid NetCDF file
  lon = np.arange(start=LON_START, stop=360., step=GRID_SIZE)
  lat = np.arange(start=LAT_START, stop=90. , step=GRID_SIZE)

  sst_obs = -9999*np.ones((QUART_DEG_LAT_DIM, QUART_DEG_LON_DIM), np.float32)
  da_sst_obs = xr.DataArray(sst_obs, name='sst', dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}, attrs=dict(_FillValue=-9999, units="degC") )

  header = np.fromfile(f, dtype='>i4', count=1)
  ict  = np.fromfile(f, dtype='>i4', count=1)
  ict  = ict[0]
  tail = np.fromfile(f, dtype='>i4', count=1)
  print('header=', header, ', ict=', ict, ', tail=', tail)

  while 0 < ict and ict <= 10000: 
    header = np.fromfile(f, dtype='>i4', count=1)
    wnum = np.fromfile(f, dtype='>f', count=int(ict))
    tail = np.fromfile(f, dtype='>i4', count=1)
    print('header=', header, ', wnum=', wnum, ', tail=', tail)

    header = np.fromfile(f, dtype='>i4', count=1)
    wsst = np.fromfile(f, dtype='>f', count=int(ict))
    tail = np.fromfile(f, dtype='>i4', count=1)
    print('header=', header, ', wsst=', wsst, ', tail=', tail)

    header = np.fromfile(f, dtype='>i4', count=1)
    wlat = np.fromfile(f, dtype='>f', count=int(ict))
    tail = np.fromfile(f, dtype='>i4', count=1)
    print('header=', header, ', wlat=', wlat, ', tail=', tail)

    header = np.fromfile(f, dtype='>i4', count=1)
    wlon = np.fromfile(f, dtype='>f', count=int(ict))
    tail = np.fromfile(f, dtype='>i4', count=1)
    print('header=', header, ', wlon=', wlon, ', tail=', tail)

    for i in range(ict):
      da_sst_obs.loc[dict(lat=wlat[i], lon=wlon[i])] = wsst[i]  # has to have 'dict' or failed.

    header = np.fromfile(f, dtype='>i4', count=1)
    if header.size < 1:  # end of file
      ict = 0
    else:
      ict  = np.fromfile(f, dtype='>i4', count=1)
      ict  = ict[0]
      tail = np.fromfile(f, dtype='>i4', count=1)
      print('header=', header, ', ict=', ict, ', tail=', tail)
    

  da_tmp = xr.where(da_sst_obs == -9999, 0, 1)
  num_obs = da_tmp.sum().values
  print('For sat_dsobs_grid, num_obs=', num_obs) 

# da_sst_obs.to_dataset(name='sst').to_netcdf('./dsobs.20160117_satA_Ligang_correct.nc', encoding={'lat': {'_FillValue': None}, 'lon': {'_FillValue': None}})
  da_sst_obs.to_dataset(name='sst').to_netcdf('./day_1D.nc', encoding={'lat': {'_FillValue': None}, 'lon': {'_FillValue': None}})


if check_nsobs_satA:
  f = open(file_nsobs_satA, 'rb')
  header = np.fromfile(f, dtype='>i4', count=1)
  jyr = np.fromfile(f, dtype='>i4', count=1)
  jmo = np.fromfile(f, dtype='>i4', count=1)
  jda = np.fromfile(f, dtype='>i4', count=1)
# str_ctype = np.fromfile(f, dtype='>b', count=8).to_string()  # doesn't work this way
# str_ctype = str(np.fromfile(f, dtype='>U8', count=1)) # failed
# str_ctype = str(np.fromfile(f, dtype='S8', count=1))  # worked
  str_ctype = np.fromfile(f, dtype=np.uint8, count=8).tostring()  # worked
  tail = np.fromfile(f, dtype='>i4', count=1)
  print('header=', header, ', jyr=', jyr, ', jmo=', jmo, ', jda=', jda, ', str_ctype=', str_ctype, ', tail=', tail)

  # Now write out the 2D grid NetCDF file
  lon = np.arange(start=LON_START, stop=360., step=GRID_SIZE)
  lat = np.arange(start=LAT_START, stop=90. , step=GRID_SIZE)

  sst_obs = -9999*np.ones((QUART_DEG_LAT_DIM, QUART_DEG_LON_DIM), np.float32)
  da_sst_obs = xr.DataArray(sst_obs, name='sst', dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}  \
    , attrs=dict(_FillValue=-9999, units="degC") )

  header = np.fromfile(f, dtype='>i4', count=1)
  ict  = np.fromfile(f, dtype='>i4', count=1)
  ict  = ict[0]
  print('ict type: ', type(ict))
  tail = np.fromfile(f, dtype='>i4', count=1)
  print('header=', header, ', ict=', ict, ', tail=', tail)

  while 0 < ict and ict <= 10000:  
    header = np.fromfile(f, dtype='>i4', count=1)
    wnum = np.fromfile(f, dtype='>f', count=int(ict))
    tail = np.fromfile(f, dtype='>i4', count=1)
    print('header=', header, ', wnum=', wnum, ', tail=', tail)

    header = np.fromfile(f, dtype='>i4', count=1)
    wsst = np.fromfile(f, dtype='>f', count=int(ict))
    tail = np.fromfile(f, dtype='>i4', count=1)
    print('header=', header, ', wsst=', wsst, ', tail=', tail)

    header = np.fromfile(f, dtype='>i4', count=1)
    wlat = np.fromfile(f, dtype='>f', count=int(ict))
    tail = np.fromfile(f, dtype='>i4', count=1)
    print('header=', header, ', wlat=', wlat, ', tail=', tail)

    header = np.fromfile(f, dtype='>i4', count=1)
    wlon = np.fromfile(f, dtype='>f', count=int(ict))
    tail = np.fromfile(f, dtype='>i4', count=1)
    print('header=', header, ', wlon=', wlon, ', tail=', tail)

    for i in range(ict):
      da_sst_obs.loc[dict(lat=wlat[i], lon=wlon[i])] = wsst[i]  # has to have 'dict' or failed.

    header = np.fromfile(f, dtype='>i4', count=1)
    if header.size < 1:
      ict = 0
    else:
      ict  = np.fromfile(f, dtype='>i4', count=1)
      ict  = ict[0]
      print('ict type: ', type(ict))
      tail = np.fromfile(f, dtype='>i4', count=1)
      print('header=', header, ', ict=', ict, ', tail=', tail)

  da_tmp = xr.where(da_sst_obs == -9999, 0, 1)
  num_obs = da_tmp.sum().values
  print('For sat_nsobs_grid, num_obs=', num_obs) 

# da_sst_obs.to_dataset(name='sst').to_netcdf('./nsobs.20160117_satA_Ligang_correct.nc', encoding={'lat': {'_FillValue': None}, 'lon': {'_FillValue': None}})
  da_sst_obs.to_dataset(name='sst').to_netcdf('./night_1D.nc', encoding={'lat': {'_FillValue': None}, 'lon': {'_FillValue': None}})


if check_icecon720x360:
  f = open(file_icecon720x360, 'rb')
  header = np.fromfile(f, dtype='>i', count=1)
  jyr = np.fromfile(f, dtype='>i', count=1)
  jmo = np.fromfile(f, dtype='>i', count=1)
  jda = np.fromfile(f, dtype='>i', count=1)
  data = np.fromfile(f, dtype='>f', count=HALF_DEG_LON_DIM*HALF_DEG_LAT_DIM)
  sst_obs = np.reshape(data, (HALF_DEG_LAT_DIM, HALF_DEG_LON_DIM), order='C')
  tail= np.fromfile(f, dtype='>i', count=1)
  print('header=', header, ', jyr=', jyr, ', jmo=', jmo, ', jda=', jda, ', tail=', tail)

  lon = np.arange(start=HALF_DEG_LON_START, stop=359.5, step=HALF_DEG_GRID_SIZE)
  lat = np.arange(start=HALF_DEG_LAT_START, stop=89.5 , step=HALF_DEG_GRID_SIZE)

  da_sst_obs = xr.DataArray(sst_obs, name='ice_con', dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}  \
    , attrs=dict(_FillValue=-1.0E30) )
  da_sst_obs.to_dataset(name='ice_con').to_netcdf('./icecon720x360.20160117_Ligang.nc') 

  da_tmp = xr.where(da_sst_obs == -1.0E30, 0, 1)
  num_obs = da_tmp.sum().values
  print('For icecon720x360, num_obs=', num_obs) # printed 


if check_icecon:
  f = open(file_icecon, 'rb')
  header = np.fromfile(f, dtype='>i', count=1)
  jyr = np.fromfile(f, dtype='>i', count=1)
  jmo = np.fromfile(f, dtype='>i', count=1)
  jda = np.fromfile(f, dtype='>i', count=1)
  data = np.fromfile(f, dtype='>f', count=quart_deg_lon_dim*quart_deg_lat_dim)
  sst_obs = np.reshape(data, (quart_deg_lat_dim, quart_deg_lon_dim), order='c')
  tail= np.fromfile(f, dtype='>i', count=1)
  print('header=', header, ', jyr=', jyr, ', jmo=', jmo, ', jda=', jda, ', tail=', tail)

  lon = np.arange(start=lon_start, stop=360., step=grid_size)
  lat = np.arange(start=lat_start, stop=90. , step=grid_size)

  da_sst_obs = xr.dataarray(sst_obs, name='ice_con', dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}  \
    , attrs=dict(_fillvalue=-1.0e30) )
  da_sst_obs.to_dataset(name='ice_con').to_netcdf('./icecon.20160117_ligang.nc') 

  da_tmp = xr.where(da_sst_obs == -1.0e30, 0, 1)
  num_obs = da_tmp.sum().values
  print('for icecon, num_obs=', num_obs)  # printed 130701


if check_icemask:  # range, about 0 ~ 36
  with open(file_icemask, 'rb') as f:
    header = np.fromfile(f, dtype='>i4', count=1)
    data = np.fromfile(f, dtype='>f', count=QUART_DEG_LON_DIM*QUART_DEG_LAT_DIM)
    icemask = np.reshape(data, (QUART_DEG_LAT_DIM, QUART_DEG_LON_DIM), order='C')
    tail = np.fromfile(f, dtype='>i4', count=1)
    print('header=', header, ', tail=', tail)

  lon = np.arange(start=LON_START, stop=360., step=GRID_SIZE)
  lat = np.arange(start=LAT_START, stop=90. , step=GRID_SIZE)

  da_mask = xr.DataArray(icemask, name='icemask', dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon})
  da_mask.to_dataset(name='icemask').to_netcdf('./ice_flags_mask.dat_Ligang.nc')


if check_frzpnt:  # freeze point, -2.0134 ~ -1.617
  f = open(file_frzpnt, 'rb')
  header = np.fromfile(f, dtype='>i', count=1)
  kkdy = np.fromfile(f, dtype='>i', count=1)
  kkmo = np.fromfile(f, dtype='>i', count=1)
  kkda = np.fromfile(f, dtype='>i', count=1)
  data = np.fromfile(f, dtype='>f', count=QUART_DEG_LON_DIM*QUART_DEG_LAT_DIM)
  frzpnt = np.reshape(data, (QUART_DEG_LAT_DIM, QUART_DEG_LON_DIM), order='C')
  tail = np.fromfile(f, dtype='>i', count=1)
  print('header=', header, ', kkdy=', kkdy, ', kkmo=', kkmo, ', kkda=', kkda, ', tail=', tail)

  lon = np.arange(start=LON_START, stop=360., step=GRID_SIZE)
  lat = np.arange(start=LAT_START, stop=90. , step=GRID_SIZE)

  da_frzpnt = xr.DataArray(frzpnt, name='frzpnt', dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}  \
    , attrs=dict(_FillValue=-9999, units="degC") )
  da_frzpnt.to_dataset(name='frzpnt').to_netcdf('./daily-fzsst.0117_Ligang.nc') 

  da_tmp = xr.where(da_frzpnt == -9999, 0, 1)
  num_obs = da_tmp.sum().values
  print('For frzpnt, num_obs=', num_obs)  # printed 691150


if check_icecon7:  # 0.15 ~ 1.0
  day = np.arange(1, 7+1, dtype='i')
  lon = np.arange(start=LON_START, stop=360., step=GRID_SIZE, dtype='f')
  lat = np.arange(start=LAT_START, stop=90. , step=GRID_SIZE, dtype='f')
  icecon7 = np.zeros((7, QUART_DEG_LAT_DIM, QUART_DEG_LON_DIM), dtype='f')

  f = open(file_icecon7, 'rb')
  for i_day in range(1, 7+1):
    header = np.fromfile(f, dtype='>i4', count=1)
    kyr = np.fromfile(f, dtype='>i4', count=1)
    kmo = np.fromfile(f, dtype='>i4', count=1)
    kda = np.fromfile(f, dtype='>i4', count=1)
    data = np.fromfile(f, dtype='>f', count=QUART_DEG_LON_DIM*QUART_DEG_LAT_DIM)
    icecon7[i_day-1, :, :] = np.reshape(data, (QUART_DEG_LAT_DIM, QUART_DEG_LON_DIM), order='C')
    tail = np.fromfile(f, dtype='>i4', count=1)
    print('i_day=', i_day, 'header=', header, ', tail=', tail, 'kyr=', kyr, ', kmo=', kmo, ', kda=', kda)

  da_icecon7 = xr.DataArray(icecon7, name='icecon7', dims=('day', 'lat', 'lon')  \
    , coords={'day': day, 'lat': lat, 'lon': lon}, attrs=dict(_FillValue=-1.0E30))
  da_icecon7.to_dataset(name='icecon7').to_netcdf('./icecon7.20160117_Ligang.nc')


if check_iceConMed:  # 0.0~1.0
  f = open(file_iceConMed, 'rb')
  header = np.fromfile(f, dtype='>i', count=1)
  jyr = np.fromfile(f, dtype='>i', count=1)
  jmo = np.fromfile(f, dtype='>i', count=1)
  jda = np.fromfile(f, dtype='>i', count=1)
  data = np.fromfile(f, dtype='>f', count=QUART_DEG_LON_DIM*QUART_DEG_LAT_DIM)
  iceConMed = np.reshape(data, (QUART_DEG_LAT_DIM, QUART_DEG_LON_DIM), order='c')
  tail= np.fromfile(f, dtype='>i', count=1)
  print('header=', header, ', jyr=', jyr, ', jmo=', jmo, ', jda=', jda, ', tail=', tail)

  lon = np.arange(start=LON_START, stop=360., step=GRID_SIZE)
  lat = np.arange(start=LAT_START, stop=90. , step=GRID_SIZE)

  da_iceConMed = xr.DataArray(iceConMed, name='iceConMed', dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}  \
    , attrs=dict(_FillValue=-9999) )
  da_iceConMed.to_dataset(name='iceConMed').to_netcdf('./icecon-med.20160117_ligang.nc') 

  da_tmp = xr.where(da_iceConMed == -9999, 0, 1)
  num_obs = da_tmp.sum().values
  print('for iceConMed, num_obs=', num_obs)  # printed 1036800


if check_icesst_grid:  # -1.8 ~ -1.6
  f = open(file_icesst_grid, 'rb')
  header = np.fromfile(f, dtype='>i', count=1)
  jyr = np.fromfile(f, dtype='>i', count=1)
  jmo = np.fromfile(f, dtype='>i', count=1)
  jda = np.fromfile(f, dtype='>i', count=1)
  data = np.fromfile(f, dtype='>f', count=QUART_DEG_LON_DIM*QUART_DEG_LAT_DIM)
  icesst = np.reshape(data, (QUART_DEG_LAT_DIM, QUART_DEG_LON_DIM), order='c')
  tail= np.fromfile(f, dtype='>i', count=1)
  print('header=', header, ', jyr=', jyr, ', jmo=', jmo, ', jda=', jda, ', tail=', tail)

  lon = np.arange(start=LON_START, stop=360., step=GRID_SIZE)
  lat = np.arange(start=LAT_START, stop=90. , step=GRID_SIZE)

  da_icesst = xr.DataArray(icesst, name='icesst', dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}  \
    , attrs=dict(_FillValue=-9999) )
  da_icesst.to_dataset(name='icesst').to_netcdf('./icesst.20160117_grid_ligang.nc') 

  da_tmp = xr.where(da_icesst == -9999, 0, 1)
  num_obs = da_tmp.sum().values
  print('for icesst_grid, num_obs=', num_obs)  # printed 118616


if check_icesst_sobs:
  f = open(file_icesst_sobs, 'rb')
  header = np.fromfile(f, dtype='>i4', count=1)
  jyr = np.fromfile(f, dtype='>i4', count=1)
  jmo = np.fromfile(f, dtype='>i4', count=1)
  jda = np.fromfile(f, dtype='>i4', count=1)
# str_ctype = str(np.fromfile(f, dtype='S8', count=1))  # worked
  str_ctype = np.fromfile(f, dtype=np.uint8, count=8).tostring()  # worked
  tail = np.fromfile(f, dtype='>i4', count=1)
  print('header=', header, ', jyr=', jyr, ', jmo=', jmo, ', jda=', jda, ', str_ctype=', str_ctype, ', tail=', tail)

  # Now write out the 2D grid NetCDF file
  lon = np.arange(start=LON_START, stop=360., step=GRID_SIZE)
  lat = np.arange(start=LAT_START, stop=90. , step=GRID_SIZE)

  sst_obs = -9999*np.ones((QUART_DEG_LAT_DIM, QUART_DEG_LON_DIM), np.float32)
  da_sst_obs = xr.DataArray(sst_obs, name='sst', dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}  \
    , attrs=dict(_FillValue=-9999, units="degC") )

  header = np.fromfile(f, dtype='>i4', count=1)
  ict  = np.fromfile(f, dtype='>i4', count=1)
  ict  = ict[0]
  tail = np.fromfile(f, dtype='>i4', count=1)
  print('header=', header, ', ict=', ict, ', tail=', tail)

  while 0 < ict and ict <= 10000: 
    header = np.fromfile(f, dtype='>i4', count=1)
    wnum = np.fromfile(f, dtype='>f', count=int(ict))
    tail = np.fromfile(f, dtype='>i4', count=1)
    print('header=', header, ', wnum=', wnum, ', tail=', tail)

    header = np.fromfile(f, dtype='>i4', count=1)
    wsst = np.fromfile(f, dtype='>f', count=int(ict))
    tail = np.fromfile(f, dtype='>i4', count=1)
    print('header=', header, ', wsst=', wsst, ', tail=', tail)

    header = np.fromfile(f, dtype='>i4', count=1)
    wlat = np.fromfile(f, dtype='>f', count=int(ict))
    tail = np.fromfile(f, dtype='>i4', count=1)
    print('header=', header, ', wlat=', wlat, ', tail=', tail)

    header = np.fromfile(f, dtype='>i4', count=1)
    wlon = np.fromfile(f, dtype='>f', count=int(ict))
    tail = np.fromfile(f, dtype='>i4', count=1)
    print('header=', header, ', wlon=', wlon, ', tail=', tail)

    for i in range(ict):
      da_sst_obs.loc[dict(lat=wlat[i], lon=wlon[i])] = wsst[i]  # has to have 'dict' or failed.

    header = np.fromfile(f, dtype='>i4', count=1)
    if header.size < 1:  # end of file
      ict = 0
    else:
      ict  = np.fromfile(f, dtype='>i4', count=1)
      ict  = ict[0]
      tail = np.fromfile(f, dtype='>i4', count=1)
      print('header=', header, ', ict=', ict, ', tail=', tail)
    

  da_tmp = xr.where(da_sst_obs == -9999, 0, 1)
  num_obs = da_tmp.sum().values
  print('For icesst_sobs, num_obs=', num_obs)  # printed 118616

  da_sst_obs.to_dataset(name='sst').to_netcdf('./icesst.20160117_Ligang_sobs.nc')


if check_mask_twodeg:
  rec = np.arange(1, 3+1, dtype='i')
  lon = np.arange(start=1, stop=360., step=2, dtype='f')
  lat = np.arange(start=-89, stop=90. , step=2, dtype='f')
  mask_twodeg = np.zeros((3, 90, 180), dtype='f')

  f = open(file_mask_twodeg, 'rb')
  for i_rec in range(1, 3+1):
    header = np.fromfile(f, dtype='>i4', count=1)
    data = np.fromfile(f, dtype='>f', count=90*180)
    mask_twodeg[i_rec-1, :, :] = np.reshape(data, (90, 180), order='C')
    tail = np.fromfile(f, dtype='>i4', count=1)
    print('i_rec=', i_rec, 'header=', header, ', tail=', tail)

  da_mask_twodeg = xr.DataArray(mask_twodeg, name='mask_twodeg', dims=('rec', 'lat', 'lon')  \
    , coords={'rec': rec, 'lat': lat, 'lon': lon})
  da_mask_twodeg.to_dataset(name='mask_twodeg').to_netcdf('./lstags.twodeg_Ligang.nc')


if check_climatology:  # -1.8 ~ 31.7
  month = np.arange(1, MONTHS_IN_YEAR+1, dtype='i')
  lon = np.arange(start=ONE_DEG_LON_START, stop=359., step=ONE_DEG_GRID_SIZE, dtype='f')
  lat = np.arange(start=ONE_DEG_LAT_START, stop=90. , step=ONE_DEG_GRID_SIZE, dtype='f')
  climatology = np.zeros((MONTHS_IN_YEAR, ONE_DEG_LAT_DIM, ONE_DEG_LON_DIM), dtype='f')

  f = open(file_climatology, 'rb')
  for i_mon in range(1, MONTHS_IN_YEAR+1):
    header = np.fromfile(f, dtype='>i4', count=1)
    base_year  = np.fromfile(f, dtype='>i4', count=1)
    base_month = np.fromfile(f, dtype='>i4', count=1)
    data       = np.fromfile(f, dtype='>f', count=ONE_DEG_LON_DIM*ONE_DEG_LAT_DIM)
    climatology[i_mon-1, :, :] = np.reshape(data, (ONE_DEG_LAT_DIM, ONE_DEG_LON_DIM), order='C')
    tail = np.fromfile(f, dtype='>i4', count=1)
    print('i_mon=', i_mon, 'header=', header, ', tail=', tail, 'base_year=', base_year, ', base_month=', base_month)

  da_climatology = xr.DataArray(climatology, name='sst_climatology', dims=('month', 'lat', 'lon')  \
    , coords={'month': month, 'lat': lat, 'lon': lon}, attrs=dict(_FillValue=-9999, units="degC") )
  da_climatology.to_dataset(name='sst_climatology').to_netcdf('./clim.71.00.gdat_Ligang_sst_climatology.nc')


if check_eot_mode:  # 2-deg grid
  MAX_EOT_MODES = 130
  rec = np.arange(1, MAX_EOT_MODES+1, dtype='i')
  lon = np.arange(start=1, stop=360., step=2, dtype='f')
  lat = np.arange(start=-89, stop=90. , step=2, dtype='f')
  eot_mode = np.zeros((MAX_EOT_MODES, 90, 180), dtype='f')

  f = open(file_eot_mode, 'rb')
  for i_rec in range(1, MAX_EOT_MODES+1):
    header = np.fromfile(f, dtype='>i4', count=1)
    data = np.fromfile(f, dtype='>f', count=90*180)
    eot_mode[i_rec-1, :, :] = np.reshape(data, (90, 180), order='C')
    tail = np.fromfile(f, dtype='>i4', count=1)
    print('i_rec=', i_rec, 'header=', header, ', tail=', tail)

  da_eot_mode = xr.DataArray(eot_mode, name='eot_mode', dims=('rec', 'lat', 'lon')  \
    , coords={'rec': rec, 'lat': lat, 'lon': lon})
  da_eot_mode.to_dataset(name='eot_mode').to_netcdf('./eot6.damp-zero.ev130.ano.dat_Ligang_eot_mode.nc')


if check_eot_weight:
  MAX_EOT_MODES = 130
  NUM_ANOMALIES = 2
  GRID_LAT_DIM = 90
  n_eot_mode = np.arange(1, MAX_EOT_MODES+1, dtype='i')
  n_anomaly  = np.arange(1, NUM_ANOMALIES+1, dtype='f')
  lat = np.arange(start=-89, stop=90. , step=2, dtype='f')

  f = open(file_eot_weight, 'rb')
  header = np.fromfile(f, dtype='>i4', count=1)
  jyr = np.fromfile(f, dtype='>i4', count=1)
  jmo = np.fromfile(f, dtype='>i4', count=1)
  jda = np.fromfile(f, dtype='>i4', count=1)
  data = np.fromfile(f, dtype='>f', count=MAX_EOT_MODES*NUM_ANOMALIES)
  eot_weight= np.reshape(data, (MAX_EOT_MODES, NUM_ANOMALIES), order='C')  # -0.08 ~ 0.51
  data = np.fromfile(f, dtype='>f', count=MAX_EOT_MODES)
  supported_modes = np.reshape(data, (MAX_EOT_MODES), order='C')  # all 0.0?
  data = np.fromfile(f, dtype='>f', count=GRID_LAT_DIM)
  smoothed_zonal_biasses = np.reshape(data, (GRID_LAT_DIM), order='C')  # about -0.15 ~ 0.37
  tail = np.fromfile(f, dtype='>i4', count=1)
  print('header=', header, ', jyr=', jyr, ', jmo=', jmo, ', jda=', jda, ', tail=', tail)
  
  da_eot_weight = xr.DataArray(eot_weight, name='eot_weight', dims=('n_eot_mode', 'n_anomaly')  \
    , coords={'n_eot_mode': n_eot_mode, 'n_anomaly': n_anomaly})
  da_supported_modes = xr.DataArray(supported_modes, name='supported_modes', dims=('n_eot_mode')  \
    , coords={'n_eot_mode': n_eot_mode})
  da_smoothed_zonal_biasses = xr.DataArray(smoothed_zonal_biasses, name='smoothed_zonal_biasses', dims=('lat')  \
    , coords={'lat': lat})

  ds_eot_weight = xr.merge([da_eot_weight, da_supported_modes, da_smoothed_zonal_biasses])
  ds_eot_weight.to_netcdf('./nwt.20160117_Ligang_eot_weight.nc')


if check_mode_variance:
  NUM_EOT_MODES = 130

  f = open(file_mode_variance, 'rb')
  header = np.fromfile(f, dtype='>i', count=1)
  data = np.fromfile(f, dtype='>f', count=NUM_EOT_MODES)
  mode_variance = np.reshape(data, (NUM_EOT_MODES), order='c')
  tail= np.fromfile(f, dtype='>i', count=1)
  print('header=', header, ', tail=', tail, 'mode_variance=', mode_variance)

  n_eot_mode = np.arange(1, NUM_EOT_MODES+1, dtype='i')
  da_mode_variance = xr.DataArray(mode_variance, name='mode_variance', dims=['n_eot_mode'], coords={'n_eot_mode': n_eot_mode} )
  da_mode_variance.to_dataset(name='mode_variance').to_netcdf('./var-mode_ligang.nc') 

  da_tmp = xr.where(da_mode_variance == -9999, 0, 1)
  num_obs = da_tmp.sum().values
  print('for mode_variance, num_obs=', num_obs)  # printed 130, no missing_value.


if check_eot_bias:
  lat = np.arange(start=-89, stop=90. , step=2, dtype='f')
  lon = np.arange(start=1, stop=360 , step=2, dtype='f')

  f = open(file_eot_bias, 'rb')

  header = np.fromfile(f, dtype='>i4', count=1)
  jyr = np.fromfile(f, dtype='>i4', count=1)
  jmo = np.fromfile(f, dtype='>i4', count=1)
  jda = np.fromfile(f, dtype='>i4', count=1)
  data = np.fromfile(f, dtype='>f', count=90*180)
  bias_corrections = np.reshape(data, (90, 180), order='C')
  tail = np.fromfile(f, dtype='>i4', count=1)
  print('header=', header, ', jyr=', jyr, ', jmo=', jmo, ', jda=', jda, ', tail=', tail, ', bias_corrections=', bias_corrections)

  header = np.fromfile(f, dtype='>i4', count=1)
  jyr = np.fromfile(f, dtype='>i4', count=1)
  jmo = np.fromfile(f, dtype='>i4', count=1)
  jda = np.fromfile(f, dtype='>i4', count=1)
  data = np.fromfile(f, dtype='>f', count=90*180)
  avg_situ_super_obs = np.reshape(data, (90, 180), order='C')
  tail = np.fromfile(f, dtype='>i4', count=1)
  print('\n header=', header, ', jyr=', jyr, ', jmo=', jmo, ', jda=', jda, ', tail=', tail, ', avg_situ_super_obs=', avg_situ_super_obs)

  header = np.fromfile(f, dtype='>i4', count=1)
  jyr = np.fromfile(f, dtype='>i4', count=1)
  jmo = np.fromfile(f, dtype='>i4', count=1)
  jda = np.fromfile(f, dtype='>i4', count=1)
  data = np.fromfile(f, dtype='>f', count=90*180)
  smoothed_avg_sat_super_obs = np.reshape(data, (90, 180), order='C')
  tail = np.fromfile(f, dtype='>i4', count=1)
  print('\n header=', header, ', jyr=', jyr, ', jmo=', jmo, ', jda=', jda, ', tail=', tail, ', smoothed_avg_sat_super_obs=', smoothed_avg_sat_super_obs)

  header = np.fromfile(f, dtype='>i4', count=1)
  jyr = np.fromfile(f, dtype='>i4', count=1)
  jmo = np.fromfile(f, dtype='>i4', count=1)
  jda = np.fromfile(f, dtype='>i4', count=1)
  data = np.fromfile(f, dtype='>f', count=90*180)
  variance_errors = np.reshape(data, (90, 180), order='C')
  tail = np.fromfile(f, dtype='>i4', count=1)
  print('\n header=', header, ', jyr=', jyr, ', jmo=', jmo, ', jda=', jda, ', tail=', tail, ', variance_errors=', variance_errors)

  da_bias_corrections = xr.DataArray(bias_corrections, name='bias_corrections', dims=('lat', 'lon')  \
    , coords={'lat': lat, 'lon': lon}, attrs=dict(_FillValue=-9999) )
  da_avg_situ_super_obs = xr.DataArray(avg_situ_super_obs, name='avg_situ_super_obs', dims=('lat', 'lon')  \
    , coords={'lat': lat, 'lon': lon}, attrs=dict(_FillValue=-9999) )
  da_smoothed_avg_sat_super_obs = xr.DataArray(smoothed_avg_sat_super_obs, name='smoothed_avg_sat_super_obs', dims=('lat', 'lon')  \
    , coords={'lat': lat, 'lon': lon}, attrs=dict(_FillValue=-9999) )
  da_variance_errors = xr.DataArray(variance_errors, name='variance_errors', dims=('lat', 'lon')  \
    , coords={'lat': lat, 'lon': lon} )

  ds_eot_bias = xr.merge([da_bias_corrections, da_avg_situ_super_obs, da_smoothed_avg_sat_super_obs, da_variance_errors])
  ds_eot_bias.to_netcdf('./nbias.20160117_Ligang_eot_bias.nc')


if check_nsobsc_sat:
  f = open(file_nsobsc_sat, 'rb')
  header = np.fromfile(f, dtype='>i4', count=1)
  jyr = np.fromfile(f, dtype='>i4', count=1)
  jmo = np.fromfile(f, dtype='>i4', count=1)
  jda = np.fromfile(f, dtype='>i4', count=1)
# str_ctype = np.fromfile(f, dtype='>b', count=8).to_string()  # doesn't work this way
# str_ctype = str(np.fromfile(f, dtype='>U8', count=1)) # failed
# str_ctype = str(np.fromfile(f, dtype='S8', count=1))  # worked
  str_ctype = np.fromfile(f, dtype=np.uint8, count=8).tostring()  # worked
  tail = np.fromfile(f, dtype='>i4', count=1)
  print('header=', header, ', jyr=', jyr, ', jmo=', jmo, ', jda=', jda, ', str_ctype=', str_ctype, ', tail=', tail)

  # For writing out the 2D grid NetCDF file
  lon = np.arange(start=LON_START, stop=360., step=GRID_SIZE)
  lat = np.arange(start=LAT_START, stop=90. , step=GRID_SIZE)

  sst_obs = -9999*np.ones((QUART_DEG_LAT_DIM, QUART_DEG_LON_DIM), np.float32)
  da_sst_obs = xr.DataArray(sst_obs, name='sst', dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}  \
    , attrs=dict(_FillValue=-9999, units="degC") )

  header = np.fromfile(f, dtype='>i4', count=1)
  ict  = np.fromfile(f, dtype='>i4', count=1)
  ict  = ict[0]
  print('ict type: ', type(ict))
  tail = np.fromfile(f, dtype='>i4', count=1)
  print('header=', header, ', ict=', ict, ', tail=', tail)

  while 0 < ict and ict <= 10000:  
    header = np.fromfile(f, dtype='>i4', count=1)
    wnum = np.fromfile(f, dtype='>f', count=int(ict))
    tail = np.fromfile(f, dtype='>i4', count=1)
    print('header=', header, ', wnum=', wnum, ', tail=', tail)

    header = np.fromfile(f, dtype='>i4', count=1)
    wsst = np.fromfile(f, dtype='>f', count=int(ict))
    tail = np.fromfile(f, dtype='>i4', count=1)
    print('header=', header, ', wsst=', wsst, ', tail=', tail)

    header = np.fromfile(f, dtype='>i4', count=1)
    wlat = np.fromfile(f, dtype='>f', count=int(ict))
    tail = np.fromfile(f, dtype='>i4', count=1)
    print('header=', header, ', wlat=', wlat, ', tail=', tail)

    header = np.fromfile(f, dtype='>i4', count=1)
    wlon = np.fromfile(f, dtype='>f', count=int(ict))
    tail = np.fromfile(f, dtype='>i4', count=1)
    print('header=', header, ', wlon=', wlon, ', tail=', tail)

    for i in range(ict):
      da_sst_obs.loc[dict(lat=wlat[i], lon=wlon[i])] = wsst[i]  # has to have 'dict' or failed.

    header = np.fromfile(f, dtype='>i4', count=1)
    if header.size < 1:
      ict = 0
    else:
      ict  = np.fromfile(f, dtype='>i4', count=1)
      ict  = ict[0]
      print('ict type: ', type(ict))
      tail = np.fromfile(f, dtype='>i4', count=1)
      print('header=', header, ', ict=', ict, ', tail=', tail)

  da_tmp = xr.where(da_sst_obs == -9999, 0, 1)
  num_obs = da_tmp.sum().values
  print('For sat_nsobs_grid, num_obs=', num_obs) 

  da_sst_obs.to_dataset(name='sst').to_netcdf('./nsobsc.20160117_Ligang_BiasCorrected.nc')


if check_correlation_scale:
  f = open(file_correlation_scale, 'rb')

  header = np.fromfile(f, dtype='>i', count=1)
  data_type = np.fromfile(f, dtype=np.uint8, count=20).tostring()  # worked
  tail = np.fromfile(f, dtype='>i', count=1)
  print('header=', header, ', tail=', tail, ', data_type=', data_type)

  header = np.fromfile(f, dtype='>i', count=1)
  data = np.fromfile(f, dtype='>f', count=QUART_DEG_LON_DIM*QUART_DEG_LAT_DIM)
  lon_scale = np.reshape(data, (QUART_DEG_LAT_DIM, QUART_DEG_LON_DIM), order='C')
  tail = np.fromfile(f, dtype='>i', count=1)
  print('\n\n header=', header, ', tail=', tail, ', lon_scale=', lon_scale)

  header = np.fromfile(f, dtype='>i', count=1)
  data_type = np.fromfile(f, dtype=np.uint8, count=20).tostring()  # worked
  tail = np.fromfile(f, dtype='>i', count=1)
  print('\n\n header=', header, ', tail=', tail, ', data_type=', data_type)

  header = np.fromfile(f, dtype='>i', count=1)
  data = np.fromfile(f, dtype='>f', count=QUART_DEG_LON_DIM*QUART_DEG_LAT_DIM)
  lat_scale = np.reshape(data, (QUART_DEG_LAT_DIM, QUART_DEG_LON_DIM), order='C')
  tail = np.fromfile(f, dtype='>i', count=1)
  print('\n\n header=', header, ', tail=', tail, ', lat_scale=', lat_scale)

  header = np.fromfile(f, dtype='>i', count=1)
  data_type = np.fromfile(f, dtype=np.uint8, count=20).tostring()  # worked
  tail = np.fromfile(f, dtype='>i', count=1)
  print('\n\n header=', header, ', tail=', tail, ', data_type=', data_type)

  header = np.fromfile(f, dtype='>i', count=1)
  data = np.fromfile(f, dtype='>f', count=QUART_DEG_LON_DIM*QUART_DEG_LAT_DIM)
  fg_autocorr = np.reshape(data, (QUART_DEG_LAT_DIM, QUART_DEG_LON_DIM), order='C')
  tail = np.fromfile(f, dtype='>i', count=1)
  print('\n\n header=', header, ', tail=', tail, ', fg_autocorr=', fg_autocorr)

  lon = np.arange(start=LON_START, stop=360., step=GRID_SIZE)
  lat = np.arange(start=LAT_START, stop=90. , step=GRID_SIZE)

  da_lon_scale = xr.DataArray(lon_scale, name='lon_scale', dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}, attrs=dict(_FillValue=-9999) )
  da_lat_scale = xr.DataArray(lat_scale, name='lat_scale', dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}, attrs=dict(_FillValue=-9999) )
  da_fg_autocorr = xr.DataArray(fg_autocorr, name='fg_autocorr', dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}, attrs=dict(_FillValue=-9999) )

  ds_fg_autocorr = xr.merge([da_lon_scale, da_lat_scale, da_fg_autocorr])
  ds_fg_autocorr.to_netcdf('./cor4sm-stat-v2_Ligang_fg_autocorr.nc')

# da_tmp = xr.where(da_sst_obs == -9999, 0, 1)
# num_obs = da_tmp.sum().values
# print('For obs_buoy_grid, num_obs=', num_obs)  # printed 


if check_fg_err_variance:
  f = open(file_fg_err_variance, 'rb')

  header = np.fromfile(f, dtype='>i', count=1)
  data_type = np.fromfile(f, dtype=np.uint8, count=20).tostring()  # worked
  tail = np.fromfile(f, dtype='>i', count=1)
  print('header=', header, ', tail=', tail, ', data_type=', data_type)

  header = np.fromfile(f, dtype='>i', count=1)
  data = np.fromfile(f, dtype='>f', count=QUART_DEG_LON_DIM*QUART_DEG_LAT_DIM)
  fg_err_variance = np.reshape(data, (QUART_DEG_LAT_DIM, QUART_DEG_LON_DIM), order='C')
  tail = np.fromfile(f, dtype='>i', count=1)
  print('\n\n header=', header, ', tail=', tail, ', fg_err_variance=', fg_err_variance)

  lon = np.arange(start=LON_START, stop=360., step=GRID_SIZE)
  lat = np.arange(start=LAT_START, stop=90. , step=GRID_SIZE)

  da_fg_err_variance = xr.DataArray(fg_err_variance, name='fg_err_variance', dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}, attrs=dict(_FillValue=-9999) )
  da_fg_err_variance.to_dataset(name='fg_err_variance').to_netcdf('./path-incr-var-stat-v2_Ligang_fg_err_variance.nc')


if check_fg_correction:
  f = open(file_fg_correction, 'rb')

  header = np.fromfile(f, dtype='>i', count=1)
  data_type = np.fromfile(f, dtype=np.uint8, count=20).tostring()  # worked
  tail = np.fromfile(f, dtype='>i', count=1)
  print('header=', header, ', tail=', tail, ', data_type=', data_type)

  header = np.fromfile(f, dtype='>i', count=1)
  data = np.fromfile(f, dtype='>f', count=QUART_DEG_LON_DIM*QUART_DEG_LAT_DIM)
  fg_correction = np.reshape(data, (QUART_DEG_LAT_DIM, QUART_DEG_LON_DIM), order='C')
  tail = np.fromfile(f, dtype='>i', count=1)
  print('\n\n header=', header, ', tail=', tail, ', fg_correction=', fg_correction)

  lon = np.arange(start=LON_START, stop=360., step=GRID_SIZE)
  lat = np.arange(start=LAT_START, stop=90. , step=GRID_SIZE)

  da_fg_correction = xr.DataArray(fg_correction, name='fg_correction', dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}, attrs=dict(_FillValue=-9999) )
  da_fg_correction.to_dataset(name='fg_correction').to_netcdf('./error-cor-stat-v2_Ligang_fg_correction.nc')


if check_monthly_clim:
  month = np.arange(1, MONTHS_IN_YEAR+1, dtype='i')
  lon = np.arange(start=LON_START, stop=360., step=GRID_SIZE, dtype='f')
  lat = np.arange(start=LAT_START, stop=90. , step=GRID_SIZE, dtype='f')
  monthly_clim = np.zeros((MONTHS_IN_YEAR, QUART_DEG_LAT_DIM, QUART_DEG_LON_DIM), dtype='f')

  f = open(file_monthly_clim, 'rb')
  for i_mon in range(1, MONTHS_IN_YEAR+1):
    header = np.fromfile(f, dtype='>i4', count=1)
    climatology_year  = np.fromfile(f, dtype='>i4', count=1)
    climatology_month = np.fromfile(f, dtype='>i4', count=1)
    data = np.fromfile(f, dtype='>f', count=QUART_DEG_LON_DIM*QUART_DEG_LAT_DIM)
    monthly_clim[i_mon-1, :, :] = np.reshape(data, (QUART_DEG_LAT_DIM, QUART_DEG_LON_DIM), order='C')
    tail = np.fromfile(f, dtype='>i4', count=1)
    print('i_mon=', i_mon, 'header=', header, ', tail=', tail, 'climatology_year=', climatology_year, ', climatology_month=', climatology_month)

  da_monthly_clim = xr.DataArray(monthly_clim, name='monthly_clim_sst', dims=('month', 'lat', 'lon')  \
    , coords={'month': month, 'lat': lat, 'lon': lon}, attrs=dict(_FillValue=-9999, units="degC") )
  da_monthly_clim.to_dataset(name='monthly_clim_sst').to_netcdf('./oiclm4.mon_Ligang_sst_clim_quart_deg.nc')


if check_residual_bias:
  f = open(file_residual_bias, 'rb')

  header = np.fromfile(f, dtype='>i', count=1)
  data_type = np.fromfile(f, dtype=np.uint8, count=20).tostring()  # worked
  tail = np.fromfile(f, dtype='>i', count=1)
  print('header=', header, ', tail=', tail, ', data_type=', data_type)

  header = np.fromfile(f, dtype='>i', count=1)
  data = np.fromfile(f, dtype='>f', count=QUART_DEG_LON_DIM*QUART_DEG_LAT_DIM)
  residual_bias = np.reshape(data, (QUART_DEG_LAT_DIM, QUART_DEG_LON_DIM), order='C')
  tail = np.fromfile(f, dtype='>i', count=1)
  print('\n\n header=', header, ', tail=', tail, ', residual_bias=', residual_bias)

  lon = np.arange(start=LON_START, stop=360., step=GRID_SIZE)
  lat = np.arange(start=LAT_START, stop=90. , step=GRID_SIZE)

  da_residual_bias = xr.DataArray(residual_bias, name='residual_bias', dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}, attrs=dict(_FillValue=-9999) )
  da_residual_bias.to_dataset(name='residual_bias').to_netcdf('./residual-stat-v2_Ligang_residual_bias.nc')


if check_buoy4sm_nsr:
  f = open(file_buoy4sm_nsr, 'rb')

  header = np.fromfile(f, dtype='>i', count=1)
  data_type = np.fromfile(f, dtype=np.uint8, count=20).tostring()  # worked
  tail = np.fromfile(f, dtype='>i', count=1)
  print('header=', header, ', tail=', tail, ', data_type=', data_type)

  header = np.fromfile(f, dtype='>i', count=1)
  data = np.fromfile(f, dtype='>f', count=QUART_DEG_LON_DIM*QUART_DEG_LAT_DIM)
  buoy4sm_nsr = np.reshape(data, (QUART_DEG_LAT_DIM, QUART_DEG_LON_DIM), order='C')
  tail = np.fromfile(f, dtype='>i', count=1)
  print('\n\n header=', header, ', tail=', tail, ', buoy4sm_nsr=', buoy4sm_nsr)

  lon = np.arange(start=LON_START, stop=360., step=GRID_SIZE)
  lat = np.arange(start=LAT_START, stop=90. , step=GRID_SIZE)

  da_buoy4sm_nsr = xr.DataArray(buoy4sm_nsr, name='buoy4sm_nsr', dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}, attrs=dict(_FillValue=-9999) )
  da_buoy4sm_nsr.to_dataset(name='buoy4sm_nsr').to_netcdf('./buoy4sm-nsr-stat-v2_Ligang_Noise2SignalRatio.nc')


if check_ship4sm_nsr:
  f = open(file_ship4sm_nsr, 'rb')

  header = np.fromfile(f, dtype='>i', count=1)
  data_type = np.fromfile(f, dtype=np.uint8, count=20).tostring()  # worked
  tail = np.fromfile(f, dtype='>i', count=1)
  print('header=', header, ', tail=', tail, ', data_type=', data_type)

  header = np.fromfile(f, dtype='>i', count=1)
  data = np.fromfile(f, dtype='>f', count=QUART_DEG_LON_DIM*QUART_DEG_LAT_DIM)
  ship4sm_nsr = np.reshape(data, (QUART_DEG_LAT_DIM, QUART_DEG_LON_DIM), order='C')
  tail = np.fromfile(f, dtype='>i', count=1)
  print('\n\n header=', header, ', tail=', tail, ', ship4sm_nsr=', ship4sm_nsr)

  lon = np.arange(start=LON_START, stop=360., step=GRID_SIZE)
  lat = np.arange(start=LAT_START, stop=90. , step=GRID_SIZE)

  da_ship4sm_nsr = xr.DataArray(ship4sm_nsr, name='ship4sm_nsr', dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}, attrs=dict(_FillValue=-9999) )
  da_ship4sm_nsr.to_dataset(name='ship4sm_nsr').to_netcdf('./ship4sm-nsr-stat-v2_Ligang_Noise2SignalRatio.nc')


if check_day_path4sm_nsr:
  f = open(file_day_path4sm_nsr, 'rb')

  header = np.fromfile(f, dtype='>i', count=1)
  data_type = np.fromfile(f, dtype=np.uint8, count=20).tostring()  # worked
  tail = np.fromfile(f, dtype='>i', count=1)
  print('header=', header, ', tail=', tail, ', data_type=', data_type)

  header = np.fromfile(f, dtype='>i', count=1)
  data = np.fromfile(f, dtype='>f', count=QUART_DEG_LON_DIM*QUART_DEG_LAT_DIM)
  day_path4sm_nsr = np.reshape(data, (QUART_DEG_LAT_DIM, QUART_DEG_LON_DIM), order='C')
  tail = np.fromfile(f, dtype='>i', count=1)
  print('\n\n header=', header, ', tail=', tail, ', day_path4sm_nsr=', day_path4sm_nsr)

  lon = np.arange(start=LON_START, stop=360., step=GRID_SIZE)
  lat = np.arange(start=LAT_START, stop=90. , step=GRID_SIZE)

  da_day_path4sm_nsr = xr.DataArray(day_path4sm_nsr, name='day_path4sm_nsr', dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}, attrs=dict(_FillValue=-9999) )
  da_day_path4sm_nsr.to_dataset(name='day_path4sm_nsr').to_netcdf('./day_path4sm-nsr-stat-v2_Ligang_Satellite_Noise2SignalRatio.nc')


if check_nte_path4sm_nsr:
  f = open(file_nte_path4sm_nsr, 'rb')

  header = np.fromfile(f, dtype='>i', count=1)
  data_type = np.fromfile(f, dtype=np.uint8, count=20).tostring()  # worked
  tail = np.fromfile(f, dtype='>i', count=1)
  print('header=', header, ', tail=', tail, ', data_type=', data_type)

  header = np.fromfile(f, dtype='>i', count=1)
  data = np.fromfile(f, dtype='>f', count=QUART_DEG_LON_DIM*QUART_DEG_LAT_DIM)
  nte_path4sm_nsr = np.reshape(data, (QUART_DEG_LAT_DIM, QUART_DEG_LON_DIM), order='C')
  tail = np.fromfile(f, dtype='>i', count=1)
  print('\n\n header=', header, ', tail=', tail, ', nte_path4sm_nsr=', nte_path4sm_nsr)

  lon = np.arange(start=LON_START, stop=360., step=GRID_SIZE)
  lat = np.arange(start=LAT_START, stop=90. , step=GRID_SIZE)

  da_nte_path4sm_nsr = xr.DataArray(nte_path4sm_nsr, name='nte_path4sm_nsr', dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}, attrs=dict(_FillValue=-9999) )
  da_nte_path4sm_nsr.to_dataset(name='nte_path4sm_nsr').to_netcdf('./nte_path4sm-nsr-stat-v2_Ligang_Satellite_Noise2SignalRatio.nc')


if check_cice4sm_nsr:
  f = open(file_cice4sm_nsr, 'rb')

  header = np.fromfile(f, dtype='>i', count=1)
  data_type = np.fromfile(f, dtype=np.uint8, count=20).tostring()  # worked
  tail = np.fromfile(f, dtype='>i', count=1)
  print('header=', header, ', tail=', tail, ', data_type=', data_type)

  header = np.fromfile(f, dtype='>i', count=1)
  data = np.fromfile(f, dtype='>f', count=QUART_DEG_LON_DIM*QUART_DEG_LAT_DIM)
  cice4sm_nsr = np.reshape(data, (QUART_DEG_LAT_DIM, QUART_DEG_LON_DIM), order='C')
  tail = np.fromfile(f, dtype='>i', count=1)
  print('\n\n header=', header, ', tail=', tail, ', cice4sm_nsr=', cice4sm_nsr)

  lon = np.arange(start=LON_START, stop=360., step=GRID_SIZE)
  lat = np.arange(start=LAT_START, stop=90. , step=GRID_SIZE)

  da_cice4sm_nsr = xr.DataArray(cice4sm_nsr, name='cice4sm_nsr', dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}, attrs=dict(_FillValue=-9999) )
  da_cice4sm_nsr.to_dataset(name='cice4sm_nsr').to_netcdf('./cice4sm-nsr-stat-v2_Ligang_Satellite_Noise2SignalRatio.nc')






