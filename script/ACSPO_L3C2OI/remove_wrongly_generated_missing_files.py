'''
Description: Check missing files for ACSPO-L3C, ACSPO-L3S, PathfinderV5.3
Author: Ligang Chen
Date created: 08/04/2022
Date last modified: 08/04/2022 
'''

import os
import datetime
import glob


# To print all numpy array elements.
# np.set_printoptions(threshold=np.inf)

check_l3c = False 
check_l3s = False 
check_pf  = True  

DIR_L3C = '/glade/scratch/lgchen/data/ACSPO_L3C'
lst_sat = [('noaa07', datetime.date(1981, 9 , 1 ), datetime.date(1985, 2 , 2 ))
    ,      ('noaa09', datetime.date(1985, 1 , 31), datetime.date(1988, 11, 7 ))  # initially start at datetime.date(1985, 2 , 25)
    ,      ('noaa11', datetime.date(1988, 11, 8 ), datetime.date(1994, 9 , 13))
    ,      ('noaa12', datetime.date(1991, 9 , 16), datetime.date(1998, 12, 14))
    ,      ('noaa14', datetime.date(1995, 1 , 19), datetime.date(2001, 10, 19))
    ,      ('noaa15', datetime.date(1998, 11, 1 ), datetime.date(2021, 12, 31))
    ,      ('noaa16', datetime.date(2001, 1 , 1 ), datetime.date(2007, 9 , 17))
    ,      ('noaa17', datetime.date(2002, 7 , 10), datetime.date(2010, 3 , 8 ))
    ,      ('noaa18', datetime.date(2005, 6 , 6 ), datetime.date(2021, 12, 31))
    ,      ('noaa19', datetime.date(2009, 2 , 22), datetime.date(2021, 12, 31))]

if check_l3c:
    print('Checking L3C ...')
    for sat_nm, jday_st, jday_end in lst_sat:
        print('Checking ' + sat_nm + '...')
        
        jday = jday_st
        while jday <= jday_end:
            str_date = jday.strftime('%Y%m%d')
            fn_day = glob.glob(pathname=DIR_L3C+'/'+sat_nm+'/'+str_date+'*_D-ACSPO_*.nc')
            fn_nit = glob.glob(pathname=DIR_L3C+'/'+sat_nm+'/'+str_date+'*_N-ACSPO_*.nc')
            if len(fn_day) == 0:
                print('Removing day data for ' + str_date + '...')
                str_cmd = 'rm ' + DIR_L3C + '/l3c2oi/' + sat_nm + '/' + str_date + '*_D_ACSPO-L3C2OI*'
                os.system(str_cmd)
            if len(fn_nit) == 0:
                print('Removing nit data for ' + str_date + '...')
                str_cmd = 'rm ' + DIR_L3C + '/l3c2oi/' + sat_nm + '/' + str_date + '*_N_ACSPO-L3C2OI*'
                os.system(str_cmd)
 
            jday += datetime.timedelta(days=1) 
    



DIR_L3S = '/glade/scratch/lgchen/data/ACSPO_L3S_LEO'
lst_sat = [('am', datetime.date(2006, 12, 1 ), datetime.date(2021, 12, 31))
    ,      ('pm', datetime.date(2012, 2 , 1 ), datetime.date(2021, 12, 31))]
if check_l3s:
    print('\n\nChecking L3S ...')
    for sat_nm, jday_st, jday_end in lst_sat:
        print('Checking ' + sat_nm + '...')
        
        jday = jday_st
        while jday <= jday_end:
            str_date = jday.strftime('%Y%m%d')
            fn_day = glob.glob(pathname=DIR_L3S+'/'+sat_nm+'/'+str_date+'*_D-ACSPO_*.nc')
            fn_nit = glob.glob(pathname=DIR_L3S+'/'+sat_nm+'/'+str_date+'*_N-ACSPO_*.nc')
            if len(fn_day) == 0:
                print('Removing day data for ' + str_date + '...')
                str_cmd = 'rm ' + DIR_L3S + '/l3s2oi/' + sat_nm + '/' + str_date + '*_D_ACSPO-L3S2OI*'
                os.system(str_cmd)
            if len(fn_nit) == 0:
                print('Removing night data for ' + str_date + '...')
                str_cmd = 'rm ' + DIR_L3S + '/l3s2oi/' + sat_nm + '/' + str_date + '*_N_ACSPO-L3S2OI*'
                os.system(str_cmd)   

            jday += datetime.timedelta(days=1) 


    

DIR_PF = '/glade/scratch/lgchen/data/PathfinderV5.3'
jday_19810825 = datetime.date(1981, 8 , 25)
jday_20211231 = datetime.date(2021, 12, 31)
if check_pf:
    print('\n\nChecking PF ...')
    
    jday = jday_19810825
    while jday <= jday_20211231:
        str_date = jday.strftime('%Y%m%d')
        fn_day = glob.glob(pathname=DIR_PF+'/PFV5.3/'+str_date+'*_day-*.nc')
        fn_nit = glob.glob(pathname=DIR_PF+'/PFV5.3/'+str_date+'*_night-*.nc')
        if len(fn_day) == 0:
            print('Removing day data for ' + str_date + '...')
            str_cmd = 'rm ' + DIR_PF + '/PF2OI/' + str_date + '_day_PFV5.3-2OI.nc'
            os.system(str_cmd)
        if len(fn_nit) == 0:
            print('Removing night data for ' + str_date + '...')
            str_cmd = 'rm ' + DIR_PF + '/PF2OI/' + str_date + '_night_PFV5.3-2OI.nc'
            os.system(str_cmd)

        jday += datetime.timedelta(days=1) 
