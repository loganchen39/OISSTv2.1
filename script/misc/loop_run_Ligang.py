#!/glade/u/apps/ch/opt/python/3.7.9/gnu/9.1.0/bin/python3

import os
import datetime

run_type = 'interim'  # 'final'
jday_20160101 = datetime.date(2016, 1, 1 )
jday_20160115 = datetime.date(2016, 1, 15)
jday_20160116 = datetime.date(2016, 1, 16)
# jday_20160415 = datetime.date(2016, 4, 15)
jday_20160130 = datetime.date(2016, 1, 30)

print("Generating sobs ...")
jday = jday_20160115  # initially jday_20160101
while jday <= jday_20160116:  # initially jday_20160130
  str_curr_day   = jday.strftime('%Y%m%d')
  str_curr_dayp1 = (jday + datetime.timedelta(days=1)).strftime('%Y%m%d')
  str_cmd = "sh build/script/oisst.sh --part=sobs --when=" + str_curr_day + "," + str_curr_dayp1 + " " + run_type
  os.system(str_cmd)

  jday += datetime.timedelta(days=1)


print("Generating oi output")
jday = jday_20160115
while jday <= jday_20160116:  # initially jday_20160130
  str_curr_day   = jday.strftime('%Y%m%d')
  str_curr_dayp1 = (jday + datetime.timedelta(days=1)).strftime('%Y%m%d')

  str_cmd = "sh build/script/oisst.sh --part=eotwt --when="  + str_curr_day + "," + str_curr_dayp1 + " " + run_type
  os.system(str_cmd)
  str_cmd = "sh build/script/oisst.sh --part=eotcor --when=" + str_curr_day + "," + str_curr_dayp1 + " " + run_type
  os.system(str_cmd)
  str_cmd = "sh build/script/oisst.sh --part=oi --when="     + str_curr_day + "," + str_curr_dayp1 + " " + run_type
  os.system(str_cmd)
  str_cmd = "sh build/script/oisst.sh --part=output --when=" + str_curr_day + "," + str_curr_dayp1 + " " + run_type
  os.system(str_cmd)

  jday += datetime.timedelta(days=1)
