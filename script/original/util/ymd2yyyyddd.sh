# ymd2yyyydyyy converts yyyymmdd to yyyyddd
# usage ymd2yyyyddd.sh 19980429
 	
# if there is no command line argument, assume the date
# is coming in on a pipe and use read to collect it

# set -x
 	
if [ X$1 = X ]
then
read  dt
else
dt=$1
fi
	
# break the yyyymmdd into separate parts for year, month, and day
	
year=`expr $dt / 10000`
month=`expr \( $dt % 10000 \) / 100`
#echo $month
day=`expr $dt % 100`
#echo $day
	
# add the days in each month, up to but not including the month itself,
# into the days. For example, if the date is 19980203, extract the
# number of days in January and add it to 03. If the date is June 14, 1998,
# extract the number of days in January, February, March, April, and May
# and add them to 14.

mon=1
while [ $mon -lt $month ]
do

# calculate the days in the earlier months
case $mon in
#1|3|5|7|8|10|12) echo 31 ; days_in_month=31 ;;
1|3|5|7|8|10|12) days_in_month=31 ;;
#4|6|9|11) echo 30 ; days_in_month=30 ;;
4|6|9|11) days_in_month=30 ;;
    *) ;;
esac
  
# except for month 2, which depends on whether the year is a leap year
#find if a year is a leap year to decide the days in Feb.

if [ $mon -eq 2 ]
then

leap=0
days_in_year=365

if [ `expr $year % 400` = 0 ]
then
leap=1
days_in_year=366
  else
  if [ `expr $year % 4` = 0 -a `expr $year % 100`  != 0 ]
  then
  leap=1
  days_in_year=366
  fi
fi
#echo 'days in year: ' $days_in_year
 
case $days_in_year in
#365) echo 28 ; days_in_month=28 ;;
365) days_in_month=28 ;;
#366) echo 29 ; days_in_month=29 ;;
366) days_in_month=29 ;;
esac

fi

day=`expr $day + $days_in_month`
mon=`expr $mon + 1`
done
	
# combine the year and day back together again and you have the julian date.

julyyyyddd=`expr \( $year \* 1000 \) + $day`
#echo 'yyyyyddd: ' $julyyyyddd

jul=`expr $julyyyyddd | cut -c5-7`
echo $jul
