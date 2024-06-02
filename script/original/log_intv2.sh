#! /bin/sh
# give log file the right name and move to the right location

# Chunying Liu  Apr. 2015

. ./oi.properties
 
ulimit -s unlimited
umask 002

echo "Script to rename and move the logfile to the right folder"

curryear=`date '+%Y'`
currmon=`date '+%m'`
today=`date '+%d'`

echo $curryear,$currmon,$today

TMtoday=$curryear-$currmon-$today
echo $TMtoday

hour=`date '+%H:%M'`

TMtoday2=$TMtoday-$hour
echo $TMtoday2

# get the just finished analysis day date, i.e. The process day is

TM0=`head -1 build/oi_intv2.txt`
echo $TM0

TM=`expr $TM0 | cut -c19-27`
echo $TM

year=`expr $TM | cut -c1-4`
echo $year

logfile=$DATA_PRELIM_OUT/log/$year/oisst_prelim_$TM-$TMtoday2.log
echo $logfile

rm -f $logfile >& /dev/null

# gets the latest log file

cat $LOG/oisst_$TMtoday.log > $logfile

