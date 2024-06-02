#!/bin/bash

# -----------------------------------------------
#
#   Main OISST run script. Call this script with
# the type of run to be executed.
#
# Preliminary
#   1 day delay. Generate superobs.
#
# Final
#   15 day delay.
#
# this script does not handle reprocessing
# -----------------------------------------------

. ./oi.properties

ulimit -s unlimited

if [[ ( $1 == "--help") || $1 == "-h" ]]; then
  echo "Usage: `basename $0` final|interim [--part=sobs|eotwt|eotcor|oi|output|nosobs]"
  echo "           [--when=[lastDate][,endDate]]"
  echo "       `basename $0` --help|-h"
  echo
  echo "Run the OISST processing. The options are:"
  echo
  echo "final|interim  Do either the final or the interim processing. (Required)"
  echo "--part         Direct the software to do only part of the processing. (Optional)"
  echo "--when         Set the dates that control what data is processed. lastDate and"
  echo "               endDate both have the form YYYYMMDD. Zero, one or two dates may"
  echo "               be present. (Optional)"
  echo "--help|-h      Print this help message and exit."
  echo
  echo "If the --part option is specified, the processing will be limited to the named"
  echo "section. The section names and their meanings are:"
  echo
  echo "sobs    Run only the superobs step."
  echo "eotwt   Run only the eot weights step."
  echo "eotcor  Run only the eot correction step."
  echo "oi      Run only the oi step."
  echo "output  Run only the output generation step."
  echo "nosobs  Run all but the superobs step. (Use existing superobs.)"
  echo
  echo "If no --when option is specified, the processing will be started with the"
  echo "stopDate unset, which will cause it to be set appropriately for production"
  echo "(one day prior for interim, 15 days prior for final)."
  echo "If the --when option is specified with no dates (--when=), the processing will"
  echo "be started with both the lastProcessedDate and the stopDate unset. This will"
  echo "cause the stopDate to be set as for production, and the lastProcessedDate set to"
  echo "two days prior for interim and 16 days prior for final."
  echo "If the --when option is specified with only a lastDate argument"
  echo "(--when=YYYYMMDD), the processing will be started with the lastProcessedDate set"
  echo "to lastDate and the stopDate unset, which will cause it to be set as above."
  echo "If the --when option is specified with only an endDate argument"
  echo "(--when=,YYYYMMDD), the processing will be started with the lastProcessedDate"
  echo "unset and the stopDate set to endDate. This will cause the lastProcessedDate to"
  echo "be set as above."
  echo "If the --when option is specified with both lastDate and endDate specified"
  echo "(--when=YYYYMMDD,YYYYMMDD), the processing will be started with the"
  echo "lastProcessedDate set to lastDate and the stopDate set to endDate."

  exit 1
fi


# Parse the arguments. Exit if there is an unrecognized argument.
#
for arg in "$@"
do
    case $arg in 
        interim|final)
            runType=$arg
            ;;
        --part=*)
            runPart=${arg#--part=}
            ;;
        --when=*)
            whenSet=1          # Indicate that the when argument was seen.
            tmp=${arg#--when=} # Strip off the option part.
            lastDate=${tmp%,*} # Get the first date, which might be empty.
            tmp=(${tmp/*,/# }) # Make an array. (Will have 0-2 valid parts.)
            endDate=${tmp[1]}  # Get the second date, which might be empty.
            ;;
        *)
            echo "Error: Unrecognized argument $arg"
            exit 1
    esac
done

# Verify that the arguments are valid. Report any errors and exit.
#
if [ -z "$runType" ]
then
    echo "Error: A run type of interim or final must be specified."
    exit 1
elif [[ "$runType" != "interim" && "$runType" != "final" ]]
then
    echo "Error: A run type of $runType is not valid."
    exit 1
fi

if [[ -n "$runPart" ]]
then
    tmp="_sobs_eotwt_eotcor_oi_output_nosobs_"

    if [[ "yes" != "${tmp/*_${runPart}_*/yes}" ]]
    then
        echo "Error: A run part of $runPart is not valid."
        exit 1
    fi
else
    runPart="all"
fi

if [[ -n "$lastDate" ]]
then
    tmp=`echo $lastDate | tr [:digit:] '#'`

    if [[ '########' != "$tmp" ]]
    then
        echo "Error: A last date of $lastDate is not valid."
        exit 1
    fi
elif [[ -n "$whenSet" ]]
then
    lastDate="none"
else
    lastDate="default"
fi

if [[ -n "$endDate" ]]
then
    tmp=`echo $endDate | tr [:digit:] '#'`

    if [[ '########' != "$tmp" ]]
    then
        echo "Error: An end date of $endDate is not valid."
        exit 1
    fi

    if [[ 8 -ne ${#endDate} ]]
    then
        echo "Error: An end date of $endDate is not valid."
        exit 1
    fi
else
    endDate="none"
fi

# Construct paths for the date files.
#
interimDateFile="$( cd "$( dirname "$0" )" && cd .. && pwd )"/oi_intv2.txt
finalDateFile="$( cd "$( dirname "$0" )" && cd .. && pwd )"/oi_finv2.txt

# find the UTIL directory and source the stopcode checking function
UTIL="$( cd "$( dirname "$0" )" && pwd )"/util
. $UTIL/check_stopcode.sh


function load() {
  # use python to generate the script workfile
  python -m build.script.parseConfig $1 $2
  checkStopcode "build.script.parseConfig.py" $?
  
  # now source the generated workfile and the run scripts
  . "$( cd "$( dirname "$0" )" && cd .. && pwd )"/workfile
  echo 'loaded'
  
  case "$1" in
    'interim')
    . $SCRIPT/avhrr_interim_main.sh
    ;;
    
    'final')
    . $SCRIPT/avhrr_final_main.sh
    ;;
    
    *)
      echo 'Please provide a runtype (interim | final).'
      exit 1
  esac
}

#------------------------------------------------
#
# runtype control switch
#
#------------------------------------------------
function init() {

  local runType=$1
  local lastDate=$2
  local endDate=$3

  case ${runType} in
    'interim')
      echo 'interim run'
      type=1
      . $interimDateFile
    ;;
    
    'final')
      echo 'final run'
      type=2
      . $finalDateFile
    ;;
    
    *)
      echo 'Please provide a runtype (interim | final).'
      exit 1
  esac
  
  # enumerate the runtypes
  let interim=$(( $type == 1 ))
  let final=$(( $type == 2 ))

  # If the lastDate value is none, unset lastProcessedDate.  If it is default,
  # do nothing, and if it is some other value, set lastProcessedDate to that
  # value.
  #
  if [[ "${lastDate}" == "none" ]]
  then
    unset lastProcessedDate
  elif [[ "${lastDate}" != "default" ]]
  then
    lastProcessedDate=${lastDate}
  fi 

  # If the endDate value is none, unset stopDate. Otherwise, set stopDate to
  # that value.
  #
  if [[ "${endDate}" == "none" ]]
  then
    unset stopDate
  else
    stopDate=${endDate}
  fi 

  TMtoday=`date '+%Y%m%d'`

  # check the runtype
  # interim runtype date control
  if (( $interim  )); then
    # if no "last run" then use current day  - 2
    if [ -z "$lastProcessedDate" ]; then
      lastProcessedDate=`sh $UTIL/finddate.sh $TMtoday d-2`
    fi
    # make sure lastProcessedDate string is correct size
    if [ ${#lastProcessedDate} -lt 8 ] || [ ${#lastProcessedDate} -gt 8 ]; then
      echo "lastProcessedDate has incorrect size.  Processing aborted."
      exit 1
    fi
    # if stopDate is empty use the current day - 1 
    if [ -z "$stopDate" ]; then
      stopDate=`sh $UTIL/finddate.sh $TMtoday d-1`
    fi
    # make sure stopDate string is correct size
    if [ ${#stopDate} -lt 8 ] || [ ${#stopDate} -gt 8 ]; then
      echo "stopDate has incorrect size.  Processing aborted."
      exit 1
    fi
    # make sure stopDate occurs after lastProcessedDate
    if [ $stopDate -lt $lastProcessedDate ]; then
      echo "stopDate occurs before lastProcessedDate in date file.  Processing aborted."
      exit 1
    fi
  # final runtype date lookup
  elif (( $final )); then
    # if no "last run" then use current day  - 16    
    if [ -z "$lastProcessedDate" ]; then
      lastProcessedDate=`sh $UTIL/finddate.sh $TMtoday d-16`
    fi
    # make sure lastProcessedDate string is correct size
    if [ ${#lastProcessedDate} -lt 8 ] || [ ${#lastProcessedDate} -gt 8 ]; then
      echo "lastProcessedDate has incorrect size.  Processing aborted."
      exit 1
    fi
    # if stopDate is empty in the date file use current day - 15      
    if [ -z "$stopDate" ]; then
      stopDate=`sh $UTIL/finddate.sh $TMtoday d-15`
    fi
    # make sure date string is correct size
    if [ ${#stopDate} -lt 8 ] || [ ${#stopDate} -gt 8 ]; then
      echo "stopDate has incorrect size.  Processing aborted."
      exit 1
    fi
    # make sure stopDate occurs after lastProcessedDate
    if [ $stopDate -lt $lastProcessedDate ]; then
      echo "stopDate occurs before lastProcessedDate in date file.  Processing aborted."
      exit 1
    fi
  else
    echo 'unknown run type'
  fi
}

#
#
#
#
function process() {
  local runPart=${1/all/}

  if (( $interim )); then
    AVHRRInterim $curDate ${runPart}
    echo "lastProcessedDate=$curDate" > $interimDateFile
    echo "stopDate=$stopDate" >> $interimDateFile
  elif (( $final )); then
    AVHRRFinal $curDate ${runPart}
    echo "lastProcessedDate=$curDate" > $finalDateFile
    echo "stopDate=$stopDate" >> $finalDateFile
  else
    echo 'unknown run type'
  fi
    
}

#
#
#
#
function setup() {
  local runType=$1
  local runPart=$2

  # -----------------------------------------------
  #
  #   Current processing date.
  #
  # -----------------------------------------------
  curDate=`sh $UTIL/finddate.sh $lastProcessedDate d+1`

  echo 'processed   last: '$lastProcessedDate
  echo 'processing   end: '$stopDate

  # -----------------------------------------------
  #
  #   Loop until the stop date is reached.
  # Preliminary and final runs typically only
  # process one day.
  #
  # -----------------------------------------------
  loopDate=$curDate
  while [ "$loopDate" -le "$stopDate" ]; do
    curDate=$loopDate
    echo 'processing analysis date ' $curDate
    
    # firstguess
    fgDate=`sh $UTIL/finddate.sh $curDate d-1`
    
    # firstguess' year
    fgYear=`echo $fgDate | cut -c 1-4`

   # set the date for final run sobs caculation
    curDate12=`sh $UTIL/finddate.sh $curDate d+11`
    echo $curDate12
   
   # set up the icesst data which is the next day of the 
    curDate_ice=`sh $UTIL/finddate.sh $curDate d+1`
    curYear_ice=`expr $curDate_ice | cut -c1-4`
    curMonth_ice=`expr $curDate_ice | cut -c5-6`
    curDay_ice=`expr $curDate_ice | cut -c7-8`
    echo 'icesst process date = ' $curYear_ice $curMonth_ice $curDay_ice

   # 12th day's year, month, day
    curYear12=`expr $curDate12 | cut -c1-4`
    curYear2digit12=`expr $curDate12 | cut -c3-4`
    curMonth12=`expr $curDate12 | cut -c5-6`
    curDay12=`expr $curDate12 | cut -c7-8`
    echo $curYear12 $curMonth12 $curDay12

    # final run sobs firstguess date
    fgDate12=`sh $UTIL/finddate.sh $curDate12 d-1`
    echo $fgDate12
    
    # firstguess' year
    fgYear12=`echo $fgDate12 | cut -c 1-4`
    echo $fgYear12
    
    # current day-8
    curDateM8=`sh $UTIL/finddate.sh $curDate d-8`
    # + 8
    curDateP8=`sh $UTIL/finddate.sh $curDate d+8`
    
    # current day-3 to current day+3
    curDateM3=`sh $UTIL/finddate.sh $curDate d-3` 
    curDateM2=`sh $UTIL/finddate.sh $curDate d-2` 
    curDateM1=`sh $UTIL/finddate.sh $curDate d-1` 
    curDateP1=`sh $UTIL/finddate.sh $curDate d+1` 
    curDateP2=`sh $UTIL/finddate.sh $curDate d+2` 
    curDateP3=`sh $UTIL/finddate.sh $curDate d+3`
    
    # current run day's year, month, day
    curYear=`expr $curDate | cut -c1-4`
    curYear2digit=`expr $curDate | cut -c3-4`
    curMonth=`expr $curDate | cut -c5-6`
    curDay=`expr $curDate | cut -c7-8`

    # years
    curDateM2Year=`expr $curDateM2 | cut -c1-4`
    curDateM1Year=`expr $curDateM1 | cut -c1-4`
    curDateM1Year2digit=`expr $curDateM1 | cut -c3-4`
    curDateP1Year=`expr $curDateP1 | cut -c1-4`
    curDateP2Year=`expr $curDateP2 | cut -c1-4`
    curDateP3Year=`expr $curDateP3 | cut -c1-4`
    
    curDateM8Year=`expr $curDateM8 | cut -c1-4`
    curDateM8Year2digit=`expr $curDateM8Year | cut -c3-4`
    curDateM8Day=`expr $curDateM8 | cut -c7-8`
    curDateM8JDay=`sh $UTIL/ymd2yyyyddd.sh $curDateM8`
    
    curDateP8Year=`expr $curDateP8 | cut -c1-4`
    curDateP8Year2digit=`expr $curDateP8Year | cut -c3-4`
    curDateP8Day=`expr $curDateP8 | cut -c7-8`
    curDateP8JDay=`sh $UTIL/ymd2yyyyddd.sh $curDateP8`
    
    # julian days for current, +1, +2
    curJulian=`sh $UTIL/ymd2yyyyddd.sh $curDate`
    curJulianM1=`sh $UTIL/ymd2yyyyddd.sh $curDateM1`
    curJulianP1=`sh $UTIL/ymd2yyyyddd.sh $curDateP1`

    ## julian days for current, +1, +2 for final run
    curJulian12=`sh $UTIL/ymd2yyyyddd.sh $curDate12`

    curDate12M1=`sh $UTIL/finddate.sh $curDate12 d-1` 
    curDate12P1=`sh $UTIL/finddate.sh $curDate12 d+1`
    curJulian12M1=`sh $UTIL/ymd2yyyyddd.sh $curDate12M1`
    curJulian12P1=`sh $UTIL/ymd2yyyyddd.sh $curDate12P1`

    curDate12M1Year=`expr $curDate12M1 | cut -c1-4`
    curDate12P1Year=`expr $curDate12P1 | cut -c1-4`

    echo 'end of define dates'
    
    # input parameters: runtype and date
    load ${runType} $curDate
    
    process ${runPart}

    #find the next day
    loopDate=`sh $UTIL/finddate.sh $curDate d+1`

  case "${runType}" in
    'interim')
    echo 'create interim logfile'
    $SCRIPT/log_intv2.sh
    ;;
    
    'final')
    echo 'create final logfile'
    $SCRIPT/log_finv2.sh
    ;;

    *)
  esac

  rm -f $LOG/*.log >& /dev/null

  done
}

function main() {
    init $1 $3 $4
    
    setup $1 $2

curryear=`date '+%Y'`
currmon=`date '+%m'`
currday=`date '+%d'`
echo    "dOISST $1 updated $curryear/$currmon/$currday"                           > ~/tmp/mail_file_oi
mail -s "dOISST $1 updated $curryear/$currmon/$currday" boyin.huang@noaa.gov      < ~/tmp/mail_file_oi
mail -s "dOISST $1 updated $curryear/$currmon/$currday" chunying.liu@noaa.gov     < ~/tmp/mail_file_oi
mail -s "dOISST $1 updated $curryear/$currmon/$currday" garrett.graham@noaa.gov   < ~/tmp/mail_file_oi
mail -s "dOISST $1 updated $curryear/$currmon/$currday" huai-min.zhang@noaa.gov   < ~/tmp/mail_file_oi
}


#[[ "${BASH_SOURCE[0]}" == "${0}" ]] && echo "script ${BASH_SOURCE[0]} is being sourced ..."
[[ "${BASH_SOURCE[0]}" == "${0}" ]] && main ${runType} ${runPart} ${lastDate} ${endDate}

