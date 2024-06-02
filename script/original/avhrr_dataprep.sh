#!/bin/bash

# --------------------------------------------------
#
# QC and data prep
#
# --------------------------------------------------
createSourceString() {
  
  case $1 in
    *"ncep"*)     shipSource="GTS_SHIP-NCEP-IN_SITU"
                  buoySource="GTS_BUOY-NCEP-IN_SITU"
                  argoSource="unknown"
                  ;;
    *"icoads"*)   shipSource="ICOADS SHIPS"
                  buoySource="ICOADS BUOYS"
                  argoSource="ICOADS Argos"
                  ;;
    *)            shipSource="UNKNOWN"
                  buoySource="UNKNOWN"
                  argoSource="UNKNOWN"
                  ;;
  esac
   
  case $2 in
    *"ncep"*)     iceSource="MMAB_50KM-NCEP-ICE"
                  ;;
    *"gsfc"*)     iceSource="GSFC-ICE"
                  ;;
    *)            iceSource="UNKNOWN"   
                  ;;                             
  esac                                
  
  case $3 in
    *"navysst"*)  satSourceA="NAVO-L2P-AVHRR19_G"
                  satPlatform="AVHRR-19,MetOpA"
                  satSensor="AVHRR_GAC"
                  ;;
    *"path52"*)   satSourceA="AVHRR_Pathfinder-NODC-L3C-V5.2"     
                  satPlatform=${pathfinder.satAFile:64:69} 
                  satSensor="AVHRR"                
                  ;;
    *)            satSourceA="UNKNOWN" 
                  ;;                 
  esac
          
if [ $# -eq 4 ]; then                    
  case $4 in
    *"navysst"*)  satSourceB="NAVO-L2P-AVHRRMTA_G"
                  satPlatform="AVHRR-19,MetOpA"
                  satSensor="AVHRR_GAC"
                  ;;
    *"path52"*)   satSourceB="AVHRR_Pathfinder-NODC-L3C-V5.2"
                  satPlatform=${pathfinder.satAFile:1:6}                 
                  ;;
    *)            satSourceB="UNKNOWN"  
                  satPlatform="UNKNOWN"
                  ;;                  
  esac
  
  sourceString="$satSourceA,$satSourceB,$buoySource,$argoSource,$shipSource,$iceSource"
  
else

  sourceString="$satSourceA,$buoySource,$argoSource,$shipSource,$iceSource"
  
fi

echo "sourceString =" $sourceString
echo "satPlatform = " $satPlatform
echo "satSensor = " $satSensor


  
}

buoyshipPrep() {
  echo "Begining buoy ship obs: start time= "`date '+%T'`
  echo 'Generating buoy ship super obs.'
  
  # create the buoyship coverage file
  `> $covName`

  #paramFile=$1

  cat <<- oiEOF > $TMP/buoyship.param
	$quarterMaskIn
	$fileStdevIn
	$buoyshipSSTDaily
	$EOTfg
	$shipObsOut
	$buoyObsOut
	$argoObsOut
	$shipGridOut
	$buoyGridOut
	$argoGridOut
	$shipSobsOut
	$buoySobsOut
	$argoSobsOut
	$covName
	$curYear $curMonth $curDay
	$covBuoy
	$covArgo
	$covShip
	$covBuoyship
	oiEOF

  cat $TMP/buoyship.param

  $BIN/buoy_ship_daily_firstguessclim.x -Wl,-T < $TMP/buoyship.param
  stopcode=$?
  checkStopcode buoy_ship_daily_firstguessclim.x $stopcode
  
  # Something happened so now we need to generate empty buoy ship super obs files for the satellite bias correction
  if [ $stopcode -eq 10 ]; then
    echo 'setting empty buoy argo ship super obs file'
    cat <<- catEOF > $covName
	0.0
	0.0
	0.0
	catEOF
  fi
  
  echo "End time for buoy ship quality control code = " `date '+%T'`
  echo "Start time for run satellite quality control code =" `date '+%T'`
}

satellitePrep() {

        # count how many satellite files on the analysis day
          
          numfiles=$(ls $DATA_PRELIM_INPUT/avhrr6hr/$curYear/*$curDate* | wc -w )
          echo  $numfiles "satellite observation files on the analysis day $curYear.$curJulian"

          if [ $numfiles == 0 ]
          then
          echo "RED-STOP: No satellite observation files on the analysis day $curYear.$curJulian"
          exit 
          fi

          if [ $numfiles -lt 3 ]
          then
          echo "YELLOW-WARNING: There are less than three satellite observation files on the analysis day $curYear.$curJulian"
          fi
                 

  # create an empty sat. coverage files if it is missing
  `> $covNameSatA`
  `> $covNameSatB`
  
  # make a new file
  `> $satSSTCatObs`
  
  # concat all satellite A files
  ls $DATA_PRELIM_INPUT/avhrr6hr/$curYear/mcsst_mta_d"$curDate"_s* 
  ls $DATA_PRELIM_INPUT/avhrr6hr/$curDateM1Year/mcsst_mta_d"$curDateM1"_s2* 
  ls $DATA_PRELIM_INPUT/avhrr6hr/$curDateP1Year/mcsst_mta_d"$curDateP1"_s0* 


  cat $DATA_PRELIM_INPUT/avhrr6hr/$curYear/mcsst_mta_d"$curDate"_s* >> $satSSTCatObs
  cat $DATA_PRELIM_INPUT/avhrr6hr/$curDateM1Year/mcsst_mta_d"$curDateM1"_s2* >> $satSSTCatObs
  cat $DATA_PRELIM_INPUT/avhrr6hr/$curDateP1Year/mcsst_mta_d"$curDateP1"_s0* >> $satSSTCatObs
  
  cat <<- oiEOF > $TMP/satelliteAParam
	$quarterMaskIn
	$fileStdevIn
	$satSSTCatObs
	$EOTfg
	$satADayGridOut
	$satANightGridOut
	$satADaySobsOut
	$satANightSobsOut
	$covNameSatA
	$curYear $curMonth $curDay 
	$satRunType
	$satASource
	$covSat
	$covSat2
	oiEOF

  cat $TMP/satelliteAParam

  $BIN/sat_navy_daily_firstguessclim.x -Wl,-T < $TMP/satelliteAParam

  checkStopcode sat_navy_daily_firstguessclim.x $?

  # make new file
  `> $satSSTCatObs`

  # concat all satellite B files

  ls $DATA_PRELIM_INPUT/avhrr6hr/$curYear/mcsst_mtb_d"$curDate"_s* 
  ls $DATA_PRELIM_INPUT/avhrr6hr/$curDateM1Year/mcsst_mtb_d"$curDateM1"_s2* 
  ls $DATA_PRELIM_INPUT/avhrr6hr/$curDateP1Year/mcsst_mtb_d"$curDateP1"_s0* 


  cat $DATA_PRELIM_INPUT/avhrr6hr/$curYear/mcsst_mtb_d"$curDate"_s* >> $satSSTCatObs
  cat $DATA_PRELIM_INPUT/avhrr6hr/$curDateM1Year/mcsst_mtb_d"$curDateM1"_s2* >> $satSSTCatObs
  cat $DATA_PRELIM_INPUT/avhrr6hr/$curDateP1Year/mcsst_mtb_d"$curDateP1"_s0* >> $satSSTCatObs

  cat <<- oiEOF > $TMP/satelliteBParam
	$quarterMaskIn
	$fileStdevIn
	$satSSTCatObs
	$EOTfg
	$satBDayGridOut
	$satBNightGridOut
	$satBDaySobsOut
	$satBNightSobsOut
	$covNameSatB
	$curYear $curMonth $curDay
	$satRunType 
	$satBSource
	$covSat
	$covSat2
	oiEOF

  cat $TMP/satelliteBParam

  $BIN/sat_navy_daily_firstguessclim.x -Wl,-T < $TMP/satelliteBParam
  checkStopcode sat_navy_daily_firstguessclim.x $?
  
  echo "End time for satellite quality control code =" `date '+%T'`
}

wgribDecode() {
 #$WGRIB $engIce -d 1 -text -o $iceGlobeReplaced
 # Change from old grib modeller engIce to NCEP production grib2 Sea Ice 
 $WGRIB -order we:ns $seaIce -d 1 -text $iceGlobeReplaced  
}

interpIce() {
if [ -e  $seaIce ]; then
    # first check if the date in the header is the processing day's date
    seaIceFileDate=`$WGRIB $seaIce -d 1 | gawk '{print substr($0,index($0,"d=")+2,8)}' -`

    if [ "$seaIceFileDate" != "$curDate" ]; then
      echo 'YELLOW-WARNING: date in the header does not match today'
      echo 'analysis day curDate=' $curDate' and seaIceFileDate='$seaIceFileDate
    fi

    if [ "$seaIceFileDate" != "" ]; then
    $WGRIB $seaIce -V -d 1 > $TMP/wgrib_info
      
      # The scan direction changed. The ice data is S-N instead N-S since 1981/11/01 to 1991/12/02
      varLat=`grep "lat 89.750000 to -89.750000 by 0.500000" $TMP/wgrib_info -c`
      varLon=`grep "lon 0.250000 to 359.750000 by 0.500000 #points=259200" $TMP/wgrib_info -c`
      varScan=`grep "input WE:NS output WE:SN" $TMP/wgrib_info -c`

      if [ "$varLon" -le "0" ]; then
        echo 'Wrong starting lon or size!'
        echo $varLon
      fi

      if [ "$varLat" -le "0" ]; then
        echo 'Wrong starting lat or size!'
        echo $varLat
      fi

      if [ "$varScan" -le "0" ]; then
        echo 'Wrong scan direction!'
        echo $varScan
      fi
        
        curYear00=`expr $seaIceFileDate | cut -c1-4`
        curMonth00=`expr $seaIceFileDate | cut -c5-6`
        curDay00=`expr $seaIceFileDate | cut -c7-8`
        
        # replace date masks with values
        iceGlobeReplaced=${iceGlobe//'yyyyMMdd'/$seaIceFileDate}
        iceGlobeReplaced=${iceGlobeReplaced//'yyyy'/$curYear00}
        
      if [ "$varLat" -ge "1" -a "$varLon" -ge "1" -a "$varScan" -ge "1" ]; then
        wgribDecode        
        
        iceCon720x360Replaced=${iceCon720x360//'yyyyMMdd'/$seaIceFileDate}
        iceCon720x360Replaced=${iceCon720x360Replaced//'yyyy'/$curYear00}

        iceCon1440x720Replaced=${iceCon1440x720//'yyyyMMdd'/$seaIceFileDate}
        iceCon1440x720Replaced=${iceCon1440x720Replaced//'yyyy'/$curYear00}

        cat <<- oiEOF > $TMP/ice_read.parm
	$quarterMaskIn
	$iceGlobeReplaced
	$iceCon720x360Replaced
	$iceCon1440x720Replaced
	$curYear00 $curMonth00 $curDay00
	$iwget
	oiEOF

        cat $TMP/ice_read.parm

        $BIN/interp_ice_half_to_quarter_deg.x -Wl,-T < $TMP/ice_read.parm
        checkStopcode interp_ice_half_to_quarter_deg.x $?

      else
        echo 'YELLOW-WARNING: error found in seaice data.'$curDate  
      fi
    else
      echo 'YELLOW-WARNING: No seaIceFileDate' $seaIceFileDate
    fi # end $seaIceFileDate != ""
  else
    echo 'YELLOW-WARNING: No ice data for analysis day' $curDate
fi # end [ -e  $seaIce]
}

convertMedianToSST() {
  if [ -e  $iceCon7day ]; then
    rm $iceCon7day
  fi

  dayCnt=1
  interval=-6
  iceConDays=0

  # 7 day ice
  while [ "$dayCnt" -le "7" ]; do
    # find the next ice day based on the interval
    if [ "$interval" -ne "0" ]; then
      TM0=`sh $UTIL/finddate.sh $curDate d$interval`
    else
      TM0=$curDate
    fi

    iceConYear=`echo $TM0 | cut -c1-4`
    echo "iceConYear = " $iceConYear
    iceConVal=${iceCon//'yyyyMMdd'/$TM0}
    echo "iceConVal = " $iceConVal
    iceConVal=${iceConVal//'yyyy'/$iceConYear}
                  
    echo "Appending $iceConVal to $iceCon7day file"
    
    if [ -e  $iceConVal ]; then
      iceConDays=`expr $iceConDays + 1`
      cat $iceConVal >> $iceCon7day
    else
      echo $iceConVal 'does not exist'
    fi

    dayCnt=$(($dayCnt + 1))
    interval=$(($interval + 1))
  done
  
  #calculate ncep ice sst
  if [ $iceConDays != 0 ]; then
    if [ ! -e $covNameIce ]; then
      `> $covNameIce`
    fi
    
    if [ -e  $iceCon7day ]; then
      cat <<- sstEOF > $TMP/ice2sst.parm
	$quarterMaskExtend
	$iceMask
	$iceFrzPnt.$curMonth$curDay
	$iceCon7day
	$iceConMed
	$iceSSTGrads
	$iceSobsOut
	$covNameIce
	$curYear $curMonth $curDay $iceConDays
	$covIce
	$covIce2
	sstEOF

      cat $TMP/ice2sst.parm

      $BIN/convert_median_ice_to_sst.x -Wl,-T < $TMP/ice2sst.parm
      checkStopcode convert_median_ice_to_sst.x $?
    else
      echo $iceCon_7day ' does not exist! Ice sobs can not be calculated'
      echo 'Warning! generate empty ice sobs'
    fi
  else
    echo 'Can not find ice for 7 days, Setting empty ice coverage.'


    cat <<- catEOF > $covNameIce
	0.0
	catEOF
  fi
}

icePrep() {
  interpIce
  convertMedianToSST
}

dataPrep() {
  echo '--- Starting QC: buoyship, satellite, ice obs'

  buoyshipPrep
  satellitePrep
  icePrep

  echo "End time for run ice quality control code =" `date '+%T'`

  echo '--- End of QC: buoyship, satellite, ice obs'
}
