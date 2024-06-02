#!/bin/bash

# --------------------------------------------------
#
# AVHRR interim
#
# --------------------------------------------------

ulimit -s unlimited

. $SCRIPT/avhrr_dataprep.sh
. $SCRIPT/avhrr_output.sh
. $SCRIPT/eot_analysis.sh

runEOTWT() {

  eotWeighting $curDate $satADayWeights $satADaySuperobs
  eotWeighting $curDate $satANightWeights $satANightSuperobs
  eotWeighting $curDate $satBDayWeights $satBDaySuperobs
  eotWeighting $curDate $satBNightWeights $satBNightSuperobs
  
}

runEOTCOR() {
  weightsSmoothingFactors="$dfw1"

  eotCorrection $curDate $satADaySuperobs 'satADayWeights' $satADayAnalysis $satADayCorrected
  eotCorrection $curDate $satANightSuperobs 'satANightWeights' $satANightAnalysis $satANightCorrected
  eotCorrection $curDate $satBDaySuperobs 'satBDayWeights' $satBDayAnalysis $satBDayCorrected
  eotCorrection $curDate $satBNightSuperobs 'satBNightWeights' $satBNightAnalysis $satBNightCorrected
  
}

  
runOI() {
  echo 'ok'
	# --------------------------------------------------
	#
	# OI
	#
	# --------------------------------------------------
	cat <<- inxEOF > $TMP/oiInterimParam
		$oiTitle
		$curDate
		$rmax $nmax $ifil $rej $nbias $nsat $iwarn
		$cor4smStat
		$pathIncrVar
		$errorCorStat
		$fgOisst
		$quarterMaskExtend
		$clim4
		$oisst
		$residualStat
		$satADayAnalysis
		$satANightAnalysis
		$satBDayAnalysis
		$satBNightAnalysis
		$buoyNSR
		$buoySuperobs
		$argoNSR
		$argoSuperobs
		$shipNSR
		$shipObsCorrected
		$satADayNSR
		$satADayCorrected
		$satANightNSR
		$satANightCorrected
		$satBDayNSR
		$satBDayCorrected
		$satBNightNSR
		$satBNightCorrected
		$iceNSR
		$iceSuperobs
	inxEOF

  cat $TMP/oiInterimParam
	
	$BIN/calc_oisst.x -Wl,-T < $TMP/oiInterimParam
	checkStopcode calc_oisst.x $?
}

runOutput() {

	output $sourceString $iceSource $satPlatform $satSensor
	
}	
	
AVHRRInterim() {
	# --------------------------------------------------
	#
	# QC and data prep.
	#
	# --------------------------------------------------
	
	#createSourceString $buoyshipSSTDaily $engIce ${satAFiles[1]} ${satBFiles[1]}
	createSourceString $buoyshipSSTDaily $seaIce ${satAFiles[1]} ${satBFiles[1]}
	  
  case "$2" in
    'sobs')
      echo "Just processing superobs"
      dataPrep
      echo "Finished processing superobs"
      return 1
    ;;
    
    'eotwt')
      echo "Just processing eot weights"
      runEOTWT
      echo "Finished processing eot weight"
      return 1
    ;;
    
    'eotcor')
      echo "Just processing eot correction"
      runEOTCOR
      echo "Finished processing eot correction"
      return 1
    ;;
    
    'oi')
      echo "Just processing oi"
      runOI
      echo "Finished processing oi"
      return 1
    ;;
    
    'output')
      echo "Just processing output"
      runOutput
      echo "Finished processing output"
      return 1
    ;;
    
    'nosobs')
      echo "Using previous superobs"
    ;;
    
    *)
      dataPrep
  esac	

  runEOTWT
  
  runEOTCOR

	runOI
	
	# --------------------------------------------------
	#
	# Generate output files (GrADS, NetCDF, etc.)
	#
	# --------------------------------------------------
	output $sourceString $iceSource $satPlatform $satSensor
}
