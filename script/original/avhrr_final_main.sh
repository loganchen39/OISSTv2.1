#!/bin/bash

# --------------------------------------------------
#
# AVHRR final
#
# --------------------------------------------------

. $SCRIPT/avhrr_dataprep_finv2.sh
. $SCRIPT/eot_analysis.sh
. $SCRIPT/avhrr_output.sh

runEOTWT() {

	eotWeighting $curDateP3 $satADayWeightsP3 $satADaySuperobsPathFmt
	eotWeighting $curDateP3 $satANightWeightsP3 $satANightSuperobsPathFmt
	eotWeighting $curDateP3 $satBDayWeightsP3 $satBDaySuperobsPathFmt
	eotWeighting $curDateP3 $satBNightWeightsP3 $satBNightSuperobsPathFmt

}

runEOTCOR() {

  # EOT cor. weight filed for d+1
	satADayWTSFilesP1=(
	        [0]="$satADayWeightsM1"
	        [1]="$satADayWeights"
	        [2]="$satADayWeightsP1"
	        [3]="$satADayWeightsP2"
    	    [4]="$satADayWeightsP3" )
	satANightWTSFilesP1=(
	        [0]="$satANightWeightsM1"
	        [1]="$satANightWeights"
	        [2]="$satANightWeightsP1"
	        [3]="$satANightWeightsP2"
	        [4]="$satANightWeightsP3" )
	satBDayWTSFilesP1=(
	        [0]="$satBDayWeightsM1"
	        [1]="$satBDayWeights"
	        [2]="$satBDayWeightsP1"
	        [3]="$satBDayWeightsP2"
    	    [4]="$satBDayWeightsP3" )
	satBNightWTSFilesP1=(
	        [0]="$satBNightWeightsM1"
	        [1]="$satBNightWeights"
	        [2]="$satBNightWeightsP1"
	        [3]="$satBNightWeightsP2"
	        [4]="$satBNightWeightsP3" )
	
	# smoothing
	weightsSmoothingFactors="$dfw1 $dfw2 $dfw3 $dfw4 $dfw5"
	
  # EOT corr cur day + 1
  eotCorrection $curDateP1 $satADaySuperobsP1 'satADayWTSFilesP1' $satADayAnalysisP1 $satADayCorrectedP1
  eotCorrection $curDateP1 $satANightSuperobsP1 'satANightWTSFilesP1' $satANightAnalysisP1 $satANightCorrectedP1
  eotCorrection $curDateP1 $satBDaySuperobsP1 'satBDayWTSFilesP1' $satBDayAnalysisP1 $satBDayCorrectedP1
  eotCorrection $curDateP1 $satBNightSuperobsP1 'satBNightWTSFilesP1' $satBNightAnalysisP1 $satBNightCorrectedP1
  
}  

runOutput() {

	output $sourceString $iceSource $satPlatform $satSensor  
	
}	

runOI() {

	cat <<- inxEOF > $TMP/oiFinalParam
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
		$buoyNSRscaled
		$buoySuperobsP1
		$argoNSRscaled
		$argoSuperobsP1
		$shipNSRscaled
		$shipObsCorrectedP1
		$satDayNSRscaled
		$satADayCorrectedP1
		$satNightNSRscaled
		$satANightCorrectedP1
		$satDayNSRscaled
		$satBDayCorrectedP1
		$satNightNSRscaled
		$satBNightCorrectedP1
		$iceNSRscaled
		$iceSuperobsP1
		$buoyNSRscaled
		$buoySuperobsM1
		$argoNSRscaled
		$argoSuperobsM1
		$shipNSRscaled
		$shipObsCorrectedM1
		$satDayNSRscaled
		$satADayCorrectedM1
		$satNightNSRscaled
		$satANightCorrectedM1
		$satDayNSRscaled
		$satBDayCorrectedM1
		$satNightNSRscaled
		$satBNightCorrectedM1
		$iceNSRscaled
		$iceSuperobsM1
	inxEOF

	cat $TMP/oiFinalParam
	
	$BIN/calc_oisst.x  -Wl,-T < $TMP/oiFinalParam
	checkStopcode calc_oisst.x $?	
	
}

AVHRRFinal() {

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

  
	# --------------------------------------------------
	#
	# EOT
	#
	# --------------------------------------------------
	# Setup eot wt final runs' parameter maps.
	# These maps are used to organize the EOT weight files needed for the 1-day-back-to-3-days-ahead processing of the current day.
	# 2x satellites w/ both day and night = 20 EOT weight files needed for a single EOT correction run.
	#
	# The following are paramters for each day of EOT weighting.
	#
	# d-3
  #eotWeighting $curDateM3 $satADayWeightsM3 $satADaySuperobsPathFmt
	#eotWeighting $curDateM3 $satANightWeightsM3 $satANightSuperobsPathFmt
	#eotWeighting $curDateM3 $satBDayWeightsM3 $satBDaySuperobsPathFmt
	#eotWeighting $curDateM3 $satBNightWeightsM3 $satBNightSuperobsPathFmt
	# d-2
  #eotWeighting $curDateM2 $satADayWeightsM2 $satADaySuperobsPathFmt
	#eotWeighting $curDateM2 $satANightWeightsM2 $satANightSuperobsPathFmt
	#eotWeighting $curDateM2 $satBDayWeightsM2 $satBDaySuperobsPathFmt
	#eotWeighting $curDateM2 $satBNightWeightsM2 $satBNightSuperobsPathFmt
	# d-1

#eotWeighting $curDateM1 $satADayWeightsM1 $satADaySuperobsPathFmt
#eotWeighting $curDateM1 $satANightWeightsM1 $satANightSuperobsPathFmt
#eotWeighting $curDateM1 $satBDayWeightsM1 $satBDaySuperobsPathFmt
#eotWeighting $curDateM1 $satBNightWeightsM1 $satBNightSuperobsPathFmt
	# d

#	eotWeighting $curDate $satADayWeights $satADaySuperobsPathFmt
#	eotWeighting $curDate $satANightWeights $satANightSuperobsPathFmt
#	eotWeighting $curDate $satBDayWeights $satBDaySuperobsPathFmt
#	eotWeighting $curDate $satBNightWeights $satBNightSuperobsPathFmt
	# d+1

#eotWeighting $curDateP1 $satADayWeightsP1 $satADaySuperobsPathFmt
#eotWeighting $curDateP1 $satANightWeightsP1 $satANightSuperobsPathFmt
#eotWeighting $curDateP1 $satBDayWeightsP1 $satBDaySuperobsPathFmt
#eotWeighting $curDateP1 $satBNightWeightsP1 $satBNightSuperobsPathFmt 
	# d+2

#eotWeighting $curDateP2 $satADayWeightsP2 $satADaySuperobsPathFmt
#eotWeighting $curDateP2 $satANightWeightsP2 $satANightSuperobsPathFmt
#eotWeighting $curDateP2 $satBDayWeightsP2 $satBDaySuperobsPathFmt
#eotWeighting $curDateP2 $satBNightWeightsP2 $satBNightSuperobsPathFmt 
	# d+3

eotWeighting $curDateP3 $satADayWeightsP3 $satADaySuperobsPathFmt
eotWeighting $curDateP3 $satANightWeightsP3 $satANightSuperobsPathFmt
eotWeighting $curDateP3 $satBDayWeightsP3 $satBDaySuperobsPathFmt
eotWeighting $curDateP3 $satBNightWeightsP3 $satBNightSuperobsPathFmt
	
	# EOT cor. weight files for d-1
	satADayWTSFilesM1=(
	        [0]="$satADayWeightsM3"
	        [1]="$satADayWeightsM2"
	        [2]="$satADayWeightsM1"
	        [3]="$satADayWeights"
    	    [4]="$satADayWeightsP1" )
	satANightWTSFilesM1=(
	        [0]="$satANightWeightsM3"
	        [1]="$satANightWeightsM2"
	        [2]="$satANightWeightsM1"
	        [3]="$satANightWeights"
	        [4]="$satANightWeightsP1" )
	satBDayWTSFilesM1=(
	        [0]="$satBDayWeightsM3"
	        [1]="$satBDayWeightsM2"
	        [2]="$satBDayWeightsM1"
	        [3]="$satBDayWeights"
    	    [4]="$satBDayWeightsP1" )
	satBNightWTSFilesM1=(
	        [0]="$satBNightWeightsM3"
	        [1]="$satBNightWeightsM2"
	        [2]="$satBNightWeightsM1"
	        [3]="$satBNightWeights"
	        [4]="$satBNightWeightsP1" )
	
	# EOT cor. weight files for current day
	satADayWTSFiles=(
	        [0]="$satADayWeightsM2"
	        [1]="$satADayWeightsM1"
	        [2]="$satADayWeights"
	        [3]="$satADayWeightsP1"
    	    [4]="$satADayWeightsP2" )
	satANightWTSFiles=(
	        [0]="$satANightWeightsM2"
	        [1]="$satANightWeightsM1"
	        [2]="$satANightWeights"
	        [3]="$satANightWeightsP1"
	        [4]="$satANightWeightsP2" )
	satBDayWTSFiles=(
	        [0]="$satBDayWeightsM2"
	        [1]="$satBDayWeightsM1"
	        [2]="$satBDayWeights"
	        [3]="$satBDayWeightsP1"
    	    [4]="$satBDayWeightsP2" )
	satBNightWTSFiles=(
	        [0]="$satBNightWeightsM2"
	        [1]="$satBNightWeightsM1"
	        [2]="$satBNightWeights"
	        [3]="$satBNightWeightsP1"
	        [4]="$satBNightWeightsP2" )
    
  # EOT cor. weight filed for d+1
	satADayWTSFilesP1=(
	        [0]="$satADayWeightsM1"
	        [1]="$satADayWeights"
	        [2]="$satADayWeightsP1"
	        [3]="$satADayWeightsP2"
    	    [4]="$satADayWeightsP3" )
	satANightWTSFilesP1=(
	        [0]="$satANightWeightsM1"
	        [1]="$satANightWeights"
	        [2]="$satANightWeightsP1"
	        [3]="$satANightWeightsP2"
	        [4]="$satANightWeightsP3" )
	satBDayWTSFilesP1=(
	        [0]="$satBDayWeightsM1"
	        [1]="$satBDayWeights"
	        [2]="$satBDayWeightsP1"
	        [3]="$satBDayWeightsP2"
    	    [4]="$satBDayWeightsP3" )
	satBNightWTSFilesP1=(
	        [0]="$satBNightWeightsM1"
	        [1]="$satBNightWeights"
	        [2]="$satBNightWeightsP1"
	        [3]="$satBNightWeightsP2"
	        [4]="$satBNightWeightsP3" )
	
	# smoothing
	weightsSmoothingFactors="$dfw1 $dfw2 $dfw3 $dfw4 $dfw5"
	
  # EOT corr d-1
	#eotCorrection $curDateM1 $satADaySuperobsM1 'satADayWTSFilesM1' $satADayAnalysisM1 $satADayCorrectedM1
  #eotCorrection $curDateM1 $satANightSuperobsM1 'satANightWTSFilesM1' $satANightAnalysisM1 $satANightCorrectedM1
  #eotCorrection $curDateM1 $satBDaySuperobsM1 'satBDayWTSFilesM1' $satBDayAnalysisM1 $satBDayCorrectedM1
  #eotCorrection $curDateM1 $satBNightSuperobsM1 'satBNightWTSFilesM1' $satBNightAnalysisM1 $satBNightCorrectedM1
  # EOT corr cur day
  #eotCorrection $TM $satADaySuperobs 'satADayWTSFiles' $satADayAnalysis $satADayCorrected
  #eotCorrection $TM $satANightSuperobs 'satANightWTSFiles' $satANightAnalysis $satANightCorrected
  #eotCorrection $TM $satBDaySuperobs 'satBDayWTSFiles' $satBDayAnalysis $satBDayCorrected
  #eotCorrection $TM $satBNightSuperobs 'satBNightWTSFiles' $satBNightAnalysis $satBNightCorrected
  # EOT corr d+1

  eotCorrection $curDateP1 $satADaySuperobsP1 'satADayWTSFilesP1' $satADayAnalysisP1 $satADayCorrectedP1
  eotCorrection $curDateP1 $satANightSuperobsP1 'satANightWTSFilesP1' $satANightAnalysisP1 $satANightCorrectedP1
  eotCorrection $curDateP1 $satBDaySuperobsP1 'satBDayWTSFilesP1' $satBDayAnalysisP1 $satBDayCorrectedP1
  eotCorrection $curDateP1 $satBNightSuperobsP1 'satBNightWTSFilesP1' $satBNightAnalysisP1 $satBNightCorrectedP1


runOI

runOutput
     
}
