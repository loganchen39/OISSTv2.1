#! /bin/bash

# --------------------------------------------------
#
# EOT Analysis
#
# --------------------------------------------------

ulimit -s unlimited

function getBuoyDirLen() {
  path=${buoySuperobsPathFormat%/*}
  echo ${#path}
}

function getArgoDirLen() {
  path=${argoSuperobsPathFormat%/*}
  echo ${#path}
}

function getSatDirLen() {
  path=${satFile%/*}
  echo ${#path}
}

# --------------------------------------------------
# Procedure to run the EOT weight Fortran routines.
# --------------------------------------------------
eotWeighting() {
  local wtDate=$1
  local wtsFile=$2
  local satFile=$3

  paramCatFile=$TMP/eotWtParam
	
	buoyDirLen=$(getBuoyDirLen)
	argoDirLen=$(getArgoDirLen)
	satDirLen=$(getSatDirLen)
	yrbuoy=`expr $buoyDirLen - 3`
	yrargo=`expr $argoDirLen - 3`
	yrship=`expr $buoyDirLen - 3`
	yrsat=`expr $satDirLen - 3`

	# The 1st eotwt4 input record is 5 minz,njj,minob,ioff,joff
	# The 2nd eotwt4 input record is 5 parameters: nby,nsh,nst,ndays,nmodes
	# The 3rd record is 3 parameters: yrbuoy yrship yrsat
	# The 4th record is 3 parameters: daybuoy dayship daysat
	# The 5th record is 3 paramters: drej crit
	# The 6th input record is the date
	# All remaining records are input and output file names which must be in order
	# The 1st file name is the two-degree land/sea mask
	# The 2nd file name is the monthly climate (1-deg)
	# The 3rd file name is the input EOT modes (filled over land with 0)
	# The 4th file name is the analyzed weights (output) 
	# The 5th file name is the input buoy data 
	# The 6th file name is the input ship data 
	# The 7th file name is the input satellite data
	# Multitple dates are read in and the complete file names
	#   for these last 3 files are computed in the fortran code
	# *Note all satellite fields, weights differ for day and night
	cat <<- inxEOF > $paramCatFile
		$ndays
		$yrbuoy $yrargo $yrship $yrsat
		$wtDate
		$useFutureSuperobs
		$twoDegMask
		$twoDegClim
		$filledModes
		$wtsFile
		$buoySuperobsPathFormat
		$argoSuperobsPathFormat
		$shipObsCorrectedPathFormat
		$satFile
	inxEOF
	
	cat $paramCatFile
	
	$BIN/eotbias-wt4-col.x -Wl,-T  < $paramCatFile
	checkStopcode eotbias-wt4-col.x $?
}

# --------------------------------------------------
# Runs the EOT correction Fortran routines.
# --------------------------------------------------
eotCorrection() {
  local eotCorDate=$1
  local satInFile=$2
  local wtsArrName=$3
  local analysisFile=$4
  local satCorrFile=$5

  eval "wtsFiles=( \${$wtsArrName[@]} )"
	
  paramCatFile=$TMP/eotCorrParam
	
	#  The 1st eotbias-cor4 input record has 2 parameters: nmodes, nuse
	#  The 2nd eotbias-cor4 input record depends on nuse and has 1-9 parameters: dwf1,dwf2,...
	#  The next input record is the date: computed by script 8 digits
	#  All remaining records are input and output file names which must be in order
	#  The 1st file name is the two-degree land/sea mask
	#  The 2nd file name is the input EOT modes (filled over land with 0)
	#  The 3rd file name is the variance of each of the 130 modes
	#  Nuse (1-9) now determines the number of files read. They will be weighted by dwf
	#        The weighting smoothing factors are applied in order: dwf1, ...
	#  The next file name is the input uncorrected satellite data
	#  The next file name is the analyzed weights 
	#  The next file name is the output bias error estimate
	#  The final file name is the output bias corrected satellite data 
	#
	#  Note all satellite fields, weights and biases differ for day and night
	cat <<- inxEOF > $paramCatFile
		$modes $nuse
		$weightsSmoothingFactors
		$eotCorDate
		$twoDegMask
		$filledModes
		$modeVariance
		$satInFile
	inxEOF
	
	# append all of the WTS files
	for wtsFile in "${wtsFiles[@]}"; do
		cat <<- inxEOF >> $paramCatFile
			$wtsFile
		inxEOF
	done
	
	cat <<- inxEOF >> $paramCatFile
		$analysisFile
		$satCorrFile
	inxEOF
	
	cat $paramCatFile
	
	$BIN/eotbias-cor4-col.x -Wl,-T < $paramCatFile
	checkStopcode eotbias-cor4-col.x $?
}
