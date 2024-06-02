#! /bin/bash

. $UTIL/month_abbreviation.sh
. $UTIL/create_mnf.sh
. $SCRIPT/create_grads_ctl_files.sh

export GADDIR GASCRP

leapyear() { 
 
 [ $[$1%400] -eq 0 ] && { climpath=$climpath_leap;return;}
 [ $[$1%100] -eq 0 ] && { climpath=$climpath_noleap;return;}
 [ $[$1%4] -eq 0 ] && climpath=$climpath_leap || climpath=$climpath_noleap
 
}

output() {
	todayYear=`date '+%Y'`
	todayMon=`date '+%m'`
	today=`date '+%d'`
	todayHour=`date '+%H'`
	todayMin=`date '+%M'`
	# --------------------------------------------------
	#
	# GrADS
	#
	# --------------------------------------------------
	monthAbbreviation=$(getMonthAbbreviation $curMonth)
	
  leapyear $curYear
  
  create_grads_ctl_files
    
	echo climpath    $climpath
	
	cat <<- rdEOF > $outputRunDate
		$curDay $monthAbbreviation $curYear
	rdEOF
  
	cat <<- anomEOF > $TMP/totalParam
		run $TotalGS	
		$dailyCTL
		$outputRunDate
		$runTypeAbrv
	anomEOF
	
	cat <<- anomEOF > $TMP/anomParam
		run $AnomGS	
		$dailyCTL
		$climCTL
		$outputRunDate
		$runTypeAbrv
	anomEOF
	
	echo 'Starting GrADS'
	
		cat $TMP/totalParam
		$GRADS -bl < $TMP/totalParam

		cat $TMP/anomParam
		$GRADS -bl < $TMP/anomParam
	
		# rename the GrADS outputs
		mv navy-sst-s-$curDay$monthAbbreviation$curYear.gif $gradsOutDir/navy-sst-s-$curDate.gif
		mv navy-sst-b-$curDay$monthAbbreviation$curYear.gif $gradsOutDir/navy-sst-b-$curDate.gif
		mv navy-anom-s-$curDay$monthAbbreviation$curYear.gif $gradsOutDir/navy-anom-s-$curDate.gif
		mv navy-anom-b-$curDay$monthAbbreviation$curYear.gif $gradsOutDir/navy-anom-b-$curDate.gif

		# Copy the GrADS outputs to published directories
		cp $gradsOutDir/navy-sst-s-$curDate.gif $gradsPubDir/navy-sst-ss.gif 
		cp $gradsOutDir/navy-sst-b-$curDate.gif $gradsPubDir/navy-sst-bb.gif 
		cp $gradsOutDir/navy-anom-s-$curDate.gif $gradsPubDir/navy-anom-ss.gif 
		cp $gradsOutDir/navy-anom-b-$curDate.gif $gradsPubDir/navy-anom-bb.gif 
																
		# remove GrADS partial files
		rm *.prt
	
	echo 'GrADS output complete.'
	echo 'NetCDF output starting.'	
	# --------------------------------------------------
	#
	# NetCDF output
	#
	# --------------------------------------------------
	# read coverage files
	coverage=`cat $covName`
	buoyCov=${coverage:0:4}
	argoCov=${coverage:5:4}
	shipCov=${coverage:10:4}
	
	coverage=`cat $covNameSatA`
	noaa19DayCov=${coverage:0:4}
	noaa19NightCov=${coverage:5:4}
	
	coverage=`cat $covNameSatB`
	metopDayCov=${coverage:0:4}
	metopNightCov=${coverage:5:4}
	
	coverage=`cat $covNameIce`
	iceCov=${coverage:0:4}
	
	# NOMADS
	# double the "$curYear $curMonth $curDay"; start date == end date for interim
	cat <<- netcdfEOF > $TMP/nomadsParam
		$netCDFrunType
		$curYear $curMonth $curDay $curYear $curMonth $curDay
		$oisst
		$nomadsCDFpath
		$nomadsIEEEpath
		$climpath
		$iceConMedNetcdf
		$todayYear,$todayMon,$today,$todayHour,$todayMin
	netcdfEOF
	#$buoyCov,$shipCov,$noaa19DayCov,$noaa19NightCov,$metopDayCov,$metopNightCov,$iceCov
	
	cat $TMP/nomadsParam
	
	$BIN/process_nomads_ncfiles.x -Wl,-T < $TMP/nomadsParam
	checkStopcode process_nomads_ncfiles.x $?
	
	echo "End time for run NOMADS NetCDF =" `date '+%T'`	
	
	uuid=$(uuidgen)
	
	echo $uuid
	
	# GHRSST
	cat <<- netcdfEOF > $TMP/ghrsstParam
		$netCDFrunType
		$curYear $curMonth $curDay $curYear $curMonth $curDay
		$quarterMaskIn
		$oisst
		$ghrsstCDFpath
		$climpath 
		$iceConMedNetcdf
		$todayYear $todayMon $today
		$sourceString
		$iceSource
		$satPlatform
		$satSensor
		$uuid
	netcdfEOF
	
	cat $TMP/ghrsstParam

	$BIN/process_ghrsst_ncfiles.x -Wl,-T < $TMP/ghrsstParam
	checkStopcode process_ghrsst_ncfiles.x $?
	
	# FR files
	cat <<- netcdfEOF > $TMP/ghrsstMetaParam
		$curYear $curMonth $curDay $curYear $curMonth $curDay
		$todayYear $todayMon $today
		$frPath
	netcdfEOF
	
	cat $TMP/ghrsstMetaParam
	
	$BIN/write_ghrsst_metadata_xml.x -Wl,-T < $TMP/ghrsstMetaParam 
	checkStopcode write_ghrsst_metadata_xml.x $?
	
	echo "End time for run GHRSST NetCDF =" `date '+%T'`
	
	# ----------------------------------------------------
	#
	# Package files, create checksums and filesize if used 
	# 
	# ----------------------------------------------------
        # Previous code
	#echo "Creating NCDC ingest manifest for "$nomadsCDFpath
	#CreateMNF $nomadsCDFpath
	#echo "Zipping NOMADS and GHRSST output files"
	#gzip --force $nomadsCDFpath
	#gzip --force $nomadsIEEEpath
	#bzip2 --force $ghrsstCDFpath
        #md5sum $nomadsCDFpath > $nomadsCDFpath.mnf
        #md5sum $frPath > $frPath.md5
        #md5sum $ghrsstCDFpath > $ghrsstCDFpath.md5

	echo "Creating mnf checksums"

        # 
	# manifest GRHSST for (JPL-NASA) NetCDF and xml file with md5
        #
        # Adjustments added by WRH 20170405 to work with
        # operational government servers - oisst- (dev test prod)
        #
        currentPath=$PWD
        echo "This is the current path to set back to after the manifest work is done =" $currentPath
  
        nomadsCDFfile=$(basename $nomadsCDFpath)
        XMLfile=$(basename $frPath)
        ghrsstCDFfile=$(basename $ghrsstCDFpath)

        # Do the GHRSST either prelim/final NetCDF file - config is overloaded
        cd $ghrsstCDFpathOnly
        # filesize=$(du -b $ghrsstCDFfile | awk '{ print $1 }')
        fileMD5=$(md5sum $ghrsstCDFfile | awk '{ print $1 }')
        echo "$fileMD5 $ghrsstCDFfile" > $ghrsstCDFfile.md5
                        
        # Copy the GRHSST outputs (nc and md5) to (JPL-NASA) pickup directories where sftp happens
	# Clear out directory first. Leave commented out for now

        echo "GHRSST GDS 2.0 to NASA-JPL path =" $ghrsstCDFJPLpathOnly 
	cd $ghrsstCDFJPLpathOnly 	 
	rm -f *.nc
	rm -f *.md5
	cd $ghrsstCDFpathOnly
	cp $ghrsstCDFfile $ghrsstCDFJPLpathOnly/$ghrsstCDFfile  
	cp $ghrsstCDFfile.md5 $ghrsstCDFJPLpathOnly/$ghrsstCDFfile.md5        

        # Do the GHRSST either prelim/final XML file - config is overloaded
        cd $frPathOnly
        # filesize=$(du -b $XMLfile | awk '{ print $1 }')
        fileMD5=$(md5sum $XMLfile | awk '{ print $1 }')
        echo "$fileMD5  $XMLfile" > $XMLfile.md5

        # Do the NOMADS for (NCEI archive) either prelim/final NetCDF file - config is overloaded
        cd $nomadsCDFpathOnly
	# stat works on the FQFN
	# filesize=$(stat -c%s "$nomadsCDFpath")
        filesize=$(du -b $nomadsCDFfile | awk '{ print $1 }')
        fileMD5=$(md5sum $nomadsCDFfile | awk '{ print $1 }')
        echo "$nomadsCDFfile,$fileMD5,$filesize" > $nomadsCDFfile.mnf

        # cd back to previous path for oisst.sh script
        cd $currentPath

}

