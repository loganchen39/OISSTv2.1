#! /bin/bash

create_grads_ctl_files()  {

  sstIntv2File=$TMP/intv2.ctl
  sstFinv2File=$TMP/finv2.ctl
  anomClimFile=$TMP/oiv2clm_1440x720.ctl

  echo DSET $DATA_PRELIM_OUT/oiout/$curYear/sst4-metopab-eot-intv2.$curYear$curMonth$curDay > $sstIntv2File
  echo 'options sequential big_endian' >> $sstIntv2File
  echo '* skip fortran leading and trailing 8 bytes' >> $sstIntv2File
  echo 'XYHEADER 12' >> $sstIntv2File
  echo '* skip year and month and day' >> $sstIntv2File
  echo 'UNDEF  -9999.0' >> $sstIntv2File
  echo 'TITLE   oi 1/4 daily grid (deg C)' >> $sstIntv2File
  echo '*' >> $sstIntv2File
  echo 'XDEF 1440 LINEAR  0.125  0.25' >> $sstIntv2File
  echo '* ' >> $sstIntv2File
  echo 'YDEF 720 LINEAR  -89.875 0.25' >> $sstIntv2File
  echo '*' >> $sstIntv2File
  echo 'ZDEF  1 LEVELS   1  ' >> $sstIntv2File
  echo '* ' >> $sstIntv2File
  echo 'TDEF  1  LINEAR '$curDay$monthAbbreviation$curYear' 1dy' >> $sstIntv2File
  echo '*' >> $sstIntv2File
  echo 'VARS 3' >> $sstIntv2File
  echo 'sst      0   99   sst analysis with OI bias correction intv2' >> $sstIntv2File
  echo 'rsvar    0   99   random+sampling error      ' >> $sstIntv2File
  echo 'bsvar    0   99   bias error ' >> $sstIntv2File
  echo 'ENDVARS  ' >> $sstIntv2File

  echo DSET $DATA_FINAL_OUT/oiout/$curYear/sst4-metopab-eot-finv2.$curYear$curMonth$curDay > $sstFinv2File
  echo 'options sequential big_endian' >> $sstFinv2File
  echo '* skip fortran leading and trailing 8 bytes' >> $sstFinv2File
  echo 'XYHEADER 12' >> $sstFinv2File
  echo '* skip year and month and day' >> $sstFinv2File
  echo 'UNDEF  -9999.0' >> $sstFinv2File
  echo 'TITLE   oi 1/4 daily grid (deg C)' >> $sstFinv2File
  echo '*' >> $sstFinv2File
  echo 'XDEF 1440 LINEAR  0.125  0.25' >> $sstFinv2File
  echo '* ' >> $sstFinv2File
  echo 'YDEF 720 LINEAR  -89.875 0.25' >> $sstFinv2File
  echo '*' >> $sstFinv2File
  echo 'ZDEF  1 LEVELS   1  ' >> $sstFinv2File
  echo '* ' >> $sstFinv2File
  echo 'TDEF  1  LINEAR '$curDay$monthAbbreviation$curYear' 1dy' >> $sstFinv2File
  echo '*' >> $sstFinv2File
  echo 'VARS 3' >> $sstFinv2File
  echo 'sst        0   99   sst with OI bias correction finv2' >> $sstFinv2File
  echo 'rsvar      0   99   random+sampling error      ' >> $sstFinv2File
  echo 'bsvar      0   99   bias error ' >> $sstFinv2File
  echo 'ENDVARS  ' >> $sstFinv2File

  echo DSET $climpath  > $anomClimFile
  echo 'options sequential big_endian ' >> $anomClimFile
  echo '* skip fortran leading and trailing 8 bytes ' >> $anomClimFile
  echo 'XYHEADER 12 ' >> $anomClimFile
  echo '* skip year and month and day ' >> $anomClimFile
  echo 'UNDEF  -9999.0 ' >> $anomClimFile
  echo 'TITLE   oi 1/4 daily grid (deg C)'  >> $anomClimFile
  echo '* ' >> $anomClimFile
  echo 'XDEF 1440 LINEAR  0.125  0.25 ' >> $anomClimFile
  echo '*  ' >> $anomClimFile
  echo 'YDEF 720 LINEAR  -89.875 0.25 ' >> $anomClimFile
  echo '* ' >> $anomClimFile
  echo 'ZDEF  1 LEVELS   1   ' >> $anomClimFile
  echo '*  ' >> $anomClimFile
  echo 'TDEF  1  LINEAR '$curDay$monthAbbreviation$curYear' 1dy' >> $anomClimFile
#  echo 'TDEF 13000 LINEAR 1jan1982 1dy ' >> $anomClimFile
  echo '* ' >> $anomClimFile
  echo 'VARS 1 ' >> $anomClimFile
  echo 'clm       0   99   sst oi.v2 climate (1971-2000)'  >> $anomClimFile
  echo 'ENDVARS ' >> $anomClimFile

}

