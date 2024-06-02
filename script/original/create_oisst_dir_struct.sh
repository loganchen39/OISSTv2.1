#!/bin/bash

if [ $# -gt 0 ]
then
   YEAR=$1
else
   echo -e "Please enter year to add to dir tree:"
   read YEAR
fi

echo $YEAR

mkdir -p ./prelim/work/obs/buoyship/$YEAR
mkdir -p ./prelim/work/obs/ice/$YEAR
mkdir -p ./prelim/work/obs/satsst/$YEAR

mkdir -p ./prelim/work/sobs/buoyship/$YEAR
mkdir -p ./prelim/work/sobs/ice/$YEAR
mkdir -p ./prelim/work/sobs/metopa/$YEAR
mkdir -p ./prelim/work/sobs/metopb/$YEAR

mkdir -p ./prelim/work/eotwt/metopa/$YEAR
mkdir -p ./prelim/work/eotwt/metopb/$YEAR

mkdir -p ./prelim/work/eotbias/metopa/$YEAR
mkdir -p ./prelim/work/eotbias/metopb/$YEAR

mkdir -p ./prelim/work/eotcor/metopa/$YEAR
mkdir -p ./prelim/work/eotcor/metopb/$YEAR

mkdir -p ./prelim/work/grid/buoyship/$YEAR

mkdir -p ./prelim/work/grid/ice/con/$YEAR
mkdir -p ./prelim/work/grid/ice/con-med/$YEAR
mkdir -p ./prelim/work/grid/ice/ice-sst/$YEAR

mkdir -p ./prelim/work/grid/metopa/$YEAR
mkdir -p ./prelim/work/grid/metopb/$YEAR

mkdir -p ./prelim/out/oiout/$YEAR
mkdir -p ./prelim/out/NetCDF/$YEAR
mkdir -p ./prelim/out/NetCDF/GHRSST/$YEAR
mkdir -p ./prelim/out/log/$YEAR
mkdir -p ./prelim/out/map/$YEAR

#-------------------------------------------------------------------------------

mkdir -p ./final/work/obs/buoyship/$YEAR
mkdir -p ./final/work/obs/ice/$YEAR
mkdir -p ./final/work/obs/satsst/$YEAR

mkdir -p ./final/work/sobs/buoyship/$YEAR
mkdir -p ./final/work/sobs/ice/$YEAR
mkdir -p ./final/work/sobs/metopa/$YEAR
mkdir -p ./final/work/sobs/metopb/$YEAR

mkdir -p ./final/work/eotwt/metopa/$YEAR
mkdir -p ./final/work/eotwt/metopb/$YEAR

mkdir -p ./final/work/eotbias/metopa/$YEAR
mkdir -p ./final/work/eotbias/metopb/$YEAR

mkdir -p ./final/work/eotcor/metopa/$YEAR
mkdir -p ./final/work/eotcor/metopb/$YEAR

mkdir -p ./final/work/grid/$YEAR
mkdir -p ./final/work/grid/buoyship/$YEAR
mkdir -p ./final/work/grid/ice/con/$YEAR
mkdir -p ./final/work/grid/ice/con-med/$YEAR
mkdir -p ./final/work/grid/ice/ice-sst/$YEAR
mkdir -p ./final/work/grid/metopa/$YEAR
mkdir -p ./final/work/grid/metopb/$YEAR

mkdir -p ./final/out/oiout/$YEAR
mkdir -p ./final/out/NetCDF/$YEAR
mkdir -p ./final/out/NetCDF/GHRSST/$YEAR
mkdir -p ./final/out/log/$YEAR
mkdir -p ./final/out/map/$YEAR

