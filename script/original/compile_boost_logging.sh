#! /bin/sh

# Script to compile OISST Boost Log dependency.
#
# Fill in missing directory fields and run to create dependency lib objects and include files.

# installation directories
LIB=''
INCLUDE=''

# oisst root directory
OISST=''

boostLibs=filesystem,system,date_time,thread,regex,log

tar -C $LIB -xpf $OISST/lib/boost-log-min.tar.gz

pushd $LIB/boost-log-min
	./bootstrap.sh --libdir=$LIB --includedir=$INCLUDE --with-libraries=$boostLibs
	./b2 install -d0 -j 4 cxxflags="-fPIC -O0"
popd
