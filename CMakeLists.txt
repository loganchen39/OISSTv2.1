# Description: CMake project file for oisst_v2.1
# Author: Ligang Chen
# Created: 07/19/2021
# Last Modified: 07/19/2021

# oisstv2.1
cmake_minimum_required(VERSION 3.12)
project(oisst VERSION 2.1 LANGUAGES Fortran C CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(oisst_compiler_flags)

# Method 1 to use NetCDF library, need to provide FindNetCDF.cmake, suitable for compliated NetCDF installation.
# find_package(NetCDF REQUIRED COMPONENTS Fortran)
# # find_package(Boost REQUIRED COMPONENTS log)  # Ligang: not to use yet.
# include_directories(${NETCDF_INCLUDES})

# Method 2: simpler way to use NetCDF lib without the file FindNetCDF.cmake, more suitable here as
#   all the includes and libs are in the same directory, similar to how we write Makefile.
message("NETCDF is ${NETCDF}")  # Ligang: empty!
# set(NETCDF /glade/work/miesch/modules/gnu-9.1.0/openmpi-4.0.3/netcdf/4.7.4)
set(NETCDF /glade/u/apps/ch/opt/netcdf/4.8.1/gnu/10.1.0)
set(NETCDF_INCLUDES ${NETCDF}/include)
set(NETCDF_LIBRARIES_DIR ${NETCDF}/lib)
set(NETCDF_LIBRARIES libnetcdff.so libnetcdf.so)
include_directories(${NETCDF_INCLUDES})
link_directories(${NETCDF_LIBRARIES_DIR})


# Sources
add_subdirectory(src)


