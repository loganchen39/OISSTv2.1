# OISSTv2.1
NOAA OISSTv2.1 source code, provided to assist the project of "improving OISST with JEDI", will be modified/tested/maintained as we progress. 

## Getting Started
To build OISSTv2.1, check CMakeLists.txt files and modify them if necessary, particularly for the NetCDF library. It's been built successfully on Cheyenne Supercomputer.  
```
git clone https://github.com/UMD-AOSC/OISSTv2.1.git  OISSTv2.1
mkdir -p build
cd build
cmake ../OISSTv2.1
cmake --build .
```
