#!/bin/tcsh

#PBS  -A UMCP0009   
#PBS  -l walltime=12:00:00              
# #PBS  -l select=1:ncpus=36:mpiprocs=36
#PBS  -l select=1:ncpus=1:mpiprocs=1 
#PBS  -N NSIDC2OI
#PBS  -j oe
#PBS  -q regular
# #PBS  -q economy
#PBS  -M lchen2@umd.edu

module load conda/latest
conda activate npl
 
python regridByCDOInterp_sic_NSIDC2OI.py >&! regridByInterp_sic_NSIDC2OI_002.log
 
exit 0
