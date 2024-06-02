#!/bin/tcsh

#PBS  -A UMCP0009
#PBS  -l walltime=12:00:00              
# #PBS  -l select=1:ncpus=36:mpiprocs=36
#PBS  -l select=1:ncpus=1:mpiprocs=1 
#PBS  -N n07_sobsc_bin1d2nc2d
#PBS  -j oe
#PBS  -q regular
# #PBS  -q economy
#PBS  -M lchen2@umd.edu

module load conda/latest
conda activate npl
 
python convert_ACSPO-L3C_sobsc_bin1d2nc2d.py >&! qsub_001.log
 
exit 0
