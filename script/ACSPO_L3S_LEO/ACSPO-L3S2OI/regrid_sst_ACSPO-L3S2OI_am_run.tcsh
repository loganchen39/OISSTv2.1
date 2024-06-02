#!/bin/tcsh

#PBS  -A UMCP0009
#PBS  -l walltime=12:00:00              
# #PBS  -l select=1:ncpus=36:mpiprocs=36
#PBS  -l select=1:ncpus=1:mpiprocs=1 
#PBS  -N am_l3s2oi
#PBS  -j oe
#PBS  -q regular
# #PBS  -q economy
#PBS  -M lchen2@umd.edu

module load conda/latest
conda activate npl
 
python3 regrid_sst_ACSPO-L3S2OI_am.py >&! qsub_am_005.log
 
exit 0
