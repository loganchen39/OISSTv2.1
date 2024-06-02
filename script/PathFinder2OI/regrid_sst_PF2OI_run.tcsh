#!/bin/tcsh

#PBS  -A UMCP0009   
#PBS  -l walltime=12:00:00
# #PBS  -l select=16:ncpus=32:mpiprocs=32 
#PBS  -l select=1:ncpus=1:mpiprocs=1 
# #PBS  -l select=1:ncpus=1:mpiprocs=1:mem=109GB 
#PBS  -N PF2OI_01
#PBS  -j oe
# #PBS  -q premium
#PBS  -q regular
#PBS  -M lchen2@umd.edu

# module load ncarenv
module load conda/latest
conda activate npl
 
python3 regrid_sst_PF2OI_v1.1_RewrittenNotTested.py >&! regrid_sst_PF2OI_06.log
 
exit 0
