#!/bin/tcsh

#PBS  -A UMCP0009   
#PBS  -l walltime=12:00:00
# #PBS  -l select=16:ncpus=32:mpiprocs=32 
# #PBS  -l select=1:ncpus=1:mpiprocs=1 
#PBS  -l select=1:ncpus=1:mpiprocs=1:mem=109GB 
#PBS  -N MayJun_calc_ACSPO_L3U2OI
#PBS  -j oe
# #PBS  -q premium
#PBS  -q regular
#PBS  -M lchen2@umd.edu

module load ncarenv
# ncar_pylib --setup
 
./calc_ACSPO_L3U2OI_DailySST_Stat_MayJun.py >&! calc_ACSPO_L3U2OI_DailySST_Stat_MayJun_01.log
 
exit 0
