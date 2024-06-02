#!/bin/tcsh

#PBS  -A UMCP0009   
#PBS  -l walltime=12:00:00
# #PBS  -l select=16:ncpus=32:mpiprocs=32 
# #PBS  -l select=1:ncpus=1:mpiprocs=1 
#PBS  -l select=1:ncpus=1:mpiprocs=1:mem=109GB 
#PBS  -N plot_ACSPO
#PBS  -j oe
#PBS  -q premium
# #PBS  -q regular
#PBS  -M lchen2@umd.edu

module load ncarenv
# ncar_pylib --setup
 
# python3 compute_L3U_OSPO_DaiySST_Grid.py >&! compute_L3U_OSPO_DaiySST_Grid_01.log
# python compute_L3U_OSPO_DaiySST_Grid.py >&! compute_L3U_OSPO_DaiySST_Grid_01.log
/glade/work/lgchen/project/OISST_NOAA/oisst.v2.1-master/L3U_OSPO/plot_ACSPO_nit.py >&! plot_ACSPO_nit_01.log
 
exit 0
