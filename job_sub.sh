#!/bin/sh
#$ -S /bin/bash
#$ -cwd
#$ -pe impi 4
#$ -jc pcc-skl
## Load the Intel MPI environment variables
. /fefs/opt/x86_64/Gaussian/envset.sh

ulimit -s unlimited

#g16 H2O
export PATH="/home/yang/anaconda3/bin:$PATH"
. /fefs/opt/x86_64/intel/parallel_studio_xe_2017/impi/2017.2.174/bin64/mpivars.sh
#. /fefs/opt/x86_64/intel/parallel_studio_xe_2017/impi/2017.2.174/bin64/mpivars.sh
#. /fefs/opt/x86_64/intel/parallel_studio_xe_2017/compilers_and_libraries_2017.2.174/linux/mpi/intel64/bin/mpiexec
#export PATH="/home/yang/anaconda2/bin:$PATH"
## -pe impi argument 56 is automatically transferred to the number of executable processes. Note, however, that -bootstrap sge is required.
mpiexec -bootstrap sge -n 4 python -u d_mcts.py
