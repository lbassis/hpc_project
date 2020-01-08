#!/usr/bin/env sh

export MODULEPATH=$MODULEPATH:/cm/shared/dev/modules/generic/modulefiles

module load compiler/gcc/8.3.0
module load mpi/openmpi/4.0.1
module load compiler/cuda/10.1
module load linalg/mkl/2019_update4

module load trace/fxt/0.3.9
module load trace/eztrace/1.1-8

# Choose the right one
module load runtime/starpu/1.3.3/mpi
#module load runtime/starpu/1.3.3/mpi-fxt
#module load runtime/starpu/1.3.3/mpi-cuda
#module load runtime/starpu/1.3.3/mpi-cuda-fxt

