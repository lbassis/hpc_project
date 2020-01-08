#!/usr/bin/env sh
SIZES=$(echo '100 200 500 800 1000 1500 2000')
VERSIONS_GEMM=$(echo 'mkl seq scal_omp bloc_omp tiled_omp tiled_starpu')
VERSIONS_GETRF=$(echo 'mkl seq mp tiled_omp tiled_starpu')

for v in $VERSIONS_GEMM; do \
  for s in $SIZES; do \
    echo -n "$v "
    ./testings/perf_dgemm -v $v -M $s -N $s -K $s | tail -n 1
  done
done

for v in $VERSIONS_GETRF; do \
  for s in $SIZES; do \
    echo -n "$v "
    ./testings/perf_dgetrf -v $v -M $s -N $s | tail -n 1
  done
done
