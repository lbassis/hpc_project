#include <stdio.h>
#include <stdlib.h>
#include <mkl_cblas.h>
#include <mkl.h>

#include "util.h"
#include "perf.h"
#include "my_lib.h"


#define N 500
#define BLOC_SIZE 130

int main(int argc, char* argv[]){

  long long ipiv[N] = {};
  double a[N * N] = {};
  double b[N * N] = {};
  double c[N * N] = {};
  double **d, **e;

  double performance;
  perf_t start,stop;
  int threads = atoi(argv[1]);

  init_random(N, N, a, 1);
  init_random(N, N, b, 1);
  init_random(N, N, c, 1);
  d = lapack2tile(N, N, BLOC_SIZE, a, N);
  e = lapack2tile(N, N, BLOC_SIZE, a, N);

  perf(&start);
  LAPACKE_dgetrf(LAPACK_COL_MAJOR, N, N, a, N, ipiv);
  perf(&stop);
  perf_diff(&start, &stop);
  performance = perf_mflops(&stop, (2 * N / 3) * N * N );
  printf("getrf_mkl, %d, %lf\n", threads, performance);

  perf(&start);
  my_dgetrf_seq              (LAPACK_COL_MAJOR, N, N, b, N, NULL);
  perf(&stop);
  perf_diff(&start, &stop);
  performance = perf_mflops(&stop, (2 * N / 3) * N * N );
  printf("getrf_seq, %d, %lf\n", threads, performance);

  perf(&start);
  my_dgetrf_openmp(LAPACK_COL_MAJOR, N, N, c, N, NULL);
  perf(&stop);
  perf_diff(&start, &stop);
  performance = perf_mflops(&stop, (2 * N / 3) * N * N );
  printf("getrf_omp, %d, %lf\n", threads, performance);

  perf(&start);
  my_dgetrf_tiled         (LAPACK_COL_MAJOR, N, N, e, N, NULL);
  perf(&stop);
  perf_diff(&start, &stop);
  performance = perf_mflops(&stop, (2 * N / 3) * N * N );
  printf("getrf_tile, %d, %lf\n", threads, performance);

  perf(&start);
  my_dgetrf_tiled_openmp         (LAPACK_COL_MAJOR, N, N, e, N, NULL);
  perf(&stop);
  perf_diff(&start, &stop);
  performance = perf_mflops(&stop, (2 * N / 3) * N * N );
  printf("getrf_omp_tile, %d, %lf\n", threads, performance);

  return 0;
}
