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

  double a[N * N] = {};
  double b[N * N] = {};
  double c[N * N] = {};
  double **d, **e, **f;

  double performance;
  perf_t start,stop;
  int threads = atoi(argv[1]);

  init_random(N, N, a, 1);
  init_random(N, N, b, 1);
  init_random(N, N, c, 1);
  d = lapack2tile(N, N, BLOC_SIZE, a, N);
  e = lapack2tile(N, N, BLOC_SIZE, b, N);
  f = lapack2tile(N, N, BLOC_SIZE, c, N);

  perf(&start);
  my_dgemm_bloc(LAPACK_COL_MAJOR, CblasNoTrans, CblasNoTrans, N, N, N, 1, a, N, b, N, 1, c, N);
  perf(&stop);
  perf_diff(&start, &stop);
  performance = perf_mflops(&stop, 2 * N  * N * N );
  printf("my_dgemm, %d, %lf\n", threads, performance);

  perf(&start); 
  my_dgemm_tiled_openmp              (LAPACK_COL_MAJOR, CblasNoTrans, CblasNoTrans, N, N, N, 1, d, N, e, N,1, f, N); 
  perf(&stop);
  perf_diff(&start, &stop);
  performance = perf_mflops(&stop, 2 * N * N * N );
  printf("my_dgemm_Tile, %d, %lf\n", threads, performance);

  return 0;
}
