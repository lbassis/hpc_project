#include <stdio.h>
#include <stdlib.h>

#include "util.h"
#include "my_lib.h"

void main(int argc, char* argv[]) {

  int M = atoi(argv[1]);
  int N = M;

  double* a = (double*)malloc(sizeof(double) * M * N);

  init_random(M, N, a, 1);
  my_dgetrf_omp_trsm_gemm(LAPACK_COL_MAJOR, M, N, a, M, NULL);
  init_random(M, N, a, 1);
  my_dgetrf(LAPACK_COL_MAJOR, M, N, a, M, NULL);

  free(a);
}
