#include <stdio.h>
#include <stdlib.h>

#include "util.h"
#include "my_lib.h"


#define M 5000
#define N 5000

void main(void) {
  double* a = (double*)malloc(sizeof(double) * M * N);


	printf("op, imp, t\n");
  init_random(M, N, a, 1);
  my_dgetrf_omp_trsm_gemm(LAPACK_COL_MAJOR, M, N, a, M, NULL);
  init_random(M, N, a, 1);
  my_dgetrf(LAPACK_COL_MAJOR, M, N, a, M, NULL);

  free(a);
}
