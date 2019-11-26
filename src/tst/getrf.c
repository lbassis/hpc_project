#include <stdio.h>
#include <stdlib.h>
#include <mkl_cblas.h>
#include <mkl.h>

#include "util.h"
#include "perf.h"
#include "my_lib.h"

#ifndef SIZE
#define SIZE 10
#endif

#define M 140
#define N 150

#define BLOC_SIZE 130

int main(void){
  double a[M * N] = {};
  double b[M * N] = {};

  init_random(M, N, a, 1);
  int i = 0;
  for(i = 0; i < M * N; i++){
    b[i] = a[i];
  }
  /* Lapack interface */
  double **a_Tile = lapack2tile( M, N, BLOC_SIZE, a, M );

  //my_dgetrf(LAPACK_COL_MAJOR, M, N, a, M, NULL);
  my_dgetrf_Tile(LAPACK_COL_MAJOR, M, N, a_Tile, M, NULL);

  long long ipiv[M] = {};
  LAPACKE_dgetrf(LAPACK_COL_MAJOR, M, N, b, M, ipiv);
  

  tile2lapack( M, N, BLOC_SIZE, (const double**)a_Tile, a, M );

  printf("__\n");
  for(i = 0; i < M * N; i++){
    b[i] -= a[i];
  }

  printf("||lapacke_getrf - my_getrf_Tile||1 = %lf\n", LAPACKE_dlange(CblasColMajor, 'M', M, N, b, M));
  printf("\n");
  return 0;
}
