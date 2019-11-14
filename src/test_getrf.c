#include <stdio.h>
#include <stdlib.h>
#include <mkl_cblas.h>
#include <mkl.h>

#include "util.h"
#include "ddot.h"
#include "perf.h"

#ifndef SIZE
#define SIZE 10
#endif

#define M 10
#define N 10

int main(void){
  double a[M * N] = {};
  double b[M * N] = {};

  init_random(M, N, a, 1);
  int i = 0;
  for(i = 0; i < M * N; i++){
    b[i] = a[i];
  }



  my_dgetrf(M, N, a, M, NULL, 0);
  //affiche(SIZE, SIZE, a, SIZE, stdout);
  //printf("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
  long long ipiv[M] = {};
  LAPACKE_dgetrf(LAPACK_COL_MAJOR, M, N, b, M, ipiv);

  affiche(M, N, a, M, stdout);
  printf("__\n");
  for(i = 0; i < M * N; i++){
    b[i] -= a[i];
  }
  affiche(M, N, b, M, stdout);
  for(i = 0; i < M; i++){
    printf("%lld ", ipiv[i] - i - 1);
  }
  printf("\n");
  return 0;
}
