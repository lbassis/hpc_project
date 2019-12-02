#include <stdio.h>
#include <stdlib.h>
#include <mkl_cblas.h>
#include <mkl.h>

#include "util.h"
#include "perf.h"
#include "my_lib.h"


#define M 140
#define N 150

int main(void){
  printf("%s: \n", __FILE__);
  double a[M * N] = {};
  double b[M * N] = {};
  int IONE = 1;
  long long int   ISEED[4] = {0,0,0,1};   /* initial seed for zlarnv() */

  init_random(M, N, a, 1);

  int i = 0;
  for(i = 0; i < M * N; i++) b[i] = a[i];
  my_dgetrf_seq(LAPACK_COL_MAJOR, M, N, a, M, NULL);
  //affiche(SIZE, SIZE, a, SIZE, stdout);
  //printf("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
  long long ipiv[M] = {};
  LAPACKE_dgetrf(LAPACK_COL_MAJOR, M, N, b, M, ipiv);



  //affiche(M, N, a, M, stdout);
  //printf("__\n");
  for(i = 0; i < M * N; i++) b[i] -= a[i];
  printf("||LAPACKE_dgetrf - my_dgetrf||1 = %lf\n", LAPACKE_dlange(CblasColMajor, 'M', M, N, b, M));
  //affiche(M, N, b, M, stdout);
  for(i = 0; i < M; i++) ipiv[i] -= i + 1;
  printf("pivot - identity = %lf\n", cblas_dnrm2(M, ipiv, 1));
  return 0;
}
