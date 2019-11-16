#include <stdio.h>
#include <stdlib.h>
#include <mkl_cblas.h>
#include <mkl.h>

#include "util.h"
#include "perf.h"
#include "my_lib.h"

int main(void){

  double *a, *b, *c, *d;
  int i;
  int IONE = 1;
  long long int   ISEED[4] = {0,0,0,1};   /* initial seed for zlarnv() */

  double alpha = 1.;
  double beta = 1.3;
  int m =   4;
  int n =   5;
  int k =   2;
  int lda = m;
  int ldb = k;
  int ldc = m;

  a = alloc_mat(lda, k);
  b = alloc_mat(ldb, n);
  c = alloc_mat(ldc, n);
  d = alloc_mat(ldc, n);

  /* random a b and c = d */
  LAPACKE_dlarnv_work(IONE, ISEED, lda*k, a);
  LAPACKE_dlarnv_work(IONE, ISEED, ldb*n, b);
  LAPACKE_dlarnv_work(IONE, ISEED, ldc*n, c);
  for (i = 0; i < ldc*n; i++) c[i] = d[i];

  /* affiche(m, n, d, ldc, stdout); */
  /* c = a*b */
  my_dgemm(0, 0, 0, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

  /* d = a*b */
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, lda, b, ldb, beta, d, ldc);

  /* print d-c */
  for (i = 0; i < ldc*n; i++) {
    d[i] -= c[i];
  }
  affiche(m, n, d, ldc, stdout);

  return 0;
}
