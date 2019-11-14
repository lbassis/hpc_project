#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <mkl.h>
#include "my_blas.h"

int my_dgesv (int matrix_layout , int n , int nrhs , double *a , int lda , int * ipiv , double *b , int ldb) {

  int side = 0;
  int uplo_l = 1;
  int uplo_u = 0;
  int transA = 0;
  int diag = 1;
  double alpha = 1.;

  /* A = LU */
  my_dgetf2( n, n, a, lda, NULL );

  /* Ly = b */
  my_dtrsm (NULL, side, uplo_l, transA, 1, n,  nrhs,  alpha, a, lda, b, ldb);

  /* Ux = y */
  my_dtrsm (NULL, side, uplo_u, transA, 0, n,  nrhs,  alpha, a, lda, b, ldb);

  return 0;
}
