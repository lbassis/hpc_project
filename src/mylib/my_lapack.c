#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <mkl.h>
#include "my_blas.h"



void my_dgetf2(CBLAS_LAYOUT layout,
               const int m,
               const int n,
               double* a,
               const int lda,
               int* ipiv) {

  assert(layout == CblasColMajor);

  (void)ipiv;

  int i, j, k;

  for (k = 0; k < m; k++) {
    for (i = k+1; i < m; i++) {
      a[i+lda*k] /= a[k+lda*k];
      for (j = k+1; j < n; j++) {
	       a[i+lda*j] -= a[i+lda*k] * a[lda*j+k];
      }
    }
  }

}

void my_dgesv (CBLAS_LAYOUT matrix_layout,
              const int n,
              const int nrhs,
	      double *a,
              const int lda,
              int * ipiv,
              double *b,
              const int ldb) {

  assert(matrix_layout == CblasColMajor);


  /* A = LU */
  my_dgetf2(matrix_layout, n, n, a, lda, NULL);

  /* Ly = b */
  my_dtrsm(matrix_layout,
           CblasLeft,
           CblasLower,
           CblasNoTrans,
           CblasNonUnit,
           /* m */ n,
           /* n */ nrhs,
           /* alpha */ 1.,
           a, lda, b, ldb);

  /* Ux = y */
  my_dtrsm(matrix_layout,
           CblasLeft,
           CblasUpper,
           CblasNoTrans,
           CblasUnit,
           /* m */ n,
           /* n */ nrhs,
           /* alpha */ 1.,
           a, lda, b, ldb);

  (void)ipiv;
}
