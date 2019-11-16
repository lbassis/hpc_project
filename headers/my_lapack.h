#ifndef MY_LAPACK_H
#define MY_LAPACK_H

#include <mkl.h>

void my_dgetf2(CBLAS_LAYOUT layout,
               const int m,
               const int n,
               double* a,
               const int lda,
               int* ipiv);

void my_dgesv (CBLAS_LAYOUT matrix_layout,
             const int n,
             const int nrhs,
             const double *a,
             const int lda,
             int * ipiv,
             double *b,
             const int ldb);

#endif
