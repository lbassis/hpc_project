#ifndef MY_BLAS_H
#define MY_BLAS_H

#include <mkl.h>

double my_ddot(const long long int N,
               const double *X,
               const long long int incX,
               const double *Y,
               const long long int incY);

void my_daxpy (const int n,
              const double alpha,
              const double *x,
              const int incX,
              double *y,
              const int incY);

void my_dgemv(CBLAS_LAYOUT layout,
              CBLAS_TRANSPOSE TransA,
              const int m,
              const int n,
              const double alpha,
              const double *A,
              const int lda,
              const double *X,
              const int incX,
              const double beta,
              double *Y,
              const int incY);

void my_dger(CBLAS_LAYOUT layout,
             const int m,
             const int n,
             const double alpha,
             const double *X,
             const int incX,
             const double *Y,
             const int incY,
             double *A,
             const int lda);


void my_dtrsm(CBLAS_LAYOUT layout,
              CBLAS_SIDE side,
              CBLAS_UPLO uplo,
              CBLAS_TRANSPOSE transA,
              CBLAS_DIAG diag,
              const int m,
              const int n,
              const double alpha,
              const double *a,
              const int lda,
              double *b,
              const int ldb);

#endif
