#ifndef MY_DGEMM_H
#define MY_DGEMM_H

#include <mkl.h>

void my_dgemm_scalaire(const int m, const double *a, const double *b, double* c);

void my_dgemm_scalaire_kij(const int m, const double *a, const double *b, double* c);

void my_dgemm_scalaire_ijk(const int m, const double *a, const double *b, double* c);

void my_dgemm_scalaire_jik(const int m, const double *a, const double *b, double* c);

void my_dgemm_seq(CBLAS_LAYOUT layout,
                  CBLAS_TRANSPOSE TransA,
                  CBLAS_TRANSPOSE TransB,
                  const int m,
                  const int n,
                  const int k,
                  const double alpha,
                  const double *a,
                  const int lda,
                  const double *b,
                  const int ldb,
                  const double beta,
                  double *c,
                  const int ldc);

void my_dgemm(int transA, int transB, int m, int n, int k, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc);


#endif
