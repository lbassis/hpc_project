#ifndef MY_DDOT_H
#define MY_DDOT_H


double my_ddot(const int N, const double *X, const int incX, const double *Y, const int incY);

double *my_dgemm_scalaire(int m, double *a, double *b);

double *my_dgemm_scalaire_kij(int m, double *a, double *b);

double *my_dgemm_scalaire_ijk(int m, double *a, double *b);

double *my_dgemm_scalaire_jik(int m, double *a, double *b);

void my_dgemm(int transA, int transB, int m, int n, int k, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc);

void my_daxpy (int n, double a, double *x, int incx, double *y, int incy);

void my_dgemv(int transA, int m, int n, double alpha, double *a, int lda, double *x, int incX, double beta, double *Y, int incY);

void my_dger(int m, int n, double alpha, double *X, int incX, double *Y, int incY, double *A, int lda);

// BLAS_COL_MAJOR, ...

#endif
