#ifndef MY_DDOT_H
#define MY_DDOT_H


double my_ddot(const int N, const double *X, const int incX, const double *Y, const int incY);

void my_dgemm_scalaire    (int layout, int transA, int transB, int m, int n, int kk, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc);
void my_dgemm_scalaire_kij(int layout, int transA, int transB, int m, int n, int kk, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc);
void my_dgemm_scalaire_ijk(int layout, int transA, int transB, int m, int n, int kk, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc);
void my_dgemm_scalaire_jik(int layout, int transA, int transB, int m, int n, int kk, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc);

void my_dgemm(int transA, int transB, int m, int n, int k, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc);

void my_daxpy (int n, double a, double *x, int incx, double *y, int incy);

void my_dgemv(int transA, int m, int n, double alpha, double *a, int lda, double *x, int incX, double beta, double *Y, int incY);

void my_dger(int m, int n, double alpha, double *X, int incX, double *Y, int incY, double *A, int lda);

void my_dgetf2 (int m, int n, double* a, int lda, int* ipiv);

void my_dgetrf (int m, int n, double* a, int lda, long int* ipiv, int info);

// BLAS_COL_MAJOR, ...

#endif
