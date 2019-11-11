#include <stdlib.h>
#include "ddot.h"

double my_ddot(const int N, const double *X, const int incX, const double *Y, const int incY) {

  int i;
  double result = 0;
  for (i = 0; i < N; i++) {
    result += X[i*incX] * Y[i*incY]; // 4 flops + 3 loads + 1 store
  }

  return result;
}

double *my_dgemm_scalaire(int m, double *a, double *b) {

  int i, j, k;
  double *c = calloc(sizeof(double), m*m);

  for (i = 0; i < m; i++) { //
    for (k = 0; k < m; k++) { //
      for (j = 0; j < m; j++) { //
	c[i+m*k] += a[j+m*i]*b[j+m*k];
      }
    }
  }

  return c;
}

double *my_dgemm_scalaire_kij(int m, double *a, double *b) {

  int i, j, k;
  double *c = calloc(sizeof(double), m*m);

  for (k = 0; k < m; k++) { //
    for (i = 0; i < m; i++) { //
      for (j = 0; j < m; j++) { //
	c[i+m*k] += a[j+m*i]*b[j+m*k];
      }
    }
  }

  return c;
}

double *my_dgemm_scalaire_ijk(int m, double *a, double *b) {

  int i, j, k;
  double *c = calloc(sizeof(double), m*m);

  for (i = 0; i < m; i++) { //
    for (j = 0; j < m; j++) { //
      for (k = 0; k < m; k++) { //
	c[i+m*k] += a[j+m*i]*b[j+m*k];
      }
    }
  }

  return c;
}

double *my_dgemm_scalaire_jik(int m, double *a, double *b) {

  int i, j, k;
  double *c = calloc(sizeof(double), m*m);

  for (j = 0; j < m; j++) { //
    for (i = 0; i < m; i++) { //
      for (k = 0; k < m; k++) { //
	c[i+m*k] += a[j+m*i]*b[j+m*k];
      }
    }
  }

  return c;
}

// TODO:
void my_dgemm(int transA, int transB, int m, int n, int k, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc) {

 int i, j, k1;
 if (transA && !transB) {
   for (j = 0; j < m; j++) { //
     for (i = 0; i < m; i++) { //
       for (k = 0; k < m; k++) { //
	        c[i+m*k] += a[j+m*i]*b[j+m*k];
       }
     }
   }
 }
}

void my_daxpy (int n, double a, double *x, int incx, double *y, int incy) {
  int i;
  for (i = 0; i < n; i++) {
    y[i] += a*x[i];
  }

  return 0;
}

void my_dgemv(int transA, int m, int n, double alpha, double *a, int lda, double *x, int incX, double beta, double *y, int incY) {

 int i, j;

 if (transA) {
   for (i = 0; i < n; i++) {
     double tmp = 0;
     for (j = 0; j < m; j++) {
       tmp += a[j+lda*i]*x[j*incX];
     }
     y[i*incY] = y[i*incY]*beta + tmp*alpha;
   }
 }

 else {
   for (i = 0; i < n; i++) {
     double tmp = 0;
     for (j = 0; j < m; j++) {
       tmp += a[i+lda*j]*x[j*incX];
     }
     y[i*incY] = y[i*incY]*beta + tmp*alpha;
   }
 }
}

void my_dger(int m, int n, double alpha, double *X, int incX, double *Y, int incY, double *A, int lda) {

 int i, j;
 for (i = 0; i < n; i++) {
   for (j = 0; j < m; j++) {
     A[i*lda+j] += X[i*incX]*Y[j*incY]*alpha;
   }
 }
}
