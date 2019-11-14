#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <mkl.h>
#include "my_blas.h"

double my_ddot(const int N, const double *X, const int incX, const double *Y, const int incY) {

  int i;
  double result = 0;
  for (i = 0; i < N; i++) {
    result += X[i*incX] * Y[i*incY]; // 4 flops + 3 loads + 1 store
  }

  return result;
}

void my_daxpy (int n, double a, double *x, int incX, double *y, int incY) {
  int i;
  for (i = 0; i < n; i++) {
    y[i*incY] += a*x[i*incX];
  }

}

void my_dgemv(int transA, int m, int n, double alpha, double *A, int lda, double *X, int incX, double beta, double *Y, int incY) {

  int i, j;

  if (transA) {
    for (i = 0; i < n; i++) {
      double tmp = 0;
      for (j = 0; j < m; j++) {
	tmp += A[j+lda*i]*X[j*incX];
      }
      Y[i*incY] = Y[i*incY]*beta + tmp*alpha;
    }
  }

  else {
    for (i = 0; i < n; i++) {
      double tmp = 0;
      for (j = 0; j < m; j++) {
	tmp += A[i+lda*j]*X[j*incX];
      }
      Y[i*incY] = Y[i*incY]*beta + tmp*alpha;
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

void my_dgetf2( int m, int n, double* a, int lda, int* ipiv ) {

  int i, j, k;

  for (k = 0; k < m; k++) {
    for (i = k+1; i < m; i++) {
      a[i+lda*k] /= a[k+lda*k];
      for (j = k+1; j < n; j++) {
	a[i+lda*j] -= a[i+lda*k] * a[lda*j+k];
      }
    }
  }

  (void)ipiv;
}

void my_dtrsm (int *Layout, int side,
               int uplo, int transA,
               int diag,
               int m,
               int n,
               double alpha,
               double *a,
               int lda,
               double *b,
               int ldb) {

  int i, j, k;
  double temp, one = 1.;

  if (side == 0) { // left
    if (!transA) { // !transA
      if (uplo == 0) { // no trans, left, upper
      	for (j = 0; j < n; j++) {
      	  if (alpha != 1.) {
      	    for (i = 0; i < m; i++) {
      	      b[i+ldb*j] = alpha*b[i+ldb*j];
      	    }
      	  }
      	  for (k = m-1; k >= 0; k--) {
      	    if (b[k+ldb*j] != 0) {
      	      if (diag == 0) {
		              b[k+ldb*j] /= a[k+lda*k];
      	      }
      	      for (i = 0; i < k; i++) {
		              b[i+ldb*j] -= b[k+ldb*j]*a[i+lda*k];
      	      }
      	    }
      	  }
      	}
      }
      else { // no trans, left, lower
	for (j = 0; j < n; j++) {
	  if (alpha != 1.) {
	    for (i = 0; i < m; i++) {
	      b[i+ldb*j] = alpha*b[i+ldb*j];
	    }
	  }
	  for (k = 0; k < m; k++) {
	    if (b[k+ldb*j] != 0) {
	      if (diag == 0) {
		b[k+ldb*j] /= a[k+lda*k];
	      }
	      for (i = k+1; i < m; i++) {
		b[i+ldb*j] -= b[k+ldb*j]*a[i+lda*k];
	      }
	    }
	  }
	}
      }
    }
    else { //transA
      if (uplo == 0) { // trans, left, upper
	for (j = 0; j < n; j++) {
	  for (i = 0; i < m; i++) {
	    temp = alpha*b[i+ldb*j];
	    for (k = 0; k < i-1; k++) {
	      temp -= a[k+lda*i]*b[k+ldb*j];
	    }
	    if (diag == 0) {
		temp /= a[i+lda*i];
	    }
	    b[i+ldb*j] = temp;
	  }
	}
      }
      else { // trans, left, lower
	for (j = 0; j < n; j++) {
	  for (i = m; i > 0; i--) {
	    temp = alpha*b[i+ldb*j];
	    for (k = i+1; k < m; k++) {
	      temp -= a[k+lda*i]*b[k+ldb*j];
	    }
	    if (diag == 0) {
		temp /= a[i+lda*i];
	    }
	    b[i+ldb*j] = temp;
	  }
	}
      }
    }
  }
  else { // side = right
    if (!transA) {
      if (uplo == 0) { // no trans, right, upper
	for (j = 0; j < n; j++) {
	  if (alpha != 1.) {
	    for (i = 0; i < m; i++) {
	      b[i+ldb*j] = alpha*b[i+ldb*j];
	    }
	  }
	  for (k = 1; k < j-1; k++) {
	    if (a[k+ldb*j] != 0) {
	      for (i = 0; i < m; i++) {
		b[i+ldb*j] -= a[k+lda*j]*b[i+ldb*k];
	      }
	    }
	  }
	  if (diag == 0) {
	    temp = one/a[j+lda*j];
	    for (i = 0; i < m; i++) {
	      b[i+ldb*j] *= temp;
	    }
	  }
	}
      }
      else { // no trans, right, lower
	for (j = 0; j < n; j++) {
	  if (alpha != 1.) {
	    for (i = 0; i < m; i++) {
	      b[i+ldb*j] = alpha*b[i+ldb*j];
	    }
	  }
	  for (k = j+1; k < n; k++) {
	    if (a[k+ldb*j] != 0) {
	      for (i = 0; i < m; i++) {
		b[i+ldb*j] -= a[k+lda*j]*b[i+ldb*k];
	      }
	    }
	  }
	  if (diag == 0) {
	    temp = one/a[j+lda*j];
	    for (i = 0; i < m; i++) {
	      b[i+ldb*j] *= temp;
	    }
	  }
	}
      }
    }
    else { //transA
      if (uplo == 0) { // trans, right, upper
	for (k = n; k >0; k--) {
	  if (diag == 0) {
	    temp = one/a[k+lda*k];
	    for (i = 0; i < m; i++) {
	      b[i+ldb*k] *= temp;
	    }
	  }
	  for (j = 0; j < k-1; j++) {
	    if (a[j+lda*k] != 0) {
	      temp = a[j+lda*k];
	      for (i = 0; i < m; i++) {
		b[i+ldb*j] -= b[i+ldb*k];
	      }
	    }
	  }
	  if (alpha != 1) {
	    for (i = 0; i < m; i++) {
	      b[i+ldb*k] *= alpha;
	    }
	  }
	}
      }
      else { // trans, right, lower
	for (k = 0; k < n; k++) {
	  if (diag == 0) {
	    temp = one/a[k+lda*k];
	    for (i = 0; i < m; i++) {
	      b[i+ldb*k] *= temp;
	    }
	  }
	  for (j = k+1; j < n; j++) {
	    if (a[j+lda*k] != 0) {
	      temp = a[j+lda*k];
	      for (i = 0; i < m; i++) {
		b[i+ldb*j] -= b[i+ldb*k];
	      }
	    }
	  }
	  if (alpha != 1) {
	    for (i = 0; i < m; i++) {
	      b[i+ldb*k] *= alpha;
	    }
	  }
	}
      }
    }
  }

  (void)Layout;
}
