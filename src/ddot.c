#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <mkl.h>
#include "ddot.h"

double my_ddot(const int N, const double *X, const int incX, const double *Y, const int incY) {

  int i;
  double result = 0;
  for (i = 0; i < N; i++) {
    result += X[i*incX] * Y[i*incY]; // 2 flops
  }

  return result;
}

void my_dgemm_scalaire(int m, double *a, double *b, double* c) {

  int i, j, k;
  double temp;

  for (i = 0; i < m; i++) {
    for (k = 0; k < m; k++) {
      temp = 0;
      //printf("c[%d] = ", i+m*k);
      for (j = 0; j < m; j++) {
	//printf("a[%d]*b[%d]+", j+m*i, j+m*k);
	       temp += a[j+m*i]*b[j+m*k];
      }
      c[i+m*k] = temp;
      //printf("\n");
    }
  }

}

void my_dgemm_scalaire_kij(int m, double *a, double *b, double* c) {

  int i, j, k;

  for (k = 0; k < m; k++) {
    for (i = 0; i < m; i++) {
      for (j = 0; j < m; j++) {
	c[i+m*k] += a[j+m*i]*b[j+m*k];
      }
    }
  }

}

void my_dgemm_scalaire_ijk(int m, double *a, double *b, double* c) {

  int i, j, k;

  for (i = 0; i < m; i++) { //
    for (j = 0; j < m; j++) { //
      for (k = 0; k < m; k++) { //
	c[i+m*k] += a[j+m*i]*b[j+m*k];
      }
    }
  }

}

void my_dgemm_scalaire_jik(int m, double *a, double *b, double* c) {

  int i, j, k;

  for (j = 0; j < m; j++) {
    for (i = 0; i < m; i++) {
      for (k = 0; k < m; k++) {
	c[i+m*k] += a[j+m*i]*b[j+m*k];
      }
    }
  }

}

void my_dgemm(int transA, int transB, int m, int n, int k, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc) {

  int i, j, l;
  double tmp;

  for (i = 0; i < m; i++) {
    for (l = 0; l < n; l++) {
      tmp = 0;
      for (j = 0; j < k; j++) {
	       if (!transA && !transB) {
	          tmp += a[j*lda+i]*b[j+ldb*l];
         }else if (transA && !transB) {
            tmp += a[j+lda*i]*b[j+ldb*l];
         }else if (!transA && transB) {
            tmp += a[j * lda + i]*b[j*ldb+l];
         }else { //transA && transB
            tmp += a[j + lda * i]*b[j*ldb+l];
         }
      }
      c[i+ldc*l] = c[i+ldc*l]*beta + tmp*alpha;
    }
  }
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

void my_dtrsm (int Layout,
               int side,
               int uplo,
               int transA,
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



#ifndef BLOC_SIZE
#define BLOC_SIZE 3
#endif


void my_dgetrf(int m,
		         int n,
		         double* a,
	           int lda,
		         int* ipiv,
		         int info){


  int nb_bloc_n = (n + BLOC_SIZE - 1) / BLOC_SIZE;
  int nb_bloc_m = (m + BLOC_SIZE - 1) / BLOC_SIZE;
  int min = fmin(nb_bloc_n, nb_bloc_m);



  //ipiv = malloc(BLOC_SIZE * BLOC_SIZE * sizeof(int));

  int k = 0;
  double* bloc_dgetf2 = NULL;

  for(k = 0; k < min; k++){
    bloc_dgetf2 = a + k * BLOC_SIZE * (lda+1);
    //printf("%d %d %d\n", k, (k < nb_bloc_m - 1) ? BLOC_SIZE : m - k * BLOC_SIZE, (k < nb_bloc_n - 1) ? BLOC_SIZE : n - k * BLOC_SIZE);

    my_dgetf2((k < nb_bloc_m - 1) ? BLOC_SIZE : m - k * BLOC_SIZE,
              (k < nb_bloc_n - 1) ? BLOC_SIZE : n - k * BLOC_SIZE,
              bloc_dgetf2,
              lda,
              NULL);
    /*LAPACKE_dgetf2 (CblasColMajor,
                    (k < nb_bloc_m - 1) ? BLOC_SIZE : m - k * BLOC_SIZE,
                    (k < nb_bloc_n - 1) ? BLOC_SIZE : n - k * BLOC_SIZE,
                    bloc_dgetf2,
                    lda,
                    ipiv);*/

    int i = 0;

    for(i = k + 1; i < nb_bloc_m; i++){
      printf("__\n");
      cblas_dtrsm (CblasColMajor,
                /*A = L * X ou A = X * U */ CblasRight,
                /* L ou U ? */ CblasUpper,
                /* A transposee ? */ CblasNoTrans,
                /* diagonale unitaire ? */ CblasNonUnit,
                /* m */ (i < nb_bloc_m - 1) ? BLOC_SIZE : m - i * BLOC_SIZE,
                /* n */ BLOC_SIZE,
                /* alpha */ 1,
                /* L ou U */ bloc_dgetf2,
                /* ldl ou ldu */ lda,
                /* A -> X */ a + BLOC_SIZE * (i + k * lda),
                /* lda */ lda);
    }
    int j = 0;
    for(j = k + 1; j < nb_bloc_n; j++){
      printf("NONNNNNNNNNNNNN %d\n", j);
      cblas_dtrsm(/*int *Layout*/ CblasColMajor,
                /*int side*/      CblasLeft,
                /*int uplo*/      CblasLower,
                /*int transA*/    CblasNoTrans,
                /*int diag*/      CblasUnit,
                /*int m*/         BLOC_SIZE,
                /*int n*/         (j < nb_bloc_n - 1) ? BLOC_SIZE : n - j * BLOC_SIZE,
                /*double alpha*/  1,
                /*double *a*/     bloc_dgetf2,
                /*int lda*/       lda,
                /*double *b*/     a + BLOC_SIZE * (k + j * lda),
                /*int ldb*/       lda);
    }
    for(i = k + 1; i < nb_bloc_m; i++){
      for(j = k + 1; j < nb_bloc_n; j++){
        //my_dgemm(0, 0, BLOC_SIZE, BLOC_SIZE, BLOC_SIZE, -1, a + i * BLOC_SIZE * lda + k * lda, lda, a + k * BLOC_SIZE * lda + j * lda, lda, 1, a + j * BLOC_SIZE * lda + i * lda, lda);
        my_dgemm (//CblasColMajor,
                     0,//CblasNoTrans,
                     0,//CblasNoTrans,
                     /* m */ (i < nb_bloc_m - 1) ? BLOC_SIZE : m - i * BLOC_SIZE,
                     /* n */ (j < nb_bloc_n - 1) ? BLOC_SIZE : n - j * BLOC_SIZE,
                     /* k */ (j < nb_bloc_n - 1) ? BLOC_SIZE : n - j * BLOC_SIZE,
                     /* alpha */ -1.,
                     /* A[i][k] */ a + BLOC_SIZE * (i + k * lda),
                     lda,
                     /* B[k][j] */ a + BLOC_SIZE * (k + j * lda),
                     lda,
                     /* beta */ 1.,
                     /* C[i][j] */ a + BLOC_SIZE * (i + j * lda),
                     lda);
      }
    }
  }
}
