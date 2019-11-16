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
    result += X[i*incX] * Y[i*incY]; // 3 flops
  }

  return result;
}


void my_dgemm_scalaire(int layout, int transA, int transB, int m, int n, int kk, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc) {
  int i, j, k;
  double temp;

  for (i = 0; i < m; i++) {
    for (k = 0; k < m; k++) {
      temp = 0;
      for (j = 0; j < m; j++) {
	temp += a[j+m*i]*b[j+m*k];
      }
      c[i+m*k] = temp;
    }
  }

  (void)layout;
  (void)transA;
  (void)transB;
  (void)kk;
  (void)alpha;
  (void)beta;
  (void)lda;
  (void)ldb;
  (void)ldc;
}

void my_dgemm_scalaire_kij(int layout, int transA, int transB, int m, int n, int kk, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc) {

  int i, j, k;

  for (k = 0; k < m; k++) {
    for (i = 0; i < m; i++) {
      for (j = 0; j < m; j++) {
	c[i+m*k] += a[j+m*i]*b[j+m*k];
      }
    }
  }

  (void)layout;
  (void)transA;
  (void)transB;
  (void)kk;
  (void)alpha;
  (void)beta;
  (void)lda;
  (void)ldb;
  (void)ldc;
}

void my_dgemm_scalaire_ijk(int layout, int transA, int transB, int m, int n, int kk, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc) {

  int i, j, k;

  for (i = 0; i < m; i++) { //
    for (j = 0; j < m; j++) { //
      for (k = 0; k < m; k++) { //
	c[i+m*k] += a[j+m*i]*b[j+m*k];
      }
    }
  }

  (void)layout;
  (void)transA;
  (void)transB;
  (void)kk;
  (void)alpha;
  (void)beta;
  (void)lda;
  (void)ldb;
  (void)ldc;
}

void my_dgemm_scalaire_jik(int layout, int transA, int transB, int m, int n, int kk, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc) {

  int i, j, k;

  for (j = 0; j < m; j++) {
    for (i = 0; i < m; i++) {
      for (k = 0; k < m; k++) {
	c[i+m*k] += a[j+m*i]*b[j+m*k];
      }
    }
  }

  (void)layout;
  (void)transA;
  (void)transB;
  (void)kk;
  (void)alpha;
  (void)beta;
  (void)lda;
  (void)ldb;
  (void)ldc;
}

void my_dgemm_seq(int transA, int transB, int m, int n, int k, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc) {

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


#define BLOCK_SIZE 3
void my_dgemm(int transA, int transB, int m, int n, int k, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc) {

  int i, j, kk;

  int nb_bloc_n = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int nb_bloc_m = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int nb_bloc_k = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;

  int current_block_n;
  int current_block_m;
  int current_block_k;
  
  int start_a, start_b, start_c;
  double current_beta = beta;

  printf("nb blocks: %d, %d e %d\n", nb_bloc_m, nb_bloc_n, nb_bloc_k);
  for (j = 0; j < nb_bloc_n; j++) { // colonnes de B
    for (kk = 0; kk < nb_bloc_k; kk++) { // colonnes de A
      start_c = (kk + ldc*j)*BLOCK_SIZE;
      
      current_block_n = (j < nb_bloc_n - 1) ? BLOCK_SIZE : n - j*BLOCK_SIZE;
      current_block_k = (kk < nb_bloc_k - 1) ? BLOCK_SIZE : k - kk*BLOCK_SIZE;
      printf("C[%d] = ", start_c);

      for (i = 0; i < nb_bloc_m; i++) { // lignes de A
	current_block_m = (i < nb_bloc_m - 1) ? BLOCK_SIZE : m - i*BLOCK_SIZE;
	start_a = (kk + lda*i)*BLOCK_SIZE;
	printf("[%d + %d*%d]\n", j, ldb, kk);
	start_b = (kk + ldb*j)*BLOCK_SIZE;
	printf("+A[%d]*B[%d](%d, %d, %d) ", start_a, start_b, current_block_m, current_block_n, current_block_k);
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
		    current_block_m, current_block_n,
		    current_block_k, alpha,
		     a+start_a, lda,
		     b+start_b, ldb, current_beta,
		     c+start_c, ldc);
	current_beta = 1;
      }
      printf("\n");
      current_beta = beta;
    }
  }
}


void my_dgemm_block_fix(int transA, int transB, int m, int n, int k, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc) {

  int q, r, i;
  int nb_bloc_n = n / BLOCK_SIZE;
  int nb_bloc_m = m / BLOCK_SIZE;
  int nb_bloc_k = k / BLOCK_SIZE;
  int start_a, start_b, start_c;
  double current_beta = beta;

  for (r = 0; r < nb_bloc_n; r++) {
    for (q = 0; q < nb_bloc_m; q++) {
      start_c = (q + ldc*r)*BLOCK_SIZE;
      for (i = 0; i < nb_bloc_k; i++) {
	start_a = (q + lda*i)*BLOCK_SIZE;
	start_b = (i + ldb*r)*BLOCK_SIZE;
	my_dgemm_seq(transA, transB, BLOCK_SIZE, BLOCK_SIZE,
		     BLOCK_SIZE, alpha,
		     a+start_a, lda,
		     b+start_b, ldb, current_beta,
		     c+start_c, ldc);
	current_beta = 1;
      }
      current_beta = beta;
    }
  }
}
#define BLOC_SIZE 4
/* void my_dgemm(int transA, int transB, int m, int n, int k, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc) { */

/*   int nb_bloc_n = (n + BLOC_SIZE - 1) / BLOC_SIZE; */
/*   int nb_bloc_m = (m + BLOC_SIZE - 1) / BLOC_SIZE; */
/*   int min = fmin(nb_bloc_n, nb_bloc_m); */
/*   int i, j, l; */

/*   for (i = 0; i < nb_bloc_m; i++) { */
/*     for (l = 0; l < nb_bloc_n; l++) { */
/*       //tmp = 0; */
/*       for (j = 0; j < BLOC_SIZE; j++) { */
/*         my_dgemm (transA, */
/* 		  transB, */
/*                   (i < nb_bloc_m - 1) ? BLOC_SIZE : m - i * BLOC_SIZE, */
/*                   (j < nb_bloc_n - 1) ? BLOC_SIZE : n - j * BLOC_SIZE, */
/*                   (j < nb_bloc_n - 1) ? BLOC_SIZE : n - j * BLOC_SIZE, */
/*                   1., */
/*                   a + BLOC_SIZE * (i + l * lda), */
/* 		  lda, */
/*                   b + BLOC_SIZE * (l + j * ldb), */
/*                   ldb, */
/*                   1., */
/*                   c + BLOC_SIZE * (i + j * ldc), */
/*                   ldc); */
/*       } */
/*     } */
/*   } */
/* } */
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

int my_dgesv (int matrix_layout , int n , int nrhs , double *a , int lda , int * ipiv , double *b , int ldb) {

  int side = 0;
  int uplo_l = 1;
  int uplo_u = 0;
  int transA = 0;
  double alpha = 1.;

  /* A = LU */
  my_dgetf2( n, n, a, lda, NULL );

  /* Ly = b */
  my_dtrsm (NULL, side, uplo_l, transA, 1, n,  nrhs,  alpha, a, lda, b, ldb);

  /* Ux = y */
  my_dtrsm (NULL, side, uplo_u, transA, 0, n,  nrhs,  alpha, a, lda, b, ldb);

  return 0;

  (void) matrix_layout;
  (void) ipiv;
}

#ifndef BLOC_SIZE
#define BLOC_SIZE 1
#endif

void my_dgetrf (int m,
		int n,
		double* a,
		int lda,
		long int *ipiv,
		int info){

  int nb_bloc_x = n / BLOC_SIZE;
  int nb_bloc_y = m / BLOC_SIZE;
  int r_bloc_x = n % BLOC_SIZE;
  int r_bloc_y = m % BLOC_SIZE;

  assert(r_bloc_x == 0);
  assert(r_bloc_y == 0);

  ipiv = malloc(BLOC_SIZE*BLOC_SIZE*sizeof(int));
  int k = 0;
  for(k = 0; k < fmin(nb_bloc_x, nb_bloc_y); k++){
    //my_dgetf2(BLOC_SIZE, BLOC_SIZE, a + k * (BLOC_SIZE + lda), lda, NULL);
    LAPACKE_dgetf2 (CblasColMajor, BLOC_SIZE, BLOC_SIZE, a+k*(BLOC_SIZE+lda), lda, (int*)ipiv);
    int i = 0;
    for(i = k + 1; i < fmin(nb_bloc_x, nb_bloc_y); i++){
      cblas_dtrsm (/*int *Layout*/ CblasColMajor,
                /*int side*/ CblasLeft,
                /*int uplo*/ CblasUpper,
                /*int transA*/ CblasNoTrans,
                /*int diag*/ CblasNonUnit,
                /*int m*/ BLOC_SIZE,
                /*int n*/ BLOC_SIZE,
                /*double alpha*/ 1,
                /*double *a*/ a + k * (BLOC_SIZE + lda),
                /*int lda*/ lda,
                /*double *b*/ a + i * BLOC_SIZE + k * lda,
                /*int ldb*/ lda);
    }
    int j = 0;
    for(j = k + 1; j < fmin(nb_bloc_x, nb_bloc_y); j++){
      cblas_dtrsm (/*int *Layout*/ CblasColMajor,
                /*int side*/ CblasRight,
                /*int uplo*/ CblasLower,
                /*int transA*/ CblasNoTrans,
                /*int diag*/ CblasUnit,
                /*int m*/ BLOC_SIZE,
                /*int n*/ BLOC_SIZE,
                /*double alpha*/ 1,
                /*double *a*/ a + k * (BLOC_SIZE + lda),
                /*int lda*/ lda,
                /*double *b*/a + k * BLOC_SIZE + j * lda,
                /*int ldb*/ lda);
    }
    for(i = k + 1; i < fmin(nb_bloc_x, nb_bloc_y); i++){
      for(j = k + 1; j < fmin(nb_bloc_x, nb_bloc_y); j++){
        cblas_dgemm (CblasColMajor, CblasNoTrans, CblasNoTrans, BLOC_SIZE, BLOC_SIZE, BLOC_SIZE, 
		     -1., a+i*BLOC_SIZE+k*lda, lda, a+k*BLOC_SIZE+j*lda, lda, 1., a+j*BLOC_SIZE+i, lda);
	//my_dgemm(0, 0, BLOC_SIZE, BLOC_SIZE, BLOC_SIZE, -1, a + i * BLOC_SIZE + k * lda, lda, a + k * BLOC_SIZE + j * lda, lda, 1, a + j * BLOC_SIZE + i * lda, lda);
      }
    }
  }
  /* last bloc*/
  //my_dgetf2(BLOC_SIZE, BLOC_SIZE, a + k * (BLOC_SIZE + lda), lda, NULL);

  (void)ipiv;
  (void)info;
}
