#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <mkl.h>
#include "my_blas.h"
#include "my_gemm.h"


#ifndef BLOC_SIZE
#define BLOC_SIZE 2
#endif


void my_dgemm_scalaire(int m, double *a, double *b, double* c) {

  int i, j, k;
  double temp;

  for (i = 0; i < m; i++) {
    for (k = 0; k < m; k++) {
      temp = 0;
      printf("c[%d] = ", i+m*k);
      for (j = 0; j < m; j++) {
	printf("a[%d]*b[%d]+", j+m*i, j+m*k);
	temp += a[j+m*i]*b[j+m*k];
      }
      c[i+m*k] = temp;
      printf("\n");
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

void my_dgemm(int transA, int transB, int m, int n, int k, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc) {

  int i, j, l, q, r;

  int nb_bloc_n = n / BLOC_SIZE;
  int nb_bloc_m = m / BLOC_SIZE;
  int a_bloc_start, b_bloc_start, c_bloc_start;

  for (r = 0; r < nb_bloc_n; r++) {  
    for (q = 0; q < nb_bloc_m; q++) {

       c_bloc_start = q*BLOC_SIZE + (r*BLOC_SIZE*nb_bloc_m);

       for (i = 0; i < nb_bloc_n; i++) {
	 a_bloc_start = q*BLOC_SIZE + (i*BLOC_SIZE*nb_bloc_m);
	 b_bloc_start = i*BLOC_SIZE + (r*BLOC_SIZE*nb_bloc_m);
	 my_dgemm_backup(transA, transB, BLOC_SIZE, BLOC_SIZE, BLOC_SIZE, 1., a+a_bloc_start, BLOC_SIZE, b+b_bloc_start, BLOC_SIZE, 1., c+c_bloc_start, BLOC_SIZE);
      }
    }
  }
}
