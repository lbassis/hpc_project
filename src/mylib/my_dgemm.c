#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <mkl.h>
#include "my_blas.h"
#include "my_dgemm.h"


#ifndef BLOC_SIZE
#define BLOC_SIZE 2
#endif


void my_dgemm_scalaire(const int m, const double *a, const double *b, double* c) {

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

void my_dgemm_scalaire_kij(const int m, const double *a, const double *b, double* c) {

  int i, j, k;

  for (k = 0; k < m; k++) {
    for (i = 0; i < m; i++) {
      for (j = 0; j < m; j++) {
	       c[i+m*k] += a[j+m*i]*b[j+m*k];
      }
    }
  }

}

void my_dgemm_scalaire_ijk(const int m, const double *a, const double *b, double* c) {

  int i, j, k;

  for (i = 0; i < m; i++) { //
    for (j = 0; j < m; j++) { //
      for (k = 0; k < m; k++) { //
	       c[i+m*k] += a[j+m*i]*b[j+m*k];
      }
    }
  }

}

void my_dgemm_scalaire_jik(const int m, const double *a, const double *b, double* c) {

  int i, j, k;

  for (j = 0; j < m; j++) {
    for (i = 0; i < m; i++) {
      for (k = 0; k < m; k++) {
	       c[i+m*k] += a[j+m*i]*b[j+m*k];
      }
    }
  }

}

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
                  const int ldc) {

  assert(layout == CblasColMajor);

  int i, j, l;
  double tmp;
  int transA = (TransA == CblasNoTrans);
  int transB = (TransB == CblasNoTrans);

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

  int i, j, kk;

  int nb_bloc_n = (n + BLOC_SIZE - 1) / BLOC_SIZE;
  int nb_bloc_m = (m + BLOC_SIZE - 1) / BLOC_SIZE;
  int nb_bloc_k = (k + BLOC_SIZE - 1) / BLOC_SIZE;

  int current_block_n;
  int current_block_m;
  int current_block_k;

  int start_a, start_b, start_c;
  double current_beta = beta;

  printf("nb blocks: %d, %d e %d\n", nb_bloc_m, nb_bloc_n, nb_bloc_k);
  for (j = 0; j < nb_bloc_n; j++) { // colonnes de B
    for (kk = 0; kk < nb_bloc_k; kk++) { // colonnes de A
      start_c = (kk + ldc*j)*BLOC_SIZE;

      current_block_n = (j < nb_bloc_n - 1) ? BLOC_SIZE : n - j*BLOC_SIZE;
      current_block_k = (kk < nb_bloc_k - 1) ? BLOC_SIZE : k - kk*BLOC_SIZE;
      printf("C[%d] = ", start_c);

      for (i = 0; i < nb_bloc_m; i++) { // lignes de A
	current_block_m = (i < nb_bloc_m - 1) ? BLOC_SIZE : m - i*BLOC_SIZE;
	start_a = (kk + lda*i)*BLOC_SIZE;
	printf("[%d + %d*%d]\n", j, ldb, kk);
	start_b = (kk + ldb*j)*BLOC_SIZE;
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
