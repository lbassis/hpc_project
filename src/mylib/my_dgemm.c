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


void my_dgemm_scalaire(CBLAS_LAYOUT layout,
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

  //printf("oi\n");
  int i, j, kk;
  double temp;

  for (i = 0; i < m; i++) {
    for (kk = 0; kk < m; kk++) {
      temp = 0;
      for (j = 0; j < m; j++) {
	       temp += a[j+m*i]*b[j+m*kk];
      }
      c[i+m*kk] = temp;
    }
  }
  //printf("tchau\n");
  (void)layout;
  (void)TransA;
  (void)TransB;
  (void)n;
  (void)k;
  (void)alpha;
  (void)beta;
  (void)lda;
  (void)ldb;
  (void)ldc;

}

void my_dgemm_scalaire_kij(CBLAS_LAYOUT layout,
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

  int i, j, kk;

  for (kk = 0; kk < m; kk++) {
    for (i = 0; i < m; i++) {
      for (j = 0; j < m; j++) {
	       c[i+m*kk] += a[j+m*i]*b[j+m*kk];
      }
    }
  }

  (void)layout;
  (void)TransA;
  (void)TransB;
  (void)n;
  (void)k;
  (void)alpha;
  (void)beta;
  (void)lda;
  (void)ldb;
  (void)ldc;
}

void my_dgemm_scalaire_ijk(CBLAS_LAYOUT layout,
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

  int i, j, kk;

  for (i = 0; i < m; i++) { //
    for (j = 0; j < m; j++) { //
      for (kk = 0; kk < m; kk++) { //
	       c[i+m*kk] += a[j+m*i]*b[j+m*kk];
      }
    }
  }

  (void)layout;
  (void)TransA;
  (void)TransB;
  (void)n;
  (void)k;
  (void)alpha;
  (void)beta;
  (void)lda;
  (void)ldb;
  (void)ldc;
}

void my_dgemm_scalaire_jik(CBLAS_LAYOUT layout,
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

  int i, j, kk;

  for (j = 0; j < m; j++) {
    for (i = 0; i < m; i++) {
      for (kk = 0; kk < m; kk++) {
	       c[i+m*kk] += a[j+m*i]*b[j+m*kk];
      }
    }
  }

  (void)layout;
  (void)TransA;
  (void)TransB;
  (void)n;
  (void)k;
  (void)alpha;
  (void)beta;
  (void)lda;
  (void)ldb;
  (void)ldc;
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
  int transA = (TransA == CblasTrans);
  int transB = (TransB == CblasTrans);

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

void my_dgemm(CBLAS_LAYOUT layout,
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
                  const int ldc){


	
	int i, j, kk;

  int nb_bloc_n = (n + BLOC_SIZE - 1) / BLOC_SIZE;
  int nb_bloc_m = (m + BLOC_SIZE - 1) / BLOC_SIZE;
  int nb_bloc_k = (k + BLOC_SIZE - 1) / BLOC_SIZE;

  int current_block_n;
  int current_block_m;

  int start_c;
  double current_beta = beta;

  for (i = 0; i < nb_bloc_m; i++) { // lignes de A
		current_block_m = (i < nb_bloc_m - 1) ? BLOC_SIZE : m - i * BLOC_SIZE;

  	for (j = 0; j < nb_bloc_n; j++) { // colonnes de B
      start_c = (i + j * ldc) * BLOC_SIZE;
			current_block_n = (j < nb_bloc_n - 1) ? BLOC_SIZE : n - j * BLOC_SIZE;

			for (kk = 0; kk < nb_bloc_k; kk++) { // colonnes de A
				my_dgemm_seq(layout,
										TransA,
										TransB,
		    						/* m */ current_block_m,
										/* n */ current_block_n,
		    						/* k */ (kk < nb_bloc_k - 1) ? BLOC_SIZE : k - kk * BLOC_SIZE,
										alpha,
		     						a + (i + kk * lda) * BLOC_SIZE,
										lda,
		     						b + (kk + j * ldb) * BLOC_SIZE,
										ldb,
										current_beta,
		     						c + start_c,
										ldc);
				current_beta = 1;
      }
      current_beta = beta;
    }
  }
}
