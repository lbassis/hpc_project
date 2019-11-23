#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <mkl.h>

#include "my_blas.h"
#include "my_dgemm.h"


#ifndef BLOC_SIZE
#define BLOC_SIZE 130
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

  if (!transB) {
    if(!transA){
      for (i = 0; i < m; i++) {
	for (l = 0; l < n; l++) {
	  tmp = 0;
	  for (j = 0; j < k; j++) {
	    tmp += a[j*lda+i]*b[j+ldb*l];
	  }
	  c[i+ldc*l] = c[i+ldc*l]*beta + tmp*alpha;
	}
      }
    }else { // transA && !transB
      for (i = 0; i < m; i++) {
	for (l = 0; l < n; l++) {
	  tmp = 0;
	  for (j = 0; j < k; j++) {
	    tmp += a[j+lda*i]*b[j+ldb*l];
	  }
	  c[i+ldc*l] = c[i+ldc*l]*beta + tmp*alpha;
	}
      }
    }
  }else{ // transB
    if(!transA){
      for (i = 0; i < m; i++) { //!transA && transB
	for (l = 0; l < n; l++) {
	  tmp = 0;
	  for (j = 0; j < k; j++) {
	    tmp += a[j * lda + i]*b[j*ldb+l];
	  }
	  c[i+ldc*l] = c[i+ldc*l]*beta + tmp*alpha;
	}
      }
    }else {
      for (i = 0; i < m; i++) { //transA && transB
	for (l = 0; l < n; l++) {
	  tmp = 0;
	  for (j = 0; j < k; j++) {
	    tmp += a[j + lda * i]*b[l + j * ldb];
	  }
	  c[i+ldc*l] = c[i+ldc*l]*beta + tmp*alpha;
	}
      }
    }
  }
}

void my_dgemm_seq_omp(CBLAS_LAYOUT layout,
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

  if (!transB) {
    if(!transA){
#pragma omp parallel for collapse(2) private(tmp, j)
      for (i = 0; i < m; i++) {
	for (l = 0; l < n; l++) {
	  tmp = 0;
	  for (j = 0; j < k; j++) {
	    tmp += a[j*lda+i]*b[j+ldb*l];
	  }
	  c[i+ldc*l] = c[i+ldc*l]*beta + tmp*alpha;
	}
      }
    }else { // transA && !transB
#pragma omp parallel for collapse(2) private(tmp, j)
      for (i = 0; i < m; i++) {
	for (l = 0; l < n; l++) {
	  tmp = 0;
	  for (j = 0; j < k; j++) {
	    tmp += a[j+lda*i]*b[j+ldb*l];
	  }
	  c[i+ldc*l] = c[i+ldc*l]*beta + tmp*alpha;
	}
      }
    }
  }else{ // transB
    if(!transA){
#pragma omp parallel for collapse(2) private(tmp, j)
      for (i = 0; i < m; i++) { //!transA && transB
	for (l = 0; l < n; l++) {
	  tmp = 0;
	  for (j = 0; j < k; j++) {
	    tmp += a[j * lda + i]*b[j*ldb+l];
	  }
	  c[i+ldc*l] = c[i+ldc*l]*beta + tmp*alpha;
	}
      }
    }else {
#pragma omp parallel for collapse(2) private(tmp, j)
      for (i = 0; i < m; i++) { //transA && transB
	for (l = 0; l < n; l++) {
	  tmp = 0;
	  for (j = 0; j < k; j++) {
	    tmp += a[j + lda * i]*b[l + j * ldb];
	  }
	  c[i+ldc*l] = c[i+ldc*l]*beta + tmp*alpha;
	}
      }
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

  int transA = (TransA == CblasTrans);
  int transB = (TransB == CblasTrans);

  int current_block_n;
  int current_block_m;

  int start_a;
  int start_b;
  int start_c;
  double current_beta = beta;

  for (i = 0; i < nb_bloc_m; i++) { // lignes de A
    current_block_m = (i < nb_bloc_m - 1) ? BLOC_SIZE : m - i * BLOC_SIZE;

    for (j = 0; j < nb_bloc_n; j++) { // colonnes de B
      start_c = (i + j * ldc) * BLOC_SIZE;
      current_block_n = (j < nb_bloc_n - 1) ? BLOC_SIZE : n - j * BLOC_SIZE;

      for (kk = 0; kk < nb_bloc_k; kk++) { // colonnes de A
	if (!transA && !transB) {
	  start_a = (i + kk * lda) * BLOC_SIZE;
	  start_b = (kk + j * ldb) * BLOC_SIZE;
	}else if (transA && !transB) {
	  start_a = (kk + i * lda) * BLOC_SIZE;
	  start_b = (kk + j * ldb) * BLOC_SIZE;
	}else if (!transA && transB) {
	  start_a = (i + kk * lda) * BLOC_SIZE;
	  start_b = (j + kk * ldb) * BLOC_SIZE;
	}else { //transA && transB
	  start_a = (kk + i * lda) * BLOC_SIZE;
	  start_b = (j + kk * ldb) * BLOC_SIZE;
	}
	my_dgemm_seq(layout,
		     TransA,
		     TransB,
		     /* m */ current_block_m,
		     /* n */ current_block_n,
		     /* k */ (kk < nb_bloc_k - 1) ? BLOC_SIZE : k - kk * BLOC_SIZE,
		     alpha,
		     a + start_a,
		     lda,
		     b + start_b,
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

void my_dgemm_Tile(CBLAS_LAYOUT layout,
		   CBLAS_TRANSPOSE TransA,
		   CBLAS_TRANSPOSE TransB,
		   const int m,
		   const int n,
		   const int k,
		   const double alpha,
		   const double **a,
		   const int lda,
		   const double **b,
		   const int ldb,
		   const double beta,
		   double **c,
		   const int ldc){



  int i, j, kk;

  int nb_bloc_n = (n + BLOC_SIZE - 1) / BLOC_SIZE;
  int nb_bloc_m = (m + BLOC_SIZE - 1) / BLOC_SIZE;
  int nb_bloc_k = (k + BLOC_SIZE - 1) / BLOC_SIZE;

  int transA = (TransA == CblasTrans);
  int transB = (TransB == CblasTrans);

  int current_block_n;
  int current_block_m;
  int current_block_k;

  int start_a;
  int start_b;
  int start_c;
  double current_beta = beta;

  for (i = 0; i < nb_bloc_m; i++) { // lignes de A
    current_block_m = (i < nb_bloc_m - 1) ? BLOC_SIZE : m - i * BLOC_SIZE;

    for (j = 0; j < nb_bloc_n; j++) { // colonnes de B
      start_c = (i + j * nb_bloc_m);
      current_block_n = (j < nb_bloc_n - 1) ? BLOC_SIZE : n - j * BLOC_SIZE;

      for (kk = 0; kk < nb_bloc_k; kk++) { // colonnes de A
      current_block_k = (kk < nb_bloc_k - 1) ? BLOC_SIZE : k - kk * BLOC_SIZE;
	if (!transA && !transB) {
	  start_a = (i + kk * nb_bloc_m);
	  start_b = (kk + j * nb_bloc_k);
	}else if (transA && !transB) {
	  start_a = (kk + i * nb_bloc_m);
	  start_b = (kk + j * nb_bloc_k);
	}else if (!transA && transB) {
	  start_a = (i + kk * nb_bloc_m);
	  start_b = (j + kk * nb_bloc_k);
	}else { //transA && transB
	  start_a = (kk + i * nb_bloc_m);
	  start_b = (j + kk * nb_bloc_k);
	}
	my_dgemm_seq(layout,
		     TransA,
		     TransB,
		     /* m */ current_block_m,
		     /* n */ current_block_n,
		     /* k */ current_block_k,
		     alpha,
		     a[start_a],
		     BLOC_SIZE,
		     b[start_b],
		     BLOC_SIZE,
		     current_beta,
		     c[start_c],
		     BLOC_SIZE);
	current_beta = 1;
      }
      current_beta = beta;
    }
  }
}

void my_dgemm_bloc(CBLAS_LAYOUT layout,
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
		   const int ldc,
		   const int bloc_size){



  int i, j, kk;

  int nb_bloc_n = (n + bloc_size - 1) / bloc_size;
  int nb_bloc_m = (m + bloc_size - 1) / bloc_size;
  int nb_bloc_k = (k + bloc_size - 1) / bloc_size;

  int current_block_n;
  int current_block_m;

  int start_c;
  double current_beta = beta;

  for (i = 0; i < nb_bloc_m; i++) { // lignes de A
    current_block_m = (i < nb_bloc_m - 1) ? bloc_size : m - i * bloc_size;

    for (j = 0; j < nb_bloc_n; j++) { // colonnes de B
      start_c = (i + j * ldc) * bloc_size;
      current_block_n = (j < nb_bloc_n - 1) ? bloc_size : n - j * bloc_size;

      for (kk = 0; kk < nb_bloc_k; kk++) { // colonnes de A
	my_dgemm_seq(layout,
		     TransA,
		     TransB,
		     /* m */ current_block_m,
		     /* n */ current_block_n,
		     /* k */ (kk < nb_bloc_k - 1) ? bloc_size : k - kk * bloc_size,
		     alpha,
		     a + (i + kk * lda) * bloc_size,
		     lda,
		     b + (kk + j * ldb) * bloc_size,
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

void my_dgemm_omp(CBLAS_LAYOUT layout,
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

  int transA = (TransA == CblasTrans);
  int transB = (TransB == CblasTrans);

  int current_block_n;
  int current_block_m;

  int start_a;
  int start_b;
  int start_c;
  double current_beta;

#pragma omp parallel for collapse(2) private(current_block_m, current_block_n, start_a, start_b, start_c, current_beta, kk)
  for (i = 0; i < nb_bloc_m; i++) { // lignes de A
    for (j = 0; j < nb_bloc_n; j++) { // colonnes de B
      current_beta = beta;
      current_block_m = (i < nb_bloc_m - 1) ? BLOC_SIZE : m - i * BLOC_SIZE;
      start_c = (i + j * ldc) * BLOC_SIZE;
      current_block_n = (j < nb_bloc_n - 1) ? BLOC_SIZE : n - j * BLOC_SIZE;

      for (kk = 0; kk < nb_bloc_k; kk++) { // colonnes de A
	if (!transA && !transB) {
	  start_a = (i + kk * lda) * BLOC_SIZE;
	  start_b = (kk + j * ldb) * BLOC_SIZE;
	}else if (transA && !transB) {
	  start_a = (kk + i * lda) * BLOC_SIZE;
	  start_b = (kk + j * ldb) * BLOC_SIZE;
	}else if (!transA && transB) {
	  start_a = (i + kk * lda) * BLOC_SIZE;
	  start_b = (j + kk * ldb) * BLOC_SIZE;
	}else { //transA && transB
	  start_a = (kk + i * lda) * BLOC_SIZE;
	  start_b = (j + kk * ldb) * BLOC_SIZE;
	}
	my_dgemm_seq(layout,
		     TransA,
		     TransB,
		     /* m */ current_block_m,
		     /* n */ current_block_n,
		     /* k */ (kk < nb_bloc_k - 1) ? BLOC_SIZE : k - kk * BLOC_SIZE,
		     alpha,
		     a + start_a,
		     lda,
		     b + start_b,
		     ldb,
		     current_beta,
		     c + start_c,
		     ldc);
	current_beta = 1;
      }
    }
  }
}
