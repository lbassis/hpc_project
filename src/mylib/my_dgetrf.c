#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <mkl.h>
#include "my_blas.h"
#include "my_lapack.h"
#include "my_dgemm.h"
#include "my_dgetrf.h"



#ifndef BLOC_SIZE
#define BLOC_SIZE 130

#endif




void my_dgetrf(CBLAS_LAYOUT layout,
							 const int m,
		           const int n,
		           double* a,
	             const int lda,
		           int* ipiv){

  assert(layout == CblasColMajor);
	(void) ipiv;

  int nb_bloc_n = (n + BLOC_SIZE - 1) / BLOC_SIZE;
  int nb_bloc_m = (m + BLOC_SIZE - 1) / BLOC_SIZE;
  int min = fmin(nb_bloc_n, nb_bloc_m);
  double* bloc_dgetf2 = NULL;



  int k = 0;
  for(k = 0; k < min; k++){
    bloc_dgetf2 = a + k * BLOC_SIZE * (lda+1);

    my_dgetf2(CblasColMajor,
	      (k < nb_bloc_m - 1) ? BLOC_SIZE : m - k * BLOC_SIZE,
              (k < nb_bloc_n - 1) ? BLOC_SIZE : n - k * BLOC_SIZE,
              bloc_dgetf2,
              lda,
              NULL);

    int i = 0;

    for(i = k + 1; i < nb_bloc_m; i++){
      my_dtrsm(CblasColMajor,
               CblasRight,
               CblasUpper,
               CblasNoTrans,
               CblasNonUnit,
               /* m */ (i < nb_bloc_m - 1) ? BLOC_SIZE : m - i * BLOC_SIZE,
               /* n */ (k < nb_bloc_n - 1) ? BLOC_SIZE : n - k * BLOC_SIZE,
               /* alpha */ 1,
               /* L\U */ bloc_dgetf2,
	       lda,
               /* A[i][k] */ a + BLOC_SIZE * (i + k * lda),
               lda);
    }
    int j = 0;
    for(j = k + 1; j < nb_bloc_n; j++){
      my_dtrsm(/*int *Layout*/ CblasColMajor,
                /*int side*/      CblasLeft,
                /*int uplo*/      CblasLower,
                /*int transA*/    CblasNoTrans,
                /*int diag*/      CblasUnit,
                /*int m*/         (k < nb_bloc_m - 1) ? BLOC_SIZE : m - k * BLOC_SIZE,
                /*int n*/         (j < nb_bloc_n - 1) ? BLOC_SIZE : n - j * BLOC_SIZE,
                /*double alpha*/  1,
                /*double *a*/     bloc_dgetf2,
                /*int lda*/       lda,
                /*double *b*/     a + BLOC_SIZE * (k + j * lda),
                /*int ldb*/       lda);
    }
    for(i = k + 1; i < nb_bloc_m; i++){
      for(j = k + 1; j < nb_bloc_n; j++){
        my_dgemm (CblasColMajor,
                     CblasNoTrans,
                     CblasNoTrans,
                     /* m */ (i < nb_bloc_m - 1) ? BLOC_SIZE : m - i * BLOC_SIZE,
                     /* n */ (j < nb_bloc_n - 1) ? BLOC_SIZE : n - j * BLOC_SIZE,
                     /* k */ BLOC_SIZE,
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



void my_dgetrf_omp_gemm(CBLAS_LAYOUT layout,
							 const int m,
		           const int n,
		           double* a,
	             const int lda,
		           int* ipiv){

  assert(layout == CblasColMajor);
	(void) ipiv;

  int nb_bloc_n = (n + BLOC_SIZE - 1) / BLOC_SIZE;
  int nb_bloc_m = (m + BLOC_SIZE - 1) / BLOC_SIZE;
  int min = fmin(nb_bloc_n, nb_bloc_m);
  double* bloc_dgetf2 = NULL;



  int k = 0;
  for(k = 0; k < min; k++){
    bloc_dgetf2 = a + k * BLOC_SIZE * (lda+1);

    my_dgetf2(CblasColMajor,
						 (k < nb_bloc_m - 1) ? BLOC_SIZE : m - k * BLOC_SIZE,
              (k < nb_bloc_n - 1) ? BLOC_SIZE : n - k * BLOC_SIZE,
              bloc_dgetf2,
              lda,
              NULL);

    int i = 0;

    for(i = k + 1; i < nb_bloc_m; i++){
      my_dtrsm(CblasColMajor,
               CblasRight,
               CblasUpper,
               CblasNoTrans,
               CblasNonUnit,
               /* m */ (i < nb_bloc_m - 1) ? BLOC_SIZE : m - i * BLOC_SIZE,
               /* n */ BLOC_SIZE,
               /* alpha */ 1,
               /* L\U */ bloc_dgetf2,
							 lda,
               /* A[i][k] */ a + BLOC_SIZE * (i + k * lda),
               lda);
    }
    int j = 0;
    for(j = k + 1; j < nb_bloc_n; j++){
      my_dtrsm(/*int *Layout*/ CblasColMajor,
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
		#pragma omp parallel for collapse(2)
    for(i = k + 1; i < nb_bloc_m; i++){
      for(j = k + 1; j < nb_bloc_n; j++){
        my_dgemm (CblasColMajor,
                     CblasNoTrans,
                     CblasNoTrans,
                     /* m */ (i < nb_bloc_m - 1) ? BLOC_SIZE : m - i * BLOC_SIZE,
                     /* n */ (j < nb_bloc_n - 1) ? BLOC_SIZE : n - j * BLOC_SIZE,
                     /* k */ BLOC_SIZE,
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


void my_dgetrf_omp_trsm_gemm(CBLAS_LAYOUT layout,
							 const int m,
		           const int n,
		           double* a,
	             const int lda,
		           int* ipiv){

  assert(layout == CblasColMajor);
	(void) ipiv;

  int nb_bloc_n = (n + BLOC_SIZE - 1) / BLOC_SIZE;
  int nb_bloc_m = (m + BLOC_SIZE - 1) / BLOC_SIZE;
  int min = fmin(nb_bloc_n, nb_bloc_m);
  double* bloc_dgetf2 = NULL;



  int k = 0;
  for(k = 0; k < min; k++){
    bloc_dgetf2 = a + k * BLOC_SIZE * (lda+1);

    my_dgetf2(CblasColMajor,
						 (k < nb_bloc_m - 1) ? BLOC_SIZE : m - k * BLOC_SIZE,
              (k < nb_bloc_n - 1) ? BLOC_SIZE : n - k * BLOC_SIZE,
              bloc_dgetf2,
              lda,
              NULL);

    int i = 0;
		#pragma omp parallel for
    for(i = k + 1; i < nb_bloc_m; i++){
      my_dtrsm(CblasColMajor,
               CblasRight,
               CblasUpper,
               CblasNoTrans,
               CblasNonUnit,
               /* m */ (i < nb_bloc_m - 1) ? BLOC_SIZE : m - i * BLOC_SIZE,
               /* n */ BLOC_SIZE,
               /* alpha */ 1,
               /* L\U */ bloc_dgetf2,
							 lda,
               /* A[i][k] */ a + BLOC_SIZE * (i + k * lda),
               lda);
    }
    int j = 0;
		#pragma omp parallel for
    for(j = k + 1; j < nb_bloc_n; j++){
      my_dtrsm(/*int *Layout*/ CblasColMajor,
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
		#pragma omp parallel for collapse(2)
    for(i = k + 1; i < nb_bloc_m; i++){
      for(j = k + 1; j < nb_bloc_n; j++){
        my_dgemm (CblasColMajor,
                     CblasNoTrans,
                     CblasNoTrans,
                     /* m */ (i < nb_bloc_m - 1) ? BLOC_SIZE : m - i * BLOC_SIZE,
                     /* n */ (j < nb_bloc_n - 1) ? BLOC_SIZE : n - j * BLOC_SIZE,
                     /* k */ BLOC_SIZE,
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

void my_dgetrf_Tile(CBLAS_LAYOUT layout,
							 const int m,
		           const int n,
		           double** a,
	             const int lda,
		           int* ipiv){

  assert(layout == CblasColMajor);
	(void) ipiv;

  int nb_bloc_n = (n + BLOC_SIZE - 1) / BLOC_SIZE;
  int nb_bloc_m = (m + BLOC_SIZE - 1) / BLOC_SIZE;
  int min = fmin(nb_bloc_n, nb_bloc_m);
  double* bloc_dgetf2 = NULL;



  int k = 0;
  for(k = 0; k < min; k++){
    bloc_dgetf2 = a[k * (nb_bloc_m+1)];

    my_dgetf2(CblasColMajor,
						 (k < nb_bloc_m - 1) ? BLOC_SIZE : m - k * BLOC_SIZE,
              (k < nb_bloc_n - 1) ? BLOC_SIZE : n - k * BLOC_SIZE,
              bloc_dgetf2,
              BLOC_SIZE,
              NULL);
    
    int i = 0;

    for(i = k + 1; i < nb_bloc_m; i++){
      my_dtrsm(CblasColMajor,
               CblasRight,
               CblasUpper,
               CblasNoTrans,
               CblasNonUnit,
               /* m */ (i < nb_bloc_m - 1) ? BLOC_SIZE : m - i * BLOC_SIZE,
               /* n */ BLOC_SIZE,
               /* alpha */ 1,
               /* L\U */ bloc_dgetf2,
	       /* lda */ BLOC_SIZE,
               /* A[i][k] */ a[i + k * nb_bloc_m],
               /* ldb */ BLOC_SIZE);
    }
    int j = 0;
    for(j = k + 1; j < nb_bloc_n; j++){
      my_dtrsm(/*int *Layout*/ CblasColMajor,
                /*int side*/      CblasLeft,
                /*int uplo*/      CblasLower,
                /*int transA*/    CblasNoTrans,
                /*int diag*/      CblasUnit,
                /*int m*/         BLOC_SIZE,
                /*int n*/         (j < nb_bloc_n - 1) ? BLOC_SIZE : n - j * BLOC_SIZE,
                /*double alpha*/  1,
                /*double *a*/     bloc_dgetf2,
                /*int lda*/       BLOC_SIZE,
	        /*double *b*/     a[k + j * nb_bloc_m],
                /*int ldb*/       BLOC_SIZE);
    }
    for(i = k + 1; i < nb_bloc_m; i++){
      for(j = k + 1; j < nb_bloc_n; j++){
        my_dgemm (CblasColMajor,
                     CblasNoTrans,
                     CblasNoTrans,
                     /* m */ (i < nb_bloc_m - 1) ? BLOC_SIZE : m - i * BLOC_SIZE,
                     /* n */ (j < nb_bloc_n - 1) ? BLOC_SIZE : n - j * BLOC_SIZE,
                     /* k */ BLOC_SIZE,
                     /* alpha */ -1.,
                     /* A[i][k] */ a[i + k * nb_bloc_m],
		     /* lda */ BLOC_SIZE,
		     /* B[k][j] */ a[k + j * nb_bloc_m],
		     /* ldb */ BLOC_SIZE,
                     /* beta */ 1.,
		     /* C[i][j] */ a[i + j * nb_bloc_m],
		     /* ldc */ BLOC_SIZE);
      }
    }
  }
  (void)lda;
}
