#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <mkl.h>
#include <omp.h>
#include "my_blas.h"
#include "my_lapack.h"
#include "my_dgemm.h"
#include "my_dgetrf.h"


#ifdef PERF
#include "perf.h"
#endif


#ifndef BLOC_SIZE
#define BLOC_SIZE 130

#endif




void my_dgetrf_seq(CBLAS_LAYOUT layout,
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

	#ifdef PERF
  perf_t start, stop, total_getf2, total_dtrsm, total_gemm;
	#endif

  int k = 0;
  for(k = 0; k < min; k++){
    bloc_dgetf2 = a + k * BLOC_SIZE * (lda+1);

		#ifdef PERF
		perf(&start);
		#endif

    my_dgetf2(CblasColMajor,
	      (k < nb_bloc_m - 1) ? BLOC_SIZE : m - k * BLOC_SIZE,
              (k < nb_bloc_n - 1) ? BLOC_SIZE : n - k * BLOC_SIZE,
              bloc_dgetf2,
              lda,
              NULL);

		#ifdef PERF
		perf(&stop);
		perf_diff(&start, &stop);
		perf_add(&stop, &total_getf2);
		perf(&start);
		#endif

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

		#ifdef PERF
		perf(&stop);
		perf_diff(&start, &stop);
		perf_add(&stop, &total_dtrsm);
		perf(&start);
		#endif

    for(i = k + 1; i < nb_bloc_m; i++){
      for(j = k + 1; j < nb_bloc_n; j++){
        my_dgemm_bloc (CblasColMajor,
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

		#ifdef PERF
		perf(&stop);
		perf_diff(&start, &stop);
		perf_add(&stop, &total_gemm);
		#endif
  }

	#ifdef PERF
	printf("getf2, seq, ");
	perf_print_time(&total_getf2, 1);
	printf("\n");
	printf("trsm, seq, ");
	perf_print_time(&total_dtrsm, 1);
	printf("\n");
	printf("gemm, seq, ");
	perf_print_time(&total_gemm, 1);
	printf("\n");
	#endif
}



void my_dgetrf_openmp(CBLAS_LAYOUT layout,
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

	#ifdef PERF
  perf_t start, stop, total_getf2, total_dtrsm, total_gemm;
	#endif

  int k = 0;
  for(k = 0; k < min; k++){
    bloc_dgetf2 = a + k * BLOC_SIZE * (lda+1);

		#ifdef PERF
		perf(&start);
		#endif

    my_dgetf2(CblasColMajor,
						 (k < nb_bloc_m - 1) ? BLOC_SIZE : m - k * BLOC_SIZE,
              (k < nb_bloc_n - 1) ? BLOC_SIZE : n - k * BLOC_SIZE,
              bloc_dgetf2,
              lda,
              NULL);


		#ifdef PERF
		perf(&stop);
		perf_diff(&start, &stop);
		perf_add(&stop, &total_getf2);
		perf(&start);
		#endif

		int i = 0;
		#pragma omp parallel for private(i)
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
		#pragma omp parallel for private(j)
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


		#ifdef PERF
		perf(&stop);
		perf_diff(&start, &stop);
		perf_add(&stop, &total_dtrsm);
		perf(&start);
		#endif

		#pragma omp parallel for collapse(2) private(i, j)
    for(i = k + 1; i < nb_bloc_m; i++){
      for(j = k + 1; j < nb_bloc_n; j++){
    my_dgemm_bloc (CblasColMajor,
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

		#ifdef PERF
		perf(&stop);
		perf_diff(&start, &stop);
		perf_add(&stop, &total_gemm);
		#endif
  }
	#ifdef PERF
	printf("getf2, omp, ");
	perf_print_time(&total_getf2, 1);
	printf("\n");
	printf("trsm, omp, ");
	perf_print_time(&total_dtrsm, 1);
	printf("\n");
	printf("gemm, omp, ");
	perf_print_time(&total_gemm, 1);
	printf("\n");
	#endif
}



void my_dgetrf_openmp_gemm(CBLAS_LAYOUT layout,
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
		#pragma omp parallel for collapse(2) private(i, j)
    for(i = k + 1; i < nb_bloc_m; i++){
      for(j = k + 1; j < nb_bloc_n; j++){
        my_dgemm_bloc (CblasColMajor,
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


void my_dgetrf_omp_trsm_gemm2(CBLAS_LAYOUT layout,
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
      my_dtrsm_openmp(CblasColMajor,
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
      my_dtrsm_openmp(/*int *Layout*/ CblasColMajor,
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
		//#pragma omp parallel for collapse(2) private(i, j)
    for(i = k + 1; i < nb_bloc_m; i++){
      for(j = k + 1; j < nb_bloc_n; j++){
        my_dgemm_bloc_openmp (CblasColMajor,
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

void my_dgetrf_tiled(CBLAS_LAYOUT layout,
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
        my_dgemm_bloc (CblasColMajor,
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

void my_dgetrf_tiled_openmp(CBLAS_LAYOUT layout,
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

#pragma omp parallel for private(i)
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
#pragma omp parallel for private(j) 
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
#pragma omp parallel for collapse(2) private(i, j)
    for(i = k + 1; i < nb_bloc_m; i++){
      for(j = k + 1; j < nb_bloc_n; j++){
        my_dgemm_bloc (CblasColMajor,
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
