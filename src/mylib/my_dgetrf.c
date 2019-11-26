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

	#ifdef PERF
	double performance;
  perf_t start,stop;
	int size[2] = {};
  printf("version, n, Mflops, us\n");
	#endif

  int k = 0;
  for(k = 0; k < min; k++){
    bloc_dgetf2 = a + k * BLOC_SIZE * (lda+1);

		#ifdef PERF
		size[0] = (k < nb_bloc_m - 1) ? BLOC_SIZE : m - k * BLOC_SIZE;
		size[1] = (k < nb_bloc_n - 1) ? BLOC_SIZE : n - k * BLOC_SIZE;
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
		performance = perf_mflops(&stop, (2 * fmin(size[0], size[1]) / 3) * size[0] * size[1]);
		printf("my_dgetf2, %d, %lf, ", k, performance);
		perf_print_time(&stop, 1);
		printf("\n");
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


		#ifdef PERF
		perf(&stop);
		size[0] = (i < nb_bloc_m - 1) ? BLOC_SIZE : m - i * BLOC_SIZE;
		size[1] = (j < nb_bloc_n - 1) ? BLOC_SIZE : n - j * BLOC_SIZE;
		perf_diff(&start, &stop);
		performance = perf_mflops(&stop, 4 * (size[0] * size[0] + size[1] * size[1]) * BLOC_SIZE);
		printf("my_dtrsm, %d, %lf, ", k, performance);
		perf_print_time(&stop, 1);
		printf("\n");
		perf(&start);
		#endif

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

		#ifdef PERF
		perf(&stop);
		perf_diff(&start, &stop);
		performance = perf_mflops(&stop, 2 * (m - k * BLOC_SIZE) * (m - k * BLOC_SIZE) * (m - k * BLOC_SIZE));
		printf("my_dgemm, %d, %lf, ", k, performance);
		perf_print_time(&stop, 1);
		printf("\n");
		#endif
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
