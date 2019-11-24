#ifndef MY_DGEMM_H
#define MY_DGEMM_H

#include <mkl.h>

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
		       const int ldc);

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
			   const int ldc);

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
			   const int ldc);

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
			   const int ldc);

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
                  const int ldc);

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
                  const int ldc);

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
                  const int ldc);

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
                  const int ldc);

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
		   						const int ldc);


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
								  const int bloc_size);


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
                  const int ldc);

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
                  const int ldc);

#endif
