#ifndef MY_DGETRF
#define MY_DGETRF

#include <mkl.h>

void my_dgetrf_seq(CBLAS_LAYOUT layout,
							 const int m,
		           const int n,
		           double* a,
	             const int lda,
		           int* ipiv);

void my_dgetrf_openmp_gemm(CBLAS_LAYOUT layout,
							 const int m,
		           const int n,
		           double* a,
	             const int lda,
		           int* ipiv);

void my_dgetrf_openmp(CBLAS_LAYOUT layout,
							 const int m,
		           const int n,
		           double* a,
	             const int lda,
		           int* ipiv);

void my_dgetrf_tiled (CBLAS_LAYOUT layout,
		     const int m,
		     const int n,
		     double** a,
	             const int lda,
		     int* ipiv);

void my_dgetrf_tiled_openmp (CBLAS_LAYOUT layout,
			     const int m,
			     const int n,
			     double** a,
			     const int lda,
			     int* ipiv);

void my_pdgetrf_tiled(CBLAS_LAYOUT layout,
							 				const int m,
		           				const int n,
		           				double** a,
	             				const int lda,
		           				int* ipiv,
              				const int dim[2]);
#endif
