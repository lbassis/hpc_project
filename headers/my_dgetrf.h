#ifndef MY_DGETRF
#define MY_DGETRF

#include <mkl.h>

void my_dgetrf(CBLAS_LAYOUT layout,
							 const int m,
		           const int n,
		           double* a,
	             const int lda,
		           int* ipiv);
#endif
