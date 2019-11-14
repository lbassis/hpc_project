#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <mkl.h>
#include "my_blas.h"
#include "my_lapack.h"
#include "my_gemm.h"
#include "my_getrf.h"

void my_dgetrf (int m, int n, double* a, int lda, int* ipiv, int info){

  int nb_bloc_x = n / BLOC_SIZE;
  int nb_bloc_y = m / BLOC_SIZE;
  int r_bloc_x = n % BLOC_SIZE;
  int r_bloc_y = m % BLOC_SIZE;

  assert(r_bloc_x == 0);
  assert(r_bloc_y == 0);

  ipiv = malloc(BLOC_SIZE*BLOC_SIZE*sizeof(int));
  int k = 0;
  for(k = 0; k < fmin(nb_bloc_x, nb_bloc_y); k++){
    //my_dgetf2(BLOC_SIZE, BLOC_SIZE, a + k * (BLOC_SIZE + lda), lda, NULL);
    LAPACKE_dgetf2 (CblasColMajor, BLOC_SIZE, BLOC_SIZE, a+k*(BLOC_SIZE+lda), lda, ipiv);
    int i = 0;
    for(i = k + 1; i < fmin(nb_bloc_x, nb_bloc_y); i++){
      cblas_dtrsm (/*int *Layout*/ CblasColMajor,
                /*int side*/ CblasLeft,
                /*int uplo*/ CblasUpper,
                /*int transA*/ CblasNoTrans,
                /*int diag*/ CblasNonUnit,
                /*int m*/ BLOC_SIZE,
                /*int n*/ BLOC_SIZE,
                /*double alpha*/ 1,
                /*double *a*/ a + k * (BLOC_SIZE + lda),
                /*int lda*/ lda,
                /*double *b*/ a + i * BLOC_SIZE + k * lda,
                /*int ldb*/ lda);
    }
    int j = 0;
    for(j = k + 1; j < fmin(nb_bloc_x, nb_bloc_y); j++){
      cblas_dtrsm (/*int *Layout*/ CblasColMajor,
                /*int side*/ CblasRight,
                /*int uplo*/ CblasLower,
                /*int transA*/ CblasNoTrans,
                /*int diag*/ CblasUnit,
                /*int m*/ BLOC_SIZE,
                /*int n*/ BLOC_SIZE,
                /*double alpha*/ 1,
                /*double *a*/ a + k * (BLOC_SIZE + lda),
                /*int lda*/ lda,
                /*double *b*/a + k * BLOC_SIZE + j * lda,
                /*int ldb*/ lda);
    }
    for(i = k + 1; i < fmin(nb_bloc_x, nb_bloc_y); i++){
      for(j = k + 1; j < fmin(nb_bloc_x, nb_bloc_y); j++){
        cblas_dgemm (CblasColMajor, CblasNoTrans, CblasNoTrans, BLOC_SIZE, BLOC_SIZE, BLOC_SIZE, 
		     -1., a+i*BLOC_SIZE+k*lda, lda, a+k*BLOC_SIZE+j*lda, lda, 1., a+j*BLOC_SIZE+i, lda);
	//my_dgemm(0, 0, BLOC_SIZE, BLOC_SIZE, BLOC_SIZE, -1, a + i * BLOC_SIZE + k * lda, lda, a + k * BLOC_SIZE + j * lda, lda, 1, a + j * BLOC_SIZE + i * lda, lda);
      }
    }
  }
  /* last bloc*/
  //my_dgetf2(BLOC_SIZE, BLOC_SIZE, a + k * (BLOC_SIZE + lda), lda, NULL);


}
