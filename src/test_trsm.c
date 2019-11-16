#include <stdio.h>
#include <stdlib.h>
#include <mkl_cblas.h>
#include <mkl.h>
#include "util.h"
#include "my_lapack.h"
#include "perf.h"
#include "my_blas.h"

#ifndef SIZE
#define SIZE 10
#endif

int main(void){

  double *a, *b, *c;
  int i, j;

  a = alloc_mat(SIZE, SIZE);
  b = alloc_mat(SIZE, SIZE);
  c = alloc_mat(SIZE, SIZE);
  init_random(SIZE, SIZE, a, 2);
  init_random(SIZE, SIZE, b, 1);
  for (i = 0; i < SIZE; i++) {
    for(j = 0; j < SIZE; j++){
      c[i + j * SIZE] = b[i + SIZE * j];
    }
  }
  affiche(SIZE, SIZE, b, SIZE, stdout);
  printf("_______\n");
  my_dtrsm(/*int *Layout*/   CblasColMajor,
            /*int side*/      CblasLeft,
            /*int uplo*/      CblasUpper,
            /*int transA*/    CblasTrans,
            /*int diag*/      CblasUnit,
            /*int m*/         SIZE,
            /*int n*/         SIZE,
            /*double alpha*/  1.0,
            /*double *a*/     a,
            /*int lda*/       SIZE,
            /*double *b*/     b,
            /*int ldb*/       SIZE);
  affiche(SIZE, SIZE, b, SIZE, stdout);
  printf("_______\n");
  cblas_dtrsm(/*int *Layout*/ CblasColMajor,
            /*int side*/      CblasLeft,
            /*int uplo*/      CblasUpper,
            /*int transA*/    CblasTrans,
            /*int diag*/      CblasUnit,
            /*int m*/         SIZE,
            /*int n*/         SIZE,
            /*double alpha*/  1.0,
            /*double *a*/     a,
            /*int lda*/       SIZE,
            /*double *b*/     c,
            /*int ldb*/       SIZE);

  for (i = 0; i < SIZE*SIZE; i++) {
    b[i] -= c[i];
  }

  affiche(SIZE, SIZE, b, SIZE, stdout);
  free(a);
  free(b);
  free(c);
  return 0;
}
