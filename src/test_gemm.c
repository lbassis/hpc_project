#include <stdio.h>
#include <stdlib.h>
#include <mkl_cblas.h>
#include <mkl.h>
#include "util.h"
#include "ddot.h"
#include "perf.h"

#ifndef SIZE
#define SIZE 3
#endif

int main(void){

  double *a, *b, *c;
  double alpha = 1.;
  double beta = 0.;
  int i;

  a = alloc_mat(SIZE, SIZE);
  b = alloc_mat(SIZE, SIZE);
  c = alloc_mat(SIZE, SIZE);

  init_random(SIZE, SIZE, &a, 1);
  //  init_random(SIZE, SIZE, &c, 2);
  init_identity(SIZE, SIZE, &c);

  /* a = b */
  for (i = 0; i < SIZE*SIZE; i++) {
    b[i] = a[i];
  }

  /* a = a*c */
  my_dgemm_scalaire(SIZE, a, c, a);
  //my_dgemm(0, 0, SIZE, SIZE, SIZE, alpha, a, SIZE, c, SIZE, beta, a, SIZE);

  /* b = b*c */
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, SIZE, SIZE, SIZE, alpha, b, SIZE, c, SIZE, beta, b, SIZE);

  /* print a-b */
  for (i = 0; i < SIZE*SIZE; i++) {
    b[i] -= a[i];
  }
  affiche(SIZE, SIZE, a, SIZE, stdout);

  return 0;
}
