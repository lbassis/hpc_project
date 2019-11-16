#include <stdio.h>
#include <stdlib.h>
#include <mkl_cblas.h>
#include <mkl.h>

#include "util.h"
#include "perf.h"
#include "my_lib.h"


#ifndef SIZE
#define SIZE 3
#endif

int main(void){

  double *a, *b, *c, *d;
  double alpha = 1.;
  double beta = 0.;
  int i;

  a = alloc_mat(SIZE, SIZE);
  b = alloc_mat(SIZE, SIZE);
  c = alloc_mat(SIZE, SIZE);
  d = alloc_mat(SIZE, SIZE);

  init_random(SIZE, SIZE, &a, 1);
  init_random(SIZE, SIZE, &c, 3);

  /* a = b */
  for (i = 0; i < SIZE*SIZE; i++) {
    b[i] = a[i];
  }

  /* d = a*c */
  my_dgemm_scalaire(SIZE, a, c, d);

  /* b = b*c */
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, SIZE, SIZE, SIZE, alpha, b, SIZE, c, SIZE, beta, b, SIZE);

  /* print b-d */
  /* for (i = 0; i < SIZE*SIZE; i++) { */
  /*   b[i] -= d[i]; */
  /* } */
  /* affiche(SIZE, SIZE, d, SIZE, stdout); */
  /* printf("__________\n"); */

  printf("result lapack:\n");
  affiche(SIZE, SIZE, b, SIZE, stdout);
  printf("my result:\n");
  affiche(SIZE, SIZE, d, SIZE, stdout);


  return 0;
}
