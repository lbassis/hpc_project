#include <stdio.h>
#include <stdlib.h>
#include <mkl_cblas.h>
#include <mkl.h>
#include "util.h"
#include "ddot.h"
#include "perf.h"

#ifndef SIZE
#define SIZE 20
#endif

int main(void){

  double *a, *b;
  int i;
  long long ipiv[SIZE] = {};

  a = alloc_mat(SIZE, SIZE);
  b = alloc_mat(SIZE, SIZE);
  init_random(SIZE, SIZE, &a, 1);
  for (i = 0; i < SIZE*SIZE; i++) {
    b[i] = a[i];
  }

  my_dgetf2(SIZE, SIZE, a, SIZE, NULL);
  LAPACKE_dgetf2(LAPACK_COL_MAJOR, SIZE, SIZE, b, SIZE, ipiv);

  for (i = 0; i < SIZE*SIZE; i++) {
    b[i] -= a[i];
  }

  affiche(SIZE, SIZE, b, SIZE, stdout);
  return 0;
}
