#include <stdio.h>
#include <stdlib.h>
#include <mkl_cblas.h>
#include <mkl.h>

#include "util.h"
#include "ddot.h"
#include "perf.h"

#define INIT_VEC(N, ...) {int i = 0;              \
                          for(i = 0; i < N; i++)  \
                            __VA_ARGS__;          \
                         }
#ifndef SIZE
#define SIZE 5
#endif

int main(void){
  double a[SIZE * SIZE] = {};
  double b[SIZE * SIZE] = {};
  INIT_VEC(SIZE * SIZE, a[i] = b[i] = i+1;)

  affiche(SIZE, SIZE, a, SIZE, stdout);
  printf("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
  affiche(SIZE, SIZE, b, SIZE, stdout);
  printf("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");


  my_dgetf2(SIZE, SIZE, a, SIZE, NULL);
  affiche(SIZE, SIZE, a, SIZE, stdout);
  printf("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
  long long ipiv[SIZE] = {};
  LAPACKE_dgetf2(LAPACK_COL_MAJOR, SIZE, SIZE, b, SIZE, ipiv);

  affiche(SIZE, SIZE, b, SIZE, stdout);
  return 0;
}
