#include <stdlib.h>
#include "ddot.h"

double my_ddot(const int N, const double *X, const int incX, const double *Y, const int incY) {
  
  int i;
  double result = 0;
  for (i = 0; i < N; i++) {
    result += X[i*incX] * Y[i*incY]; // 4 flops + 3 loads + 1 store
  }

  return result;
}

double *my_dgemm_scalaire(int m, double *a, double *b) {
 
  int i, j, k;
  double *c = calloc(sizeof(double), m*m);
  
  for (i = 0; i < m; i++) { //
    for (j = 0; j < m; j++) { // 
      for (k = 0; k < m; k++) { // 
	c[i+m*j] += a[k+m*i]*b[k+m*j];
      }
    }
  }

  return c;
}
