#include "ddot.h"

double my_ddot(const int N, const double *X, const int incX, const double *Y, const int incY) {
  
  int i;
  double result = 0;
  for (i = 0; i < N; i++) {
    result += X[i*incX] * Y[i*incY]; // 4 flops + 3 loads + 1 store
  }

  return result;
}
