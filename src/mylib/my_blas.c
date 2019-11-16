#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <mkl.h>
#include "my_blas.h"

double my_ddot(const int N,
               const double *X,
               const int incX,
               const double *Y,
               const int incY) {

  int i;
  double result = 0;
  for (i = 0; i < N; i++) {
    result += X[i*incX] * Y[i*incY]; // 2 flops
  }

  return result;
}

void my_daxpy (const int n,
               const double alpha,
               const double *x,
               const int incX,
               double *y,
               const int incY) {

  int i;
  for (i = 0; i < n; i++) {
    y[i*incY] += alpha*x[i*incX];
  }

}

void my_dgemv(CBLAS_LAYOUT layout,
              CBLAS_TRANSPOSE TransA,
              const int m,
              const int n,
              const double alpha,
              const double *A,
              const int lda,
              const double *X,
              const int incX,
              const double beta,
              double *Y,
              const int incY){

  assert(layout == CblasColMajor);

  int i, j;
  int transA = (TransA == CblasTrans);

  if (transA) {
    for (i = 0; i < n; i++) {
      double tmp = 0;
      for (j = 0; j < m; j++) {
	       tmp += A[j+lda*i]*X[j*incX];
      }
      Y[i*incY] = Y[i*incY]*beta + tmp*alpha;
    }
  } else {
    for (i = 0; i < n; i++) {
      double tmp = 0;
      for (j = 0; j < m; j++) {
	       tmp += A[i+lda*j]*X[j*incX];
      }
      Y[i*incY] = Y[i*incY]*beta + tmp*alpha;
    }
  }
}


void my_dger(CBLAS_LAYOUT layout,
             const int m,
             const int n,
             const double alpha,
             const double *X,
             const int incX,
             const double *Y,
             const int incY,
             double *A,
             const int lda) {

  assert(layout == CblasColMajor);

  int i, j;
  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      A[i*lda+j] += X[i*incX]*Y[j*incY]*alpha;
    }
  }
}

void my_dtrsm(CBLAS_LAYOUT layout,
              CBLAS_SIDE side,
              CBLAS_UPLO uplo,
              CBLAS_TRANSPOSE transA,
              CBLAS_DIAG diag,
              const int m,
              const int n,
              const double alpha,
              const double *a,
              const int lda,
              double *b,
              const int ldb){

    assert(layout == CblasColMajor);

    if (n <= 0 || m <= 0) {
      fprintf(stderr, "my_dtrsm matrix dimentions invalid\n");
      return;
    }

    int i = 0, j = 0, k = 0;
    if (alpha == 0.) {
      for (j = 0; j < n; ++j) {
        for (i = 0; i < m; ++i) {
          b[i + j * ldb] = 0.;
        }
      }
      return;
    }

    int left_side = (side == CblasLeft);
    int upper = (uplo == CblasUpper);
    int unknown_diag = (diag == CblasNonUnit);
    int transa = (transA == CblasTrans);

    if (left_side) {
      if (transa) {
         if (upper) {
            for (j = 0; j < n; j++) {
                if (alpha != 1.) {
                   for (i = 0; i < m; i++) {
                       b[i + j * ldb] *= alpha;
                   }
                }
                for (k = m-1; k >= 0; k--) { // left trans upper
                   if (unknown_diag) {
                      b[k + j * ldb] /= a[k + k * lda];
                   }
                   for (i = 0; i < k; i++) {
                       b[i + j * ldb] -= b[k + j * ldb] * a[i + k * lda];
                   }
                }
            }
          } else {
          for (j = 0; j < n; j++) {
              if (alpha != 1.) {
                  for (i = 0; i < m; i++) {
                    b[i + j * ldb] *= alpha;
                  }
                }
                for (k = 0; k < m; k++) { // left trans lower
                    if (unknown_diag) {
                         b[k + j * ldb] /= a[k + k * lda];
                    }
                    for (i = k + 1; i < m; i++) {
                         b[i + j * ldb] -= b[k + j * ldb] * a[i + k * lda];
                    }

               }
          }
         }
       } else {
         if (upper) {
            for (j = 0; j < n; j++) {
               for (i = 0; i < m; i++) {
		              //if(alpha != 1.){
                      b[i + j * ldb] *= alpha;
		              //}
                  for (k = 0; k < i; k++) { // left !trans upper
                      b[i + j * ldb] -= a[k + i * lda] * b[k + j * ldb];
                  }
                  if (unknown_diag) {
                    b[i + j * ldb] /= a[i + i * lda];
                  }
                }
             }
          } else {
            for (j = 0; j < n; j++) {
               for (i = m-1; i >= 0; i--) { // left !trans lower
		              //if(alpha != 1.){
		                b[i + j * ldb] *= alpha;
		              //}
                  for (k = i+1; k < m; k++) {
                      b[i + j * ldb] -= a[k + i * lda] * b[k + j * ldb];
                    }
                    if (unknown_diag) {
                     b[i + j * ldb] /= a[i + i * lda];
                   }
                }
             }
         }
      }
    } else {
      if (transa) {
         if (upper) {
            for (j = 0; j < n; j++) {
                if (alpha != 1.) {
                   for (i = 0; i < m; i++) {
                     b[i + j * ldb] *= alpha;
                   }
                }
               for (k = 0; k < j; k++) { // right trans upper
                  for (i = 0; i < m; i++) {
                      b[i + j * ldb] -= a[k + j * lda] * b[i + k * ldb];
                  }
               }
               if (unknown_diag) {
                   for (i = 0; i < m; i++) {
                     b[i + j * ldb] /= a[j + j * lda];
                   }
               }
             }
         } else {
            for (j = n-1; j >= 0; j--) {
                if (alpha != 1.) {
                   for (i = 0; i < m; i++) {
                       b[i + j * ldb] *= alpha;
                  }
                }
                for (k = j+1; k < n; k++) { // right trans lower
                    for (i = 0; i < m; i++) {
                        b[i + j * ldb] -= a[k + j * lda] * b[i + k * ldb];
                    }
                }
                if (unknown_diag) {
                  for (i = 0; i < m; i++) {
                     b[i + j * ldb] /= a[j + j * lda];
                  }
                }
            }
          }
      } else {
          if (upper) {
            for (k = n-1; k >= 0; k--) {
               if (unknown_diag) {
                   for (i = 0; i < m; i++) {
                     b[i + k * ldb] /= a[k + k * lda];
                   }
                }
                for (j = 0; j < k; j++) { // right !trans upper
                   for (i = 0; i < m; i++) {
                      b[i + j * ldb] -= a[j + k * lda] * b[i + k * ldb];
                  }
               }
               if (alpha != 1.) {
                  for (i = 0; i < m; i++) {
                      b[i + k * ldb] *= alpha;
                    }
               }
             }
         } else {
            for (k = 0; k < n; k++) {
               if (unknown_diag) {
                 for (i = 0; i < m; i++) {
                     b[i + k * ldb] /= a[k + k *lda];
                  }
               }
               for (j = k+1; j < n; j++) { // right !trans lower
                  for (i = 0; i < m; i++) {
                     b[i + j * ldb] -= a[j + k * lda] * b[i + k * ldb];
                  }
               }
               if (alpha != 1.) {
                   for (i = 0; i < m; i++) {
                      b[i + k * ldb] *= alpha;
                   }
               }
             }
          }
       }
    }
}
