#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <mkl.h>
#include "my_blas.h"

#define UNROLL 7

double my_ddot(const long long int N,
               const double *X,
               const long long int incX,
               const double *Y,
               const long long int incY) {

  long long int i;
  double result = 0;

  if (N < UNROLL) {
    for (i = 0; i < N; i++) {
      result += X[i*incX] * Y[i*incY]; // 2 flops
    }
  } else {
    long long int remaining = N % UNROLL;
    long long int n = N/UNROLL;

    if (incX == 1 && incY == 1) {
      for (i = 0; i < N-remaining; i+=UNROLL) {
	       result  += X[i] * Y[i] +
	         + X[i+1] * Y[i+1]
	         + X[i+2] * Y[i+2]
	         + X[i+3] * Y[i+3]
	         + X[i+4] * Y[i+4]
	         + X[i+5] * Y[i+5]
	         + X[i+6] * Y[i+6];
      }

      for (i = n*UNROLL; i < N; i++) {
	       result += X[i] * Y[i];
      }
    } else {
      for (i = 0; i < N-remaining; i+=UNROLL) {
	       result  += X[i*incX] * Y[i*incY] +
	         + X[(i+1)*incX] * Y[(i+1)*incY]
	         + X[(i+2)*incX] * Y[(i+2)*incY]
	         + X[(i+3)*incX] * Y[(i+3)*incY]
	         + X[(i+4)*incX] * Y[(i+4)*incY]
	         + X[(i+5)*incX] * Y[(i+5)*incY]
	         + X[(i+6)*incX] * Y[(i+6)*incY];
      }
      for (i = n*UNROLL; i < N; i++) {
	       result += X[i*incX] * Y[i*incY];
      }
    }
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

void my_dtrsm2(CBLAS_LAYOUT layout,
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
    int transa = (transA != CblasTrans);//?

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
                  if (unknown_diag) {
                    b[i + j * ldb] /= a[i + i * lda];
                  }
                  for (k = 0; k < i; k++) { // left !trans upper
                      b[i + j * ldb] -= a[k + i * lda] * b[k + j * ldb];
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
      //#pragma omp parallel for collapse(2) private(i, j)
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
    int transa = (transA != CblasTrans);

    if (left_side) {
      if (transa) {
         if (upper) {
           //#pragma omp parallel for private(i, k)
            for (j = 0; j < n; j++) {
                double* b_start = b + j * ldb;
                if (alpha != 1.) {
                   for (i = 0; i < m; i++) {
                       b_start[i] *= alpha;
                   }
                }
                for (k = m-1; k >= 0; k--) { // left trans upper
                   if (unknown_diag) {
                      b_start[k] /= a[k + k * lda];
                   }
                   const double* ak = a + k *lda;
                   for (i = 0; i < k; i++) {
                       b_start[i] -= b_start[k] * ak[i];
                   }
                }
            }
          } else {
          //#pragma omp parallel for private(i, k)
          for (j = 0; j < n; j++) {
              double* b_start = b + j * ldb;
              if (alpha != 1.) {
                  for (i = 0; i < m; i++) {
                    b_start[i] *= alpha;
                  }
                }
                for (k = 0; k < m; k++) { // left trans lower
                    if (unknown_diag) {
                         b_start[k] /= a[k + k * lda];
                    }
                    const double* ak = a + k *lda;
                    for (i = k + 1; i < m; i++) {
                         b_start[i] -= b_start[k] * ak[i];
                    }

               }
          }
         }
       } else {
         if (upper) {//?
            //#pragma omp parallel for private(i, k)
            for (j = 0; j < n; j++) {
              double* b_start = b + j * ldb;
               for (i = 0; i < m; i++) {
		              //if(alpha != 1.){
                      b_start[i] *= alpha;
		              //}
                  if (unknown_diag) {
                    b_start[i] /= a[i + i * lda];
                  }
                  const double* ai = a + i *lda;
                  for (k = 0; k < i; k++) { // left !trans upper
                      b_start[i] -= ai[k] * b_start[k];
                  }
                }
             }
          } else {
            //#pragma omp parallel for private(i, k)
            for (j = 0; j < n; j++) {
              double* b_start = b + j * ldb;
               for (i = m-1; i >= 0; i--) { // left !trans lower
		              //if(alpha != 1.){
		                b_start[i] *= alpha;
		              //}
                    //if(akj != 0.){
                      const double* ai = a + i *lda;
                      for (k = i+1; k < m; k++) {
                        b_start[i] -= ai[k] * b_start[k];
                      }
                    //}
                    if (unknown_diag) {
                     b_start[i] /= a[i + i * lda];
                   }
                }
             }
         }
      }
    } else { // right side
      if (transa) {
         if (upper) {
           //#pragma omp parallel for private(i, k)
            for (j = 0; j < n; j++) {
                double* b_start = b + j * ldb;
                if (alpha != 1.) {
                   for (i = 0; i < m; i++) {
                     b_start[i] *= alpha;
                   }
                }
                for (k = 0; k < j; k++) { // right trans upper
                  double* bk = b + k * ldb;
                  double akj = a[k + j * lda];
                  if(akj != 0.){
                    for (i = 0; i < m; i++) {
                        b_start[i] -= akj * bk[i];
                    }
                  }
               }
               if (unknown_diag) {
                   double ajj = a[j + j * lda];
                   for (i = 0; i < m; i++) {
                     b_start[i] /= ajj;
                   }
               }
             }
         } else {
           //#pragma omp parallel for private(i, k)
            for (j = n-1; j >= 0; j--) {
              double* b_start = b + j * ldb;
                if (alpha != 1.) {
                   for (i = 0; i < m; i++) {
                       b_start[i] *= alpha;
                  }
                }
                for (k = j+1; k < n; k++) { // right trans lower
                  double* bk = b + k * ldb;
                  double akj = a[k + j * lda];
                  if(akj != 0.){
                    for (i = 0; i < m; i++) {
                        b_start[i] -= akj * bk[i];
                    }
                  }
                }
                if (unknown_diag) {
                  double ajj = a[j + j * lda];
                  for (i = 0; i < m; i++) {
                     b_start[i] /= ajj;
                  }
                }
            }
          }
      } else {
          if (upper) {
            //#pragma omp parallel for private(i, j)
            for (k = n-1; k >= 0; k--) {
               double* b_start = b + k * ldb;
               if (unknown_diag) {
                   double akk = a[k + k * lda];
                   for (i = 0; i < m; i++) {
                     b_start[i] /= akk;
                   }
                }
                for (j = 0; j < k; j++) { // right !trans upper
                  double* bj = b + j * ldb;
                  double ajk = a[j + k * lda];
                  if(ajk != 0.){
                    for (i = 0; i < m; i++) {
                        bj[i] -= ajk * b_start[i];
                    }
                  }
               }
               if (alpha != 1.) {
                  for (i = 0; i < m; i++) {
                      b_start[i] *= alpha;
                    }
               }
             }
         } else {
           //#pragma omp parallel for private(i, j)
           for (k = 0; k < n; k++) {
               double* b_start = b + k * ldb;
               if (unknown_diag) {
                 double akk = a[k + k *lda];
                 for (i = 0; i < m; i++) {
                     b_start[i] /= akk;
                  }
               }
               for (j = k+1; j < n; j++) { // right !trans lower
                 double ajk = a[j + k * lda];
                 if(ajk != 0.){
                   double* bj = b + j * ldb;
                   for (i = 0; i < m; i++) {
                     bj[i] -= ajk * b_start[i];
                   }
                 }
               }
               if (alpha != 1.) {
                   for (i = 0; i < m; i++) {
                      b_start[i] *= alpha;
                   }
               }
             }
          }
       }
    }
}

void my_dtrsm_openmp(CBLAS_LAYOUT layout,
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
      #pragma omp parallel for collapse(2)
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
    int transa = (transA != CblasTrans);

    if (left_side) {
      if (transa) {
         if (upper) {
           #pragma omp parallel for private(i, k)
            for (j = 0; j < n; j++) {
                double* b_start = b + j * ldb;
                if (alpha != 1.) {
                   for (i = 0; i < m; i++) {
                       b_start[i] *= alpha;
                   }
                }
                for (k = m-1; k >= 0; k--) { // left trans upper
                   if (unknown_diag) {
                      b_start[k] /= a[k + k * lda];
                   }
                   const double* ak = a + k *lda;
                   for (i = 0; i < k; i++) {
                       b_start[i] -= b_start[k] * ak[i];
                   }
                }
            }
          } else {
          #pragma omp parallel for private(i, k)
          for (j = 0; j < n; j++) {
              double* b_start = b + j * ldb;
              if (alpha != 1.) {
                  for (i = 0; i < m; i++) {
                    b_start[i] *= alpha;
                  }
                }
                for (k = 0; k < m; k++) { // left trans lower
                    if (unknown_diag) {
                         b_start[k] /= a[k + k * lda];
                    }
                    const double* ak = a + k *lda;
                    for (i = k + 1; i < m; i++) {
                         b_start[i] -= b_start[k] * ak[i];
                    }

               }
          }
         }
       } else {
         if (upper) {
            #pragma omp parallel for private(i, k)
            for (j = 0; j < n; j++) {
              double* b_start = b + j * ldb;
               for (i = 0; i < m; i++) {
		              //if(alpha != 1.){
                      b_start[i] *= alpha;
		              //}
                  const double* ai = a + i *lda;
                  for (k = 0; k < i; k++) { // left !trans upper
                      b_start[i] -= ai[k] * b_start[k];
                  }
                  if (unknown_diag) {
                    b_start[i] /= a[i + i * lda];
                  }
                }
             }
          } else {
            #pragma omp parallel for private(i, k)
            for (j = 0; j < n; j++) {
              double* b_start = b + j * ldb;
               for (i = m-1; i >= 0; i--) { // left !trans lower
		              //if(alpha != 1.){
		                b_start[i] *= alpha;
		              //}
                    //if(akj != 0.){
                      const double* ai = a + i *lda;
                      for (k = i+1; k < m; k++) {
                        b_start[i] -= ai[k] * b_start[k];
                      }
                    //}
                    if (unknown_diag) {
                     b_start[i] /= a[i + i * lda];
                   }
                }
             }
         }
      }
    } else { // right side
      if (transa) {
         if (upper) {
           #pragma omp parallel for private(i, k)
            for (j = 0; j < n; j++) {
                double* b_start = b + j * ldb;
                if (alpha != 1.) {
                   for (i = 0; i < m; i++) {
                     b_start[i] *= alpha;
                   }
                }
                for (k = 0; k < j; k++) { // right trans upper
                  double* bk = b + k * ldb;
                  double akj = a[k + j * lda];
                  if(akj != 0.){
                    for (i = 0; i < m; i++) {
                        b_start[i] -= akj * bk[i];
                    }
                  }
               }
               if (unknown_diag) {
                   double ajj = a[j + j * lda];
                   for (i = 0; i < m; i++) {
                     b_start[i] /= ajj;
                   }
               }
             }
         } else {
           #pragma omp parallel for private(i, k)
            for (j = n-1; j >= 0; j--) {
              double* b_start = b + j * ldb;
                if (alpha != 1.) {
                   for (i = 0; i < m; i++) {
                       b_start[i] *= alpha;
                  }
                }
                for (k = j+1; k < n; k++) { // right trans lower
                  double* bk = b + k * ldb;
                  double akj = a[k + j * lda];
                  if(akj != 0.){
                    for (i = 0; i < m; i++) {
                        b_start[i] -= akj * bk[i];
                    }
                  }
                }
                if (unknown_diag) {
                  double ajj = a[j + j * lda];
                  for (i = 0; i < m; i++) {
                     b_start[i] /= ajj;
                  }
                }
            }
          }
      } else { // !trans
          if (upper) {
            #pragma omp parallel for private(i, j)
            for (k = n-1; k >= 0; k--) {
               double* b_start = b + k * ldb;
               if (unknown_diag) {
                   double akk = a[k + k * lda];
                   for (i = 0; i < m; i++) {
                     b_start[i] /= akk;
                   }
                }
                for (j = 0; j < k; j++) { // right !trans upper
                  double* bj = b + j * ldb;
                  double ajk = a[j + k * lda];
                  if(ajk != 0.){
                    for (i = 0; i < m; i++) {
                        bj[i] -= ajk * b_start[i];
                    }
                  }
               }
               if (alpha != 1.) {
                  for (i = 0; i < m; i++) {
                      b_start[i] *= alpha;
                    }
               }
             }
         } else {
           #pragma omp parallel for private(i, j)
           for (k = 0; k < n; k++) {
               double* b_start = b + k * ldb;
               if (unknown_diag) {
                 double akk = a[k + k *lda];
                 for (i = 0; i < m; i++) {
                     b_start[i] /= akk;
                  }
               }
               for (j = k+1; j < n; j++) { // right !trans lower
                 double ajk = a[j + k * lda];
                 if(ajk != 0.){
                   double* bj = b + j * ldb;
                   for (i = 0; i < m; i++) {
                     bj[i] -= ajk * b_start[i];
                   }
                 }
               }
               if (alpha != 1.) {
                   for (i = 0; i < m; i++) {
                      b_start[i] *= alpha;
                   }
               }
             }
          }
       }
    }
}
