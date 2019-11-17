#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mkl_cblas.h>

#include "util.h"
#include "my_lib.h"

#define TEST_MAT(NAME, N, M, INIT, ...) void test_ ## NAME ## _(void){                        \
                                          double *a = alloc_mat(N, M), *b = alloc_mat(N, M);  \
                                          INIT_MAT(N, M, INIT)                                \
                                          CHECK(__VA_ARGS__)                                  \
                                          free(a);                                            \
                                          free(b);                                            \
                                        }

#define TEST_VEC(NAME, N, INIT, ...) void test_ ## NAME ## _(void){                           \
                                        double *a = alloc_vec(N), *b = alloc_vec(N);          \
                                        INIT_VEC(N, INIT)                                     \
                                        CHECK(__VA_ARGS__)                                    \
                                        free(a);                                              \
                                        free(b);                                              \
                                      }



#define INIT_MAT(N, M, ...) {int j = 0;                   \
                             for(j = 0; j < M; j++)       \
                                INIT_VEC(N, __VA_ARGS__)  \
                            }

#define INIT_VEC(N, ...) {int i = 0;              \
                          for(i = 0; i < N; i++)  \
                            __VA_ARGS__;          \
                         }


#define CHECK(...) if(fabs(my_ddot(__VA_ARGS__) - cblas_ddot(__VA_ARGS__)) > ESP) { \
                      printf("[FAILED] %s\n", __func__);                            \
                      nb_fail++;                                                    \
                    }

#define CALL(NAME) test_ ## NAME ## _();

#define ESP 1.11e-16
#define SIZE 100
#define LDA 10






int nb_fail = 0;


TEST_VEC(ddot_zero, SIZE, a[i] = 1; b[i] = 0, SIZE, a, 1, a, 1)

TEST_VEC(ddot_sum_of_i, SIZE, a[i] = i, SIZE, a, 1, a, 1)

TEST_VEC(ddot_with_lda_vec, LDA, a[i] = 1 / (i+1); b[i] = i * i, LDA/2, a, LDA/2, b, 1)

TEST_MAT(ddot_with_lda_mat, LDA, 2, a[LDA * j + i] = i, LDA/2, a, LDA/2, a + LDA * sizeof(double), LDA/2)


int main(void){
  printf("Tests :\n");

  CALL(ddot_zero)
  CALL(ddot_sum_of_i)
  CALL(ddot_with_lda_vec)
  CALL(ddot_with_lda_mat)

  if(nb_fail == 0){
    printf("[SUCCESS]\n");
  }

  return 0;
}
